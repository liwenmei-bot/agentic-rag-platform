import { defineStore } from 'pinia'
import * as api from '../api/client'

function makeBaseMessage(role, content = '') {
  return {
    role,
    content,
    sources: [],
    files: [],
    toolSteps: [],

    // Knowledge-base RAG trace
    ragSteps: [],
    retrievalInfo: null,
    originalQuestion: '',
    pipelineStatus: '',

    // Planner / Actor / Reflector explicit trace
    planner: null,
    actorTrace: [],
    reflection: null,
    reflectionHistory: [],
  }
}

function actionLabel(action) {
  const map = {
    direct_answer: 'Direct Answer',
    vector_retrieval: 'Vector Retrieval / Chroma',
    graph_retrieval: 'Graph Retrieval / Neo4j',
    context_judge: 'Reflector / Context Judge',
    answer_generation: 'Answer Generation',
  }
  return map[action] || action
}

function routeName(route) {
  return String(route || 'unknown').toUpperCase()
}

function buildRagSteps(message) {
  const info = message.retrievalInfo
  const planner = message.planner || info?.planner || null
  const actorTrace = message.actorTrace?.length
    ? message.actorTrace
    : (info?.actor_trace || [])
  const reflectionHistory = message.reflectionHistory?.length
    ? message.reflectionHistory
    : (info?.reflection_history || [])

  if (!info && !planner && actorTrace.length === 0 && reflectionHistory.length === 0) {
    return []
  }

  const steps = []
  const route = info?.route || planner?.route || 'unknown'

  // 1. Router
  steps.push({
    key: 'router',
    label: 'Agent Router',
    status: 'done',
    detail: `已选择 ${routeName(route)} 路线。${info?.route_reason ? ` ${info.route_reason}` : ''}`,
  })

  // 2. Planner
  if (planner) {
    const planLabels = (planner.steps || [])
      .map((step) => step.label || actionLabel(step.action))
      .filter(Boolean)
      .join(' → ')

    steps.push({
      key: 'planner',
      label: 'Planner',
      status: 'done',
      detail: planLabels
        ? `执行计划：${planLabels}`
        : `已根据 ${routeName(route)} 路线生成执行计划。`,
    })
  }

  if (info?.skipped_retrieval) {
    steps.push({
      key: 'actor-direct',
      label: 'Actor',
      status: 'done',
      detail: 'DIRECT 路线不访问 Chroma / Neo4j，直接执行回答生成。',
    })

    steps.push({
      key: 'reflector-direct',
      label: 'Reflector',
      status: 'done',
      detail: 'DIRECT 路线无需知识库证据，允许直接回答。',
    })

    steps.push({
      key: 'generate',
      label: 'Answer Generation',
      status: 'running',
      detail: '正在生成回答。',
    })

    return steps
  }

  // 3. Actor - every real retrieval attempt
  actorTrace.forEach((trace, index) => {
    const actions = (trace.actions || [])
      .map(actionLabel)
      .join(' + ')

    const evidence = [
      `Vector hits: ${trace.vector_hit_count ?? 0}`,
      `Graph: ${trace.graph_context_available ? 'YES' : 'NO'}`,
    ].join(' · ')

    steps.push({
      key: `actor-${trace.attempt || index + 1}`,
      label: `Actor · 第 ${trace.attempt || index + 1} 轮执行`,
      status: 'done',
      detail: `${actions || routeName(trace.route)}；${evidence}`,
    })
  })

  // Backward compatibility: if only retrieval_info exists, still show Actor.
  if (actorTrace.length === 0 && info?.completed) {
    steps.push({
      key: 'actor-fallback',
      label: 'Actor',
      status: 'done',
      detail: `已按 ${routeName(route)} 路线执行检索。`,
    })
  }

  // 4. Reflector - every context judgment
  reflectionHistory.forEach((reflection, index) => {
    steps.push({
      key: `reflector-${reflection.round_index ?? index + 1}-${index}`,
      label: `Reflector · 第 ${reflection.round_index ?? index + 1} 轮反思`,
      status: reflection.sufficient ? 'done' : 'warning',
      detail: reflection.reason || (
        reflection.sufficient
          ? '当前证据充分，可以进入回答生成。'
          : '当前证据不足，需要调整查询或安全拒答。'
      ),
    })
  })

  // Backward compatibility: if no explicit reflector event exists.
  if (reflectionHistory.length === 0 && info?.completed) {
    steps.push({
      key: 'reflector-fallback',
      label: 'Reflector / Context Judge',
      status: info.context_sufficient === false ? 'warning' : 'done',
      detail: info.context_sufficient === false
        ? '当前证据不足。'
        : '当前证据满足回答条件。',
    })
  }

  // 5. Query Rewrite
  if (info?.rewrite_attempted) {
    steps.push({
      key: 'rewrite',
      label: 'Query Rewrite',
      status: info.rewrite_candidate ? 'done' : 'warning',
      detail: info.rewrite_candidate || '已尝试查询改写，但没有生成可用候选查询。',
    })

    if (info.rewritten_query) {
      steps.push({
        key: 'rewrite-adopt',
        label: 'Rewrite Adoption',
        status: 'done',
        detail: `已采用改写查询：${info.rewritten_query}`,
      })
    } else {
      steps.push({
        key: 'rewrite-adopt',
        label: 'Rewrite Adoption',
        status: 'neutral',
        detail: '候选改写未优于原查询结果，最终保留原检索上下文。',
      })
    }
  }

  // 6. Final answer
  steps.push({
    key: 'generate',
    label: 'Answer Generation',
    status: 'running',
    detail: '正在根据最终证据生成回答。',
  })

  return steps
}

function rebuildRagSteps(message) {
  message.ragSteps = buildRagSteps(message)
}

function finishGenerateStep(message) {
  const step = message.ragSteps.find((item) => item.key === 'generate')
  if (step) {
    step.status = 'done'
    step.detail = '回答生成完成。'
  }
}

export const useChatStore = defineStore('chat', {
  state: () => ({
    sessions: [],
    currentSessionId: null,
    messages: [],
    isStreaming: false,
    uploadedFiles: [],
    // knowledge = Basic RAG; agentic = Planner/Actor/Reflector; tools = legacy tool Agent
    chatMode: 'knowledge',
  }),

  actions: {
    async loadSessions() {
      this.sessions = await api.listSessions()

      if (!this.currentSessionId && this.sessions.length > 0) {
        await this.switchSession(this.sessions[0].id)
      } else if (this.sessions.length === 0) {
        await this.startNewSession()
      }
    },

    async startNewSession() {
      const session = await api.createSession()
      this.sessions.unshift(session)
      this.currentSessionId = session.id
      this.messages = []
    },

    async switchSession(sessionId) {
      if (this.isStreaming) return
      this.currentSessionId = sessionId
      const rawMessages = await api.getSessionMessages(sessionId)

      this.messages = rawMessages.map((m) => {
        const message = makeBaseMessage(m.role, m.content)
        let sources = []
        let files = []

        if (m.sources) {
          try {
            const parsed = JSON.parse(m.sources)

            if (Array.isArray(parsed)) {
              sources = parsed
            } else if (parsed && parsed.files) {
              files = parsed.files.map((f) => ({
                filename: f.filename,
                downloadUrl: `/files/${sessionId}/${f.filename}`,
              }))
            }
          } catch (e) {
            // 历史记录中 sources 解析失败时保持空数组。
          }
        }

        message.sources = sources
        message.files = files

        // 当前 session 历史仍只保存回答和 sources；
        // Planner / Actor / Reflector Trace 暂不持久化。
        return message
      })
    },

    async removeSession(sessionId) {
      await api.deleteSession(sessionId)
      this.sessions = this.sessions.filter((s) => s.id !== sessionId)

      if (this.currentSessionId === sessionId) {
        this.currentSessionId = null
        this.messages = []

        if (this.sessions.length > 0) {
          await this.switchSession(this.sessions[0].id)
        } else {
          await this.startNewSession()
        }
      }
    },

    async sendMessage(question) {
      if (!question.trim() || this.isStreaming) return
      if (!this.currentSessionId) await this.startNewSession()
      const sessionId = this.currentSessionId
      const mode = this.chatMode

      const userMessage = makeBaseMessage('user', question)
      this.messages.push(userMessage)

      const assistantMessage = makeBaseMessage('assistant')
      assistantMessage.originalQuestion = question
      this.messages.push(assistantMessage)

      this.isStreaming = true

      try {
        if (mode === 'knowledge') {
          // Basic RAG：只做一次 Chroma 检索，不展示 Agent Trace。
          await api.streamBasicChat(sessionId, question, (event) => {
            if (event.type === 'stage') {
              assistantMessage.pipelineStatus = '正在检索知识库…'
            } else if (event.type === 'memory') {
              assistantMessage.pipelineStatus = '正在检索知识库…'
            } else if (event.type === 'basic_info') {
              assistantMessage.pipelineStatus = '正在生成回答…'
            } else if (event.type === 'done') {
              assistantMessage.pipelineStatus = ''
            }
            if (event.type === 'sources') {
              assistantMessage.sources = event.data || []
            } else if (event.type === 'content') {
              assistantMessage.content += event.data || ''
            }
          })

        } else if (mode === 'agentic') {
          // Agentic RAG：Router V5 -> Planner -> Actor -> Reflector -> Rewrite/Retry。
          await api.streamAgenticChat(sessionId, question, (event) => {
            if (event.type === 'stage') {
              const labels = {
                memory: '正在读取会话上下文…',
                router: '正在选择检索路线…',
                actor: `正在执行第 ${event.data?.attempt || 1} 轮检索…`,
                reflector: '正在判断证据是否充分…',
                rewrite: '正在改写检索问题…',
                answer: '正在生成回答…',
              }
              assistantMessage.pipelineStatus = labels[event.data?.stage] || ''
            } else if (event.type === 'router') {
              assistantMessage.retrievalInfo = { ...event.data }
              rebuildRagSteps(assistantMessage)
            } else if (event.type === 'memory') {
              assistantMessage.resolvedQuestion = event.data?.resolved_question
            } else if (event.type === 'rewrite') {
              assistantMessage.retrievalInfo = {
                ...assistantMessage.retrievalInfo,
                rewrite_attempted: true,
                rewrite_candidate: event.data?.candidate,
              }
              rebuildRagSteps(assistantMessage)
            } else if (event.type === 'planner') {
              assistantMessage.planner = event.data || null
              rebuildRagSteps(assistantMessage)

            } else if (event.type === 'actor') {
              if (event.data) {
                assistantMessage.actorTrace.push(event.data)
              }
              rebuildRagSteps(assistantMessage)

            } else if (event.type === 'reflector') {
              if (event.data) {
                assistantMessage.reflection = event.data
                assistantMessage.reflectionHistory.push(event.data)
              }
              rebuildRagSteps(assistantMessage)

            } else if (event.type === 'retrieval_info') {
              assistantMessage.retrievalInfo = event.data ? { ...event.data, completed: true } : null

              if (!assistantMessage.planner && event.data?.planner) {
                assistantMessage.planner = event.data.planner
              }
              if (assistantMessage.actorTrace.length === 0 && event.data?.actor_trace) {
                assistantMessage.actorTrace = [...event.data.actor_trace]
              }
              if (
                assistantMessage.reflectionHistory.length === 0 &&
                event.data?.reflection_history
              ) {
                assistantMessage.reflectionHistory = [...event.data.reflection_history]
                assistantMessage.reflection = event.data.reflection ||
                  assistantMessage.reflectionHistory.at(-1) || null
              }

              rebuildRagSteps(assistantMessage)

            } else if (event.type === 'sources') {
              assistantMessage.sources = event.data || []

            } else if (event.type === 'content') {
              assistantMessage.content += event.data || ''

            } else if (event.type === 'done') {
              assistantMessage.pipelineStatus = ''
              finishGenerateStep(assistantMessage)
            }
          })

          // 兼容后端异常未发送 done 的情况。
          finishGenerateStep(assistantMessage)

        } else {
          // 旧 Tool Agent：保留 knowledge_search / web_search / generate_report 等能力。
          await api.streamToolAgentChat(sessionId, question, (event) => {
            if (event.type === 'tool_call') {
              assistantMessage.toolSteps.push({
                name: event.data.name,
                id: event.data.id,
                status: 'calling',
                result: '',
              })
            } else if (event.type === 'tool_result') {
              const step = [...assistantMessage.toolSteps]
                .reverse()
                .find((s) => s.id === event.data.id && s.status === 'calling')

              if (step) {
                step.status = event.data.ok === false ? 'failed' : 'done'
                step.result = event.data.result
              }
            } else if (event.type === 'file') {
              assistantMessage.files.push({
                filename: event.data.filename,
                downloadUrl: `/files/${sessionId}/${event.data.filename}`,
              })
            } else if (event.type === 'content') {
              assistantMessage.content += event.data || ''
            }
          })
        }

        const session = this.sessions.find((s) => s.id === sessionId)
        if (session && session.title === '新对话') {
          session.title = question.slice(0, 20)
        }
      } catch (e) {
        assistantMessage.content = `本轮失败，未保存到会话。原因：${e.message || '请检查后端服务'}`
        assistantMessage.pipelineStatus = ''

        if (mode === 'agentic') {
          const generateStep = assistantMessage.ragSteps.find((item) => item.key === 'generate')
          if (generateStep) {
            generateStep.status = 'warning'
            generateStep.detail = '回答生成失败，请检查后端服务。'
          }
        }

        console.error(e)
      } finally {
        this.isStreaming = false
      }
    },

    async uploadFile(file) {
      const result = await api.uploadDocument(file)
      this.uploadedFiles.push({
        filename: result.filename,
        tripleCount: result.triple_count || 0,
      })
      return result
    },
  },
})
