import { defineStore } from 'pinia'
import * as api from '../api/client'

function makeBaseMessage(role, content = '') {
  return {
    role,
    content,
    sources: [],
    files: [],
    toolSteps: [],
    ragSteps: [],
    retrievalInfo: null,
    originalQuestion: '',
  }
}

function routeName(route) {
  const value = (route || '').toLowerCase()
  if (value === 'direct') return 'DIRECT'
  if (value === 'vector') return 'VECTOR'
  if (value === 'graph') return 'GRAPH'
  if (value === 'hybrid') return 'HYBRID'
  return 'UNKNOWN'
}

function buildRagSteps(info) {
  if (!info) return []

  const route = (info.route || (info.skipped_retrieval ? 'direct' : 'hybrid')).toLowerCase()
  const routeText = routeName(route)

  const steps = [
    {
      key: 'router',
      label: 'Agent Router',
      status: 'done',
      detail: `${routeText} · ${info.route_reason || '已根据用户问题选择执行路线。'}`,
    },
  ]

  // DIRECT：不访问 Chroma / Neo4j，也不需要 Context Judge / Query Rewrite。
  if (route === 'direct' || info.skipped_retrieval) {
    steps.push(
      {
        key: 'vector',
        label: 'Vector Retrieval',
        status: 'neutral',
        detail: 'DIRECT 路由，不执行 Chroma 文档向量检索。',
      },
      {
        key: 'graph',
        label: 'Graph Retrieval',
        status: 'neutral',
        detail: 'DIRECT 路由，不执行 Neo4j 知识图谱检索。',
      },
      {
        key: 'judge',
        label: 'Context Judge',
        status: 'neutral',
        detail: 'DIRECT 路由无需检索，因此跳过上下文充分性判断。',
      },
      {
        key: 'rewrite',
        label: 'Query Rewrite',
        status: 'neutral',
        detail: 'DIRECT 路由无需 Query Rewrite。',
      },
      {
        key: 'generate',
        label: '直接回答',
        status: 'running',
        detail: '正在直接生成回答。',
      }
    )

    return steps
  }

  // Router 真正决定检索分支：VECTOR / GRAPH / HYBRID。
  steps.push({
    key: 'vector',
    label: 'Vector Retrieval',
    status: route === 'vector' || route === 'hybrid' ? 'done' : 'neutral',
    detail:
      route === 'vector' || route === 'hybrid'
        ? '已执行 Chroma 文档向量检索。'
        : '当前为 GRAPH 路由，未执行 Chroma 文档向量检索。',
  })

  steps.push({
    key: 'graph',
    label: 'Graph Retrieval',
    status: route === 'graph' || route === 'hybrid' ? 'done' : 'neutral',
    detail:
      route === 'graph' || route === 'hybrid'
        ? '已执行 Neo4j 知识图谱关系检索。'
        : '当前为 VECTOR 路由，未执行 Neo4j 知识图谱检索。',
  })

  if (info.rewrite_attempted) {
    steps.push({
      key: 'judge',
      label: 'Context Judge',
      status: 'warning',
      detail: '首次检索上下文不足，进入 Semantic-safe Query Rewrite。',
    })

    steps.push({
      key: 'rewrite',
      label: 'Query Rewrite',
      status: 'done',
      detail: info.rewrite_candidate || '已生成更适合当前知识库的候选检索查询。',
    })

    steps.push({
      key: 'retrieve-second',
      label: `${routeText} 二次检索`,
      status: info.rewritten_query ? 'done' : 'neutral',
      detail: info.rewritten_query
        ? `候选改写效果更好，已按 ${routeText} 路由重新检索并采用改写查询。`
        : `已按 ${routeText} 路由尝试二次检索，但结果未优于原查询，因此保留第一次检索结果。`,
    })
  } else {
    steps.push({
      key: 'judge',
      label: 'Context Judge',
      status: 'done',
      detail: '当前检索上下文充分。',
    })

    steps.push({
      key: 'rewrite',
      label: 'Query Rewrite',
      status: 'neutral',
      detail: '当前上下文充分，无需 Query Rewrite。',
    })
  }

  steps.push({
    key: 'generate',
    label: '生成回答',
    status: 'running',
    detail:
      route === 'hybrid'
        ? '正在综合文档片段与知识图谱关系生成回答。'
        : route === 'graph'
          ? '正在根据知识图谱关系生成回答。'
          : '正在根据文档检索结果生成回答。',
  })

  return steps
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
    agentMode: false,
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

        // 当前后端的 session 历史只保存回答和 sources，
        // 尚未持久化 retrieval_info，因此刷新页面后旧消息不显示 RAG 执行轨迹。
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

      const userMessage = makeBaseMessage('user', question)
      this.messages.push(userMessage)

      const assistantMessage = makeBaseMessage('assistant')
      assistantMessage.originalQuestion = question
      this.messages.push(assistantMessage)

      this.isStreaming = true

      try {
        if (this.agentMode) {
          await api.streamAgentChat(this.currentSessionId, question, (event) => {
            if (event.type === 'tool_call') {
              assistantMessage.toolSteps.push({
                name: event.data.name,
                status: 'calling',
                result: '',
              })
            } else if (event.type === 'tool_result') {
              const step = [...assistantMessage.toolSteps]
                .reverse()
                .find((s) => s.name === event.data.name && s.status === 'calling')

              if (step) {
                step.status = 'done'
                step.result = event.data.result
              }
            } else if (event.type === 'file') {
              assistantMessage.files.push({
                filename: event.data.filename,
                downloadUrl: `/files/${this.currentSessionId}/${event.data.filename}`,
              })
            } else if (event.type === 'content') {
              assistantMessage.content += event.data
            }
          })
        } else {
          await api.streamChat(this.currentSessionId, question, (event) => {
            if (event.type === 'retrieval_info') {
              assistantMessage.retrievalInfo = event.data || null
              assistantMessage.ragSteps = buildRagSteps(event.data)
            } else if (event.type === 'sources') {
              assistantMessage.sources = event.data || []
            } else if (event.type === 'content') {
              assistantMessage.content += event.data || ''
            } else if (event.type === 'done') {
              finishGenerateStep(assistantMessage)
            }
          })

          // 兼容后端在异常情况下没有发送 done 的场景。
          finishGenerateStep(assistantMessage)
        }

        const session = this.sessions.find((s) => s.id === this.currentSessionId)
        if (session && session.title === '新对话') {
          session.title = question.slice(0, 20)
        }
      } catch (e) {
        assistantMessage.content = '抱歉，回答生成失败，请检查后端服务是否正常运行。'

        if (!this.agentMode) {
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
