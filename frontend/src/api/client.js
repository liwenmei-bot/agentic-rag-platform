const BASE_URL = '/api'

export async function createSession() {
  const res = await fetch(`${BASE_URL}/sessions`, { method: 'POST' })
  if (!res.ok) throw new Error('创建会话失败')
  return res.json()
}

export async function listSessions() {
  const res = await fetch(`${BASE_URL}/sessions`)
  if (!res.ok) throw new Error('获取会话列表失败')
  return res.json()
}

export async function getSessionMessages(sessionId) {
  const res = await fetch(`${BASE_URL}/sessions/${sessionId}/messages`)
  if (!res.ok) throw new Error('获取历史消息失败')
  return res.json()
}

export async function deleteSession(sessionId) {
  const res = await fetch(`${BASE_URL}/sessions/${sessionId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('删除会话失败')
  return res.json()
}

export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${BASE_URL}/upload`, { method: 'POST', body: formData })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || '文档上传失败')
  }
  return res.json()
}

export async function getGraph() {
  const res = await fetch(`${BASE_URL}/graph`)
  if (!res.ok) throw new Error('获取知识图谱失败')
  return res.json()
}

/**
 * Agent 模式的流式对话。
 * 当前事件类型：
 * - tool_call
 * - tool_result
 * - file
 * - content
 * - done
 */
export async function streamAgentChat(sessionId, question, onEvent) {
  const res = await fetch(`${BASE_URL}/agent/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, question }),
  })

  if (!res.ok || !res.body) {
    throw new Error('Agent 请求失败')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() || ''

    for (const part of parts) {
      const line = part.trim()
      if (!line.startsWith('data:')) continue

      const jsonStr = line.slice(5).trim()
      try {
        const event = JSON.parse(jsonStr)
        onEvent(event)
      } catch (e) {
        console.error('解析 Agent SSE 数据失败', e, jsonStr)
      }
    }
  }
}

/**
 * 普通知识库模式的流式问答。
 *
 * 后端 /chat/stream 当前会返回：
 * - retrieval_info：Agent Router / Context Judge / Query Rewrite 的检索决策
 * - sources：引用来源
 * - content：流式回答正文
 * - done：本轮结束
 *
 * EventSource 原生只支持 GET，而这里是 POST，
 * 因此继续使用 fetch + ReadableStream 手动解析 SSE。
 */
export async function streamChat(sessionId, question, onEvent) {
  const res = await fetch(`${BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, question }),
  })

  if (!res.ok || !res.body) {
    throw new Error('问答请求失败')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // SSE 每条事件由空行分隔；一次网络包可能包含多条或半条消息。
    const parts = buffer.split('\n\n')
    buffer = parts.pop() || ''

    for (const part of parts) {
      const line = part.trim()
      if (!line.startsWith('data:')) continue

      const jsonStr = line.slice(5).trim()
      try {
        const event = JSON.parse(jsonStr)
        onEvent(event)
      } catch (e) {
        console.error('解析 Chat SSE 数据失败', e, jsonStr)
      }
    }
  }
}
