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

async function streamSse(url, body, onEvent, errorMessage) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok || !res.body) {
    throw new Error(errorMessage)
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
        onEvent(JSON.parse(jsonStr))
      } catch (e) {
        console.error('解析 SSE 数据失败', e, jsonStr)
      }
    }
  }
}

// 1) 基础知识库问答：一次 Chroma 检索 -> LLM。
export async function streamBasicChat(sessionId, question, onEvent) {
  return streamSse(
    `${BASE_URL}/chat/basic/stream`,
    { session_id: sessionId, question },
    onEvent,
    '知识库问答请求失败',
  )
}

// 2) Agentic RAG：Router V5 -> Planner -> Actor -> Reflector -> Rewrite/Retry -> Answer。
export async function streamAgenticChat(sessionId, question, onEvent) {
  return streamSse(
    `${BASE_URL}/chat/stream`,
    { session_id: sessionId, question },
    onEvent,
    'Agentic RAG 请求失败',
  )
}

// 3) 旧工具 Agent：knowledge_search / web_search / generate_report 等工具调用。
export async function streamToolAgentChat(sessionId, question, onEvent) {
  return streamSse(
    `${BASE_URL}/agent/chat/stream`,
    { session_id: sessionId, question },
    onEvent,
    '工具 Agent 请求失败',
  )
}

// 兼容旧调用名。
export async function streamChat(sessionId, question, onEvent) {
  return streamAgenticChat(sessionId, question, onEvent)
}

export async function streamAgentChat(sessionId, question, onEvent) {
  return streamToolAgentChat(sessionId, question, onEvent)
}
