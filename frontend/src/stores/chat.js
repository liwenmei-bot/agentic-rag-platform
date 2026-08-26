import { defineStore } from 'pinia'
import * as api from '../api/client'

export const useChatStore = defineStore('chat', {
  state: () => ({
    sessions: [],
    currentSessionId: null,
    messages: [],       // 当前会话的消息列表
    isStreaming: false,
    uploadedFiles: [],  // 本次会话里上传过的文件名，用于侧边栏展示
  }),

  actions: {
    async loadSessions() {
      this.sessions = await api.listSessions()
      // 如果还没有任何会话，或者当前没选中会话，自动开一个新的
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
      this.messages = rawMessages.map((m) => ({
        role: m.role,
        content: m.content,
        sources: m.sources ? JSON.parse(m.sources) : [],
      }))
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

      this.messages.push({ role: 'user', content: question, sources: [] })
      const assistantMessage = { role: 'assistant', content: '', sources: [] }
      this.messages.push(assistantMessage)

      this.isStreaming = true
      try {
        await api.streamChat(this.currentSessionId, question, (event) => {
          if (event.type === 'sources') {
            assistantMessage.sources = event.data
          } else if (event.type === 'content') {
            assistantMessage.content += event.data
          }
        })

        // 第一次提问后，把会话标题自动改成问题的前 20 个字，方便侧边栏识别
        const session = this.sessions.find((s) => s.id === this.currentSessionId)
        if (session && session.title === '新对话') {
          session.title = question.slice(0, 20)
        }
      } catch (e) {
        assistantMessage.content = '抱歉，回答生成失败，请检查后端服务是否正常运行。'
        console.error(e)
      } finally {
        this.isStreaming = false
      }
    },

    async uploadFile(file) {
      const result = await api.uploadDocument(file)
      this.uploadedFiles.push({ filename: result.filename, tripleCount: result.triple_count || 0 })
      return result
    },
  },
})
