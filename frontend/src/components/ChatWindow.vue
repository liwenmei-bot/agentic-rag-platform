<script setup>
import { ref, nextTick, watch } from 'vue'
import { useChatStore } from '../stores/chat'
import MessageBubble from './MessageBubble.vue'

const store = useChatStore()
const input = ref('')
const scrollContainer = ref(null)

function scrollToBottom() {
  nextTick(() => {
    if (scrollContainer.value) {
      scrollContainer.value.scrollTop = scrollContainer.value.scrollHeight
    }
  })
}

watch(() => store.messages.length, scrollToBottom)
watch(
  () => store.messages[store.messages.length - 1]?.content,
  scrollToBottom
)

async function handleSend() {
  const question = input.value
  input.value = ''
  await store.sendMessage(question)
}

function handleKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}
</script>

<template>
  <div class="chat-window">
    <div class="mode-bar">
      <button
        class="mode-btn"
        :class="{ active: !store.agentMode }"
        @click="store.agentMode = false"
      >
        知识库问答
      </button>
      <button
        class="mode-btn"
        :class="{ active: store.agentMode }"
        @click="store.agentMode = true"
      >
        Agent 模式
      </button>
    </div>

    <div ref="scrollContainer" class="message-scroll">
      <div v-if="store.messages.length === 0" class="empty-state">
        <p class="empty-mark">§</p>
        <p class="empty-title">
          {{ store.agentMode ? 'Agent 会自主判断该用什么工具' : '知识库空空如也，从提问开始' }}
        </p>
        <p class="empty-hint">
          {{
            store.agentMode
              ? '试试问一个需要检索、搜索或生成报告的问题，观察它的执行过程。'
              : '先在左侧上传一份文档，再向它提问——回答会标注引用出处。'
          }}
        </p>
      </div>

      <div v-else class="message-list">
        <MessageBubble
          v-for="(msg, i) in store.messages"
          :key="i"
          :role="msg.role"
          :content="msg.content"
          :sources="msg.sources"
          :tool-steps="msg.toolSteps"
          :files="msg.files"
          :is-streaming="store.isStreaming && i === store.messages.length - 1 && msg.role === 'assistant'"
        />
      </div>
    </div>

    <div class="input-bar">
      <textarea
        v-model="input"
        class="input-box"
        :placeholder="store.agentMode ? '让 Agent 帮你检索、搜索或生成报告…（Enter 发送）' : '向知识库提问…（Enter 发送，Shift+Enter 换行）'"
        rows="1"
        @keydown="handleKeydown"
      />
      <button
        class="send-btn"
        :disabled="!input.trim() || store.isStreaming"
        @click="handleSend"
      >
        {{ store.isStreaming ? '生成中…' : '发送' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.chat-window {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: var(--paper-50);
  min-width: 0;
}

.mode-bar {
  display: flex;
  gap: var(--space-2);
  padding: var(--space-3) var(--space-6) 0;
  max-width: 760px;
  margin: 0 auto;
  width: 100%;
}

.mode-btn {
  background: transparent;
  border: 1px solid rgba(42, 38, 32, 0.15);
  color: var(--paper-ink);
  border-radius: 999px;
  padding: 6px var(--space-4);
  font-size: 12px;
  font-weight: 600;
  opacity: 0.6;
}
.mode-btn.active {
  background: var(--amber-500);
  border-color: var(--amber-500);
  color: var(--ink-950);
  opacity: 1;
}

.message-scroll {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-6) var(--space-6) 0;
}

.message-list {
  max-width: 760px;
  margin: 0 auto;
}

.empty-state {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: var(--paper-ink);
  opacity: 0.75;
}
.empty-mark {
  font-family: var(--font-display);
  font-size: 48px;
  color: var(--amber-600);
  margin: 0 0 var(--space-2);
}
.empty-title {
  font-family: var(--font-display);
  font-size: 20px;
  margin: 0 0 var(--space-2);
}
.empty-hint {
  font-size: 13px;
  color: var(--text-faint);
  margin: 0;
  max-width: 360px;
}

.input-bar {
  max-width: 760px;
  width: 100%;
  margin: 0 auto;
  padding: var(--space-4) var(--space-6) var(--space-6);
  display: flex;
  gap: var(--space-3);
  align-items: flex-end;
}

.input-box {
  flex: 1;
  resize: none;
  border: 1px solid var(--ink-700);
  border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-4);
  font-family: var(--font-body);
  font-size: 14px;
  background: #fff;
  color: var(--paper-ink);
  max-height: 160px;
  line-height: 1.5;
}
.input-box:focus {
  outline: none;
  border-color: var(--amber-500);
}

.send-btn {
  background: var(--amber-500);
  color: var(--ink-950);
  border: none;
  border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-5);
  font-size: 14px;
  font-weight: 600;
  transition: background 0.15s;
  white-space: nowrap;
}
.send-btn:hover:not(:disabled) {
  background: var(--amber-400);
}
.send-btn:disabled {
  opacity: 0.5;
  cursor: default;
}
</style>
