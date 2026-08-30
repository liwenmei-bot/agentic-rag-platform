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

function stepIcon(status) {
  if (status === 'done') return '✓'
  if (status === 'warning') return '!'
  if (status === 'running') return '…'
  return '—'
}

function routeName(route) {
  const value = (route || '').toLowerCase()
  if (value === 'direct') return 'DIRECT'
  if (value === 'vector') return 'VECTOR'
  if (value === 'graph') return 'GRAPH'
  if (value === 'hybrid') return 'HYBRID'
  return 'UNKNOWN'
}

function routeClass(route) {
  return `route-${(route || 'unknown').toLowerCase()}`
}

watch(() => store.messages.length, scrollToBottom)

watch(
  () => store.messages[store.messages.length - 1]?.content,
  scrollToBottom
)

watch(
  () => store.messages[store.messages.length - 1]?.ragSteps?.length,
  scrollToBottom
)

async function handleSend() {
  const question = input.value.trim()
  if (!question) return

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
        <div
          v-for="(msg, i) in store.messages"
          :key="i"
          class="message-entry"
        >
          <!--
            普通知识库模式的 Agent Router / Context Judge / Query Rewrite 可视化。
            Agent 工具调用模式仍继续使用 MessageBubble 原有的 toolSteps 展示。
          -->
          <div
            v-if="msg.role === 'assistant' && msg.ragSteps?.length"
            class="rag-trace"
          >
            <div class="rag-trace-header">
              <div>
                <div class="rag-trace-title">Agent 检索过程</div>
                <div class="rag-trace-subtitle">
                  Router · Retrieval · Context Judge · Query Rewrite
                </div>
              </div>

              <div class="trace-badges">
                <span
                  v-if="msg.retrievalInfo?.route"
                  class="route-badge"
                  :class="routeClass(msg.retrievalInfo.route)"
                >
                  {{ routeName(msg.retrievalInfo.route) }}
                </span>

                <span
                  v-if="msg.retrievalInfo?.rewrite_attempted"
                  class="rewrite-badge"
                >
                  Query Rewrite
                </span>

                <span
                  v-else-if="msg.retrievalInfo && !msg.retrievalInfo.skipped_retrieval"
                  class="enough-badge"
                >
                  Context OK
                </span>
              </div>
            </div>

            <div
              v-if="msg.retrievalInfo?.route"
              class="router-summary"
            >
              <div class="router-summary-main">
                <span class="router-summary-label">当前路由</span>
                <strong>{{ routeName(msg.retrievalInfo.route) }}</strong>
              </div>
              <div class="router-summary-reason">
                {{ msg.retrievalInfo.route_reason || '已根据问题意图选择检索路线。' }}
              </div>
            </div>

            <div class="rag-steps">
              <div
                v-for="step in msg.ragSteps"
                :key="step.key"
                class="rag-step"
              >
                <span
                  class="rag-step-icon"
                  :class="`status-${step.status}`"
                >
                  {{ stepIcon(step.status) }}
                </span>

                <div class="rag-step-body">
                  <div class="rag-step-label">{{ step.label }}</div>
                  <div class="rag-step-detail">{{ step.detail }}</div>
                </div>
              </div>
            </div>

            <div
              v-if="msg.retrievalInfo?.rewrite_attempted"
              class="rewrite-panel"
            >
              <div class="rewrite-row">
                <span class="rewrite-label">原查询</span>
                <span class="rewrite-value">{{ msg.originalQuestion }}</span>
              </div>

              <div class="rewrite-row">
                <span class="rewrite-label">候选改写</span>
                <span class="rewrite-value">
                  {{ msg.retrievalInfo.rewrite_candidate || '未生成候选查询' }}
                </span>
              </div>

              <div class="rewrite-row">
                <span class="rewrite-label">最终采用</span>
                <span class="rewrite-value">
                  {{
                    msg.retrievalInfo.rewritten_query
                      ? msg.retrievalInfo.rewritten_query
                      : '未采用候选改写，继续使用原查询结果'
                  }}
                </span>
              </div>
            </div>
          </div>

          <MessageBubble
            :role="msg.role"
            :content="msg.content"
            :sources="msg.sources"
            :tool-steps="msg.toolSteps"
            :files="msg.files"
            :is-streaming="
              store.isStreaming &&
              i === store.messages.length - 1 &&
              msg.role === 'assistant'
            "
          />
        </div>
      </div>
    </div>

    <div class="input-bar">
      <textarea
        v-model="input"
        class="input-box"
        :placeholder="
          store.agentMode
            ? '让 Agent 帮你检索、搜索或生成报告…（Enter 发送）'
            : '向知识库提问…（Enter 发送，Shift+Enter 换行）'
        "
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

.message-entry {
  min-width: 0;
}

.rag-trace {
  margin: 0 0 var(--space-3);
  border: 1px solid rgba(42, 38, 32, 0.12);
  border-radius: var(--radius-md);
  background: rgba(255, 255, 255, 0.72);
  padding: var(--space-4);
}

.rag-trace-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-3);
  margin-bottom: var(--space-4);
}

.rag-trace-title {
  color: var(--paper-ink);
  font-size: 13px;
  font-weight: 700;
}

.rag-trace-subtitle {
  margin-top: 2px;
  color: var(--text-faint);
  font-size: 11px;
}

.trace-badges {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
}

.route-badge,
.rewrite-badge,
.enough-badge {
  flex: none;
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 10px;
  font-weight: 700;
}

.route-badge {
  border: 1px solid rgba(42, 38, 32, 0.16);
  color: var(--paper-ink);
  background: rgba(255, 255, 255, 0.65);
}

.route-badge.route-hybrid,
.route-badge.route-graph {
  border-color: var(--amber-500);
}

.route-badge.route-vector {
  border-color: rgba(42, 38, 32, 0.28);
}

.route-badge.route-direct {
  opacity: 0.72;
}

.rewrite-badge {
  background: var(--amber-500);
  color: var(--ink-950);
}

.enough-badge {
  border: 1px solid rgba(42, 38, 32, 0.14);
  color: var(--paper-ink);
  background: transparent;
}

.router-summary {
  margin: calc(var(--space-4) * -0.35) 0 var(--space-4);
  padding: 10px 12px;
  border: 1px solid rgba(42, 38, 32, 0.1);
  border-radius: var(--radius-md);
  background: rgba(255, 255, 255, 0.48);
}

.router-summary-main {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--paper-ink);
  font-size: 11px;
}

.router-summary-label {
  color: var(--text-faint);
  font-weight: 600;
}

.router-summary-reason {
  margin-top: 4px;
  color: var(--text-faint);
  font-size: 11px;
  line-height: 1.5;
}

.rag-steps {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.rag-step {
  display: flex;
  align-items: flex-start;
  gap: var(--space-3);
}

.rag-step-icon {
  width: 20px;
  height: 20px;
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  border: 1px solid rgba(42, 38, 32, 0.18);
  font-size: 11px;
  font-weight: 800;
  color: var(--paper-ink);
}

.rag-step-icon.status-done {
  border-color: var(--amber-500);
  color: var(--amber-600);
}

.rag-step-icon.status-warning {
  border-color: var(--amber-500);
  background: var(--amber-500);
  color: var(--ink-950);
}

.rag-step-icon.status-running {
  border-color: var(--amber-500);
  color: var(--amber-600);
}

.rag-step-icon.status-neutral {
  border-color: rgba(42, 38, 32, 0.1);
  color: var(--text-faint);
  opacity: 0.62;
}

.rag-step:has(.rag-step-icon.status-neutral) .rag-step-label,
.rag-step:has(.rag-step-icon.status-neutral) .rag-step-detail {
  opacity: 0.62;
}

.rag-step-body {
  min-width: 0;
  padding-top: 1px;
}

.rag-step-label {
  color: var(--paper-ink);
  font-size: 12px;
  font-weight: 650;
}

.rag-step-detail {
  margin-top: 2px;
  color: var(--text-faint);
  font-size: 11px;
  line-height: 1.55;
  overflow-wrap: anywhere;
}

.rewrite-panel {
  margin-top: var(--space-4);
  padding-top: var(--space-3);
  border-top: 1px dashed rgba(42, 38, 32, 0.14);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.rewrite-row {
  display: grid;
  grid-template-columns: 64px minmax(0, 1fr);
  gap: var(--space-2);
  align-items: start;
}

.rewrite-label {
  color: var(--text-faint);
  font-size: 11px;
  font-weight: 600;
}

.rewrite-value {
  color: var(--paper-ink);
  font-size: 11px;
  line-height: 1.55;
  overflow-wrap: anywhere;
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

@media (max-width: 720px) {
  .mode-bar,
  .message-scroll,
  .input-bar {
    padding-left: var(--space-3);
    padding-right: var(--space-3);
  }

  .rewrite-row {
    grid-template-columns: 1fr;
    gap: 2px;
  }
}
</style>
