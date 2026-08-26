<script setup>
import { ref } from 'vue'
import { useChatStore } from '../stores/chat'

const activeView = defineModel('activeView', { default: 'chat' })

const store = useChatStore()
const fileInput = ref(null)
const isUploading = ref(false)
const uploadError = ref('')

function triggerUpload() {
  fileInput.value?.click()
}

async function handleFileChange(e) {
  const file = e.target.files[0]
  if (!file) return
  isUploading.value = true
  uploadError.value = ''
  try {
    await store.uploadFile(file)
  } catch (err) {
    uploadError.value = err.message
  } finally {
    isUploading.value = false
    e.target.value = ''
  }
}

function formatTime(isoString) {
  const d = new Date(isoString)
  return `${d.getMonth() + 1}/${d.getDate()}`
}
</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <span class="brand-mark">§</span>
      <span class="brand-name">Agentic RAG</span>
    </div>

    <button class="new-chat-btn" @click="store.startNewSession(); activeView = 'chat'">
      <span class="plus">+</span> 新对话
    </button>

    <div class="view-tabs">
      <button
        class="view-tab"
        :class="{ active: activeView === 'chat' }"
        @click="activeView = 'chat'"
      >
        对话
      </button>
      <button
        class="view-tab"
        :class="{ active: activeView === 'graph' }"
        @click="activeView = 'graph'"
      >
        知识图谱
      </button>
    </div>

    <div class="upload-zone">
      <input
        ref="fileInput"
        type="file"
        accept=".pdf,.docx,.txt"
        style="display: none"
        @change="handleFileChange"
      />
      <button class="upload-btn" @click="triggerUpload" :disabled="isUploading">
        {{ isUploading ? '处理中…' : '上传文档到知识库' }}
      </button>
      <p v-if="uploadError" class="upload-error">{{ uploadError }}</p>
      <ul v-if="store.uploadedFiles.length" class="uploaded-list">
        <li v-for="(file, i) in store.uploadedFiles" :key="i" class="uploaded-item">
          <span class="doc-icon">📄</span>{{ file.filename }}
          <span class="triple-count">+{{ file.tripleCount }} 关系</span>
        </li>
      </ul>
    </div>

    <div class="session-list-label">历史对话</div>
    <ul class="session-list">
      <li
        v-for="session in store.sessions"
        :key="session.id"
        class="session-item"
        :class="{ active: session.id === store.currentSessionId }"
        @click="store.switchSession(session.id)"
      >
        <span class="session-title">{{ session.title }}</span>
        <span class="session-time">{{ formatTime(session.created_at) }}</span>
        <button
          class="session-delete"
          title="删除会话"
          @click.stop="store.removeSession(session.id)"
        >
          ×
        </button>
      </li>
    </ul>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 260px;
  flex-shrink: 0;
  background: var(--ink-900);
  border-right: 1px solid var(--ink-700);
  display: flex;
  flex-direction: column;
  padding: var(--space-4);
  gap: var(--space-4);
}

.brand {
  display: flex;
  align-items: baseline;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-1) var(--space-3);
  border-bottom: 1px solid var(--ink-700);
}

.brand-mark {
  font-family: var(--font-display);
  font-size: 22px;
  color: var(--amber-400);
}

.brand-name {
  font-family: var(--font-display);
  font-size: 16px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--text-primary);
}

.new-chat-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  background: transparent;
  border: 1px solid var(--ink-600);
  color: var(--text-primary);
  border-radius: var(--radius-md);
  padding: var(--space-3);
  font-size: 14px;
  font-weight: 500;
  transition: border-color 0.15s, background 0.15s;
}
.new-chat-btn:hover {
  border-color: var(--amber-500);
  background: rgba(217, 165, 74, 0.08);
}
.plus {
  color: var(--amber-400);
  font-size: 16px;
}

.view-tabs {
  display: flex;
  gap: 4px;
  padding: 4px;
  background: var(--ink-800);
  border-radius: var(--radius-md);
}

.view-tab {
  flex: 1;
  background: transparent;
  border: none;
  color: var(--text-muted);
  border-radius: var(--radius-sm);
  padding: var(--space-2);
  font-size: 13px;
  font-weight: 500;
  transition: background 0.15s, color 0.15s;
}
.view-tab.active {
  background: var(--ink-700);
  color: var(--amber-400);
}
.view-tab:hover:not(.active) {
  color: var(--text-primary);
}

.upload-zone {
  padding: var(--space-3);
  background: var(--ink-800);
  border-radius: var(--radius-md);
  border: 1px dashed var(--ink-600);
}

.upload-btn {
  width: 100%;
  background: var(--amber-500);
  color: var(--ink-950);
  border: none;
  border-radius: var(--radius-sm);
  padding: var(--space-2) var(--space-3);
  font-size: 13px;
  font-weight: 600;
  transition: background 0.15s;
}
.upload-btn:hover:not(:disabled) {
  background: var(--amber-400);
}
.upload-btn:disabled {
  opacity: 0.6;
  cursor: default;
}

.upload-error {
  margin: var(--space-2) 0 0;
  font-size: 12px;
  color: var(--danger);
}

.uploaded-list {
  list-style: none;
  margin: var(--space-3) 0 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}
.uploaded-item {
  font-size: 12px;
  color: var(--text-muted);
  display: flex;
  align-items: center;
  gap: var(--space-1);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.doc-icon {
  font-size: 12px;
}

.triple-count {
  margin-left: auto;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--amber-600);
  flex-shrink: 0;
}

.session-list-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-faint);
  padding: 0 var(--space-1);
}

.session-list {
  list-style: none;
  margin: 0;
  padding: 0;
  overflow-y: auto;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.session-item {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-sm);
  cursor: pointer;
  position: relative;
}
.session-item:hover {
  background: var(--ink-800);
}
.session-item.active {
  background: var(--ink-800);
  box-shadow: inset 3px 0 0 var(--amber-500);
}

.session-title {
  flex: 1;
  font-size: 13px;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-time {
  font-size: 11px;
  color: var(--text-faint);
}

.session-delete {
  display: none;
  background: transparent;
  border: none;
  color: var(--text-faint);
  font-size: 16px;
  line-height: 1;
  padding: 0 var(--space-1);
}
.session-item:hover .session-delete {
  display: block;
}
.session-delete:hover {
  color: var(--danger);
}
</style>
