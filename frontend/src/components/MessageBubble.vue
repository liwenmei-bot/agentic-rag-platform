<script setup>
defineProps({
  role: { type: String, required: true },       // 'user' | 'assistant'
  content: { type: String, required: true },
  sources: { type: Array, default: () => [] },
  toolSteps: { type: Array, default: () => [] },
  files: { type: Array, default: () => [] },
  isStreaming: { type: Boolean, default: false },
})

const TOOL_LABELS = {
  search_knowledge_base: '检索知识库',
  web_search: '联网搜索',
  generate_report: '生成报告',
}

function toolLabel(name) {
  return TOOL_LABELS[name] || name
}
</script>

<template>
  <div class="message-row" :class="role">
    <div class="avatar" :class="role">
      {{ role === 'user' ? '我' : 'AI' }}
    </div>
    <div class="bubble" :class="role">
      <!-- Agent 工具调用步骤：像"过程日志"一样展示在正文上方 -->
      <div v-if="toolSteps.length" class="tool-steps">
        <div v-for="(step, i) in toolSteps" :key="i" class="tool-step" :class="step.status">
          <span class="tool-icon">{{ step.status === 'calling' ? '⋯' : '✓' }}</span>
          <span class="tool-name">{{ toolLabel(step.name) }}</span>
          <span v-if="step.status === 'done'" class="tool-result">{{ step.result }}</span>
        </div>
      </div>

      <p class="content">{{ content }}<span v-if="isStreaming" class="cursor">▍</span></p>

      <!-- Agent 生成的可下载文件 -->
      <div v-if="files.length" class="files">
        <a
          v-for="(file, i) in files"
          :key="i"
          class="file-chip"
          :href="file.downloadUrl"
          :download="file.filename"
          target="_blank"
        >
          📎 {{ file.filename }}
        </a>
      </div>

      <div v-if="sources.length" class="sources">
        <span class="sources-label">引用来源</span>
        <span
          v-for="(src, i) in sources"
          :key="i"
          class="source-tag"
          :title="`相关度 ${src.score}`"
        >
          {{ i + 1 }}. {{ src.filename }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.message-row {
  display: flex;
  gap: var(--space-3);
  padding: var(--space-4) 0;
  align-items: flex-start;
}

.avatar {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 600;
  flex-shrink: 0;
}
.avatar.user {
  background: var(--ink-700);
  color: var(--text-primary);
}
.avatar.assistant {
  background: var(--amber-500);
  color: var(--ink-950);
}

.bubble {
  max-width: 680px;
  border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-4);
}
.bubble.user {
  background: var(--paper-100);
  color: var(--paper-ink);
}
.bubble.assistant {
  background: transparent;
  color: var(--paper-ink);
  padding-left: 0;
}

.content {
  margin: 0;
  font-size: 15px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

.cursor {
  color: var(--amber-600);
  animation: blink 1s step-start infinite;
}
@keyframes blink {
  50% { opacity: 0; }
}

/* Agent 工具调用步骤：琥珀色圆点表示进行中，打勾表示完成，像一条"施工日志" */
.tool-steps {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: var(--space-3);
  padding: var(--space-2) var(--space-3);
  background: rgba(217, 165, 74, 0.08);
  border-left: 2px solid var(--amber-500);
  border-radius: 4px;
}
.tool-step {
  display: flex;
  align-items: baseline;
  gap: var(--space-2);
  font-size: 12px;
  font-family: var(--font-mono);
}
.tool-icon {
  color: var(--amber-600);
  width: 12px;
}
.tool-step.calling .tool-icon {
  animation: pulse 1s ease-in-out infinite;
}
@keyframes pulse {
  50% { opacity: 0.3; }
}
.tool-name {
  color: var(--paper-ink);
  font-weight: 600;
}
.tool-result {
  color: var(--text-faint);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 400px;
}

/* 生成的文件：像一个可点击的档案标签 */
.files {
  margin-top: var(--space-3);
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}
.file-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--amber-500);
  color: var(--ink-950);
  text-decoration: none;
  font-size: 12px;
  font-weight: 600;
  padding: var(--space-1) var(--space-3);
  border-radius: 999px;
}
.file-chip:hover {
  background: var(--amber-400);
}

.sources {
  margin-top: var(--space-3);
  padding-top: var(--space-2);
  border-top: 1px dotted var(--ink-600);
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  align-items: center;
}
.sources-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-faint);
}
.source-tag {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--amber-600);
  border-bottom: 1px solid var(--amber-500);
  padding-bottom: 1px;
  cursor: default;
}
</style>
