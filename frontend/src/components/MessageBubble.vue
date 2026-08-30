<script setup>
import { computed } from 'vue'
import DOMPurify from 'dompurify'
import MarkdownIt from 'markdown-it'

const props = defineProps({
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

const markdown = new MarkdownIt({
  html: false,       // 不允许模型直接注入原始 HTML
  linkify: true,
  typographer: false,
  breaks: true,      // 单换行也按换行展示，更符合聊天体验
})

function toolLabel(name) {
  return TOOL_LABELS[name] || name
}

const renderedContent = computed(() => {
  if (props.role !== 'assistant') return ''

  const html = markdown.render(props.content || '')

  // markdown-it 已禁用原始 HTML；DOMPurify 再做一层兜底，
  // 避免以后配置变化或链接内容带来 XSS 风险。
  return DOMPurify.sanitize(html)
})
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

      <!--
        用户消息保持纯文本；AI 回答使用安全 Markdown 渲染。
        这样 **粗体**、列表、代码、小标题等不会再把 Markdown 符号直接显示出来。
      -->
      <div v-if="role === 'assistant'" class="content-wrap">
        <div class="content markdown-content" v-html="renderedContent"></div>
        <span v-if="isStreaming" class="cursor">▍</span>
      </div>
      <p v-else class="content">{{ content }}<span v-if="isStreaming" class="cursor">▍</span></p>

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
  word-break: break-word;
}

.content-wrap {
  min-width: 0;
}

.bubble.user .content {
  white-space: pre-wrap;
}

/*
 * v-html 生成的节点不会自动带 scoped 属性，
 * 因此 Markdown 内部元素必须使用 :deep(...) 设置样式。
 */
.markdown-content {
  white-space: normal;
}

.markdown-content :deep(p) {
  margin: 0 0 0.75em;
}

.markdown-content :deep(p:last-child) {
  margin-bottom: 0;
}

.markdown-content :deep(strong) {
  font-weight: 700;
  color: inherit;
}

.markdown-content :deep(em) {
  font-style: italic;
}

.markdown-content :deep(h1),
.markdown-content :deep(h2),
.markdown-content :deep(h3),
.markdown-content :deep(h4) {
  margin: 1em 0 0.5em;
  line-height: 1.35;
  color: inherit;
}

.markdown-content :deep(h1) {
  font-size: 1.35em;
}

.markdown-content :deep(h2) {
  font-size: 1.22em;
}

.markdown-content :deep(h3) {
  font-size: 1.12em;
}

.markdown-content :deep(h4) {
  font-size: 1.04em;
}

.markdown-content :deep(ul),
.markdown-content :deep(ol) {
  margin: 0.55em 0 0.85em;
  padding-left: 1.55em;
}

.markdown-content :deep(li) {
  margin: 0.25em 0;
}

.markdown-content :deep(li > p) {
  margin: 0;
}

.markdown-content :deep(code) {
  font-family: var(--font-mono);
  font-size: 0.9em;
  padding: 0.12em 0.38em;
  border-radius: 4px;
  background: rgba(127, 127, 127, 0.12);
}

.markdown-content :deep(pre) {
  margin: 0.8em 0;
  padding: 0.85em 1em;
  overflow-x: auto;
  border-radius: var(--radius-md);
  background: rgba(18, 25, 38, 0.08);
}

.markdown-content :deep(pre code) {
  padding: 0;
  background: transparent;
  white-space: pre;
}

.markdown-content :deep(blockquote) {
  margin: 0.75em 0;
  padding-left: 0.9em;
  border-left: 3px solid var(--amber-500);
  color: var(--text-faint);
}

.markdown-content :deep(a) {
  color: var(--amber-600);
  text-decoration: underline;
  text-underline-offset: 2px;
}

.markdown-content :deep(hr) {
  margin: 1em 0;
  border: 0;
  border-top: 1px solid rgba(127, 127, 127, 0.25);
}

.markdown-content :deep(table) {
  width: 100%;
  margin: 0.8em 0;
  border-collapse: collapse;
  font-size: 0.94em;
}

.markdown-content :deep(th),
.markdown-content :deep(td) {
  padding: 0.45em 0.6em;
  border: 1px solid rgba(127, 127, 127, 0.22);
  text-align: left;
  vertical-align: top;
}

.markdown-content :deep(th) {
  font-weight: 700;
  background: rgba(127, 127, 127, 0.08);
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
