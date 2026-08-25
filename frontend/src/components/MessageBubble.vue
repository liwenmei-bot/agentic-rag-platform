<script setup>
defineProps({
  role: { type: String, required: true },       // 'user' | 'assistant'
  content: { type: String, required: true },
  sources: { type: Array, default: () => [] },
  isStreaming: { type: Boolean, default: false },
})
</script>

<template>
  <div class="message-row" :class="role">
    <div class="avatar" :class="role">
      {{ role === 'user' ? '我' : 'AI' }}
    </div>
    <div class="bubble" :class="role">
      <p class="content">{{ content }}<span v-if="isStreaming" class="cursor">▍</span></p>

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

/* 引用来源：做成"档案脚注"的样式——小号、琥珀色下划线、像图书馆卡片目录 */
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
