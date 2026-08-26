<script setup>
import { ref, onMounted } from 'vue'
import { useChatStore } from './stores/chat'
import Sidebar from './components/Sidebar.vue'
import ChatWindow from './components/ChatWindow.vue'
import GraphView from './components/GraphView.vue'

const store = useChatStore()
const activeView = ref('chat') // 'chat' | 'graph'

onMounted(() => {
  store.loadSessions()
})
</script>

<template>
  <div class="app-shell">
    <Sidebar v-model:active-view="activeView" />
    <ChatWindow v-if="activeView === 'chat'" />
    <GraphView v-else />
  </div>
</template>

<style scoped>
.app-shell {
  height: 100vh;
  display: flex;
}
</style>
