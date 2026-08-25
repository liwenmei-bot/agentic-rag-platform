<script setup>
import { onMounted } from 'vue'
import { useChatStore } from './stores/chat'
import Sidebar from './components/Sidebar.vue'
import ChatWindow from './components/ChatWindow.vue'

const store = useChatStore()

onMounted(() => {
  store.loadSessions()
})
</script>

<template>
  <div class="app-shell">
    <Sidebar />
    <ChatWindow />
  </div>
</template>

<style scoped>
.app-shell {
  height: 100vh;
  display: flex;
}
</style>
