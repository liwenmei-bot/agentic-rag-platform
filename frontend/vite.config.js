import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      // 开发环境下，前端 /api 请求自动转发到后端 8000 端口，避免手动写完整 URL
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // Agent 生成的文件（比如 generate_report 产出的 .md）也存在后端，
      // 同样需要转发，否则浏览器会向前端自己的开发服务器（5173）请求，
      // 找不到文件时 Vite 会 fallback 返回 index.html，导致下载到的是网页源码而不是真正的文件
      '/files': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
