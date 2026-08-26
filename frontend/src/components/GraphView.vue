<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts'
import * as api from '../api/client'

const chartContainer = ref(null)
const isLoading = ref(true)
const isEmpty = ref(false)
const errorMsg = ref('')
let chartInstance = null

async function loadGraph() {
  if (chartInstance) {
    chartInstance.dispose()
    chartInstance = null
  }

  isLoading.value = true
  isEmpty.value = false
  errorMsg.value = ''

  try {
    const data = await api.getGraph()

    if (!data.nodes || data.nodes.length === 0) {
      isEmpty.value = true
      isLoading.value = false
      return
    }

    isLoading.value = false
    await nextTick()

    renderChart({
      nodes: data.nodes,
      links: data.links || [],
    })
  } catch (e) {
    errorMsg.value = e?.message || '获取知识图谱失败'
    isLoading.value = false
  }
}

function renderChart(data) {
  if (!chartContainer.value) return
  if (!chartInstance) {
    chartInstance = echarts.init(chartContainer.value)
  }

  const nodes = data.nodes.map((n) => ({
    id: n.id,
    name: n.name,
    symbolSize: 36,
    itemStyle: { color: '#d9a54a' },
    label: { color: '#e8e6df' },
  }))

  const links = data.links.map((l) => ({
    source: l.source,
    target: l.target,
    label: { show: true, formatter: l.relation, fontSize: 10, color: '#8b93a3' },
    lineStyle: { color: '#3d4759', curveness: 0.15 },
  }))

  chartInstance.setOption({
    backgroundColor: 'transparent',
    tooltip: {},
    series: [
      {
        type: 'graph',
        layout: 'force',
        roam: true,
        draggable: true,
        data: nodes,
        links: links,
        force: {
          repulsion: 260,
          edgeLength: 120,
          gravity: 0.1,
        },
        label: {
          show: true,
          position: 'right',
          fontFamily: 'Inter, sans-serif',
          fontSize: 12,
        },
        edgeSymbol: ['none', 'arrow'],
        edgeSymbolSize: 6,
        emphasis: {
          focus: 'adjacency',
          lineStyle: { width: 3 },
        },
      },
    ],
  })
}

function handleResize() {
  chartInstance?.resize()
}

onMounted(() => {
  loadGraph()
  window.addEventListener('resize', handleResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  chartInstance?.dispose()
})
</script>

<template>
  <div class="graph-view">
    <div class="graph-header">
      <h2 class="graph-title">知识图谱</h2>
      <button class="refresh-btn" @click="loadGraph">刷新</button>
    </div>

    <div v-if="isLoading" class="graph-state">加载中…</div>
    <div v-else-if="errorMsg" class="graph-state error">
      {{ errorMsg }}（确认后端服务和 Neo4j 是否都在运行）
    </div>
    <div v-else-if="isEmpty" class="graph-state">
      知识图谱还是空的——上传文档后，系统会自动抽取实体关系存入图谱。
    </div>
    <div v-else ref="chartContainer" class="chart-container"></div>
  </div>
</template>

<style scoped>
.graph-view {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: var(--ink-950);
  min-width: 0;
  min-height: 0;
}

.graph-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-5) var(--space-6);
  border-bottom: 1px solid var(--ink-700);
}

.graph-title {
  font-family: var(--font-display);
  font-size: 18px;
  color: var(--text-primary);
  margin: 0;
}

.refresh-btn {
  background: transparent;
  border: 1px solid var(--ink-600);
  color: var(--text-primary);
  border-radius: var(--radius-sm);
  padding: var(--space-2) var(--space-4);
  font-size: 13px;
}
.refresh-btn:hover {
  border-color: var(--amber-500);
}

.graph-state {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: 14px;
  text-align: center;
  padding: var(--space-6);
}
.graph-state.error {
  color: var(--danger);
}

.chart-container {
  flex: 1;
  width: 100%;
  min-height: 0;
}
</style>
