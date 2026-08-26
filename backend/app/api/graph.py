"""
知识图谱查询接口，返回适合前端可视化组件（ECharts graph）使用的数据结构。
"""
from fastapi import APIRouter

from app.services import graph_service

router = APIRouter()


@router.get("/graph")
async def get_graph(limit: int = 200):
    return graph_service.get_full_graph(limit=limit)
