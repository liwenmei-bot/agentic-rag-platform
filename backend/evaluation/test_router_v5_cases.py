# -*- coding: utf-8 -*-
from pathlib import Path
import sys

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.rag_service import _route_query

cases = [
    # Holdout-03 observed failure categories
    ("配置文件有没有记录 Neo4j 的连接地址？", "vector"),
    ("先查 README 对 Router 的说明，再沿 Neo4j 关系验证它连接了哪些检索分支。", "hybrid"),
    ("请把配置文档里的 Chroma 信息和图谱中的组件关系合并说明。", "hybrid"),
    ("同时参考代码说明和 Neo4j 边，讲清 Vector Retrieval 与 Graph Retrieval 怎么协作。", "hybrid"),
    ("不要只给概念说明，也要给关系图中的连接，解释 HYBRID 路线。", "hybrid"),
    ("请综合 README 片段和实体关系，说明 Chroma 在整个 RAG 流程中的位置和作用。", "hybrid"),
    ("从文档内容与关系路径两个角度解释一次完整查询如何到达 Answer Generation。", "hybrid"),
    ("请一边引用资料中的配置说明，一边依据 Neo4j 关系解释 Docker 服务之间的联系。", "hybrid"),
    ("把知识库片段里对 embedding 的描述和图里的 Chroma 连接关系放在一起回答。", "hybrid"),
    ("需要文本事实也需要图谱边：说明 Router 选 GRAPH 后后端会经过哪些组件。", "hybrid"),
    ("正文里找项目用到的向量库，再去图谱中说明它与 embedding 的联系。", "hybrid"),
    ("别只看 Neo4j，也别只看文档，两边都用来解释检索失败后的处理流程。", "hybrid"),

    # Pure GRAPH controls
    ("Router 选择 HYBRID 后，Vector 和 Graph 两个分支如何汇合？", "graph"),
    ("文档 Chunk 在图谱中连向哪些实体类型？", "graph"),
    ("Query Rewrite 和 Context Judge 在流程里前后怎么衔接？", "graph"),

    # DIRECT / VECTOR controls
    ("早呀！", "direct"),
    ("把“vector database”翻成中文。", "direct"),
    ("想一句不超过十五个字的团队口号。", "direct"),
    ("向量检索里的 embedding 是什么意思？", "vector"),
    ("Docker Compose 里定义了哪些运行服务？", "vector"),
]

correct = 0
for i, (question, expected) in enumerate(cases, start=1):
    route, reason = _route_query(question)
    ok = route == expected
    correct += int(ok)
    print(f"[{i}/{len(cases)}] {question}")
    print(f"  expected={expected.upper():<6} predicted={route.upper():<6} {'PASS' if ok else 'FAIL'}")
    print(f"  reason={reason}")
    print()

print(f"Result: {correct}/{len(cases)} correct")
