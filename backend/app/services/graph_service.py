"""
知识图谱服务：封装 Neo4j 的连接、三元组写入、子图查询。

设计上跟 vector_store_service.py 是同样的思路：单独封装一层，
上层代码（rag_service.py）不需要知道底层是 Neo4j 还是别的图数据库。
"""
from functools import lru_cache

from neo4j import GraphDatabase

from app.core.config import settings


@lru_cache(maxsize=1)
def get_driver():
    """懒加载 + 缓存驱动实例，避免每次调用都重新建立连接。"""
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def close_driver() -> None:
    driver = get_driver()
    driver.close()


def add_triples(triples: list[dict], source_filename: str) -> int:
    """
    把一批三元组写入 Neo4j。

    每个 triple 形如 {"subject": "苹果公司", "relation": "创始人", "object": "史蒂夫·乔布斯"}

    用 MERGE 而不是 CREATE：MERGE 是"不存在则创建，存在则复用"，
    避免同一个实体（比如"苹果公司"在多份文档里都出现）被重复创建成多个节点。
    """
    driver = get_driver()
    written_count = 0

    with driver.session() as session:
        for triple in triples:
            subject = triple.get("subject", "").strip()
            relation = triple.get("relation", "").strip()
            obj = triple.get("object", "").strip()

            if not subject or not relation or not obj:
                continue  # 跳过抽取质量不好、字段缺失的三元组

            # Cypher 里关系类型不能用参数化传入变量名，只能用字符串拼接，
            # 这里做了简单的清洗（只保留中英文、数字、下划线）防止拼出非法的 Cypher 语法
            safe_relation = "".join(c for c in relation if c.isalnum() or c == "_") or "关联"

            session.run(
                f"""
                MERGE (a:Entity {{name: $subject}})
                MERGE (b:Entity {{name: $object}})
                MERGE (a)-[r:`{safe_relation}`]->(b)
                ON CREATE SET r.source = $source
                """,
                subject=subject,
                object=obj,
                source=source_filename,
            )
            written_count += 1

    return written_count


def find_related_entities(entity_names: list[str], depth: int = 1) -> list[dict]:
    """
    给定一批实体名，查询它们在图谱里的邻居节点和关系。
    depth=1 表示只查直接相连的一跳邻居，避免子图过大、检索结果太发散。
    """
    if not entity_names:
        return []

    driver = get_driver()
    results = []

    with driver.session() as session:
        records = session.run(
            f"""
            MATCH (a:Entity)-[r]-(b:Entity)
            WHERE a.name IN $names
            RETURN a.name AS source, type(r) AS relation, b.name AS target
            LIMIT 30
            """,
            names=entity_names,
        )
        for record in records:
            results.append({
                "source": record["source"],
                "relation": record["relation"],
                "target": record["target"],
            })

    return results


def list_all_entity_names() -> list[str]:
    """
    取出图谱里所有实体名称。

    用途：问答时需要判断"用户的问题里提到了哪些已知实体"，
    最简单可靠的办法不是再调一次 LLM 做实体识别（增加延迟和成本），
    而是直接查库里已有哪些实体名，再看它们是否作为子串出现在问题文本里。
    图谱规模不大时，这个办法足够快、足够准。
    """
    driver = get_driver()
    with driver.session() as session:
        records = session.run("MATCH (a:Entity) RETURN DISTINCT a.name AS name")
        return [record["name"] for record in records]


def get_full_graph(limit: int = 200) -> dict:
    """
    查询整个图谱（限制关系数量，避免图太大卡住前端），
    返回适合前端可视化组件（ECharts graph）直接使用的 nodes/links 格式。
    """
    driver = get_driver()
    nodes: dict[str, dict] = {}
    links: list[dict] = []

    # 避免前端传入 0、负数或过大的 limit。
    limit = max(1, min(int(limit), 1000))

    with driver.session() as session:
        records = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            RETURN
                a.name AS source,
                type(r) AS relation,
                b.name AS target
            LIMIT $limit
            """,
            limit=limit,
        )

        for record in records:
            source = record["source"]
            relation = record["relation"]
            target = record["target"]

            nodes[source] = {"id": source, "name": source}
            nodes[target] = {"id": target, "name": target}
            links.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                }
            )

    return {
        "nodes": list(nodes.values()),
        "links": links,
    }
