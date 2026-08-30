"""
知识图谱服务：封装 Neo4j 的连接、三元组写入、实体匹配、关系查询和整图查询。

本版重点修复：
1. 用户问题中的英文实体匹配改为大小写不敏感：
   "Agent" 可以命中 Neo4j 里的 "agent"。
2. 关系查询保留 Neo4j 中原始的有向关系：
   (source)-[relation]->(target)。
3. 查询结果使用 DISTINCT，减少重复关系。
"""

import re
from functools import lru_cache

from neo4j import GraphDatabase

from app.core.config import settings


@lru_cache(maxsize=1)
def get_driver():
    """懒加载 + 缓存 Neo4j Driver，避免每次调用都重新建立连接。"""
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

    新版关系会维护 sources 列表，允许同一关系由多份文档共同支持。
    同时保留 source 字段，兼容现有代码和旧数据。
    """
    driver = get_driver()
    written_count = 0
    seen: set[tuple[str, str, str]] = set()

    with driver.session() as session:
        for triple in triples:
            subject = str(triple.get("subject", "")).strip()
            relation = str(triple.get("relation", "")).strip()
            obj = str(triple.get("object", "")).strip()

            if not subject or not relation or not obj:
                continue

            key = (subject.casefold(), relation.casefold(), obj.casefold())
            if key in seen:
                continue
            seen.add(key)

            safe_relation = "".join(
                c for c in relation
                if c.isalnum() or c == "_"
            ) or "关联"

            session.run(
                f"""
                MERGE (a:Entity {{name: $subject}})
                MERGE (b:Entity {{name: $object}})
                MERGE (a)-[r:`{safe_relation}`]->(b)
                ON CREATE SET
                    r.source = $source,
                    r.sources = [$source]
                ON MATCH SET
                    r.sources = CASE
                        WHEN r.sources IS NULL THEN
                            CASE
                                WHEN r.source IS NULL THEN [$source]
                                WHEN r.source = $source THEN [$source]
                                ELSE [r.source, $source]
                            END
                        WHEN NOT $source IN r.sources THEN r.sources + $source
                        ELSE r.sources
                    END,
                    r.source = CASE
                        WHEN r.source IS NULL THEN $source
                        ELSE r.source
                    END
                """,
                subject=subject,
                object=obj,
                source=source_filename,
            )
            written_count += 1

    return written_count


def delete_graph_by_source(source_filename: str) -> dict:
    """
    删除/解除某个文档对知识图谱关系的贡献。

    - 如果某条关系只由该文档支持：删除关系。
    - 如果某条关系同时由其他文档支持：只从 sources 中移除该文件。
    - 最后清理没有任何关系的孤立 Entity 节点。

    兼容旧版仅有 r.source、没有 r.sources 的关系。
    """
    driver = get_driver()
    deleted_relations = 0
    updated_relations = 0
    deleted_nodes = 0

    with driver.session() as session:
        records = list(
            session.run(
                """
                MATCH ()-[r]->()
                WHERE r.source = $source
                   OR $source IN coalesce(r.sources, [])
                RETURN
                    elementId(r) AS relationship_id,
                    r.source AS legacy_source,
                    r.sources AS sources
                """,
                source=source_filename,
            )
        )

        for record in records:
            relationship_id = record["relationship_id"]
            sources = list(record["sources"] or [])
            legacy_source = record["legacy_source"]

            if not sources and legacy_source:
                sources = [legacy_source]

            remaining = [s for s in sources if s != source_filename]
            remaining = list(dict.fromkeys(remaining))

            if remaining:
                session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE elementId(r) = $relationship_id
                    SET r.sources = $sources,
                        r.source = $primary_source
                    """,
                    relationship_id=relationship_id,
                    sources=remaining,
                    primary_source=remaining[0],
                )
                updated_relations += 1
            else:
                session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE elementId(r) = $relationship_id
                    DELETE r
                    """,
                    relationship_id=relationship_id,
                )
                deleted_relations += 1

        cleanup = session.run(
            """
            MATCH (n:Entity)
            WHERE NOT (n)--()
            WITH collect(n) AS nodes
            FOREACH (n IN nodes | DELETE n)
            RETURN size(nodes) AS deleted_nodes
            """
        ).single()
        if cleanup:
            deleted_nodes = int(cleanup["deleted_nodes"] or 0)

    return {
        "deleted_relations": deleted_relations,
        "updated_relations": updated_relations,
        "deleted_nodes": deleted_nodes,
    }

def _normalize_for_match(text: str) -> str:
    """
    用于实体匹配的轻量标准化：
    - 去首尾空格
    - 连续空白合并
    - casefold()：英文大小写不敏感

    中文内容不会因为 casefold 而改变。
    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text.casefold()


def list_all_entity_names() -> list[str]:
    """返回 Neo4j 中所有 Entity 节点的唯一名称。"""
    driver = get_driver()

    with driver.session() as session:
        records = session.run(
            """
            MATCH (a:Entity)
            WHERE a.name IS NOT NULL
            RETURN DISTINCT a.name AS name
            """
        )
        return [
            record["name"]
            for record in records
            if record["name"]
        ]


def match_entity_names(question: str, limit: int = 20) -> list[str]:
    """
    从用户问题中识别已经存在于 Neo4j 的实体。

    与旧版：
        name in question

    不同，本版使用大小写不敏感匹配，因此：
        Neo4j: "agent"
        用户:  "Agent 和多轮执行是什么关系？"
    可以正确命中 "agent" 和 "多轮执行"。

    返回值始终使用 Neo4j 中保存的原始实体名，
    这样后续 Cypher 查询可以直接精确匹配。
    """
    normalized_question = _normalize_for_match(question)

    if not normalized_question:
        return []

    matched = []

    for name in list_all_entity_names():
        normalized_name = _normalize_for_match(name)

        if not normalized_name:
            continue

        if normalized_name in normalized_question:
            matched.append(name)

    # 优先保留更具体、更长的实体；同时去重。
    matched = sorted(
        set(matched),
        key=lambda value: len(value),
        reverse=True,
    )

    return matched[:max(1, int(limit))]


def find_related_entities(
    entity_names: list[str],
    depth: int = 1,
    limit: int = 30,
) -> list[dict]:
    """
    查询指定实体的一跳关系。

    这里用有向 MATCH：
        (a)-[r]->(b)

    但 WHERE 同时允许 a 或 b 是命中的实体，因此：
    - 能找到实体发出的关系
    - 也能找到指向实体的关系
    - 返回时仍保留图谱真实方向

    depth 参数目前保留接口兼容性，V1 仍固定使用一跳查询。
    """
    if not entity_names:
        return []

    canonical_names = list(dict.fromkeys(entity_names))
    limit = max(1, min(int(limit), 200))

    driver = get_driver()
    results = []

    with driver.session() as session:
        records = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE a.name IN $names OR b.name IN $names
            RETURN DISTINCT
                a.name AS source,
                type(r) AS relation,
                b.name AS target
            LIMIT $limit
            """,
            names=canonical_names,
            limit=limit,
        )

        for record in records:
            results.append(
                {
                    "source": record["source"],
                    "relation": record["relation"],
                    "target": record["target"],
                }
            )

    return results


def get_full_graph(limit: int = 200) -> dict:
    """
    查询整个图谱，返回前端 ECharts graph 可以直接使用的 nodes / links。
    """
    driver = get_driver()
    nodes: dict[str, dict] = {}
    links: list[dict] = []

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

            nodes[source] = {
                "id": source,
                "name": source,
            }
            nodes[target] = {
                "id": target,
                "name": target,
            }

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
