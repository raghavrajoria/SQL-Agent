"""
schema_catalog/rank_schema_context.py

Combines semantic retrieval (ChromaDB distance) with FK-graph expansion
(expand_schema.py) into a single ranked, capped table list -- the final
schema context handed to SQL generation.

Why this exists:
Pure FK expansion (expand_schema.py alone) is correct but unbounded --
hub tables (e.g. `film` in Sakila, or any heavily-referenced table in a
~200-table production schema) can pull in dozens of structurally-connected
but semantically-irrelevant neighbors. Without ranking + a hard cap, this
either produces a bloated prompt (cost, latency, and accuracy all suffer)
or -- at production scale, not just this database -- can produce a runaway
expansion that pulls a large fraction of the schema into every request.

This module fixes both problems:
1. Converts ChromaDB distance to a similarity score.
2. Expands via FK graph, decaying score per hop, so a table reached
   only through the graph never outranks a directly relevant one.
3. Hard-caps BFS frontier size and total hops so expansion time and
   prompt size stay bounded regardless of how connected the schema is.
4. Returns only the top N tables by combined score.

IMPORTANT ON SCORE DIRECTION:
retrieve_schema.py returns ChromaDB `distance` (lower = more relevant).
Verified against real output: Chroma's default distance space is squared
L2 ("l2"), not cosine distance, even though embeddings are normalized.
distance_to_similarity() converts via `1 - distance/2`, which is the
correct recovery of cosine similarity from squared L2 distance on
unit-length vectors. If retrieve_schema.py or build_vector_index.py ever
change the embedding normalization or the collection's distance space,
this conversion must be re-verified -- there is no error thrown on a
mismatch, just quietly wrong table selection. Sanity check: the most
relevant table in test output should have the LOWEST distance and,
after conversion, the HIGHEST similarity, roughly in a 0.2-0.6 range for
this embedding model on short table descriptions.
"""

from schema_catalog.expand_schema import load_schema_catalog, build_fk_graph

# Tunable constants -- documented, not magic numbers
HOP_DECAY = 0.7        # score multiplier per FK hop away from a direct match
MAX_HOPS = 2            # hard cap on expansion depth
MAX_FRONTIER = 40       # hard cap on tables visited during BFS, regardless
                         # of schema size or connectivity -- this is the
                         # safety valve for ~200-table production schemas
TOP_N = 8                # final number of tables handed to SQL generation
BRIDGE_BONUS_WEIGHT = 0.05  # per extra anchor connection beyond the first,
                             # for non-anchor tables -- breaks ties between
                             # structural bridge tables and peripheral leaves


def distance_to_similarity(distance: float) -> float:
    """
    Converts ChromaDB distance to a 0-1 similarity score.

    Chroma's default distance space is squared L2 ("l2"), not cosine
    distance, even when the input embeddings are normalized. For unit-length
    vectors, squared L2 relates to cosine similarity as:

        distance = ||a - b||^2 = 2 - 2 * cos_similarity

    So distance ranges roughly 0-4 (not 0-2), and cos_similarity is
    recovered as:

        cos_similarity = 1 - distance / 2

    Verified against real output: distance=1.195 -> similarity=0.40,
    distance=1.49 -> similarity=0.255 -- consistent, differentiated
    scores in the expected 0.2-0.6 range for MiniLM sentence embeddings.

    If your Chroma collection is ever explicitly created with
    `metadata={"hnsw:space": "cosine"}`, this formula must change back
    to `1 - distance` -- check collection creation in build_vector_index.py
    before touching this if scores look wrong again.
    """
    similarity = 1.0 - (distance / 2.0)
    return max(0.0, min(1.0, similarity))


def rank_schema_context(
    retrieval_results: list[dict],
    fk_graph: dict,
    max_hops: int = MAX_HOPS,
    decay: float = HOP_DECAY,
    top_n: int = TOP_N,
    max_frontier: int = MAX_FRONTIER,
) -> dict:
    """
    Args:
        retrieval_results: output of retrieve_schema.retrieve_schema()
                             -- list of {"table_name", "distance", ...}
        fk_graph: output of expand_schema.build_fk_graph()
        max_hops: BFS depth cap
        decay: per-hop score multiplier
        top_n: max tables returned
        max_frontier: hard cap on total tables visited during BFS --
                        prevents runaway expansion on densely connected
                        production schemas, independent of max_hops

    Returns:
        {
          "ranked_tables": [
              {"table_name": "customer", "score": 0.82, "hops": 0},
              {"table_name": "rental", "score": 0.71, "hops": 0},
              {"table_name": "payment", "score": 0.497, "hops": 1},
              ...
          ],
          "table_names": ["customer", "rental", "payment", ...],
          "join_paths": ["rental.customer_id -> customer.customer_id", ...],
          "truncated": False   # True if max_frontier was hit before
                                 # max_hops completed -- signals the cap
                                 # actually kicked in, worth logging in prod
        }
    """
    scores: dict[str, float] = {}
    hop_reached: dict[str, int] = {}
    join_paths: set[str] = set()

    for r in retrieval_results:
        name = r["table_name"]
        sim = distance_to_similarity(r["distance"])
        scores[name] = sim
        hop_reached[name] = 0

    visited = set(scores.keys())
    frontier = list(scores.items())
    truncated = False
    hop = 0

    while frontier and hop < max_hops:
        next_frontier = []
        for table, score in frontier:
            if len(visited) >= max_frontier:
                truncated = True
                break

            for neighbor, edge_info in fk_graph.get(table, {}).items():
                if len(visited) >= max_frontier:
                    truncated = True
                    break

                candidate_score = score * decay
                join_paths.add(edge_info["via"])

                if neighbor not in scores or candidate_score > scores[neighbor]:
                    is_new = neighbor not in visited
                    scores[neighbor] = candidate_score
                    hop_reached[neighbor] = hop + 1
                    visited.add(neighbor)
                    if is_new:
                        next_frontier.append((neighbor, candidate_score))

        frontier = next_frontier
        hop += 1

    # Bridge-connectivity bonus.
    #
    # Problem this fixes: same-hop neighbors of a hub table (e.g. everything
    # 1 hop off `film`) all get identical decayed scores, because decay only
    # looks at ONE parent's score. It can't tell "inventory bridges film and
    # rental, so it's structurally necessary" apart from "language only
    # touches film, so it's a peripheral leaf" -- both score identically.
    #
    # Fix: after the BFS settles, boost any non-anchor table that connects
    # via FK to MORE THAN ONE of the original semantic anchors (hop 0
    # tables). A table bridging two independently-confirmed-relevant tables
    # is very likely a required join path; a table touching only one anchor
    # is more likely peripheral. Anchors themselves are untouched -- their
    # ranking stays purely semantic.
    #
    # This does not attempt to fully solve join-path relevance (that's a
    # Steiner-tree problem, overkill here) -- it only breaks the specific,
    # common tie between "necessary bridge table" and "single-connection
    # leaf table" that pure hop-decay can't distinguish.
    anchors = {name for name, h in hop_reached.items() if h == 0}
    for table in list(scores.keys()):
        if hop_reached[table] == 0:
            continue  # anchors keep pure semantic score, no bonus
        connection_count = sum(
            1 for anchor in anchors if anchor in fk_graph.get(table, {})
        )
        if connection_count > 1:
            scores[table] += (connection_count - 1) * BRIDGE_BONUS_WEIGHT

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    ranked_table_names = {name for name, _ in ranked}

    ranked_out = [
        {"table_name": name, "score": round(score, 4), "hops": hop_reached[name]}
        for name, score in ranked
    ]

    # Only keep join paths where BOTH ends survived the top-N cut --
    # a join path to a table that got cut is dead weight in the prompt.
    relevant_join_paths = sorted(
        jp for jp in join_paths
        if any(t in jp for t in ranked_table_names)
        and all(
            part.split(".")[0] in ranked_table_names
            for part in jp.split(" -> ")
        )
    )

    return {
        "ranked_tables": ranked_out,
        "table_names": [t["table_name"] for t in ranked_out],
        "join_paths": relevant_join_paths,
        "truncated": truncated,
    }


if __name__ == "__main__":
    from schema_catalog.retrieve_schema import retrieve_schema

    schema_catalog = load_schema_catalog()
    fk_graph = build_fk_graph(schema_catalog)

    test_questions = [
        "Which customers rented the most movies?",
        "Which actors appeared in the most movies?",
        "Which films generated the most rental revenue?",
    ]

    for question in test_questions:
        retrieval_results = retrieve_schema(question, top_k=3)
        result = rank_schema_context(retrieval_results, fk_graph)

        print(f"\nQ: {question}")
        print(f"  ranked tables:")
        for t in result["ranked_tables"]:
            print(f"    {t['table_name']:20s} score={t['score']:.4f}  hops={t['hops']}")
        print(f"  join paths:")
        for jp in result["join_paths"]:
            print(f"    {jp}")
        if result["truncated"]:
            print(f"  WARNING: expansion was truncated by max_frontier cap")