#!/usr/bin/env python3
"""
JusticeFusion Weaviate FULL Schema + Data Integrity Diagnostics

Goal:
Confirm the database creation/indexing pipeline inserts:
- Document chunks
- Legal articles (if used)
- GraphNodes
- GraphEdges
...and that nodes and edges can be CONNECTED reliably for visualization / retrieval.

This script prints:
- effective config endpoints + collection names
- weaviate /v1/meta info
- schema properties for each collection
- counts + samples for each collection
- vector presence (best-effort)
- GraphNodes near_vector search (using your EmbeddingModelManager)
- CRITICAL: edge->node join integrity tests with multiple strategies
- per-file summary (top files by edge count): nodes/edges counts + join rates

Exit code:
- 0 on success (even with warnings)
- 1 on fatal failures (cannot connect, missing required collections, missing required fields)
"""

import sys
import re
import json
import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Set

import weaviate
from weaviate import WeaviateClient

import config

# --- ADDED: Class to duplicate terminal output to a file ---
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def __del__(self):
        try:
            self.log_file.close()
        except Exception:
            pass
# -----------------------------------------------------------

# Optional v4 query helpers
try:
    from weaviate.classes.query import Filter, MetadataQuery
except Exception:
    Filter = None  # type: ignore
    MetadataQuery = None  # type: ignore


# --------------------------------------------------------------------------------------
# Logging: FORCE config so other modules cannot suppress earlier logs.
# --------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,  # <<< IMPORTANT: overrides any previous logging config
)
log = logging.getLogger("schema_inspect")

# Reduce noisy loggers (optional)
for noisy in ("httpx", "weaviate", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# --- ADDED: Suppress pdfminer warnings ---
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfminer.pdfpage").setLevel(logging.ERROR)
# -----------------------------------------

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def hr(char: str = "=") -> str:
    return char * 110


def p(section_title: str) -> None:
    # always-visible section header
    print("\n" + hr("="))
    print(section_title)
    print(hr("="), flush=True)


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def connect_client() -> WeaviateClient:
    connection_params = weaviate.connect.ConnectionParams.from_params(
        http_host=config.WEAVIATE_HOST,
        http_port=config.WEAVIATE_PORT,
        http_secure=config.WEAVIATE_SECURE,
        grpc_host=config.WEAVIATE_GRPC_HOST,
        grpc_port=config.WEAVIATE_GRPC_PORT,
        grpc_secure=config.WEAVIATE_GRPC_SECURE,
    )
    client = WeaviateClient(connection_params=connection_params)
    client.connect()
    if not client.is_ready():
        raise RuntimeError("Weaviate client connected but instance is NOT ready.")
    return client


def http_url() -> str:
    scheme = "https" if config.WEAVIATE_SECURE else "http"
    return f"{scheme}://{config.WEAVIATE_HOST}:{config.WEAVIATE_PORT}"


def grpc_url() -> str:
    scheme = "grpcs" if config.WEAVIATE_GRPC_SECURE else "grpc"
    return f"{scheme}://{config.WEAVIATE_GRPC_HOST}:{config.WEAVIATE_GRPC_PORT}"


def aggregate_count(col) -> Optional[int]:
    try:
        agg = col.aggregate.over_all(total_count=True)
        return safe_getattr(agg, "total_count", None)
    except Exception as e:
        log.warning("Aggregate total_count failed: %s", e)
        return None


def fetch_objects(col, props: List[str], limit: int = 5) -> List[Dict[str, Any]]:
    res = col.query.fetch_objects(limit=limit, return_properties=props)
    objs = safe_getattr(res, "objects", []) or []
    out: List[Dict[str, Any]] = []
    for o in objs:
        out.append(safe_getattr(o, "properties", {}) or {})
    return out


def fetch_objects_with_metadata_vector(col, props: List[str], limit: int = 1) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    Best-effort: fetch sample + vector dimension if supported by this client.
    """
    samples: List[Dict[str, Any]] = []
    vec_dim: Optional[int] = None

    if MetadataQuery is None:
        return fetch_objects(col, props, limit=limit), None

    try:
        res = col.query.fetch_objects(
            limit=limit,
            return_properties=props,
            return_metadata=MetadataQuery(vector=True),
        )
        objs = safe_getattr(res, "objects", []) or []
        for o in objs:
            samples.append(safe_getattr(o, "properties", {}) or {})

            v = safe_getattr(o, "vector", None)
            if v is None:
                vs = safe_getattr(o, "vectors", None)
                if isinstance(vs, dict) and vs:
                    v = next(iter(vs.values()))
            if isinstance(v, list):
                vec_dim = len(v)
    except Exception as e:
        log.warning("Vector metadata fetch failed (non-fatal): %s", e)
        samples = fetch_objects(col, props, limit=limit)
        vec_dim = None

    return samples, vec_dim


def print_collection_schema(col, name: str) -> List[str]:
    cfg = col.config.get()
    props = safe_getattr(cfg, "properties", []) or []

    print(f"\n--- Schema: {name} ---", flush=True)
    vectorizer = safe_getattr(cfg, "vectorizer", None)
    vector_index_type = safe_getattr(cfg, "vector_index_type", None)
    if vectorizer is not None:
        print(f"{name}.vectorizer         : {vectorizer}", flush=True)
    if vector_index_type is not None:
        print(f"{name}.vector_index_type : {vector_index_type}", flush=True)

    prop_names: List[str] = []
    for prop in props:
        pname = safe_getattr(prop, "name", None)
        if not pname:
            continue
        prop_names.append(pname)
        dtype = safe_getattr(prop, "data_type", None)
        tokenization = safe_getattr(prop, "tokenization", None)
        index_filterable = safe_getattr(prop, "index_filterable", None)
        index_searchable = safe_getattr(prop, "index_searchable", None)

        print(
            f"  - {pname:20s} | data_type={dtype} | tokenization={tokenization} | "
            f"filterable={index_filterable} | searchable={index_searchable}",
            flush=True
        )

    print(f"{name}.properties: {prop_names}", flush=True)
    return prop_names


def ensure_required_fields(collection_name: str, prop_names: List[str], required: List[str]) -> None:
    missing = [r for r in required if r not in prop_names]
    if missing:
        raise RuntimeError(f"FAIL: Collection '{collection_name}' missing required properties: {missing}")


def has_filter_support() -> bool:
    return Filter is not None


def filter_eq(prop: str, value: Any):
    if Filter is None:
        raise RuntimeError("Filter API not available. Upgrade weaviate-client v4 or adjust script.")
    return Filter.by_property(prop).equal(value)


def fetch_where(col, where_filter, props: List[str], limit: int = 2000) -> List[Dict[str, Any]]:
    res = col.query.fetch_objects(limit=limit, filters=where_filter, return_properties=props)
    objs = safe_getattr(res, "objects", []) or []
    return [safe_getattr(o, "properties", {}) or {} for o in objs]


def endpoint_style_stats(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    pat = re.compile(r"^n\d+$", re.IGNORECASE)
    endpoints: List[str] = []
    for e in edges:
        s = str(e.get("source_entity", "")).strip()
        t = str(e.get("target_entity", "")).strip()
        if s:
            endpoints.append(s)
        if t:
            endpoints.append(t)
    nx_like = sum(1 for x in endpoints if pat.match(x))
    return {
        "total_endpoints": len(endpoints),
        "nx_like": nx_like,
        "nx_like_pct": round(100.0 * nx_like / max(1, len(endpoints)), 2),
    }


def build_nodes_index_by_group(
    nodes: List[Dict[str, Any]],
    group_key: str,
    keys: List[str],
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Builds: group_value -> key_name -> set(values)
    Example: filename -> {"node_id": {"n1","n2"}, "node_name": {"X","Y"}}
    """
    idx: Dict[str, Dict[str, Set[str]]] = {}
    for n in nodes:
        gv = str(n.get(group_key, "")).strip()
        if not gv:
            continue
        if gv not in idx:
            idx[gv] = {k: set() for k in keys}
        for k in keys:
            v = str(n.get(k, "")).strip()
            if v:
                idx[gv][k].add(v)
    return idx


def join_integrity_stats(
    node_col,
    edge_col,
    node_prop_names: List[str],
    sample_edges_limit: int = 300,
) -> Dict[str, Any]:
    """
    Try to resolve edge endpoints to nodes. We test four strategies:
      1) source_id + node_id
      2) source_id + node_name
      3) source_filename + node_id
      4) source_filename + node_name

    Returns detailed stats + top unresolved endpoint examples.
    """
    edge_props = ["source_entity", "target_entity", "relationship_type", "source_id", "source_filename"]
    edges = fetch_objects(edge_col, edge_props, limit=min(sample_edges_limit, 300))
    out: Dict[str, Any] = {
        "sample_edges": len(edges),
        "strategies": {},
        "unresolved_examples": {},
        "notes": [],
    }
    if not edges:
        out["notes"].append("No edges in sample; cannot test joins.")
        return out

    node_has_node_id = "node_id" in node_prop_names
    node_keys = ["node_name"]
    if node_has_node_id:
        node_keys.append("node_id")

    # If server-side filtering is available, do per-group fetch.
    # If not, fetch all nodes (bounded) and do local grouping.
    node_props = ["node_name", "node_type", "description", "source_id", "source_filename"] + node_keys

    if has_filter_support():
        # Cache nodes per group to avoid repeated calls
        cache: Dict[Tuple[str, str], Dict[str, Set[str]]] = {}  # (group_key, group_val) -> key -> set()

        def get_group_index(group_key: str, group_val: str) -> Dict[str, Set[str]]:
            ck = (group_key, group_val)
            if ck in cache:
                return cache[ck]
            wf = filter_eq(group_key, group_val)
            rows = fetch_where(node_col, wf, node_props, limit=3000)
            group_idx = {k: set() for k in node_keys}
            for r in rows:
                for k in node_keys:
                    v = str(r.get(k, "")).strip()
                    if v:
                        group_idx[k].add(v)
            cache[ck] = group_idx
            return group_idx

        def test_strategy(label: str, group_key: str, node_key: str) -> None:
            tested = 0
            src_ok = 0
            tgt_ok = 0
            both_ok = 0
            unresolved_src: Dict[str, int] = {}
            unresolved_tgt: Dict[str, int] = {}

            for e in edges:
                gv = str(e.get(group_key, "")).strip()
                s = str(e.get("source_entity", "")).strip()
                t = str(e.get("target_entity", "")).strip()
                if not gv or not s or not t:
                    continue
                tested += 1
                gidx = get_group_index(group_key, gv)

                s_res = s in gidx.get(node_key, set())
                t_res = t in gidx.get(node_key, set())
                if s_res:
                    src_ok += 1
                else:
                    unresolved_src[s] = unresolved_src.get(s, 0) + 1
                if t_res:
                    tgt_ok += 1
                else:
                    unresolved_tgt[t] = unresolved_tgt.get(t, 0) + 1
                if s_res and t_res:
                    both_ok += 1

            out["strategies"][label] = {
                "tested_edges": tested,
                "resolved_source_pct": round(100.0 * src_ok / max(1, tested), 2),
                "resolved_target_pct": round(100.0 * tgt_ok / max(1, tested), 2),
                "both_endpoints_resolved_pct": round(100.0 * both_ok / max(1, tested), 2),
            }
            # Save top unresolved examples
            out["unresolved_examples"][label] = {
                "top_unresolved_source": sorted(unresolved_src.items(), key=lambda x: x[1], reverse=True)[:10],
                "top_unresolved_target": sorted(unresolved_tgt.items(), key=lambda x: x[1], reverse=True)[:10],
            }

        # strategy list
        if node_has_node_id:
            test_strategy("source_id + node_id", "source_id", "node_id")
        test_strategy("source_id + node_name", "source_id", "node_name")
        if node_has_node_id:
            test_strategy("source_filename + node_id", "source_filename", "node_id")
        test_strategy("source_filename + node_name", "source_filename", "node_name")

        return out

    # Fallback: local join (fetch all nodes in one go; safe for your ~2k nodes)
    out["notes"].append("Filter API not available; using local join (fetch all nodes).")
    all_nodes = fetch_objects(node_col, node_props, limit=10000)
    idx_by_filename = build_nodes_index_by_group(all_nodes, "source_filename", node_keys)
    idx_by_sourceid = build_nodes_index_by_group(all_nodes, "source_id", node_keys)

    def test_strategy_local(label: str, group_key: str, node_key: str) -> None:
        tested = 0
        src_ok = 0
        tgt_ok = 0
        both_ok = 0

        gidx = idx_by_filename if group_key == "source_filename" else idx_by_sourceid

        for e in edges:
            gv = str(e.get(group_key, "")).strip()
            s = str(e.get("source_entity", "")).strip()
            t = str(e.get("target_entity", "")).strip()
            if not gv or not s or not t:
                continue
            tested += 1
            group = gidx.get(gv, {})
            s_res = s in group.get(node_key, set())
            t_res = t in group.get(node_key, set())
            if s_res:
                src_ok += 1
            if t_res:
                tgt_ok += 1
            if s_res and t_res:
                both_ok += 1

        out["strategies"][label] = {
            "tested_edges": tested,
            "resolved_source_pct": round(100.0 * src_ok / max(1, tested), 2),
            "resolved_target_pct": round(100.0 * tgt_ok / max(1, tested), 2),
            "both_endpoints_resolved_pct": round(100.0 * both_ok / max(1, tested), 2),
        }

    if node_has_node_id:
        test_strategy_local("source_id + node_id", "source_id", "node_id")
    test_strategy_local("source_id + node_name", "source_id", "node_name")
    if node_has_node_id:
        test_strategy_local("source_filename + node_id", "source_filename", "node_id")
    test_strategy_local("source_filename + node_name", "source_filename", "node_name")

    return out


def top_filenames_by_edge_count(edge_col, limit_edges: int = 2000) -> List[Tuple[str, int]]:
    """
    Best-effort: fetch a bunch of edges and count by source_filename.
    (Weaviate doesn't always support group-by aggregate easily across versions.)
    """
    edges = fetch_objects(edge_col, ["source_filename"], limit=min(limit_edges, 5000))
    counts: Dict[str, int] = {}
    for e in edges:
        fn = str(e.get("source_filename", "")).strip()
        if not fn:
            continue
        counts[fn] = counts.get(fn, 0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]


def per_file_summary(node_col, edge_col, node_prop_names: List[str]) -> None:
    """
    Prints per-file sanity for top 5 files by edge count:
    - edges count
    - nodes count (if filter supported)
    - quick join success by filename strategy
    """
    print("\nPer-file graph sanity (top files by edge count):", flush=True)
    top_files = top_filenames_by_edge_count(edge_col)
    if not top_files:
        print("  (No edges found to compute per-file summary.)", flush=True)
        return

    node_has_node_id = "node_id" in node_prop_names

    for fn, ecount in top_files[:5]:
        print(f"\n  File: {fn}", flush=True)
        print(f"    edges (sampled count): {ecount}", flush=True)

        if has_filter_support():
            try:
                n_where = filter_eq("source_filename", fn)
                ncount = aggregate_count_with_filter(node_col, n_where)
                print(f"    nodes (by filter): {ncount if ncount is not None else '(unknown)'}", flush=True)

                # join check: fetch nodes for that filename -> sets
                node_keys = ["node_name"] + (["node_id"] if node_has_node_id else [])
                node_rows = fetch_where(
                    node_col,
                    n_where,
                    ["node_name", "source_filename"] + node_keys,
                    limit=3000,
                )
                sets = {k: set() for k in node_keys}
                for r in node_rows:
                    for k in node_keys:
                        v = str(r.get(k, "")).strip()
                        if v:
                            sets[k].add(v)

                # edges for that filename
                e_where = filter_eq("source_filename", fn)
                edge_rows = fetch_where(
                    edge_col,
                    e_where,
                    ["source_entity", "target_entity", "source_filename"],
                    limit=3000,
                )

                def join_pct(node_key: str) -> Tuple[float, float, float]:
                    tested = 0
                    src_ok = 0
                    tgt_ok = 0
                    both_ok = 0
                    for e in edge_rows:
                        s = str(e.get("source_entity", "")).strip()
                        t = str(e.get("target_entity", "")).strip()
                        if not s or not t:
                            continue
                        tested += 1
                        s_res = s in sets.get(node_key, set())
                        t_res = t in sets.get(node_key, set())
                        if s_res:
                            src_ok += 1
                        if t_res:
                            tgt_ok += 1
                        if s_res and t_res:
                            both_ok += 1
                    return (
                        round(100.0 * src_ok / max(1, tested), 2),
                        round(100.0 * tgt_ok / max(1, tested), 2),
                        round(100.0 * both_ok / max(1, tested), 2),
                    )

                # print join by node_id if present
                if node_has_node_id:
                    s, t, b = join_pct("node_id")
                    print(f"    join by filename+node_id:  src={s}% tgt={t}% both={b}%", flush=True)

                s, t, b = join_pct("node_name")
                print(f"    join by filename+node_name: src={s}% tgt={t}% both={b}%", flush=True)

            except Exception as e:
                print(f"    (per-file summary failed: {e})", flush=True)
        else:
            print("    (Filter not available; per-file summary skipped.)", flush=True)


def aggregate_count_with_filter(col, where_filter) -> Optional[int]:
    try:
        agg = col.aggregate.over_all(total_count=True, filters=where_filter)
        return safe_getattr(agg, "total_count", None)
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> int:
    client: Optional[WeaviateClient] = None
    embedder = None

    try:
        p("JusticeFusion Weaviate FULL Diagnostics")

        now = datetime.now(timezone.utc).isoformat()
        print(f"Generated at: {now}", flush=True)
        print(f"HTTP  : {http_url()}", flush=True)
        print(f"gRPC  : {grpc_url()}", flush=True)
        print(
            f"Classes: Document='{config.WEAVIATE_DOC_CLASS}', "
            f"LegalArticle='{config.WEAVIATE_ARTICLE_CLASS}', "
            f"GraphNodes='{config.WEAVIATE_NODE_CLASS}', "
            f"GraphEdges='{config.WEAVIATE_EDGE_CLASS}'",
            flush=True
        )
        print(f"RUN_MODE: {getattr(config, 'RUN_MODE', '(unknown)')}", flush=True)

        p("Step 1: Connect + readiness + /v1/meta")

        client = connect_client()
        print("Weaviate client connected. is_ready=True", flush=True)

        # Try /v1/meta via client (best-effort)
        try:
            # weaviate-client exposes meta on v4 under client.misc or client.get_meta in some versions
            meta = None
            if hasattr(client, "misc") and hasattr(client.misc, "meta"):
                meta = client.misc.meta()
            elif hasattr(client, "get_meta"):
                meta = client.get_meta()
            if meta:
                print("Meta (best-effort):", flush=True)
                print(json.dumps(meta, indent=2)[:2000], flush=True)
            else:
                print("Meta: (not available via this client version; /v1/meta was tested in db_connection_diagnostics)", flush=True)
        except Exception as e:
            print(f"Meta fetch failed (non-fatal): {e}", flush=True)

        p("Step 2: Collection existence")

        doc_class = config.WEAVIATE_DOC_CLASS
        art_class = config.WEAVIATE_ARTICLE_CLASS
        node_class = config.WEAVIATE_NODE_CLASS
        edge_class = config.WEAVIATE_EDGE_CLASS

        for cname in [doc_class, art_class, node_class, edge_class]:
            exists = client.collections.exists(cname)
            print(f"{cname:12s}: exists={exists}", flush=True)
            if not exists:
                raise RuntimeError(f"FAIL: Required collection '{cname}' does not exist.")

        doc_col = client.collections.get(doc_class)
        art_col = client.collections.get(art_class)
        node_col = client.collections.get(node_class)
        edge_col = client.collections.get(edge_class)

        p("Step 3: Schema dump (properties + types)")

        doc_props = print_collection_schema(doc_col, doc_class)
        art_props = print_collection_schema(art_col, art_class)
        node_props = print_collection_schema(node_col, node_class)
        edge_props = print_collection_schema(edge_col, edge_class)

        # Required fields that are crucial for your pipeline to work
        ensure_required_fields(doc_class, doc_props, ["filename", "chunk_id", "source_id"])
        ensure_required_fields(node_class, node_props, ["node_name", "node_type", "description", "source_id", "source_filename"])
        ensure_required_fields(edge_class, edge_props, ["source_entity", "target_entity", "relationship_type", "source_id", "source_filename"])

        node_has_node_id = "node_id" in node_props
        print(f"\nGraphNodes has 'node_id' property: {node_has_node_id}", flush=True)
        if not node_has_node_id:
            print("WARNING: If GraphEdges endpoints look like 'n7', your schema MUST include node_id or edges won't connect.", flush=True)

        p("Step 4: Counts + samples + vector presence (best-effort)")

        def report_collection(name: str, col, sample_props: List[str]) -> None:
            cnt = aggregate_count(col)
            sample, vec_dim = fetch_objects_with_metadata_vector(col, sample_props, limit=1)
            print(f"\n[{name}]", flush=True)
            print(f"  total_count : {cnt if cnt is not None else '(unknown)'}", flush=True)
            if sample:
                print(f"  sample      : {sample[0]}", flush=True)
            else:
                print("  sample      : 0 objects", flush=True)
            print(f"  vector_dim  : {vec_dim if vec_dim is not None else '(unknown/not returned)'}", flush=True)

        report_collection(doc_class, doc_col, ["filename", "chunk_id", "source_id"])
        report_collection(art_class, art_col, ["law_title", "article_number", "source_id"])
        node_sample_props = ["node_name", "node_type", "description", "source_id", "source_filename"] + (["node_id"] if node_has_node_id else [])
        report_collection(node_class, node_col, node_sample_props)
        report_collection(edge_class, edge_col, ["source_entity", "target_entity", "relationship_type", "source_id", "source_filename"])

        if (aggregate_count(art_col) or 0) == 0:
            print("\nWARNING: LegalArticle is empty. This means law ingestion/classification likely isn't populating LegalArticle.", flush=True)

        p("Step 5: GraphEdges endpoint style + samples")

        edge_samples = fetch_objects(edge_col, ["source_entity", "target_entity", "relationship_type", "source_id", "source_filename"], limit=10)
        print(f"Fetched {len(edge_samples)} edge samples:", flush=True)
        for i, e in enumerate(edge_samples, 1):
            print(f"  Edge#{i}: {e}", flush=True)

        style = endpoint_style_stats(edge_samples if edge_samples else [])
        print("\nEdge endpoint style stats (sample=10):", style, flush=True)
        if style["nx_like_pct"] >= 50.0:
            print(
                "WARNING: Many endpoints look like n7/n12. That usually means edges reference GraphNodes.node_id.\n"
                "         If your UI joins by node_name or UUID, edges will appear disconnected.",
                flush=True
            )

        p("Step 6: CRITICAL edge->node join integrity")

        join_stats = join_integrity_stats(
            node_col=node_col,
            edge_col=edge_col,
            node_prop_names=node_props,
            sample_edges_limit=300,
        )
        print("Join stats (JSON):", flush=True)
        print(json.dumps(join_stats, indent=2)[:4000], flush=True)  # prevent insanely long dumps

        # Quick human diagnosis from join stats
        strat = join_stats.get("strategies", {}) or {}
        s_id_nid = strat.get("source_id + node_id")
        s_fn_nid = strat.get("source_filename + node_id")
        s_id_nname = strat.get("source_id + node_name")
        s_fn_nname = strat.get("source_filename + node_name")

        print("\nDiagnosis hints:", flush=True)

        if s_id_nid and s_id_nname:
            if s_id_nid.get("both_endpoints_resolved_pct", 0) > 50 and s_id_nname.get("both_endpoints_resolved_pct", 0) < 5:
                print(
                    "- Edges resolve by node_id but NOT by node_name. Your visualization must join by node_id OR store node UUID refs on edges\n"
                    "  OR rewrite edge endpoints from node_id -> node_name during insertion.",
                    flush=True
                )
        if s_fn_nid and s_id_nid:
            if s_fn_nid.get("both_endpoints_resolved_pct", 0) > s_id_nid.get("both_endpoints_resolved_pct", 0) + 25:
                print(
                    "- Joining by source_filename works much better than by source_id.\n"
                    "  This often means GraphEdges.source_id is inconsistent/overwritten (edge UUID dedup across docs).",
                    flush=True
                )

        p("Step 7: Per-file graph sanity (top files by edge count)")

        per_file_summary(node_col, edge_col, node_props)

        p("Step 8: near_vector smoke test on GraphNodes (embedding model)")

        print("Loading EmbeddingModelManager...", flush=True)
        from embedding_model import EmbeddingModelManager  # lazy import
        embedder = EmbeddingModelManager()

        test_query = "ustavno pravo bosna i hercegovina specifičnosti"
        vecs = embedder.get_embeddings([test_query], prompt_name=config.EMBEDDING_PROMPT_QUERY, batch_size=1)
        if not vecs or not isinstance(vecs, list) or not vecs[0]:
            raise RuntimeError("FAIL: EmbeddingModelManager returned no vector for test query.")

        qvec = vecs[0]
        print(f"Query vector ready. dim={len(qvec)}", flush=True)

        print("Running near_vector query on GraphNodes...", flush=True)
        if MetadataQuery is not None:
            res2 = node_col.query.near_vector(
                near_vector=qvec,
                limit=5,
                return_properties=["node_name", "node_type", "description", "source_filename"],
                return_metadata=MetadataQuery(distance=True),
            )
        else:
            res2 = node_col.query.near_vector(
                near_vector=qvec,
                limit=5,
                return_properties=["node_name", "node_type", "description", "source_filename"],
            )

        objs2 = safe_getattr(res2, "objects", []) or []
        print(f"near_vector returned {len(objs2)} objects.", flush=True)
        for i, o in enumerate(objs2[:5], 1):
            props = safe_getattr(o, "properties", {}) or {}
            md = safe_getattr(o, "metadata", None)
            dist = safe_getattr(md, "distance", None) if md else None
            print(f"Result #{i} distance={dist} props={props}", flush=True)

        p("SUCCESS: FULL diagnostics completed")

        print(
            "If your UI still shows disconnected graphs, the join integrity section above tells you exactly why.\n"
            "Most common root causes:\n"
            "  1) Edge endpoints store node_id (n7) but GraphNodes.node_id missing or UI joins by node_name.\n"
            "  2) GraphEdges.source_id overwritten due to edge UUID dedup (source_id excluded from edge UUID).\n"
            "  3) UI queries wrong property names for edges.",
            flush=True
        )

        return 0

    except Exception as e:
        p("FAILURE")
        print(str(e), flush=True)
        log.error("Exception", exc_info=True)
        return 1

    finally:
        try:
            if embedder:
                embedder.close()
        except Exception:
            pass
        try:
            if client and client.is_connected():
                client.close()
        except Exception:
            pass


if __name__ == "__main__":
    # --- ADDED: Redirect stdout and stderr to save to file ---
    sys.stdout = TeeLogger("diagnostic_output.txt")
    sys.stderr = sys.stdout  # Capture stack traces as well
    # ---------------------------------------------------------
    sys.exit(main())