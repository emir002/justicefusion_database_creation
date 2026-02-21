#!/usr/bin/env python3
"""
Schema + data smoke test for Weaviate GraphNodes.

Checks:
- GraphNodes schema includes 'description'
- GraphNodes has at least 1 object (optional warning if 0)
- Can fetch objects returning 'description'
- Can run near_vector on GraphNodes using a real embedding vector

Exit code:
- 0 on success
- 1 on failure
"""

import sys
import logging

import weaviate
from weaviate import WeaviateClient

import config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("schema_inspect")


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


def main() -> int:
    client = None
    embedder = None
    try:
        log.info("Connecting to Weaviate...")
        client = connect_client()
        log.info("Connected and ready.")

        node_class = config.WEAVIATE_NODE_CLASS
        if not client.collections.exists(node_class):
            raise RuntimeError(f"Collection '{node_class}' does not exist.")

        col = client.collections.get(node_class)

        # ---- Schema inspection ----
        cfg = col.config.get()
        prop_names = [p.name for p in getattr(cfg, "properties", [])]
        log.info("GraphNodes properties: %s", prop_names)

        if "description" not in prop_names:
            raise RuntimeError(
                "FAIL: GraphNodes schema missing required property 'description'. "
                "Recreate schema in FIRST_RUN after adding Property(name='description', ...)."
            )
        log.info("OK: 'description' property exists in GraphNodes schema.")

        # ---- Count objects (best-effort) ----
        total_count = None
        try:
            agg = col.aggregate.over_all(total_count=True)
            total_count = getattr(agg, "total_count", None)
        except Exception as e:
            log.warning("Could not aggregate total_count: %s", e)

        if total_count is not None:
            log.info("GraphNodes total_count: %s", total_count)
        else:
            log.info("GraphNodes total_count: (unknown)")

        # ---- Fetch sample objects ----
        try:
            res = col.query.fetch_objects(
                limit=3,
                return_properties=["node_name", "node_type", "description", "source_id", "source_filename"],
            )
            objs = getattr(res, "objects", []) or []
            log.info("Fetched %d GraphNodes objects (sample).", len(objs))
            for i, o in enumerate(objs[:3], start=1):
                props = getattr(o, "properties", {}) or {}
                log.info("Sample #%d props: %s", i, props)
        except Exception as e:
            raise RuntimeError(f"FAIL: fetch_objects failed: {e}")

        # ---- Build a query vector using the embedding model ----
        log.info("Loading EmbeddingModelManager for query vector...")
        from embedding_model import EmbeddingModelManager  # LAZY IMPORT (avoid stanza init until needed)

        embedder = EmbeddingModelManager()

        test_query = "ustavno pravo bosna i hercegovina specifičnosti"
        vecs = embedder.get_embeddings([test_query], prompt_name=config.EMBEDDING_PROMPT_QUERY, batch_size=1)
        if not vecs or not isinstance(vecs, list) or not vecs[0]:
            raise RuntimeError("FAIL: EmbeddingModelManager returned no vector for test query.")
        qvec = vecs[0]
        log.info("Query vector ready. dim=%d", len(qvec))

        # ---- near_vector smoke test ----
        log.info("Running near_vector query on GraphNodes...")
        res2 = col.query.near_vector(
            near_vector=qvec,
            limit=5,
            return_properties=["node_name", "node_type", "description", "source_filename"],
        )
        objs2 = getattr(res2, "objects", []) or []
        log.info("near_vector returned %d objects.", len(objs2))

        for i, o in enumerate(objs2[:5], start=1):
            props = getattr(o, "properties", {}) or {}
            md = getattr(o, "metadata", None)
            dist = getattr(md, "distance", None) if md else None
            log.info("Result #%d distance=%s props=%s", i, dist, props)

        if len(objs2) == 0:
            log.warning(
                "near_vector returned 0 results. This usually means GraphNodes vectors were not stored "
                "(or the collection is empty). Check main_indexer upsert vectors_list for GraphNodes."
            )

        log.info("SUCCESS: Schema + fetch + near_vector checks completed.")
        return 0

    except Exception as e:
        log.error(str(e), exc_info=True)
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
    sys.exit(main())
