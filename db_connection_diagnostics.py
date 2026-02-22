#!/usr/bin/env python3
"""Weaviate connectivity diagnostics for JusticeFusion."""

from __future__ import annotations

import json
import os
import socket
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import weaviate
from weaviate import WeaviateClient

import config


def _kv_line(key: str, value: Any, width: int = 34) -> str:
    return f"{key:<{width}}: {value}"


def _to_bool_env(raw_value: Optional[str], default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _effective_config_snapshot() -> Dict[str, Any]:
    http_scheme = "https" if config.WEAVIATE_SECURE else "http"
    grpc_scheme = "grpcs" if config.WEAVIATE_GRPC_SECURE else "grpc"
    return {
        "RUN_MODE": config.RUN_MODE,
        "WEAVIATE_HOST": config.WEAVIATE_HOST,
        "WEAVIATE_PORT": config.WEAVIATE_PORT,
        "WEAVIATE_SECURE": config.WEAVIATE_SECURE,
        "WEAVIATE_GRPC_HOST": config.WEAVIATE_GRPC_HOST,
        "WEAVIATE_GRPC_PORT": config.WEAVIATE_GRPC_PORT,
        "WEAVIATE_GRPC_SECURE": config.WEAVIATE_GRPC_SECURE,
        "HTTP_URL": f"{http_scheme}://{config.WEAVIATE_HOST}:{config.WEAVIATE_PORT}",
        "GRPC_URL": f"{grpc_scheme}://{config.WEAVIATE_GRPC_HOST}:{config.WEAVIATE_GRPC_PORT}",
        "WEAVIATE_DOC_CLASS": config.WEAVIATE_DOC_CLASS,
        "WEAVIATE_ARTICLE_CLASS": config.WEAVIATE_ARTICLE_CLASS,
        "WEAVIATE_NODE_CLASS": config.WEAVIATE_NODE_CLASS,
        "WEAVIATE_EDGE_CLASS": config.WEAVIATE_EDGE_CLASS,
    }


def _capture_env_vars() -> List[Tuple[str, bool, Optional[str]]]:
    vars_to_check = [
        "WEAVIATE_HOST", "WEAVIATE_PORT", "WEAVIATE_SECURE",
        "WEAVIATE_GRPC_HOST", "WEAVIATE_GRPC_PORT", "WEAVIATE_GRPC_SECURE",
        "WEAVIATE_GRPC_GRPC_SECURE",
    ]
    return [(name, name in os.environ, os.environ.get(name)) for name in vars_to_check]


def _socket_reachability(host: str, port: int, timeout: int = 3) -> Tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "reachable"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _http_meta_check(http_url: str, timeout: int = 5) -> Dict[str, Any]:
    target = f"{http_url}/v1/meta"
    try:
        with urllib.request.urlopen(target, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            keys = sorted(parsed.keys()) if isinstance(parsed, dict) else []
            version_fields = {
                k: parsed.get(k)
                for k in ("version", "gitHash", "modules")
                if isinstance(parsed, dict) and k in parsed
            }
            return {
                "ok": True,
                "status": response.status,
                "keys": keys,
                "version_fields": version_fields,
                "error": None,
            }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {
            "ok": False,
            "status": None,
            "keys": [],
            "version_fields": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _build_connection_params():
    return weaviate.connect.ConnectionParams.from_params(
        http_host=config.WEAVIATE_HOST,
        http_port=config.WEAVIATE_PORT,
        http_secure=config.WEAVIATE_SECURE,
        grpc_host=config.WEAVIATE_GRPC_HOST,
        grpc_port=config.WEAVIATE_GRPC_PORT,
        grpc_secure=config.WEAVIATE_GRPC_SECURE,
    )


def _safe_total_count(collection: Any) -> Tuple[Optional[int], Optional[str]]:
    try:
        agg = collection.aggregate.over_all(total_count=True)
        total_count = getattr(agg, "total_count", None)
        return total_count, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _safe_fetch_one(collection: Any, return_properties: List[str]) -> Tuple[Optional[Any], Optional[str]]:
    try:
        result = collection.query.fetch_objects(limit=1, return_properties=return_properties)
        objects = getattr(result, "objects", []) or []
        if not objects:
            return None, None
        obj = objects[0]
        props = getattr(obj, "properties", obj)
        return props, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def build_diagnostics_report() -> str:
    lines: List[str] = []
    lines.append("=" * 88)
    lines.append("JusticeFusion DB Connection Diagnostics")
    lines.append(_kv_line("Generated at", datetime.utcnow().isoformat() + "Z"))
    lines.append("=" * 88)

    snap = _effective_config_snapshot()
    lines.append("\n[1] Effective config values (as used by code)")
    for key in [
        "RUN_MODE",
        "WEAVIATE_HOST", "WEAVIATE_PORT", "WEAVIATE_SECURE",
        "WEAVIATE_GRPC_HOST", "WEAVIATE_GRPC_PORT", "WEAVIATE_GRPC_SECURE",
        "HTTP_URL", "GRPC_URL",
        "WEAVIATE_DOC_CLASS", "WEAVIATE_ARTICLE_CLASS", "WEAVIATE_NODE_CLASS", "WEAVIATE_EDGE_CLASS",
    ]:
        lines.append(_kv_line(key, snap[key]))

    lines.append("\n[2] Environment variable presence and values")
    env_rows = _capture_env_vars()
    for name, present, value in env_rows:
        rendered = value if value is not None else "<unset>"
        lines.append(_kv_line(name, f"present={present}; value={rendered}"))
    if os.getenv("WEAVIATE_GRPC_GRPC_SECURE") is not None:
        typo_val = os.getenv("WEAVIATE_GRPC_GRPC_SECURE")
        canonical_val = os.getenv("WEAVIATE_GRPC_SECURE")
        resolved = _to_bool_env(canonical_val, default=_to_bool_env(typo_val, default=False))
        lines.append("WARNING: WEAVIATE_GRPC_GRPC_SECURE is set and is likely a typo.")
        lines.append(_kv_line("Canonical env (preferred)", f"WEAVIATE_GRPC_SECURE={canonical_val}"))
        lines.append(_kv_line("Resolved config.WEAVIATE_GRPC_SECURE", resolved))

    lines.append("\n[3] Network reachability checks (stdlib socket.create_connection)")
    http_reachable, http_msg = _socket_reachability(config.WEAVIATE_HOST, config.WEAVIATE_PORT)
    grpc_reachable, grpc_msg = _socket_reachability(config.WEAVIATE_GRPC_HOST, config.WEAVIATE_GRPC_PORT)
    lines.append(_kv_line("HTTP reachability", f"{'PASS' if http_reachable else 'FAIL'} ({http_msg})"))
    lines.append(_kv_line("gRPC reachability", f"{'PASS' if grpc_reachable else 'FAIL'} ({grpc_msg})"))

    lines.append("\n[4] HTTP sanity check (urllib GET /v1/meta)")
    meta = _http_meta_check(snap["HTTP_URL"])
    if meta["ok"]:
        lines.append(_kv_line("HTTP status", meta["status"]))
        lines.append(_kv_line("Parsed JSON keys", ", ".join(meta["keys"]) if meta["keys"] else "<none>"))
        lines.append(_kv_line("Version-related fields", json.dumps(meta["version_fields"], ensure_ascii=False)))
    else:
        lines.append(_kv_line("/v1/meta", f"FAIL ({meta['error']})"))
        lines.append("Hint: is Weaviate running on that host/port? firewall? docker port mapping?")

    lines.append("\n[5] Weaviate Python client check")
    client: Optional[WeaviateClient] = None
    try:
        connection_params = _build_connection_params()
        client = WeaviateClient(connection_params=connection_params)
        client.connect()
        ready = client.is_ready()
        lines.append(_kv_line("client.connect()", "OK"))
        lines.append(_kv_line("client.is_ready()", ready))

        if not ready:
            lines.append("Instance reported NOT ready.")
            try:
                nodes = client.cluster.get_nodes_status()
                lines.append(_kv_line("cluster.get_nodes_status()", nodes))
            except Exception as nodes_exc:
                lines.append(_kv_line("cluster.get_nodes_status()", f"ERROR: {type(nodes_exc).__name__}: {nodes_exc}"))
            lines.append("Hints:")
            lines.append("- Check docker logs")
            lines.append(f"- curl {snap['HTTP_URL']}/v1/meta")
            lines.append("- confirm ports 8080 and 50051 exposed")

        lines.append("\n[6] Collection existence + counts + tiny read")
        collection_specs = [
            ("Document", config.WEAVIATE_DOC_CLASS, ["filename", "chunk_id", "source_id"]),
            ("LegalArticle", config.WEAVIATE_ARTICLE_CLASS, ["law_title", "article_number", "source_id"]),
            ("GraphNodes", config.WEAVIATE_NODE_CLASS, ["node_name", "node_type", "source_id", "source_filename"]),
            ("GraphEdges", config.WEAVIATE_EDGE_CLASS, ["source_entity", "target_entity", "relationship_type", "source_id", "source_filename"]),
        ]
        for label, collection_name, properties in collection_specs:
            exists = client.collections.exists(collection_name)
            lines.append(f"- {label} ({collection_name})")
            lines.append(_kv_line("  exists", exists))
            if not exists:
                continue
            collection = client.collections.get(collection_name)
            total_count, count_error = _safe_total_count(collection)
            if count_error:
                lines.append(_kv_line("  total_count", f"ERROR: {count_error}"))
            else:
                lines.append(_kv_line("  total_count", total_count if total_count is not None else "<unknown>"))

            sample, sample_error = _safe_fetch_one(collection, properties)
            if sample_error:
                lines.append(_kv_line("  sample fetch", f"ERROR: {sample_error}"))
            elif sample is None:
                lines.append(_kv_line("  sample fetch", "0 objects"))
            else:
                lines.append(_kv_line("  sample fetch", json.dumps(sample, ensure_ascii=False, default=str)))

    except Exception as exc:
        lines.append(_kv_line("client.connect()/usage", f"FAIL ({type(exc).__name__}: {exc})"))
        lines.append("Hints:")
        lines.append("- Check docker logs")
        lines.append(f"- curl {snap['HTTP_URL']}/v1/meta")
        lines.append("- confirm ports 8080 and 50051 exposed")
        lines.append("Traceback:")
        lines.extend([f"    {line}" for line in traceback.format_exc().strip().splitlines()])
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    lines.append("\n[7] How to use Weaviate in this project")
    lines.append("```python")
    lines.append("import config")
    lines.append("from weaviate_client import WeaviateManager")
    lines.append("")
    lines.append('wm = WeaviateManager(run_mode="UPDATE")')
    lines.append("col = wm.client.collections.get(config.WEAVIATE_DOC_CLASS)")
    lines.append('res = col.query.fetch_objects(limit=3, return_properties=["filename", "chunk_id", "source_id"])')
    lines.append("for obj in res.objects:")
    lines.append("    print(obj.properties)")
    lines.append("wm.close()")
    lines.append("```")

    return "\n".join(lines)


def main() -> None:
    print(build_diagnostics_report())


if __name__ == "__main__":
    main()
