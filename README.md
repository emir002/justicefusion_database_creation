# justicefusion_database_creation

## Graph data contract (updated)

To render per-file graphs correctly, consume graph data with these keys:

- Node unique id: `node_key` (`{source_id}::{node_id}`)
- Node label: `node_name`
- Edge source: `source_node_key`
- Edge target: `target_node_key`

Fallback (if UI update is delayed):

- Build node id as `f"{source_id}::{node_id}"`
- Join edges against the same composite key

Do **not** join edges to nodes by `node_name` or UUID.

## Reindex procedure after graph schema update

1. Run with `FIRST_RUN` to recreate schema (or set `WEAVIATE_RECREATE_GRAPH_SCHEMA=1` in `UPDATE` mode to recreate graph collections).
2. Re-index documents.
3. Run `schema_inspect.py` and verify:
   - edge/node joins by `node_key` are near-complete
   - graph edge counts are stable across reruns
   - per-file graph renders non-zero edges
