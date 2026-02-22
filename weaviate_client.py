#!/usr/bin/env python3
# weaviate_client.py

import logging
import os
from typing import List, Dict, Any, Optional
from uuid import UUID
import time
import sys

# Third-party libraries
import weaviate
from weaviate import WeaviateClient
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from weaviate.exceptions import WeaviateBaseError, UnexpectedStatusCodeError, WeaviateStartUpError
from weaviate.collections.classes.filters import Filter

# Local imports
import config
import utils
import metrics

logger = logging.getLogger(__name__)

class WeaviateManager:
    """
    Manages connection to Weaviate, schema creation/verification, and data operations.
    """
    DEFAULT_MIN_BATCH_SIZE = 5

    def __init__(self, run_mode: str = "UPDATE"):
        logger.info(f"Initializing WeaviateManager in '{run_mode}' mode.")
        self.run_mode = run_mode
        self.client: Optional[WeaviateClient] = self._connect()
        if self.client:
            self._ensure_schema()
            logger.info("WeaviateManager initialized successfully.")
        else:
            logger.error("WeaviateManager initialization failed: Could not connect to Weaviate.")

    def _connect(self) -> Optional[WeaviateClient]:
        logger.info(
            f"Attempting to connect to Weaviate: http://{config.WEAVIATE_HOST}:{config.WEAVIATE_PORT} "
            f"(gRPC: grpc://{config.WEAVIATE_GRPC_HOST}:{config.WEAVIATE_GRPC_PORT})"
        )
        connection_params = weaviate.connect.ConnectionParams.from_params(
            http_host=config.WEAVIATE_HOST, http_port=config.WEAVIATE_PORT, http_secure=config.WEAVIATE_SECURE,
            grpc_host=config.WEAVIATE_GRPC_HOST, grpc_port=config.WEAVIATE_GRPC_PORT, grpc_secure=config.WEAVIATE_GRPC_SECURE,
        )

        auth_config = None
        # Example for API key auth:
        # if hasattr(config, 'WEAVIATE_API_KEY') and config.WEAVIATE_API_KEY:
        #     auth_config = weaviate.auth.AuthApiKey(api_key=config.WEAVIATE_API_KEY)

        client = WeaviateClient(
            connection_params=connection_params,
            auth_client_secret=auth_config,
        )

        try:
            client.connect()
            logger.info("Successfully established initial connection with Weaviate.")
            if not client.is_ready():
                logger.error("Weaviate client connected, but instance reports NOT ready.")
                try:
                    cluster_status = client.cluster.get_nodes_status()
                    logger.error(f"Weaviate cluster nodes status: {cluster_status}")
                except WeaviateBaseError as cluster_e:
                    logger.error(f"Could not retrieve Weaviate cluster status: {cluster_e}")
                client.close()
                return None
            logger.info("Weaviate instance is connected and ready.")
            return client
        except WeaviateStartUpError as e_startup:
            logger.error(f"Weaviate failed to start/connect: {e_startup}", exc_info=True)
        except WeaviateBaseError as e_base:
            logger.error(f"Weaviate-specific error during connection: {e_base}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error connecting to Weaviate: {e}", exc_info=True)

        if client and client.is_connected():
            client.close()
        return None

    def is_connected(self) -> bool:
        connected = self.client is not None and self.client.is_connected()
        return connected

    def close(self):
        if self.client and self.client.is_connected():
            logger.info("Closing Weaviate client connection.")
            try:
                self.client.close()
                logger.info("Weaviate client connection closed.")
            except Exception as e:
                logger.error(f"Error closing Weaviate connection: {e}", exc_info=True)
        elif self.client:
            logger.info("Weaviate client existed but was not connected. No close action taken.")
        else:
            logger.debug("No Weaviate client instance to close.")

    def _collection_exists(self, collection_name: str) -> bool:
        if not self.is_connected() or not self.client:
            logger.error(f"Cannot check collection '{collection_name}': Not connected.")
            return False
        try:
            return self.client.collections.exists(collection_name)
        except Exception as e:
            logger.error(f"Error checking existence of collection '{collection_name}': {e}", exc_info=True)
            return False

    def _create_collection(self, name: str, properties: List[Property], description: Optional[str] = None):
        if not self.is_connected() or not self.client:
            logger.error(f"Cannot create collection '{name}': Not connected.")
            return

        logger.info(f"Attempting to create Weaviate collection: '{name}'.")

        prop_details = []
        for p_item in properties:
            prop_name = p_item.name
            data_type_repr = "UNKNOWN_OR_ERROR"
            try:
                dt_attr = getattr(p_item, 'data_type', None)
                if dt_attr is not None:
                    if hasattr(dt_attr, 'value'):
                        data_type_repr = dt_attr.value
                    else:
                        data_type_repr = str(dt_attr)
                else:
                    data_type_repr = "DATATYPE_ATTRIBUTE_NOT_FOUND_VIA_GETATTR"
            except Exception as e_log_prop:
                data_type_repr = f"ERROR_LOGGING_DATATYPE: {type(e_log_prop).__name__}"
            prop_details.append((prop_name, data_type_repr))
        logger.debug(f"Properties for collection '{name}': {prop_details}")

        try:
            self.client.collections.create(
                name=name,
                properties=properties,
                description=description,
                vectorizer_config=Configure.Vectorizer.none()
            )
            logger.info(f"Collection '{name}' created successfully.")
        except UnexpectedStatusCodeError as e:
            if e.status_code == 422 and "already exists" in str(e).lower():  # type: ignore
                logger.warning(f"Collection '{name}' creation reported 422 but seems to already exist. Message: {e}")
            else:
                logger.error(f"Failed to create collection '{name}' (status {e.status_code}): {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error creating collection '{name}': {e}", exc_info=True)

    def _delete_collection(self, collection_name: str):
        if not self.is_connected() or not self.client:
            logger.error(f"Cannot delete collection '{collection_name}': Not connected.")
            return
        if self._collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' exists. Proceeding with deletion...")
            try:
                self.client.collections.delete(collection_name)
                logger.info(f"Collection '{collection_name}' deleted successfully.")
            except Exception as e:
                logger.error(f"Failed to delete collection '{collection_name}': {e}", exc_info=True)
        else:
            logger.info(f"Collection '{collection_name}' does not exist. No deletion needed.")

    @staticmethod
    def _canonical_graph_schema() -> Dict[str, Dict[str, DataType]]:
        return {
            config.WEAVIATE_NODE_CLASS: {
                "node_id": DataType.TEXT,
                "node_name": DataType.TEXT,
                "node_type": DataType.TEXT,
                "source_id": DataType.TEXT,
                "source_filename": DataType.TEXT,
                "description": DataType.TEXT,
            },
            config.WEAVIATE_EDGE_CLASS: {
                "source_entity": DataType.TEXT,
                "target_entity": DataType.TEXT,
                "relationship_type": DataType.TEXT,
                "source_id": DataType.TEXT,
                "source_filename": DataType.TEXT,
            },
        }

    def _extract_collection_properties(self, collection_name: str) -> Dict[str, str]:
        """Best-effort extraction of collection property names/types from Weaviate config object."""
        if not self.client:
            return {}
        try:
            collection = self.client.collections.get(collection_name)
            config_obj = collection.config.get()
            props = getattr(config_obj, "properties", None)
            if props is None and hasattr(config_obj, "to_dict"):
                props = config_obj.to_dict().get("properties", [])

            out: Dict[str, str] = {}
            if isinstance(props, list):
                for p in props:
                    if isinstance(p, dict):
                        name = str(p.get("name", ""))
                        dtype = p.get("dataType") or p.get("data_type") or p.get("dataType", [])
                        if isinstance(dtype, list) and dtype:
                            dtype = dtype[0]
                        out[name] = str(dtype).lower()
                    else:
                        name = getattr(p, "name", "")
                        dtype = getattr(p, "data_type", "")
                        dtype_str = ""
                        if isinstance(dtype, list) and dtype:
                            dtype_str = str(dtype[0])
                        else:
                            dtype_str = str(dtype)
                        out[str(name)] = dtype_str.lower()
            return out
        except Exception as e:
            logger.error(f"Failed reading schema for '{collection_name}': {e}", exc_info=True)
            return {}

    def _ensure_collection_properties(self, collection_name: str, required_properties: List[Property]):
        if not self.is_connected() or not self.client:
            logger.error(f"Cannot ensure properties for '{collection_name}': Not connected.")
            return

        existing = self._extract_collection_properties(collection_name)
        if not existing:
            logger.warning(f"Could not read existing properties for '{collection_name}'. Skipping property sync.")
            return

        collection = self.client.collections.get(collection_name)
        for prop in required_properties:
            if prop.name in existing:
                continue
            try:
                collection.config.add_property(prop)
                logger.info(f"Added missing property '{prop.name}' to collection '{collection_name}'.")
            except Exception as e:
                logger.error(
                    f"Failed to add missing property '{prop.name}' to collection '{collection_name}': {e}",
                    exc_info=True,
                )

    def _graph_schema_is_compatible(self) -> bool:
        expected = self._canonical_graph_schema()
        node_props = self._extract_collection_properties(config.WEAVIATE_NODE_CLASS)
        edge_props = self._extract_collection_properties(config.WEAVIATE_EDGE_CLASS)
        if not node_props or not edge_props:
            return False

        if "node_id" not in node_props:
            return False

        legacy_uuid_fields = {"source", "target"}
        if any(f in edge_props and "uuid" in edge_props[f] for f in legacy_uuid_fields):
            return False

        for coll_name, coll_expected in expected.items():
            actual = node_props if coll_name == config.WEAVIATE_NODE_CLASS else edge_props
            for prop in coll_expected.keys():
                if prop not in actual:
                    return False
        return True

    def _filter_properties_for_collection(self, collection_name: str, props: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(props, dict):
            return {}
        schema = self._canonical_graph_schema()
        allowed = set(schema.get(collection_name, {}).keys())
        if not allowed:
            return props
        return {k: v for k, v in props.items() if k in allowed}

    def _ensure_schema(self):
        if not self.is_connected():
            logger.error("Cannot ensure schema: Not connected.")
            return

        logger.info(f"Ensuring Weaviate schema. Run Mode: '{self.run_mode}'.")
        prop_tokenization = (
            Tokenization.WORD if config.WEAVIATE_PROPERTY_TOKENIZATION.lower() == "word"
            else Tokenization.WHITESPACE
        )
        collections_to_ensure = {
            config.WEAVIATE_DOC_CLASS: {
                "properties": [
                    Property(name="text", data_type=DataType.TEXT, tokenization=prop_tokenization),
                    Property(name="filename", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="title", data_type=DataType.TEXT, tokenization=prop_tokenization),
                    Property(name="doc_summary", data_type=DataType.TEXT, tokenization=prop_tokenization),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="keywords", data_type=DataType.TEXT_ARRAY, tokenization=prop_tokenization),
                    Property(name="source_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="chunk_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="page_start", data_type=DataType.INT),
                    Property(name="page_end", data_type=DataType.INT),
                    Property(name="prop_start", data_type=DataType.INT),
                    Property(name="prop_end", data_type=DataType.INT),
                    Property(name="chunk_strategy", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                ], "description": "Stores chunks of general documents."
            },
            config.WEAVIATE_ARTICLE_CLASS: {
                "properties": [
                    Property(name="law_title", data_type=DataType.TEXT, tokenization=prop_tokenization),
                    Property(name="article_number", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="article_text", data_type=DataType.TEXT, tokenization=prop_tokenization),
                    Property(name="doc_summary", data_type=DataType.TEXT, tokenization=prop_tokenization),
                    Property(name="source_filename", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="keywords", data_type=DataType.TEXT_ARRAY, tokenization=prop_tokenization),
                    Property(name="source_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="chunk_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="page_start", data_type=DataType.INT),
                    Property(name="page_end", data_type=DataType.INT),
                    Property(name="chunk_strategy", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                ], "description": "Stores individual legal articles."
            },
            config.WEAVIATE_NODE_CLASS: {
                "properties": [
                    Property(name="node_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="node_name", data_type=DataType.TEXT, tokenization=prop_tokenization),
                    Property(name="node_type", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="source_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="source_filename", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="description", data_type=DataType.TEXT, tokenization=prop_tokenization),
                ], "description": "Stores knowledge graph nodes (legal entities)."
            },
            config.WEAVIATE_EDGE_CLASS: {
                "properties": [
                    Property(name="source_entity", data_type=DataType.TEXT, tokenization=prop_tokenization),
                    Property(name="target_entity", data_type=DataType.TEXT, tokenization=prop_tokenization),
                    Property(name="relationship_type", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="source_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    # NEW: tie edge rows to file for reliable cleanup
                    Property(name="source_filename", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                ], "description": "Stores knowledge graph edges (relationships)."
            },
        }

        if self.run_mode == "UPDATE":
            node_exists = self._collection_exists(config.WEAVIATE_NODE_CLASS)
            edge_exists = self._collection_exists(config.WEAVIATE_EDGE_CLASS)
            if node_exists and edge_exists and not self._graph_schema_is_compatible():
                recreate_graph = os.getenv("WEAVIATE_RECREATE_GRAPH_SCHEMA", "0").strip().lower() in {"1", "true", "yes"}
                if recreate_graph:
                    logger.warning("Graph schema mismatch detected in UPDATE mode. Recreating GraphNodes/GraphEdges due to WEAVIATE_RECREATE_GRAPH_SCHEMA=1.")
                    self._delete_collection(config.WEAVIATE_NODE_CLASS)
                    self._delete_collection(config.WEAVIATE_EDGE_CLASS)
                else:
                    msg = (
                        "Graph schema mismatch detected for GraphNodes/GraphEdges in UPDATE mode. "
                        "Set WEAVIATE_RECREATE_GRAPH_SCHEMA=1 to recreate only graph collections, "
                        "or run FIRST_RUN mode."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

        for name, spec in collections_to_ensure.items():
            if self.run_mode == "FIRST_RUN":
                self._delete_collection(name)
                time.sleep(0.2)
                self._create_collection(name, spec["properties"], spec["description"])
            elif self.run_mode == "UPDATE":
                if not self._collection_exists(name):
                    self._create_collection(name, spec["properties"], spec["description"])
                else:
                    logger.info(f"Collection '{name}' exists. Ensuring required properties in UPDATE mode.")
                    self._ensure_collection_properties(name, spec["properties"])
        if os.getenv("DEBUG_SCHEMA", "0").strip().lower() in {"1", "true", "yes"}:
            node_schema = self._extract_collection_properties(config.WEAVIATE_NODE_CLASS)
            logger.info(f"DEBUG_SCHEMA=1 | GraphNodes properties: {sorted(node_schema.keys())}")
        logger.info("Schema setup/verification process completed.")

    def delete_objects_by_filter(self, collection_name: str, weaviate_filter: Filter) -> int:
        deleted_count = 0
        if not self.is_connected() or not self.client:
            return 0
        if not self._collection_exists(collection_name):
            return 0

        try:
            collection = self.client.collections.get(collection_name)
            result = collection.data.delete_many(where=weaviate_filter)
            deleted_count = result.successful
            logger.info(f"Deletion from '{collection_name}': Matched={result.matches}, Deleted={deleted_count}, Failed={result.failed}.")
            if result.failed > 0:
                logger.warning(f"{result.failed} objects failed delete from '{collection_name}'. Errors: {result.errors}")
            metrics.WEAVIATE_DELETES.labels(collection=collection_name, filter_type="by_property").inc(deleted_count)
        except Exception as e:
            logger.error(f"Error deleting from '{collection_name}': {e}", exc_info=True)
        return deleted_count

    def delete_data_for_file(self, filename: str):
        """
        Delete all data rows associated with a given source filename.
        Document:     property 'filename' == filename
        LegalArticle: property 'source_filename' == filename
        GraphNodes:   property 'source_filename' == filename  (CHANGED)
        GraphEdges:   property 'source_filename' == filename  (CHANGED)
        """
        if not filename or not filename.strip():
            return
        logger.info(f"Deleting all data associated with source filename: '{filename}'")

        # Documents by filename
        self.delete_objects_by_filter(
            config.WEAVIATE_DOC_CLASS,
            Filter.by_property("filename").equal(filename)
        )
        # Legal articles by source_filename (already present in your schema)
        self.delete_objects_by_filter(
            config.WEAVIATE_ARTICLE_CLASS,
            Filter.by_property("source_filename").equal(filename)
        )
        # FIX: Nodes and Edges must delete by source_filename (not source_id)
        self.delete_objects_by_filter(
            config.WEAVIATE_NODE_CLASS,
            Filter.by_property("source_filename").equal(filename)
        )
        self.delete_objects_by_filter(
            config.WEAVIATE_EDGE_CLASS,
            Filter.by_property("source_filename").equal(filename)
        )
        logger.info(f"Deletion process completed for filename: '{filename}'")

    def upsert_data_objects(
        self,
        collection_name: str,
        data_properties_list: List[Dict[str, Any]],
        vectors_list: List[Optional[List[float]]],
        uuids_list: List[UUID],
        min_batch_size: int = DEFAULT_MIN_BATCH_SIZE
    ):
        if not self.is_connected() or not self.client:
            logger.error(f"Cannot upsert to '{collection_name}': Not connected.")
            return
        if not (len(data_properties_list) == len(vectors_list) == len(uuids_list)):
            logger.error(f"Data integrity error for '{collection_name}': List lengths mismatch. Aborting.")
            return
        if not data_properties_list:
            logger.info(f"No data to upsert into '{collection_name}'.")
            return

        logger.info(f"Preparing to upsert {len(data_properties_list)} objects into '{collection_name}'.")
        metrics.WEAVIATE_UPSERTS_ATTEMPTED.labels(collection=collection_name).inc(len(data_properties_list))

        try:
            collection = self.client.collections.get(collection_name)
        except Exception as e:
            logger.error(f"Failed to get collection '{collection_name}' for upsert: {e}", exc_info=True)
            metrics.WEAVIATE_UPSERTS_FAILED_SERVER_SIDE.labels(collection=collection_name).inc(len(data_properties_list))
            return

        successfully_added_to_batch_count = 0
        objects_failed_client_validation = []

        final_successful_upserts = 0
        final_failed_upserts = 0

        try:
            batch_manager = collection.batch
            with batch_manager.dynamic() as batch_context:
                logger.info(f"Starting dynamic batch upsert for {len(data_properties_list)} objects into '{collection_name}'.")
                batch_start_time = time.time()

                for i in range(len(data_properties_list)):
                    try:
                        current_vector = vectors_list[i]
                        filtered_props = self._filter_properties_for_collection(collection_name, data_properties_list[i])
                        batch_context.add_object(
                            properties=filtered_props,
                            vector=current_vector,
                            uuid=uuids_list[i]
                        )
                        successfully_added_to_batch_count += 1
                    except Exception as e_add_obj:
                        logger.error(f"Error adding object {i} (UUID: {uuids_list[i]}) to batch for '{collection_name}': {e_add_obj}", exc_info=True)
                        objects_failed_client_validation.append({
                            "uuid": uuids_list[i],
                            "properties_snippet": str(data_properties_list[i])[:200],
                            "error": str(e_add_obj)
                        })

            batch_duration = time.time() - batch_start_time
            metrics.WEAVIATE_BATCH_LATENCY.labels(collection=collection_name).observe(batch_duration)

            server_side_errors_list = batch_manager.failed_objects
            num_server_side_failures = len(server_side_errors_list) if server_side_errors_list else 0

            client_side_failures = len(objects_failed_client_validation)

            final_successful_upserts = (successfully_added_to_batch_count - client_side_failures) - num_server_side_failures
            final_failed_upserts = client_side_failures + num_server_side_failures

            metrics.WEAVIATE_UPSERTS_SUCCESSFUL.labels(collection=collection_name).inc(final_successful_upserts)
            if client_side_failures > 0:
                metrics.WEAVIATE_UPSERTS_FAILED_CLIENT_SIDE.labels(collection=collection_name).inc(client_side_failures)
                logger.warning(f"{client_side_failures} objects failed client-side validation for '{collection_name}'.")
                for failed_info in objects_failed_client_validation[:3]:
                    logger.debug(f"Client-side fail: {failed_info}")
            if num_server_side_failures > 0:
                metrics.WEAVIATE_UPSERTS_FAILED_SERVER_SIDE.labels(collection=collection_name).inc(num_server_side_failures)
                logger.warning(f"{num_server_side_failures} objects failed server-side during batch to '{collection_name}'. Errors from failed_objects: {server_side_errors_list[:3]}")
                if collection_name == config.WEAVIATE_EDGE_CLASS:
                    for failed_obj in server_side_errors_list[:5]:
                        logger.warning(f"GraphEdges failed_object: {failed_obj}")

            logger.info(
                f"Batch to '{collection_name}' finished. Sent: {successfully_added_to_batch_count - client_side_failures}. "
                f"Server-side OK: {final_successful_upserts}. Server-side Fails: {num_server_side_failures}. Client-side Fails: {client_side_failures}."
            )

        except WeaviateBaseError as e_batch_op:
            logger.error(f"Weaviate error during batch operation for '{collection_name}': {e_batch_op}", exc_info=True)
            num_attempted_in_batch = successfully_added_to_batch_count - len(objects_failed_client_validation)
            final_failed_upserts = len(objects_failed_client_validation) + num_attempted_in_batch
            final_successful_upserts = 0
            metrics.WEAVIATE_UPSERTS_FAILED_SERVER_SIDE.labels(collection=collection_name).inc(num_attempted_in_batch)
            if len(objects_failed_client_validation) > 0:
                metrics.WEAVIATE_UPSERTS_FAILED_CLIENT_SIDE.labels(collection=collection_name).inc(len(objects_failed_client_validation))

        except Exception as e_unexpected:
            logger.error(f"Unexpected error during batch upsert to '{collection_name}': {e_unexpected}", exc_info=True)
            final_failed_upserts = len(data_properties_list)
            final_successful_upserts = 0
            metrics.WEAVIATE_UPSERTS_FAILED_SERVER_SIDE.labels(collection=collection_name).inc(len(data_properties_list))

        logger.info(
            f"Upsert operation summary for '{collection_name}': "
            f"Total objects input: {len(data_properties_list)}, "
            f"Successful (estimated): {final_successful_upserts}, "
            f"Failed (estimated): {final_failed_upserts}."
        )
