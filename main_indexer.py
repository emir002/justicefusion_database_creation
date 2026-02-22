#!/usr/bin/env python3
# main_indexer.py

import logging
import logging.config  # For applying dictConfig
import time
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional

# Imports for graceful stop and memory management
import signal
import gc
import shutil
import psutil
import os
import faulthandler
faulthandler.enable(file=open("fatal_crash.log", "w"))

# Optional torch import for empty_cache on GPU boxes
try:
    import torch
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_AVAILABLE = False

# Local module imports
try:
    import config  # General configuration, paths, API keys, logging setup
    # Initialize logging as early as possible
    logging.config.dictConfig(config.LOGGING_CONFIG)
    logger = logging.getLogger(__name__)

    import utils  # Utility functions
    from llm_interaction import LLMManager  # For interacting with Gemini LLM
    from embedding_model import EmbeddingModelManager  # For generating text embeddings
    from weaviate_client import WeaviateManager  # For Weaviate DB operations
    from text_processing import FileLoader, TextProcessor  # For loading and processing text
    import metrics  # For Prometheus metrics

except ImportError as e:
    temp_fallback_logger = logging.getLogger("init_fallback_logger")
    if not temp_fallback_logger.hasHandlers():
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s [%(levelname)s] %(name)s (%(lineno)d): %(message)s'
        )
    logging.error(
        f"CRITICAL: Error importing one or more essential local modules: {e}. "
        f"Ensure all required .py files are in the same directory or PYTHONPATH. Script will exit.",
        exc_info=True
    )
    sys.exit(1)
except Exception as e_cfg:
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s [%(levelname)s] %(name)s (%(lineno)d): %(message)s'
    )
    logging.error(
        f"CRITICAL: An unexpected error occurred during initial setup or import: {e_cfg}. Script will exit.",
        exc_info=True
    )
    sys.exit(1)

# --- Graceful-stop handling ---
STOP_REQUESTED = False

def _request_stop(signum, frame):
    """Signal handler to request a graceful stop."""
    global STOP_REQUESTED
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        logger.warning(
            f"Stop signal ({signal.Signals(signum).name}) received. "
            f"The script will finish the current file and then exit gracefully. "
            f"Please wait..."
        )

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)

def _memnote(tag=""):
    p = psutil.Process(os.getpid())
    logging.getLogger(__name__).info(
        f"[MEM]{tag}: rss={p.memory_info().rss/1024/1024:.1f} MB"
    )

# --- Main Application Class ---
class DocumentIndexer:
    """
    Orchestrates the entire document processing and indexing workflow.
    """

    def __init__(self):
        logger.info("======================================================================")
        logger.info("========== Initializing Document Indexer Orchestrator ==========")
        logger.info("======================================================================")

        try:
            logger.info("Initializing LLMManager...")
            self.llm_manager = LLMManager(config.GEMINI_API_KEYS, config.GEMINI_MODEL_NAME)
            logger.info("LLMManager initialized.")

            logger.info("Initializing EmbeddingModelManager...")
            self.embedding_manager = EmbeddingModelManager()
            logger.info("EmbeddingModelManager initialized.")

            logger.info("Initializing WeaviateManager...")
            self.weaviate_manager = WeaviateManager(run_mode=config.RUN_MODE)
            logger.info("WeaviateManager initialized.")

            logger.info("Text processing utilities (FileLoader, TextProcessor) initialized.")

            self._validate_initialization()
            logger.info("Document Indexer Orchestrator initialized successfully and components validated.")

        except (ConnectionError, RuntimeError, ValueError) as e:
            logger.critical(
                f"A critical {type(e).__name__} occurred during DocumentIndexer initialization: {e}. Cannot proceed.",
                exc_info=True
            )
            raise
        except Exception as e:
            logger.critical(
                f"An unexpected critical error during DocumentIndexer initialization: {e}. Cannot proceed.",
                exc_info=True
            )
            raise

    @staticmethod
    def _flush_mem(extra_msg: str = ""):
        """Force-free Python objects and clear GPU cache if available."""
        if not getattr(config, "AGGRESSIVE_MEMORY_CLEANUP", False):
            if (extra_msg in ["after full LAW document", "after full general document", "after run_indexing completes"]
                    or extra_msg.startswith("end of file loop for")):
                pass
            else:
                return

        log_msg = f"Flushing memory ({extra_msg})" if extra_msg else "Flushing memory"
        logger.debug(log_msg)

        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared PyTorch CUDA cache.")

    def _validate_initialization(self):
        logger.info("Performing post-initialization validation of core components...")
        if not self.weaviate_manager or not self.weaviate_manager.is_connected():
            raise ConnectionError("Critical component failure: Weaviate connection is not established.")
        logger.info("Validation OK: WeaviateManager is connected.")

        if not self.embedding_manager or not self.embedding_manager.model:
            raise RuntimeError("Critical component failure: Embedding model is not loaded.")

        if (getattr(self.embedding_manager, "dataset_embeddings", None) is None and
            config.EMBEDDING_MODEL_NAME_OR_PATH and
            "jxm/cde-small-v2" in config.EMBEDDING_MODEL_NAME_OR_PATH):
            raise RuntimeError("Critical component failure: Dataset embeddings for CDE model are missing or failed to load.")
        logger.info("Validation OK: EmbeddingModelManager and its model (and dataset embeddings if applicable) are loaded.")

        if not self.llm_manager:
            raise RuntimeError("Critical component failure: LLMManager is not available.")
        logger.info("Validation OK: LLMManager appears initialized.")
        logger.info("All core components validated successfully.")

    def _process_law_document(self, file_path: Path, original_full_text: str):
        filename = file_path.name
        logger.info(f"--- Starting LAW document processing for: '{filename}' ---")
        processing_start_time = time.time()
        full_text_for_splitting_and_llm = utils.clean_text(original_full_text, is_pre_splitting=True)

        if not full_text_for_splitting_and_llm.strip():
            logger.warning(f"Full text for '{filename}' is empty after pre-cleaning. Skipping file.")
            return

        law_document_title = self.llm_manager.extract_document_title(full_text_for_splitting_and_llm)
        if not law_document_title or law_document_title.startswith("Error:"):
            law_document_title = utils.clean_text(filename.rsplit('.', 1)[0], is_pre_splitting=False)

        law_document_summary = self.llm_manager.generate_summary(full_text_for_splitting_and_llm)
        if not law_document_summary or law_document_summary.startswith("Error:"):
            law_document_summary = ""

        articles_data = TextProcessor.split_law_articles(full_text_for_splitting_and_llm)
        if not articles_data:
            logger.warning(f"No articles found in LAW document '{filename}'. Skipping indexing of articles.")
            return
        logger.info(f"Split '{filename}' into {len(articles_data)} articles.")

        all_data_objects, all_nodes, all_edges = [], [], []
        skipped_articles_count = 0
        source_id = utils.make_source_id_from_filename(filename)

        for i, (marker, body) in enumerate(articles_data):
            article_proc_start_time = time.time()
            logger.debug(f"Processing article {i+1}/{len(articles_data)}: '{marker}' from '{filename}'")
            refined_text = TextProcessor.refine_and_lemmatize(body, f"law_article_{filename}_{marker.replace(' ', '_')}")
            if not refined_text.strip():
                logger.warning(f"Article '{marker}' from '{filename}' empty after refinement. Skipping.")
                skipped_articles_count += 1
                continue

            embeddings = self.embedding_manager.get_embeddings(
                [refined_text],
                prompt_name=config.EMBEDDING_PROMPT_DOCUMENT
            )
            article_embedding: Optional[List[float]] = None
            if embeddings and len(embeddings) > 0:
                article_embedding = embeddings[0]

            if article_embedding is None:
                logger.error(f"Failed embedding for article '{marker}'. Skipping.")
                metrics.EMBEDDINGS_GENERATED.labels(status='failed').inc()
                skipped_articles_count += 1
                continue
            metrics.EMBEDDINGS_GENERATED.labels(status='success').inc()

            article_uuid: str = utils.generate_legal_article_id(law_document_title, marker, filename)
            entity_source_id: str = str(article_uuid)
            entities = self.llm_manager.extract_entities_and_relationships(refined_text, entity_source_id, source_filename=filename)

            # Upsert objects
            all_data_objects.append({
                "properties": {
                    "law_title": law_document_title,
                    "article_number": marker,
                    "article_text": refined_text,
                    "doc_summary": law_document_summary,
                    "source_filename": filename,
                    "keywords": [],
                    "source_id": source_id,
                    "chunk_id": f"{source_id}#{marker.replace(' ', '')}",
                    "page_start": None,
                    "page_end": None,
                    "chunk_strategy": "article",
                },
                "vector": article_embedding,
                "uuid": article_uuid
            })

            # Add source_filename to nodes/edges for reliable cleanup
            if entities and isinstance(entities.get("nodes"), list):
                for idx, n in enumerate(entities["nodes"]):
                    if not isinstance(n, dict):
                        logger.warning(f"Skipping non-dict node for '{filename}' article {i} at index {idx}.")
                        continue
                    node_props = {
                        "node_id": str(n.get("node_id", "")).strip() or f"n{idx + 1}",
                        "node_name": str(n.get("node_name", "")).strip(),
                        "node_type": str(n.get("node_type", "")).strip(),
                        "description": str(n.get("description", "")).strip(),
                        "source_id": str(n.get("source_id", entity_source_id)).strip() or entity_source_id,
                        "source_filename": filename,
                    }
                    if not node_props["description"]:
                        base = node_props.get("node_name") or node_props.get("node_type") or node_props.get("node_id") or "Entity"
                        t = node_props.get("node_type", "").strip()
                        node_props["description"] = f"{base} ({t})".strip() if t else base
                    all_nodes.append({
                        "properties": node_props,
                        "uuid": utils.generate_graph_node_id(
                            node_props.get("node_name", ""),
                            node_props.get("node_type", ""),
                            node_props.get("source_id", entity_source_id),
                        ),
                    })
            if entities and isinstance(entities.get("edges"), list):
                for idx, e in enumerate(entities["edges"]):
                    if not isinstance(e, dict):
                        logger.warning(f"Skipping non-dict edge for '{filename}' article {i} at index {idx}.")
                        continue
                    edge_props = {
                        "source_entity": str(e.get("source_entity", "")).strip(),
                        "target_entity": str(e.get("target_entity", "")).strip(),
                        "relationship_type": str(e.get("relationship_type", "")).strip(),
                        "source_id": str(e.get("source_id", entity_source_id)).strip() or entity_source_id,
                        "source_filename": filename,
                    }
                    if not edge_props["source_entity"] or not edge_props["target_entity"]:
                        logger.warning(f"Skipping edge missing endpoints for '{filename}' article {i} at index {idx}.")
                        continue
                    all_edges.append({
                        "properties": edge_props,
                        "uuid": utils.generate_graph_edge_id(
                            edge_props["source_entity"],
                            edge_props["target_entity"],
                            edge_props["relationship_type"],
                            edge_props["source_id"],
                        ),
                    })

            logger.debug(f"Finished article '{marker}' in {time.time() - article_proc_start_time:.2f}s.")

            _memnote(f"after article {i+1}/{len(articles_data)} from {filename}")

            del refined_text, embeddings, article_embedding, entities
            if getattr(config, "AGGRESSIVE_MEMORY_CLEANUP", False):
                self._flush_mem(f"after article {i+1}/{len(articles_data)}")

        if articles_data:
            logger.info(f"{skipped_articles_count}/{len(articles_data)} articles skipped for '{filename}'.")

        if all_data_objects:
            self.weaviate_manager.upsert_data_objects(
                config.WEAVIATE_ARTICLE_CLASS,
                [o["properties"] for o in all_data_objects],
                [o["vector"] for o in all_data_objects],
                [o["uuid"] for o in all_data_objects]
            )
        if all_nodes:
            node_texts = [
                f"{n['properties'].get('node_name', '')}\n{n['properties'].get('node_type', '')}\n{n['properties'].get('description', '')}".strip()
                for n in all_nodes
            ]
            node_vectors = self.embedding_manager.get_embeddings(
                node_texts,
                prompt_name=config.EMBEDDING_PROMPT_DOCUMENT
            )
            if not node_vectors or len(node_vectors) != len(all_nodes):
                logger.error(
                    f"Node embedding batch mismatch for '{filename}': vectors={len(node_vectors) if node_vectors else 0}, nodes={len(all_nodes)}. Falling back to per-node embeddings."
                )
                node_vectors = []
                for text in node_texts:
                    single = self.embedding_manager.get_embeddings([text], prompt_name=config.EMBEDDING_PROMPT_DOCUMENT)
                    node_vectors.append(single[0] if single and len(single) > 0 else None)
            self.weaviate_manager.upsert_data_objects(
                config.WEAVIATE_NODE_CLASS,
                [n["properties"] for n in all_nodes],
                node_vectors,
                [n["uuid"] for n in all_nodes]
            )
        if all_edges:
            self.weaviate_manager.upsert_data_objects(
                config.WEAVIATE_EDGE_CLASS,
                [e["properties"] for e in all_edges],
                [None] * len(all_edges),
                [e["uuid"] for e in all_edges]
            )
        logger.info(f"--- Finished LAW document processing for: '{filename}' in {time.time() - processing_start_time:.2f}s ---")
        if getattr(config, "AGGRESSIVE_MEMORY_CLEANUP", False):
            self._flush_mem("after full LAW document")

    def _process_other_document(self, file_path: Path, original_full_text: str):
        filename = file_path.name
        logger.info(f"--- Starting general document processing for: '{filename}' ---")
        processing_start_time = time.time()
        full_text_for_splitting_and_llm = utils.clean_text(original_full_text, is_pre_splitting=True)
        if not full_text_for_splitting_and_llm.strip():
            logger.warning(f"Full text for '{filename}' empty after pre-cleaning. Skipping.")
            return

        document_title = self.llm_manager.extract_document_title(full_text_for_splitting_and_llm)
        if not document_title or document_title.startswith("Error:"):
            document_title = utils.clean_text(filename.rsplit('.', 1)[0], is_pre_splitting=False)

        document_summary = self.llm_manager.generate_summary(full_text_for_splitting_and_llm)
        if not document_summary or document_summary.startswith("Error:"):
            document_summary = ""

        chunk_dicts = TextProcessor.split_general_text(full_text_for_splitting_and_llm, llm_manager=self.llm_manager)
        if not chunk_dicts:
            logger.warning(f"No chunks found in document '{filename}'. Skipping.")
            return
        logger.info(f"Split '{filename}' into {len(chunk_dicts)} chunks using strategy='{config.CHUNKING_STRATEGY}'.")

        all_data_objects, all_nodes, all_edges = [], [], []
        skipped_chunks_count = 0
        source_id = utils.make_source_id_from_filename(filename)

        for i, ch in enumerate(chunk_dicts):
            chunk_text = ch["text"]
            page_start = ch.get("page_start")
            page_end = ch.get("page_end")
            prop_start = ch.get("prop_start")
            prop_end = ch.get("prop_end")
            chunk_strategy = ch.get("chunk_strategy", config.CHUNKING_STRATEGY)
            chunk_proc_start_time = time.time()
            logger.debug(f"Processing chunk {i+1}/{len(chunk_dicts)} from '{filename}'")
            refined_text = TextProcessor.refine_and_lemmatize(chunk_text, f"other_chunk_{filename}_{i}")
            if not refined_text.strip():
                logger.warning(f"Chunk {i} from '{filename}' empty after refinement. Skipping.")
                skipped_chunks_count += 1
                continue

            embeddings = self.embedding_manager.get_embeddings(
                [refined_text],
                prompt_name=config.EMBEDDING_PROMPT_DOCUMENT
            )
            chunk_embedding: Optional[List[float]] = None
            if embeddings and len(embeddings) > 0:
                chunk_embedding = embeddings[0]

            if chunk_embedding is None:
                logger.error(f"Failed embedding for chunk {i}. Skipping.")
                metrics.EMBEDDINGS_GENERATED.labels(status='failed').inc()
                skipped_chunks_count += 1
                continue
            metrics.EMBEDDINGS_GENERATED.labels(status='success').inc()

            chunk_uuid: str = utils.generate_doc_chunk_id_v2(filename, i, page_start, page_end, prop_start, prop_end)
            chunk_id = utils.make_chunk_id(source_id, i, page_start, page_end, prop_start, prop_end)
            entity_source_id: str = str(chunk_uuid)
            entities = self.llm_manager.extract_entities_and_relationships(refined_text, entity_source_id, source_filename=filename)

            all_data_objects.append({
                "properties": {
                    "text": refined_text,
                    "filename": filename,
                    "title": document_title,
                    "doc_summary": document_summary,
                    "chunk_index": i,
                    "keywords": [],
                    "source_id": source_id,
                    "chunk_id": chunk_id,
                    "page_start": page_start,
                    "page_end": page_end,
                    "prop_start": prop_start,
                    "prop_end": prop_end,
                    "chunk_strategy": chunk_strategy,
                },
                "vector": chunk_embedding,
                "uuid": chunk_uuid
            })

            # Add source_filename to nodes/edges for reliable cleanup
            if entities and isinstance(entities.get("nodes"), list):
                for idx, n in enumerate(entities["nodes"]):
                    if not isinstance(n, dict):
                        logger.warning(f"Skipping non-dict node for '{filename}' article {i} at index {idx}.")
                        continue
                    node_props = {
                        "node_id": str(n.get("node_id", "")).strip() or f"n{idx + 1}",
                        "node_name": str(n.get("node_name", "")).strip(),
                        "node_type": str(n.get("node_type", "")).strip(),
                        "description": str(n.get("description", "")).strip(),
                        "source_id": str(n.get("source_id", entity_source_id)).strip() or entity_source_id,
                        "source_filename": filename,
                    }
                    if not node_props["description"]:
                        base = node_props.get("node_name") or node_props.get("node_type") or node_props.get("node_id") or "Entity"
                        t = node_props.get("node_type", "").strip()
                        node_props["description"] = f"{base} ({t})".strip() if t else base
                    all_nodes.append({
                        "properties": node_props,
                        "uuid": utils.generate_graph_node_id(
                            node_props.get("node_name", ""),
                            node_props.get("node_type", ""),
                            node_props.get("source_id", entity_source_id),
                        ),
                    })
            if entities and isinstance(entities.get("edges"), list):
                for idx, e in enumerate(entities["edges"]):
                    if not isinstance(e, dict):
                        logger.warning(f"Skipping non-dict edge for '{filename}' article {i} at index {idx}.")
                        continue
                    edge_props = {
                        "source_entity": str(e.get("source_entity", "")).strip(),
                        "target_entity": str(e.get("target_entity", "")).strip(),
                        "relationship_type": str(e.get("relationship_type", "")).strip(),
                        "source_id": str(e.get("source_id", entity_source_id)).strip() or entity_source_id,
                        "source_filename": filename,
                    }
                    if not edge_props["source_entity"] or not edge_props["target_entity"]:
                        logger.warning(f"Skipping edge missing endpoints for '{filename}' article {i} at index {idx}.")
                        continue
                    all_edges.append({
                        "properties": edge_props,
                        "uuid": utils.generate_graph_edge_id(
                            edge_props["source_entity"],
                            edge_props["target_entity"],
                            edge_props["relationship_type"],
                            edge_props["source_id"],
                        ),
                    })

            logger.debug(f"Finished chunk {i} in {time.time() - chunk_proc_start_time:.2f}s.")

            _memnote(f"after chunk {i+1}/{len(chunk_dicts)} from {filename}")

            del refined_text, embeddings, chunk_embedding, entities
            if getattr(config, "AGGRESSIVE_MEMORY_CLEANUP", False):
                self._flush_mem(f"after chunk {i+1}/{len(chunk_dicts)}")

        if chunk_dicts:
            logger.info(f"{skipped_chunks_count}/{len(chunk_dicts)} chunks skipped for '{filename}'.")

        if all_data_objects:
            self.weaviate_manager.upsert_data_objects(
                config.WEAVIATE_DOC_CLASS,
                [o["properties"] for o in all_data_objects],
                [o["vector"] for o in all_data_objects],
                [o["uuid"] for o in all_data_objects]
            )
        if all_nodes:
            node_texts = [
                f"{n['properties'].get('node_name', '')}\n{n['properties'].get('node_type', '')}\n{n['properties'].get('description', '')}".strip()
                for n in all_nodes
            ]
            node_vectors = self.embedding_manager.get_embeddings(
                node_texts,
                prompt_name=config.EMBEDDING_PROMPT_DOCUMENT
            )
            if not node_vectors or len(node_vectors) != len(all_nodes):
                logger.error(
                    f"Node embedding batch mismatch for '{filename}': vectors={len(node_vectors) if node_vectors else 0}, nodes={len(all_nodes)}. Falling back to per-node embeddings."
                )
                node_vectors = []
                for text in node_texts:
                    single = self.embedding_manager.get_embeddings([text], prompt_name=config.EMBEDDING_PROMPT_DOCUMENT)
                    node_vectors.append(single[0] if single and len(single) > 0 else None)
            self.weaviate_manager.upsert_data_objects(
                config.WEAVIATE_NODE_CLASS,
                [n["properties"] for n in all_nodes],
                node_vectors,
                [n["uuid"] for n in all_nodes]
            )
        if all_edges:
            self.weaviate_manager.upsert_data_objects(
                config.WEAVIATE_EDGE_CLASS,
                [e["properties"] for e in all_edges],
                [None] * len(all_edges),
                [e["uuid"] for e in all_edges]
            )
        logger.info(f"--- Finished general document processing for: '{filename}' in {time.time() - processing_start_time:.2f}s ---")
        if getattr(config, "AGGRESSIVE_MEMORY_CLEANUP", False):
            self._flush_mem("after full general document")

    def run_indexing(self):
        logger.info("#####================================================#####")
        logger.info("##### Starting Document Indexing Run                     #####")
        logger.info("#####================================================#####")
        logger.info(f"Run Mode:                 '{config.RUN_MODE}'")
        logger.info(f"Input Folder:             '{config.INPUT_FOLDER}'")
        logger.info(f"Finished Folder:          '{config.FINISHED_FOLDER}'")
        logger.info(f"Quarantine Folder:        '{getattr(config, 'QUARANTINE_FOLDER', 'Not Set')}'")
        logger.info(f"Aggressive Memory Cleanup: {getattr(config, 'AGGRESSIVE_MEMORY_CLEANUP', False)}")

        processed_files, failed_files, skipped_items = 0, 0, 0
        try:
            config.ensure_dirs_exist()
        except Exception as e_dir:
            logger.error(f"Failed to ensure directories exist: {e_dir}. Aborting.", exc_info=True)
            return

        try:
            files_to_process = sorted([f for f in config.INPUT_FOLDER.iterdir() if f.is_file()])
            all_items_in_dir = list(config.INPUT_FOLDER.iterdir())
            logger.info(f"Found {len(all_items_in_dir)} items in input. Processing {len(files_to_process)} files.")
            non_files = [item for item in all_items_in_dir if not item.is_file()]
            if non_files:
                logger.info(f"Skipping {len(non_files)} non-file items: {[item.name for item in non_files]}")
                skipped_items += len(non_files)
        except FileNotFoundError:
            logger.error(f"Input folder '{config.INPUT_FOLDER}' not found. Please create it.", exc_info=True)
            return
        except Exception as e_scan:
            logger.error(f"Error scanning input folder '{config.INPUT_FOLDER}': {e_scan}", exc_info=True)
            return

        if not files_to_process:
            logger.info("No files found in input folder.")

        for file_path in files_to_process:
            if STOP_REQUESTED:
                logger.warning("Stop request acknowledged. Halting processing loop.")
                break

            file_start_time = time.time()
            logger.info(f"\n>>> Processing file: '{file_path.name}' (Path: '{file_path}')")
            processed_file_successfully_data_wise = False

            try:
                load_result = FileLoader.load_text(file_path)
                if load_result is None:
                    failed_files += 1
                    metrics.FILES_PROCESSED.labels(status='failed_loading').inc()
                    continue
                original_full_text, file_type_suffix = load_result
                logger.info(f"Loaded '{file_path.name}' (type: '{file_type_suffix}', len: {len(original_full_text)}).")

                if not original_full_text or not original_full_text.strip():
                    logger.warning(f"File '{file_path.name}' is empty. Attempting to move.")
                    dest_path_finished = config.FINISHED_FOLDER / file_path.name
                    moved_empty_file = False
                    try:
                        shutil.move(str(file_path), str(dest_path_finished))
                        logger.info(f"Moved empty file '{file_path.name}' to finished: '{dest_path_finished}'.")
                        moved_empty_file = True
                    except Exception as e_move_empty_finished:
                        logger.error(f"Failed to move empty file '{file_path.name}' to finished: {e_move_empty_finished}. Attempting quarantine.", exc_info=True)
                        if hasattr(config, 'QUARANTINE_FOLDER'):
                            dest_path_quarantine = config.QUARANTINE_FOLDER / file_path.name
                            try:
                                config.QUARANTINE_FOLDER.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(file_path), str(dest_path_quarantine))
                                logger.info(f"Moved empty file '{file_path.name}' to quarantine: '{dest_path_quarantine}'.")
                                moved_empty_file = True
                            except Exception as e_move_empty_quarantine:
                                logger.error(f"Failed to move empty file '{file_path.name}' to quarantine: {e_move_empty_quarantine}. File remains in input.", exc_info=True)
                        else:
                            logger.error("QUARANTINE_FOLDER not configured. Empty file remains in input.")

                    if moved_empty_file:
                        skipped_items += 1
                        metrics.FILES_PROCESSED.labels(status='skipped_empty').inc()
                    else:
                        failed_files += 1
                        metrics.FILES_PROCESSED.labels(status='error_moving_empty_file').inc()
                    continue

                if self.weaviate_manager.is_connected():
                    logger.info(f"Pre-deleting data for '{file_path.name}'...")
                    self.weaviate_manager.delete_data_for_file(file_path.name)
                else:
                    logger.error(f"Weaviate not connected. Cannot delete data for '{file_path.name}'.")

                temp_text_for_classification = utils.clean_text(original_full_text, is_pre_splitting=True)
                classification_snippet = temp_text_for_classification[:config.LLM_CLASSIFICATION_SNIPPET_MAX_LEN]
                if getattr(config, "DEBUG", False):
                    logger.debug(
                        "Classification snippet selected | filename='%s' | snippet='%s'",
                        file_path.name,
                        classification_snippet[:300].replace("\n", " "),
                    )

                doc_type = ""
                doc_type = self.llm_manager.classify_document(classification_snippet, filename=file_path.name)

                logger.info(f"Document '{file_path.name}' classified as: '{doc_type}'.")

                if doc_type == 'LAW':
                    self._process_law_document(file_path, original_full_text)
                else:
                    if doc_type == 'MIXED':
                        logger.info(f"'{file_path.name}' is '{doc_type}', using general workflow.")
                    self._process_other_document(file_path, original_full_text)

                processed_file_successfully_data_wise = True

                dest_path_finished = config.FINISHED_FOLDER / file_path.name
                try:
                    shutil.move(str(file_path), str(dest_path_finished))
                    processed_files += 1
                    metrics.FILES_PROCESSED.labels(status='success').inc()
                    logger.info(f"Processed and moved '{file_path.name}' to '{dest_path_finished}'.")
                except Exception as e_move:
                    logger.error(f"Failed to move processed file '{file_path.name}' to finished folder: {e_move}. Attempting to quarantine.", exc_info=True)
                    if hasattr(config, 'QUARANTINE_FOLDER'):
                        dest_path_quarantine = config.QUARANTINE_FOLDER / file_path.name
                        try:
                            config.QUARANTINE_FOLDER.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(file_path), str(dest_path_quarantine))
                            processed_files += 1
                            metrics.FILES_PROCESSED.labels(status='success_quarantined').inc()
                            logger.info(f"Processed file '{file_path.name}' successfully moved to quarantine: '{dest_path_quarantine}'.")
                        except Exception as e_quarantine:
                            failed_files += 1
                            metrics.FILES_PROCESSED.labels(status='failure_stuck_in_input').inc()
                            logger.error(f"CRITICAL: Failed to move processed file '{file_path.name}' to quarantine: {e_quarantine}. File remains in input. Please move manually.", exc_info=True)
                    else:
                        failed_files += 1
                        metrics.FILES_PROCESSED.labels(status='failure_stuck_in_input').inc()
                        logger.error("CRITICAL: QUARANTINE_FOLDER not configured. Processed file remains in input. Please move manually.")

            except (ConnectionError, RuntimeError) as e_crit:
                logger.critical(f"Critical {type(e_crit).__name__} on '{file_path.name}': {e_crit}. Aborting.", exc_info=True)
                failed_files += 1
                metrics.FILES_PROCESSED.labels(status='failed_processing_critical').inc()
                break
            except Exception as e_file:
                logger.error(f"Unexpected error on '{file_path.name}': {e_file}. Skipping.", exc_info=True)
                failed_files += 1
                metrics.FILES_PROCESSED.labels(status='failed_processing_unexpected').inc()
            finally:
                if getattr(config, "AGGRESSIVE_MEMORY_CLEANUP", False) or not processed_file_successfully_data_wise:
                    self._flush_mem(f"end of file loop for {file_path.name}")
            logger.info(f"<<< Finished '{file_path.name}' in {time.time() - file_start_time:.2f}s <<<\n")

        logger.info("#####================================================#####")
        logger.info("##### Document Indexing Run Finished                     #####")
        logger.info(f"Summary: Processed files moved out of input = {processed_files}")
        logger.info(f"         Files failed during processing      = {failed_files}")
        logger.info(f"         Skipped items (non-file/empty)    = {skipped_items}")
        logger.info("#####================================================#####")
        if getattr(config, "AGGRESSIVE_MEMORY_CLEANUP", True):
            self._flush_mem("after run_indexing completes")

    def close(self):
        logger.info("--- Document Indexer shutting down. Closing resources... ---")
        if hasattr(self, 'weaviate_manager') and self.weaviate_manager:
            logger.info("Closing WeaviateManager...")
            self.weaviate_manager.close()
        if hasattr(self, 'embedding_manager') and self.embedding_manager:
            logger.info("Closing EmbeddingModelManager...")
            self.embedding_manager.close()
        logger.info("--- All resources closed. Document Indexer shutdown complete. ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("############################################################")
    logger.info("##### Main Indexer Script Execution Started              #####")
    logger.info("############################################################")
    main_start_time = time.time()
    indexer: Optional[DocumentIndexer] = None
    exit_code = 0
    try:
        indexer = DocumentIndexer()
        indexer.run_indexing()
    except (ConnectionError, RuntimeError, ValueError) as e_critical_setup:
        logger.critical(f"CRITICAL {type(e_critical_setup).__name__} during setup/run: {e_critical_setup}. Aborting.", exc_info=True)
        exit_code = 1
    except Exception as e_unexpected_main:
        logger.critical(f"Unexpected CRITICAL error at main level: {e_unexpected_main}. Aborting.", exc_info=True)
        exit_code = 1
    finally:
        if indexer:
            logger.info("Closing DocumentIndexer resources...")
            indexer.close()
        else:
            logger.warning("Indexer instance not created. No resources to close at main level.")
        logger.info(f"Total script execution time: {time.time() - main_start_time:.2f} seconds.")
        logger.info(f"##### Main Indexer Script Execution Finished. Exit code: {exit_code} #####")
    sys.exit(exit_code)
