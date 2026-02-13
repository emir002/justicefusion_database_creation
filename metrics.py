#!/usr/bin/env python3
# metrics.py

import logging
from prometheus_client import Counter, Histogram, Info, start_http_server
# import config # No longer strictly needed here if APP_VERSION is not used or passed differently

logger = logging.getLogger(__name__)

# --- Server Initialization ---
PROMETHEUS_PORT = 8000 # Define the port, ensure it's not conflicting

try:
    start_http_server(PROMETHEUS_PORT)
    logger.info(f"Prometheus metrics server started successfully on port {PROMETHEUS_PORT}.")
except OSError as e: 
    if "address already in use" in str(e).lower():
        logger.warning(
            f"Prometheus metrics server port {PROMETHEUS_PORT} is already in use. "
            f"Assuming server is already started by another instance or a previous run."
        )
    else:
        logger.error(f"Failed to start Prometheus metrics server on port {PROMETHEUS_PORT}: {e}", exc_info=True)
except Exception as e:
    logger.error(f"An unexpected error occurred while starting Prometheus metrics server on port {PROMETHEUS_PORT}: {e}", exc_info=True)


# --- General Application Info ---
APP_INFO = Info('document_indexer_app', 'Information about the Document Indexer application')
# Example: You can set info at runtime if needed, e.g., from config or build process
# APP_INFO.info({'version': '1.0.0', 'build_date': '2025-05-11'})


# --- LLM Metrics ---
LLM_CALLS = Counter(
    'llm_calls_total',
    'Total number of LLM calls, categorized by endpoint/model and status.',
    ['endpoint', 'status'] # status: 'success', 'api_error', 'retryable_error', 'blocked_or_empty', 'other_error'
)
LLM_LATENCY = Histogram(
    'llm_call_latency_seconds',
    'Latency of LLM calls, categorized by endpoint/model.',
    ['endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, float('inf')]
)

# --- Embedding Model Metrics ---
EMBED_CACHE_HITS = Counter(
    'embedding_dataset_cache_hits_total', 
    'Number of dataset embedding cache hits (first-stage CDE).',
)
EMBED_CACHE_MISSES = Counter(
    'embedding_dataset_cache_misses_total', 
    'Number of dataset embedding cache misses (first-stage CDE).'
)
EMBEDDING_GENERATION_LATENCY = Histogram(
    'embedding_generation_latency_seconds',
    'Latency of generating embeddings for a batch of texts.',
    ['model_name', 'stage'], # stage: 'first_stage_dataset' or 'second_stage_texts'
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, float('inf')]
)
# This is the metric that was missing from your execution context:
EMBEDDINGS_GENERATED = Counter(
    'embeddings_generated_total',
    'Total number of embeddings generated (for chunks/articles), by status.',
    ['status'] # e.g., 'success', 'failed'
)

# --- Weaviate Client Metrics ---
WEAVIATE_UPSERTS_ATTEMPTED = Counter(
    'weaviate_upsert_objects_attempted_total',
    'Total number of objects attempted to be upserted to Weaviate, by collection.',
    ['collection']
)
WEAVIATE_UPSERTS_FAILED_CLIENT_SIDE = Counter(
    'weaviate_upsert_objects_failed_client_total',
    'Objects that failed validation or preparation before being sent to Weaviate.',
    ['collection']
)
WEAVIATE_UPSERTS_FAILED_SERVER_SIDE = Counter(
    'weaviate_upsert_objects_failed_server_total',
    'Objects that Weaviate reported as failed during a batch operation.',
    ['collection']
)
WEAVIATE_UPSERTS_SUCCESSFUL = Counter(
    'weaviate_upsert_objects_successful_total',
    'Objects successfully accepted by Weaviate (estimated for batch operations).',
    ['collection']
)
WEAVIATE_BATCH_LATENCY = Histogram(
    'weaviate_batch_latency_seconds',
    'Latency of Weaviate batch upsert operations (full batch from client perspective), by collection.',
    ['collection'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 15, 20, 30, 60, float('inf')]
)
WEAVIATE_DELETES = Counter(
    'weaviate_delete_objects_total',
    'Total number of objects successfully deleted from Weaviate, by collection and filter type.',
    ['collection', 'filter_type'] # e.g., filter_type='by_filename'
)

# --- Document Processing Metrics ---
FILES_PROCESSED = Counter(
    'files_processed_total',
    'Total number of files encountered, categorized by final processing status.',
    ['status'] # e.g., 'success', 'failed_loading', 'failed_processing_critical', 'failed_processing_unexpected', 'skipped_empty', 'skipped_non_file'
)
DOC_PROCESSING_TIME = Histogram(
    'doc_processing_duration_seconds',
    'Time taken to process each document through its pipeline (load to finish/move), by document type.',
    ['doc_type'], # e.g., 'law', 'other', 'unclassified_or_failed'
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120, 300, 600, 900, 1200, 1800, float('inf')] 
)
TEXT_PROCESSING_STAGES_LATENCY = Histogram(
    'text_processing_stage_latency_seconds',
    'Latency of specific text processing stages.',
    ['stage_name'], # e.g., 'pre_clean_for_splitting', 'final_clean_lemmatize', 'chunking', 'article_splitting', 'stanza_lemmatization'
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, float('inf')]
)

logger.info("All Prometheus metrics definitions loaded and server attempted to start.")