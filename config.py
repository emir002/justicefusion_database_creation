#!/usr/bin/env python3
# config.py

"""Central configuration module.

Goals:
  - Keep the module *easy to scan* and *safe to tweak*.
  - Provide a single switch to run **LLM calls** either:
        (a) externally (Gemini), or
        (b) locally via Ollama.

Notes:
  - Other modules import constants from here; so we keep existing names for backwards
    compatibility while reorganizing and adding the new LLM switches.
"""

import os
import logging  # Basic fallback logging if config is imported before dictConfig is applied
from pathlib import Path

import torch


logger = logging.getLogger(__name__)  # Module logger

DEBUG = os.getenv("DEBUG", "0").strip().lower() in {"1", "true", "yes"}


# -----------------------------------------------------------------------------
# General
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()
LOG_FILE = BASE_DIR / "processing.log"


#####################################################################################
RUN_MODE = "FIRST_RUN"  # Options: "FIRST_RUN", "UPDATE"
#####################################################################################


# Ensure logger is configured if this is the first import, before logging anything.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

logger.info(f"Configuration loaded. BASE_DIR: {BASE_DIR}, RUN_MODE: {RUN_MODE}")


# -----------------------------------------------------------------------------
# Input / Output folders
# -----------------------------------------------------------------------------

INPUT_FOLDER = BASE_DIR / "data" / "input_documents"
FINISHED_FOLDER = BASE_DIR / "data" / "finished_documents"
CACHE_DIR = BASE_DIR / "cache"
QUARANTINE_FOLDER = BASE_DIR / "data" / "quarantine_folder"

############################################################################################
REPRESENTATIVE_CORPUS_DIR = Path("C:/cde-small-v2_graph_v2/5_4 _test2/data/legal_documents")
############################################################################################

logger.info(f"Input folder: {INPUT_FOLDER}")
logger.info(f"Finished folder: {FINISHED_FOLDER}")
logger.info(f"Cache folder: {CACHE_DIR}")
logger.info(f"Quarantine folder: {QUARANTINE_FOLDER}")
logger.info(f"Representative corpus folder: {REPRESENTATIVE_CORPUS_DIR}")


# -----------------------------------------------------------------------------
# Caching
# -----------------------------------------------------------------------------

EMBEDDING_CACHE_PATH = CACHE_DIR / "embedding_cache.lmdb"  # Cache for individual text embeddings (currently unused)

############################################################################################
DATASET_EMBEDDING_CACHE_PATH = Path("C:/cde-small-v2_graph_v2/5_4 _test2/embedding_cache.lmdb")
############################################################################################

RATE_LIMIT_STATE_FILE = CACHE_DIR / "rate_limit_state.json"

logger.info(f"Individual embedding cache path (unused by EmbeddingModelManager): {EMBEDDING_CACHE_PATH}")
logger.info(f"Dataset embedding cache path: {DATASET_EMBEDDING_CACHE_PATH}")
logger.info(f"Rate limit state file: {RATE_LIMIT_STATE_FILE}")


# -----------------------------------------------------------------------------
# Text processing
# -----------------------------------------------------------------------------

CHUNK_MAX_CHARS = 1500
CHUNK_OVERLAP = 300
STANZA_LANGUAGES = ['hr', 'sr']

# -----------------------------------------------------------------------------
# Chunking strategy
# -----------------------------------------------------------------------------

# "sentence" (current) or "proposition" (new)
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "proposition").strip().lower()

# For proposition packing
PROP_MAX_CHARS_PER_CHUNK = int(os.getenv("PROP_MAX_CHARS_PER_CHUNK", "1500"))
PROP_OVERLAP_PROPOSITIONS = int(os.getenv("PROP_OVERLAP_PROPOSITIONS", "2"))  # overlap by N propositions
PROP_MAX_PROPOSITIONS_PER_CHUNK = int(os.getenv("PROP_MAX_PROPOSITIONS_PER_CHUNK", "12"))

# LLM proposition extraction limits
PROP_LLM_MAX_INPUT_CHARS = int(os.getenv("PROP_LLM_MAX_INPUT_CHARS", "6000"))
PROP_LLM_MAX_OUTPUT_TOKENS = int(os.getenv("PROP_LLM_MAX_OUTPUT_TOKENS", "6144"))

# Safer defaults for gpt-oss:* and other reasoning-heavy models
OLLAMA_JSON_MODE_FOR_PROPOSITIONS = os.getenv(
    "OLLAMA_JSON_MODE_FOR_PROPOSITIONS", "1"
).strip().lower() in {"1", "true", "yes"}

# Bound proposition extraction output (prevents token-limit cutoffs)
PROP_MAX_PROPS_PER_CALL = int(os.getenv("PROP_MAX_PROPS_PER_CALL", "60"))
PROP_MAX_CHARS_EACH = int(os.getenv("PROP_MAX_CHARS_EACH", "220"))

# Bound graph extraction output (prevents giant JSON payloads)
GRAPH_MAX_NODES_PER_CHUNK = int(os.getenv("GRAPH_MAX_NODES_PER_CHUNK", "20"))
GRAPH_MAX_EDGES_PER_CHUNK = int(os.getenv("GRAPH_MAX_EDGES_PER_CHUNK", "30"))
GRAPH_DESC_MAX_WORDS = int(os.getenv("GRAPH_DESC_MAX_WORDS", "12"))

# If true, try LLM proposition extraction; if false, use heuristic splitting only
ENABLE_LLM_PROPOSITION_EXTRACTION = os.getenv("ENABLE_LLM_PROPOSITION_EXTRACTION", "1").strip() not in {"0","false","False"}

# Max characters of text snippet for LLM document type classification
LLM_CLASSIFICATION_SNIPPET_MAX_LEN = 3000

logger.info(
    f"Text processing: CHUNK_MAX_CHARS={CHUNK_MAX_CHARS}, CHUNK_OVERLAP={CHUNK_OVERLAP}, "
    f"STANZA_LANGUAGES={STANZA_LANGUAGES}, LLM_CLASSIFICATION_SNIPPET_MAX_LEN={LLM_CLASSIFICATION_SNIPPET_MAX_LEN}"
)


# -----------------------------------------------------------------------------
# Embedding model (cde-small-v2)
# -----------------------------------------------------------------------------

########################################################################################################
LOCAL_MODEL_PATH = Path("C:/cde-small-v2_graph_v2/5_4 _test2/cde-small-v2")
########################################################################################################

EMBEDDING_MODEL_NAME_OR_PATH = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else "jxm/cde-small-v2"
EXPECTED_MINICORPUS_SIZE = 512
EMBEDDING_PROMPT_DOCUMENT = "document"
EMBEDDING_PROMPT_QUERY = "query"

logger.info(f"Embedding model path/name: {EMBEDDING_MODEL_NAME_OR_PATH}, Expected minicorpus size: {EXPECTED_MINICORPUS_SIZE}")
logger.info(f"Embedding prompts: Document='{EMBEDDING_PROMPT_DOCUMENT}', Query='{EMBEDDING_PROMPT_QUERY}'")


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
USE_GPU = (DEVICE == "cuda" or DEVICE == "mps")
logger.info(f"Device selected: {DEVICE}, USE_GPU_explicitly_enabled: {USE_GPU}")


# -----------------------------------------------------------------------------
# LLM switching (NEW)
# -----------------------------------------------------------------------------

# Pick where **all** LLM calls go:
#   - "local"    -> only Ollama
#   - "external" -> Gemini (with existing Ollama fallback logic in llm_interaction.py)
LLM_MODE = "local"  # "local" | "external"

# Choose the external model name by typing it here.
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

# Choose the local Ollama model name by typing it here.
# (Requested default for local-only usage)
LOCAL_LLM_MODEL_NAME = "qwen2.5:14b-instruct-q6_K"

# Ollama endpoint (OpenAI-style model server is NOT used here; this is Ollama's HTTP API).
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_ENDPOINT = "/api/generate"
# Prefer chat endpoint for instruct/chat models (e.g., gpt-oss) in local mode.
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_USE_CHAT = os.getenv("OLLAMA_USE_CHAT", "1").strip() not in {"0", "false", "False"}
# Reasoning models can return empty output if token budget is too small.
MIN_LOCAL_MAX_OUTPUT_TOKENS_FOR_REASONING = int(os.getenv("MIN_LOCAL_MAX_OUTPUT_TOKENS_FOR_REASONING", "256"))
REASONING_MODEL_PREFIXES = [p.strip() for p in os.getenv("REASONING_MODEL_PREFIXES", "gpt-oss").split(",") if p.strip()]

# Local generation defaults
LOCAL_LLM_TEMPERATURE = 0.5
LOCAL_LLM_MAX_OUTPUT_TOKENS = 8192
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_TIMEOUT = 240  # seconds


# -----------------------------------------------------------------------------
# External LLM (Gemini)
# -----------------------------------------------------------------------------

GEMINI_API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
logger.info(f"LLM Model: {GEMINI_MODEL_NAME}, Number of API keys: {len(GEMINI_API_KEYS)}")

LLM_RATE_LIMIT_PER_MINUTE = 15  # rate per key
LLM_TOKEN_LIMIT_PER_MINUTE = 250_000  # Gemini input tokens per minute per key
LLM_DAILY_LIMIT_PER_KEY = 1000  # LLM daily rate limit per key
LLM_RETRY_ATTEMPTS = 5
LLM_RETRY_WAIT_INITIAL = 2  # seconds
LLM_RETRY_WAIT_MULTIPLIER = 2
LLM_RETRY_WAIT_MAX = 60     # seconds
LLM_MAX_RETRIES_PER_ITEM = 3  # Added from discussion, max retries for a specific LLM task on an item (e.g. summary for one chunk)
LLM_MAX_RETRIES = LLM_MAX_RETRIES_PER_ITEM
LLM_BACKOFF_BASE_SECONDS = 0.8
OLLAMA_JSON_MODE_FOR_ENTITY_EXTRACTION = True
logger.info(
    f"LLM Rate Limits: {LLM_RATE_LIMIT_PER_MINUTE}/min per key, {LLM_DAILY_LIMIT_PER_KEY}/day per key. "
    f"Call Retries: {LLM_RETRY_ATTEMPTS} attempts. Max item retries: {LLM_MAX_RETRIES_PER_ITEM}"
)

# Max input character lengths for different LLM tasks (adjust as needed)
LLM_SUMMARY_MAX_INPUT_LENGTH = 7000  # Max characters from document to send for summary
LLM_TITLE_SNIPPET_MAX_LEN = 2000     # Max characters from start of doc for title extraction
LLM_ENTITY_MAX_INPUT_LENGTH = int(os.getenv("LLM_ENTITY_MAX_INPUT_LENGTH", "3000"))
# Keep JSON tasks deterministic and give graph extraction enough output budget.
LLM_ENTITY_EXTRACTION_TEMPERATURE = float(os.getenv("LLM_ENTITY_EXTRACTION_TEMPERATURE", "0.1"))
LLM_ENTITY_EXTRACTION_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_ENTITY_EXTRACTION_MAX_OUTPUT_TOKENS", "1024"))
LLM_ENTITY_EXTRACTION_RETRY_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_ENTITY_EXTRACTION_RETRY_MAX_OUTPUT_TOKENS", "4096"))

# -----------------------------------------------------------------------------
# Local LLM (Ollama) – backwards-compatible names (used by llm_interaction.py)
# -----------------------------------------------------------------------------

# Keep the old names so you don't have to touch other modules.
LIGHTRAG_MODEL_NAME = LOCAL_LLM_MODEL_NAME
LIGHTRAG_OLLAMA_URL = f"{OLLAMA_BASE_URL}{OLLAMA_GENERATE_ENDPOINT}"
LIGHTRAG_TEMPERATURE = LOCAL_LLM_TEMPERATURE
LIGHTRAG_MAX_OUTPUT_TOKENS = LOCAL_LLM_MAX_OUTPUT_TOKENS

# How long (in seconds) we wait before re-testing Gemini once it starts
# returning 429 / quota / service errors.
PRIMARY_LLM_COOLDOWN       = 600           # 10 minutes is a sensible default

# --- Weaviate ---
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_SECURE = os.getenv("WEAVIATE_SECURE", "false").lower() == "true"
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "localhost")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
_weaviate_grpc_secure_value = os.getenv("WEAVIATE_GRPC_SECURE")
_weaviate_grpc_typo_secure_value = os.getenv("WEAVIATE_GRPC_GRPC_SECURE")
if _weaviate_grpc_typo_secure_value is not None:
    logger.warning(
        "Detected env var WEAVIATE_GRPC_GRPC_SECURE (likely typo). "
        "Prefer WEAVIATE_GRPC_SECURE. Canonical resolution uses WEAVIATE_GRPC_SECURE first, "
        "then falls back to WEAVIATE_GRPC_GRPC_SECURE for backwards compatibility."
    )
_resolved_weaviate_grpc_secure_raw = (
    _weaviate_grpc_secure_value
    if _weaviate_grpc_secure_value is not None
    else (_weaviate_grpc_typo_secure_value if _weaviate_grpc_typo_secure_value is not None else "false")
)
WEAVIATE_GRPC_SECURE = str(_resolved_weaviate_grpc_secure_raw).strip().lower() == "true"
logger.info(f"Weaviate HTTP: host={WEAVIATE_HOST}, port={WEAVIATE_PORT}, secure={WEAVIATE_SECURE}")
logger.info(f"Weaviate gRPC: host={WEAVIATE_GRPC_HOST}, port={WEAVIATE_GRPC_PORT}, secure={WEAVIATE_GRPC_SECURE}")

WEAVIATE_ARTICLE_CLASS = "LegalArticle"
WEAVIATE_DOC_CLASS = "Document"
WEAVIATE_NODE_CLASS = "GraphNodes"
WEAVIATE_EDGE_CLASS = "GraphEdges"
logger.info(
    f"Weaviate classes: Article='{WEAVIATE_ARTICLE_CLASS}', Document='{WEAVIATE_DOC_CLASS}', "
    f"Node='{WEAVIATE_NODE_CLASS}', Edge='{WEAVIATE_EDGE_CLASS}'"
)

WEAVIATE_PROPERTY_TOKENIZATION = "word"  # For BM25/keyword search
logger.info(f"Weaviate property tokenization: {WEAVIATE_PROPERTY_TOKENIZATION}")

# --- Initialization ---
def ensure_dirs_exist():
    """Creates necessary directories if they don't exist."""
    logger.info("Ensuring all necessary directories exist...")
    dirs_to_check = [INPUT_FOLDER, FINISHED_FOLDER, CACHE_DIR, REPRESENTATIVE_CORPUS_DIR]
    if not LOG_FILE.parent.exists():  # Ensure log file directory exists
        dirs_to_check.append(LOG_FILE.parent)

    for d_path in dirs_to_check:
        try:
            d_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory '{d_path}' ensured.")
        except Exception as e_dir:
            logger.error(f"Failed to create or ensure directory '{d_path}': {e_dir}", exc_info=True)
            raise  # Reraise if critical directory creation fails

ensure_dirs_exist()  # Call it once when config is loaded.

# --- Logging Configuration Dictionary ---
# This dictionary is used by logging.config.dictConfig() in main_indexer.py
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s (%(module)s.%(funcName)s:%(lineno)d): %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "level": "INFO",  # Console level
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": "DEBUG",  # File level - set to DEBUG to capture more details
            "formatter": "standard",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_FILE,
            "maxBytes": 10 * 1024 * 1024,  # Increased to 10MB
            "backupCount": 5,              # Increased backup count
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": "DEBUG",  # Root logger level - set to DEBUG
            "propagate": True,
        },
        # Quieten overly verbose libraries by sending their logs mainly to file if above WARNING
        "httpx": {"handlers": ["file"], "level": "WARNING", "propagate": False},
        "pdfminer": {"handlers": ["file"], "level": "WARNING", "propagate": False},
        "PIL": {"handlers": ["file"], "level": "WARNING", "propagate": False},
        "stanza": {"handlers": ["file"], "level": "WARNING", "propagate": False},  # Stanza can be very verbose at INFO/DEBUG
        "urllib3": {"handlers": ["file"], "level": "WARNING", "propagate": False},
        "google.generativeai": {"handlers": ["file"], "level": "WARNING", "propagate": False},
        "sentence_transformers": {"handlers": ["file"], "level": "WARNING", "propagate": False},
        "langdetect": {"handlers": ["file"], "level": "WARNING", "propagate": False},  # If it gets noisy
    },
}
# logger.info("LOGGING_CONFIG dictionary defined.") # Avoid logging this with the default basicConfig

# If this script is imported and dictConfig hasn't run yet from main_indexer,
# the initial log messages from this script might use a basic config.
# main_indexer.py should be the one to call dictConfig(config.LOGGING_CONFIG) first.
