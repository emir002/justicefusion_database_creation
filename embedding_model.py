#!/usr/bin/env python3
# embedding_model.py (Aligned with embedding_generator.py logic - V2 with central_error_handler)

import logging
import hashlib
import pickle
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import os
import sys
import time
import re

# Third-party libraries
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import lmdb
from tqdm import tqdm
import concurrent.futures

# Imports for text processing (mimicking embedding_generator.py)
import nltk
import stanza
try:
    from langdetect import detect, LangDetectException
except ImportError:
    logging.getLogger(__name__).warning("langdetect library not found. Please install it (`pip install langdetect`) for EmbeddingModelManager.")
    def detect(text): raise LangDetectException("langdetect not installed")
    class LangDetectException(Exception): pass  # type: ignore

try:
    from unidecode import unidecode
except ImportError:
    logging.getLogger(__name__).warning("unidecode library not found for EmbeddingModelManager. Cyrillic transliteration will be skipped.")
    def unidecode(s): return s  # Fallback

# Local imports from the original embedding_model.py context
import config
import utils
import metrics

logger = logging.getLogger(__name__)

# --- NEW: single-source-of-truth decision ---
# We lemmatize in TextProcessor (text_processing.py) and NOT here.
LEMMATIZE_IN_EMBEDDER = False

# --- Central Error Handler ---
def _central_error_handler_internal(message: str, exception: Exception):
    if isinstance(exception, FileNotFoundError):
        logger.error(f"{message}: {exception}", exc_info=False)
    elif isinstance(exception, PermissionError):
        logger.error(f"{message}: {exception}", exc_info=False)
    elif isinstance(exception, (ValueError, TypeError)):
        logger.warning(f"{message}: {exception}", exc_info=False)
    elif isinstance(exception, lmdb.Error):
        logger.error(f"LMDB Specific Error - {message}: {exception}", exc_info=True)
    else:
        logger.error(f"Unhandled Error - {message}: {exception}", exc_info=True)

# --- Stanza Initialization (only when lemmatization in embedder is explicitly enabled) ---
_nlp_pipelines_internal: Dict[str, stanza.Pipeline] = {}
_supported_stanza_languages_internal = getattr(config, 'STANZA_LANGUAGES', ['hr', 'sr'])
_use_gpu_internal = torch.cuda.is_available()

if LEMMATIZE_IN_EMBEDDER:
    try:
        logger.info(f"Initializing Stanza for EmbeddingModelManager. GPU available: {_use_gpu_internal}")
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("NLTK 'punkt' not found for EmbeddingModelManager, downloading...")
            nltk.download('punkt', quiet=True)

        for lang_code in _supported_stanza_languages_internal:
            logger.info(f"Downloading/loading Stanza model for: {lang_code} in EmbeddingModelManager")
            stanza.download(lang_code, processors='tokenize,pos,lemma', verbose=False, logging_level='WARN')
            _nlp_pipelines_internal[lang_code] = stanza.Pipeline(
                lang_code,
                processors='tokenize,pos,lemma',
                use_gpu=_use_gpu_internal,
                verbose=False,
                logging_level='WARN'
            )
        logger.info("Stanza pipelines initialized successfully for EmbeddingModelManager.")
    except Exception as e_stanza_init:
        _central_error_handler_internal("Error downloading or initializing Stanza models in EmbeddingModelManager", e_stanza_init)
        logger.warning("Stanza lemmatization will be unavailable in EmbeddingModelManager.")
        _nlp_pipelines_internal = {}
else:
    logger.info("Skipping Stanza initialization in EmbeddingModelManager (LEMMATIZE_IN_EMBEDDER=False).")

# --- Text Processing Functions ---
_email_pattern_internal = re.compile(r'\S+@\S+\.\S+')
_url_pattern_internal = re.compile(r'http[s]?://\S+|www\.\S+')
_header_footer_pattern_internal = re.compile(r'(Page \d+ of \d+|Document Title)')
_empty_lines_pattern_internal = re.compile(r'\n\s*\n')
_special_char_pattern_internal = re.compile(r'\s{2,}')
_cyrillic_pattern_internal = re.compile(r'[\u0400-\u04FF]')

def _contains_cyrillic_internal(text: str) -> bool:
    return bool(_cyrillic_pattern_internal.search(text))

def _refine_text_advanced_internal(text: str, source_doc_type_for_logging: str = "EmbeddingModelManager_refined_text") -> str:
    """
    Clean text for embeddings. IMPORTANT: No lemmatization here if LEMMATIZE_IN_EMBEDDER=False
    to avoid double-lemmatization (TextProcessor handles it).
    """
    if not isinstance(text, str):
        return ""
    try:
        text = _email_pattern_internal.sub('', text)
        text = _url_pattern_internal.sub('', text)
        text = _header_footer_pattern_internal.sub('', text)
        text = _empty_lines_pattern_internal.sub('\n', text)
        text = _special_char_pattern_internal.sub(' ', text)
        text = text.strip()

        if _contains_cyrillic_internal(text):
            text = unidecode(text)

        # --- DISABLED LEMMATIZATION IN EMBEDDER (see LEMMATIZE_IN_EMBEDDER flag) ---
        if LEMMATIZE_IN_EMBEDDER and _nlp_pipelines_internal and text:
            detected_lang_code = None
            try:
                snippet_for_lang_detect = text[:min(len(text), 500)]
                if snippet_for_lang_detect.strip():
                    detected_lang_code = detect(snippet_for_lang_detect)
            except LangDetectException as lde:
                logger.debug(f"Langdetect failed for '{source_doc_type_for_logging}': {lde}. Snippet: '{snippet_for_lang_detect[:30]}...'")
            except Exception as e_detect:
                logger.warning(f"Unexpected error during langdetect for '{source_doc_type_for_logging}': {e_detect}.")

            stanza_lang_to_use = None
            if detected_lang_code and detected_lang_code in _supported_stanza_languages_internal and detected_lang_code in _nlp_pipelines_internal:
                stanza_lang_to_use = detected_lang_code
            else:
                if 'sr' in _supported_stanza_languages_internal and 'sr' in _nlp_pipelines_internal and _contains_cyrillic_internal(text[:500]):
                    stanza_lang_to_use = 'sr'
                elif 'hr' in _supported_stanza_languages_internal and 'hr' in _nlp_pipelines_internal:
                    stanza_lang_to_use = 'hr'
                elif _supported_stanza_languages_internal and _nlp_pipelines_internal and len(_nlp_pipelines_internal) > 0:
                    stanza_lang_to_use = next(iter(_nlp_pipelines_internal.keys()), None)

            if stanza_lang_to_use and stanza_lang_to_use in _nlp_pipelines_internal:
                logger.debug(f"Lemmatizing with Stanza pipeline: {stanza_lang_to_use} for '{source_doc_type_for_logging}'")
                pipeline = _nlp_pipelines_internal[stanza_lang_to_use]
                doc = pipeline(text)
                lemmatized_words = [word.lemma for sentence_obj in doc.sentences for word in sentence_obj.words if word.lemma and word.lemma.strip()]
                if lemmatized_words:
                    text = ' '.join(lemmatized_words)
                else:
                    logger.debug(f"Lemmatization resulted in empty text for '{source_doc_type_for_logging}', keeping pre-lemmatized text.")
        # ---------------------------------------------------------------------------

    except Exception as e_refine:
        _central_error_handler_internal(f"Error in _refine_text_advanced_internal for '{source_doc_type_for_logging}'", e_refine)
        return text.strip().lower() if isinstance(text, str) else ""
    return text.strip().lower()

def _preprocess_text_internal(text: str, source_doc_type_for_logging: str = "EmbeddingModelManager_preprocessed_text") -> str:
    return _refine_text_advanced_internal(text, source_doc_type_for_logging)

def _load_legal_documents_internal(directory_path: Path) -> Tuple[List[str], List[str]]:
    legal_docs_raw: List[str] = []
    loaded_filenames: List[str] = []
    if not directory_path.is_dir():
        logger.error(f"Provided path is not a directory for representative corpus: {directory_path}")
        return [], []

    files = list(directory_path.glob('*.*'))
    logger.info(f"Found {len(files)} potential items in representative corpus: {directory_path}")
    processed_count = 0
    for file_path in tqdm(files, desc="Loading representative documents"):
        if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.docx']:
            text_content: Optional[str] = None
            try:
                logger.debug(f"Reading representative file: {file_path.name}")
                if file_path.suffix.lower() == '.txt':
                    text_content = file_path.read_text(encoding='utf-8', errors='ignore')
                elif file_path.suffix.lower() == '.pdf':
                    try:
                        from pdfminer.high_level import extract_pages
                        from pdfminer.layout import LTTextContainer, LAParams
                        text_parts = []
                        laparams = LAParams(line_margin=0.4, boxes_flow=0.5, char_margin=2.0, detect_vertical=False)
                        for page_layout in extract_pages(str(file_path), laparams=laparams):
                            for element in page_layout:
                                if isinstance(element, LTTextContainer):
                                    text_parts.append(element.get_text())
                        text_content = "".join(text_parts) if text_parts else ""
                    except ImportError:
                        logger.warning("pdfminer.six not fully available for detailed PDF extraction.")
                    except Exception as pdf_e:
                        _central_error_handler_internal(f"PDFMiner (extract_pages) failed for {file_path.name}", pdf_e)
                elif file_path.suffix.lower() == '.docx':
                    try:
                        from docx import Document as DocxDocument
                        doc = DocxDocument(str(file_path))
                        text_content = "\n".join([para.text for para in doc.paragraphs if para.text and para.text.strip()])
                    except ImportError:
                        logger.warning("python-docx not installed. Cannot read .docx files.")
                    except Exception as docx_e:
                        _central_error_handler_internal(f"Error reading DOCX {file_path.name}", docx_e)

                if text_content and text_content.strip():
                    legal_docs_raw.append(text_content.strip())
                    loaded_filenames.append(file_path.name)
                    processed_count += 1
                else:
                    logger.warning(f"No text extracted or text was empty for file: {file_path.name}")
            except Exception as e_load_file:
                _central_error_handler_internal(f"Error processing representative file {file_path.name}", e_load_file)
    logger.info(f"Successfully loaded text from {processed_count} representative files out of {len(files)} items scanned.")
    return legal_docs_raw, loaded_filenames

class EmbeddingModelManager:
    def __init__(self):
        logger.info("Initializing EmbeddingModelManager (aligned with embedding_generator.py logic)...")
        self.model_name_or_path = config.EMBEDDING_MODEL_NAME_OR_PATH
        self.device = config.DEVICE
        self.model = self._load_model_aligned_with_generator()
        if not self.model:
            raise RuntimeError("Embedding model failed to load. Cannot proceed.")

        self.dataset_cache_path_str = str(config.DATASET_EMBEDDING_CACHE_PATH)
        ds_cache_map_size_gb = 1
        ds_cache_map_size_bytes = ds_cache_map_size_gb * 1024 * 1024 * 1024
        logger.info(f"Opening LMDB dataset embedding cache at '{self.dataset_cache_path_str}' "
                    f"with map_size={ds_cache_map_size_gb}GB (subdir=False alignment).")
        try:
            Path(self.dataset_cache_path_str).parent.mkdir(parents=True, exist_ok=True)
            self.ds_cache_env = lmdb.open(
                self.dataset_cache_path_str, map_size=ds_cache_map_size_bytes,
                subdir=False, lock=True, readonly=False, meminit=False, map_async=True
            )
            logger.info("LMDB dataset embedding cache opened successfully.")
        except lmdb.Error as e:
            _central_error_handler_internal(f"Failed to open LMDB dataset embedding cache at '{self.dataset_cache_path_str}'", e)
            self.ds_cache_env = None
            logger.warning("Proceeding without LMDB dataset embedding cache. Performance will be affected.")
        except Exception as e_mkdir:
            _central_error_handler_internal(f"Failed to create parent directory for LMDB file '{self.dataset_cache_path_str}'", e_mkdir)
            self.ds_cache_env = None
            logger.warning("Proceeding without LMDB dataset embedding cache due to directory error.")

        self.dataset_embeddings = self._compute_or_load_dataset_embeddings_aligned_with_generator()
        if self.model and self.dataset_embeddings is not None:
            logger.info("EmbeddingModelManager initialized successfully (aligned version).")
        else:
            logger.error("EmbeddingModelManager (aligned) init failed: Model or dataset embeddings missing.")

    def _load_model_aligned_with_generator(self) -> Optional[SentenceTransformer]:
        logger.info(f"Loading embedding model (aligned): {self.model_name_or_path} onto device: {self.device}")
        try:
            model_instance = SentenceTransformer(self.model_name_or_path, device=self.device, trust_remote_code=True)
            logger.info(f"Embedding model '{self.model_name_or_path}' loaded successfully (aligned).")

            if hasattr(model_instance, 'max_seq_length'):
                logger.info(f"Model directly has max_seq_length: {model_instance.max_seq_length}")
            elif hasattr(model_instance, 'get_max_seq_length') and callable(model_instance.get_max_seq_length):
                model_instance.max_seq_length = model_instance.get_max_seq_length()
                logger.info(f"Model max_seq_length from get_max_seq_length(): {model_instance.max_seq_length}")
            elif hasattr(model_instance, 'tokenizer') and hasattr(model_instance.tokenizer, 'model_max_length'):
                model_instance.max_seq_length = model_instance.tokenizer.model_max_length
            elif (len(model_instance) > 0 and hasattr(model_instance[0], 'auto_model') and
                  hasattr(model_instance[0].auto_model, 'config') and
                  hasattr(model_instance[0].auto_model.config, 'max_position_embeddings')):
                model_instance.max_seq_length = model_instance[0].auto_model.config.max_position_embeddings
            else:
                model_instance.max_seq_length = 512
                logger.warning(f"Assuming max_seq_length: {model_instance.max_seq_length}")
            logger.info(f"Effective max_seq_length for model: {model_instance.max_seq_length}")

            model_expected_minicorpus_size = config.EXPECTED_MINICORPUS_SIZE
            try:
                if (len(model_instance) > 0 and hasattr(model_instance[0], 'auto_model') and
                        hasattr(model_instance[0].auto_model, 'config') and
                        hasattr(model_instance[0].auto_model.config, 'transductive_corpus_size')):
                    model_instance.minicorpus_size = model_instance[0].auto_model.config.transductive_corpus_size
                    if model_instance.minicorpus_size != model_expected_minicorpus_size:
                         logger.warning(f"Model's transductive_corpus_size ({model_instance.minicorpus_size}) "
                                        f"differs from config.EXPECTED_MINICORPUS_SIZE ({model_expected_minicorpus_size}). Using model's value.")
                else:
                    model_instance.minicorpus_size = model_expected_minicorpus_size
                    logger.warning(f"Using EXPECTED_MINICORPUS_SIZE from config.py: {model_expected_minicorpus_size}")
            except Exception as e_mc_size:
                model_instance.minicorpus_size = model_expected_minicorpus_size
                _central_error_handler_internal(f"Using EXPECTED_MINICORPUS_SIZE due to error accessing model's transductive_corpus_size", e_mc_size)
            logger.info(f"Final minicorpus_size for model: {model_instance.minicorpus_size}")
            return model_instance
        except Exception as e:
            _central_error_handler_internal(f"Failed to load embedding model '{self.model_name_or_path}' (aligned)", e)
            return None

    def _get_minicorpus_size(self) -> int:
        if self.model and hasattr(self.model, 'minicorpus_size'):
            return self.model.minicorpus_size
        logger.warning("model.minicorpus_size not set. Fallback to config.EXPECTED_MINICORPUS_SIZE.")
        return config.EXPECTED_MINICORPUS_SIZE

    def _load_representative_docs_aligned_with_generator(self, corpus_dir: Path, num_required: int) -> List[str]:
        logger.info(f"Loading raw representative documents from: {corpus_dir} (num_required for selection: {num_required})")
        raw_docs, _ = _load_legal_documents_internal(corpus_dir)
        if not raw_docs:
            raise ValueError(f"Rep corpus empty or failed to load: {corpus_dir}")
        raw_docs.sort()
        logger.info(f"Loaded and sorted {len(raw_docs)} raw documents from representative corpus.")
        random.seed(42)
        if len(raw_docs) > num_required:
            logger.info(f"Sampling {num_required} raw docs from {len(raw_docs)} (seed 42).")
            return random.sample(raw_docs, num_required)
        elif len(raw_docs) < num_required:
            logger.warning(f"Raw docs ({len(raw_docs)}) < required ({num_required}). Oversampling (seed 42).")
            return random.choices(raw_docs, k=num_required)
        else:
            return raw_docs

    def _compute_or_load_dataset_embeddings_aligned_with_generator(self) -> Optional[torch.Tensor]:
        logger.info("Attempting to compute or load first-stage dataset embeddings (aligned version).")
        if not self.model:
            logger.error("Model not loaded.")
            return None

        corpus_dir = Path(config.REPRESENTATIVE_CORPUS_DIR)
        if not corpus_dir.is_dir():
            logger.error(f"Rep corpus dir '{corpus_dir}' not found.")
            return None

        _, current_filenames = _load_legal_documents_internal(corpus_dir)
        if not current_filenames:
            logger.error(f"No files in rep corpus '{corpus_dir}'.")
            return None
        current_filenames.sort()
        current_file_count = len(current_filenames)

        lmdb_key_embeddings = b'dataset_embeddings'
        lmdb_key_count = b'dataset_file_count'
        lmdb_key_names = b'dataset_file_names'
        recompute_needed = True
        cached_embeddings_tensor: Optional[torch.Tensor] = None

        if self.ds_cache_env:
            try:
                with self.ds_cache_env.begin(write=False) as txn:
                    stored_count_bytes = txn.get(lmdb_key_count)
                    stored_names_bytes = txn.get(lmdb_key_names)
                    if stored_count_bytes and stored_names_bytes:
                        try:
                            stored_count = int(stored_count_bytes.decode())
                            stored_filenames_list = pickle.loads(stored_names_bytes)
                            if current_file_count == stored_count and current_filenames == stored_filenames_list:
                                metrics.EMBED_CACHE_HITS.inc()
                                serialized_data = txn.get(lmdb_key_embeddings)
                                if serialized_data:
                                    loaded_tensor = pickle.loads(serialized_data)
                                    if isinstance(loaded_tensor, torch.Tensor):
                                        cached_embeddings_tensor = loaded_tensor.to(self.device)
                                        recompute_needed = False
                                        logger.info(f"Dataset embeddings loaded from LMDB. Shape: {cached_embeddings_tensor.shape}")
                                    else:
                                        logger.error("Cached data not PyTorch Tensor. Recomputing.")
                                else:
                                    logger.warning("Metadata matched, no embedding data. Recomputing.")
                            else:
                                logger.info("Rep corpus file count/names mismatch. Recomputing.")
                        except Exception as e_meta_load:
                            _central_error_handler_internal("Error reading metadata from LMDB cache (aligned)", e_meta_load)
                    else:
                        logger.info("No count/names in LMDB cache. Recomputing.")
            except lmdb.Error as e_lmdb_read:
                _central_error_handler_internal("LMDB error reading dataset cache (aligned)", e_lmdb_read)
        else:
            logger.warning("LMDB cache not available. Recomputing.")

        if not recompute_needed and cached_embeddings_tensor is not None:
            return cached_embeddings_tensor

        metrics.EMBED_CACHE_MISSES.inc()
        logger.info("Proceeding to compute new first-stage dataset embeddings (aligned version).")
        comp_start_time = time.time()
        minicorpus_size_needed = self._get_minicorpus_size()
        try:
            minicorpus_docs_raw_selected = self._load_representative_docs_aligned_with_generator(corpus_dir, minicorpus_size_needed)
        except ValueError as e_load_rep:
            _central_error_handler_internal("Failed to load rep docs for dataset embeddings", e_load_rep)
            return None

        preprocessed_minicorpus_docs: List[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(_preprocess_text_internal, doc, f"rep_corpus_minicorpus_{i}") for i, doc in enumerate(minicorpus_docs_raw_selected)]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Preprocessing minicorpus (aligned)"):
                try:
                    preprocessed_minicorpus_docs.append(future.result())
                except Exception as exc:
                    logger.error(f"Doc in minicorpus gen error during preprocessing: {exc}")
                    preprocessed_minicorpus_docs.append("")

        final_preprocessed_docs = [doc for doc in preprocessed_minicorpus_docs if doc.strip()]
        if not final_preprocessed_docs:
            logger.error("All minicorpus docs empty post-preprocessing.")
            return None
        logger.info(f"Successfully preprocessed {len(final_preprocessed_docs)} docs for minicorpus.")

        try:
            new_dataset_embeddings_tensor = self.model.encode(
                final_preprocessed_docs, prompt_name="document", batch_size=32,
                show_progress_bar=True, convert_to_tensor=True, device=self.device
            )
            logger.info(f"First-stage dataset embeddings computed. Shape: {new_dataset_embeddings_tensor.shape}. Took {time.time() - comp_start_time:.2f}s.")
            if self.ds_cache_env:
                try:
                    with self.ds_cache_env.begin(write=True) as txn:
                        txn.put(lmdb_key_embeddings, pickle.dumps(new_dataset_embeddings_tensor.cpu()))
                        txn.put(lmdb_key_count, str(current_file_count).encode('utf-8'))
                        txn.put(lmdb_key_names, pickle.dumps(current_filenames))
                    logger.info(f"Dataset embeddings, count ({current_file_count}), names saved to LMDB (aligned).")
                except lmdb.MapFullError:
                    logger.error("LMDB dataset cache full.")
                except Exception as e_lmdb_write:
                    _central_error_handler_internal("Failed to save dataset embeddings to LMDB (aligned)", e_lmdb_write)
            return new_dataset_embeddings_tensor
        except Exception as e_encode:
            _central_error_handler_internal("Error during first-stage dataset embedding computation (aligned)", e_encode)
            return None

    def get_embeddings(self, texts: List[str], prompt_name: str, batch_size: int = 32) -> Optional[List[List[float]]]:
        if not self.model:
            logger.error("Model not loaded.")
            return None
        if self.dataset_embeddings is None:
            logger.error("Dataset embeddings not available.")
            return None
        if not texts:
            logger.warning("Empty list for embedding.")
            return []
        if not prompt_name:
            logger.error("Prompt name needed.")
            return None

        preprocessed_texts: List[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(_preprocess_text_internal, text, f"get_embeddings_input_{i}") for i, text in enumerate(texts)]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(texts), desc="Preprocessing texts for get_embeddings"):
                try:
                    preprocessed_texts.append(future.result())
                except Exception as exc:
                    logger.error(f"Text idx gen error in get_embeddings preprocessing: {exc}")
                    preprocessed_texts.append("")

        final_texts_to_encode = [pt for pt in preprocessed_texts if pt.strip()]
        if not final_texts_to_encode:
            logger.warning("All texts empty post-preprocessing in get_embeddings.")
            return []

        logger.info(f"Generating L2 stage embeddings for {len(final_texts_to_encode)} texts, prompt: '{prompt_name}'.")
        try:
            emb_start_time = time.time()
            embeddings_tensor = self.model.encode(
                final_texts_to_encode, prompt_name=prompt_name, dataset_embeddings=self.dataset_embeddings,
                batch_size=batch_size, show_progress_bar=False, convert_to_tensor=True, device=self.device
            )
            if embeddings_tensor.ndim == 2 and embeddings_tensor.shape[0] > 0:
                embeddings_tensor = embeddings_tensor / torch.linalg.norm(embeddings_tensor, ord=2, dim=1, keepdim=True)
            elif embeddings_tensor.ndim == 1 and embeddings_tensor.shape[0] > 0:
                embeddings_tensor = embeddings_tensor / torch.linalg.norm(embeddings_tensor, ord=2)
            logger.debug("Second-stage embeddings L2 normalized.")

            embeddings_list = embeddings_tensor.cpu().tolist()
            logger.info(f"Generated/normalized {len(embeddings_list)} L2 stage embeddings in {time.time() - emb_start_time:.2f}s.")
            return embeddings_list
        except Exception as e:
            _central_error_handler_internal(f"Error generating L2 stage embeddings", e)
            return None

    def get_embedding_dimension(self) -> Optional[int]:
        if not self.model:
            logger.error("Model not loaded.")
            return None
        try:
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                dimension = self.model.get_sentence_embedding_dimension()
            elif hasattr(self.model, 'embedding_size'):
                dimension = self.model.embedding_size
            elif len(self.model) > 0 and hasattr(self.model[0], 'auto_model'):
                dimension = self.model[0].auto_model.config.hidden_size
            else:
                if self.dataset_embeddings is not None and self.dataset_embeddings.ndim == 2:
                    dimension = self.dataset_embeddings.shape[1]
                else:
                    logger.error("Could not determine embedding dimension.")
                    return None
            logger.info(f"Reported embedding dimension: {dimension}")
            return dimension
        except Exception as e:
            _central_error_handler_internal("Could not get embedding dimension from model", e)
            if self.dataset_embeddings is not None and self.dataset_embeddings.ndim == 2:
                return self.dataset_embeddings.shape[1]
            return None

    def close(self):
        if hasattr(self, 'ds_cache_env') and self.ds_cache_env:
            logger.info(f"Closing dataset embedding cache (LMDB) at '{self.dataset_cache_path_str}'.")
            try:
                self.ds_cache_env.close()
                logger.info("LMDB cache closed.")
            except lmdb.Error as e:
                _central_error_handler_internal("Error closing LMDB dataset cache", e)

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(name)s (%(module)s.%(funcName)s:%(lineno)d): %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    logger.info("--- Starting EmbeddingModelManager Self-Test (Aligned Version V2) ---")
    if not hasattr(config, 'BASE_DIR'):
        logger.error("Config incomplete.")
        sys.exit(1)
    try:
        Path(config.REPRESENTATIVE_CORPUS_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.DATASET_EMBEDDING_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    except Exception as e_dir_test:
        logger.error(f"Test dir creation failed: {e_dir_test}")
        sys.exit(1)

    rep_corpus_path_for_test = Path(config.REPRESENTATIVE_CORPUS_DIR)
    if not list(rep_corpus_path_for_test.glob('*.*')):
        logger.warning(f"Rep corpus '{rep_corpus_path_for_test}' empty. Creating dummy files.")
        dummy_text = "Testni dokument broj {}. Svrha testiranja."
        num_files_to_create = config.EXPECTED_MINICORPUS_SIZE if config.EXPECTED_MINICORPUS_SIZE > 0 else 10
        for i in range(num_files_to_create):
            try:
                (rep_corpus_path_for_test / f"dummy_rep_doc_{i+1}.txt").write_text(dummy_text.format(i+1), encoding='utf-8')
            except Exception as e_create:
                logger.error(f"Failed to create dummy_rep_doc_{i+1}.txt: {e_create}")
        logger.info(f"Created {num_files_to_create} dummy files in '{rep_corpus_path_for_test}'.")

    embedder_instance = None
    try:
        embedder_instance = EmbeddingModelManager()
        if embedder_instance.model and embedder_instance.dataset_embeddings is not None:
            logger.info(f"Model embedding dimension: {embedder_instance.get_embedding_dimension()}")
            logger.info(f"Dataset_embeddings shape: {embedder_instance.dataset_embeddings.shape}")
            test_docs_raw = ["Prva testna rečenica.", "Porez na dodatu vrijednost?", "Zakon o radu propisuje."]

            refined_test_docs = [_preprocess_text_internal(doc, "main_test_doc") for doc in test_docs_raw]
            logger.info(f"Test sentences for DOCUMENT prompt (preprocessed by _preprocess_text_internal for this test): {refined_test_docs}")

            doc_embeddings = embedder_instance.get_embeddings(test_docs_raw, prompt_name=config.EMBEDDING_PROMPT_DOCUMENT)
            if doc_embeddings and len(doc_embeddings) == len(test_docs_raw):
                logger.info(f"Doc embeddings generated. First: {doc_embeddings[0][:5] if doc_embeddings[0] else 'N/A'}")
            else:
                logger.error(f"Doc embedding failed. Got: {len(doc_embeddings) if doc_embeddings else 'None'}")

            test_queries_raw = ["Uslovi za penziju?", "Izmjene obligacionih odnosa."]
            query_embeddings = embedder_instance.get_embeddings(test_queries_raw, prompt_name=config.EMBEDDING_PROMPT_QUERY)
            if query_embeddings and len(query_embeddings) == len(test_queries_raw):
                logger.info(f"Query embeddings generated. First: {query_embeddings[0][:5] if query_embeddings[0] else 'N/A'}")
            else:
                logger.error(f"Query embedding failed. Got: {len(query_embeddings) if query_embeddings else 'None'}")
        else:
            logger.error("Test failed: Model or dataset_embeddings not initialized.")
    except Exception as e_main_test:
        _central_error_handler_internal("Critical error during EmbeddingModelManager self-test (Aligned V2)", e_main_test)
    finally:
        if embedder_instance:
            logger.info("Closing embedder (aligned test V2)...")
            embedder_instance.close()
    logger.info("--- EmbeddingModelManager Self-Test Finished (Aligned Version V2) ---")
