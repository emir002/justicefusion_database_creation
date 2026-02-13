#!/usr/bin/env python3
# utils.py

import re
import json
import logging
import hashlib
from uuid import UUID # Kept for context, though functions return str
import unicodedata
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

# Third-party libraries
from weaviate.util import generate_uuid5 as weaviate_generate_uuid5
from json_repair import repair_json

logger = logging.getLogger(__name__)

# --- Text Cleaning Patterns (Consolidated & Refined) ---
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b')
URL_PATTERN = re.compile(r'(?:http[s]?://|www\.)\S+')
EMPTY_LINES_PATTERN = re.compile(r'\n\s*\n+')
EXTRA_WHITESPACE_PATTERN = re.compile(r'[ \t]+')
HYPHENATED_LINE_BREAK_PATTERN = re.compile(r'-\n\s*')
CYRILLIC_PATTERN = re.compile(r'[\u0400-\u04FF]')
REPLACEMENT_CHAR_PATTERN = re.compile(chr(0xFFFD))
DB3_DOKUMENTI_PREGLED_PATTERN = re.compile(r'^Dokumenti pregled', re.IGNORECASE | re.MULTILINE)


def normalize_unicode_text(text: str) -> str:
    """Apply Unicode normalization (NFKC) to regularize characters."""
    if not isinstance(text, str):
        return ""
    try:
        return unicodedata.normalize('NFKC', text)
    except Exception as e:
        logger.error(f"Error during Unicode normalization: {e}", exc_info=True)
        return text # Return original text on error

def contains_cyrillic(text: str) -> bool:
    """Return True if the text contains any Cyrillic characters."""
    if not isinstance(text, str):
        return False
    return bool(CYRILLIC_PATTERN.search(text))

def clean_text(text: str, is_pre_splitting: bool = False) -> str:
    """
    Cleans text with two modes:
    - is_pre_splitting=True: Applies minimal, safe cleaning before sentence tokenization
                             or regex-based splitting. Focuses on normalization, fixing broken
                             words, essential whitespace, and careful unidecode.
    - is_pre_splitting=False: Applies more comprehensive cleaning (removes emails, URLs etc.)
                              suitable for final text content of chunks/articles after splitting.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Apply Unicode Normalization (NFKC)
    text = normalize_unicode_text(text)

    # Replace specific Unicode replacement characters and fix hyphenated line breaks
    text = REPLACEMENT_CHAR_PATTERN.sub(' ', text)
    text = HYPHENATED_LINE_BREAK_PATTERN.sub('', text)

    # Conditionally transliterate Cyrillic using unidecode
    if contains_cyrillic(text):
        # Initialize fallback variable *before* the try block
        text_before_unidecode = text
        try:
            from unidecode import unidecode # Import only when needed
            text = unidecode(text)
        except ImportError:
            logger.warning("Module 'unidecode' is not installed. Cannot transliterate Cyrillic text.")
        except Exception as e_unidecode:
            logger.error(f"Error during Cyrillic transliteration with unidecode: {e_unidecode}", exc_info=True)
            # Fallback to text before the unidecode attempt if it fails
            text = text_before_unidecode

    if not is_pre_splitting:
        # Apply more thorough cleaning rules, typically after initial splitting
        text = EMAIL_PATTERN.sub(' ', text)
        text = URL_PATTERN.sub(' ', text)
        text = DB3_DOKUMENTI_PREGLED_PATTERN.sub('', text)

    # Perform essential whitespace cleanup for all cases
    text = EMPTY_LINES_PATTERN.sub('\n', text)
    text = EXTRA_WHITESPACE_PATTERN.sub(' ', text)
    text = text.strip()
    
    return text

# --- UUID Generation Functions ---
def generate_uuid_from_string(identifier: str, namespace: Optional[str] = None) -> str:
    """Generates a deterministic UUID string using Weaviate's library, optionally with a namespace."""
    # weaviate_generate_uuid5 handles namespace=None by using its internal default (NAMESPACE_DNS).
    # If a string namespace is provided, it uses that.
    return weaviate_generate_uuid5(identifier, namespace)

def generate_doc_chunk_id(filename: str, chunk_index: int) -> str:
    """Generates a deterministic UUID string for a document chunk."""
    identifier = f"document_chunk::filename:{filename}::chunk_index:{chunk_index}"
    return generate_uuid_from_string(identifier)

def generate_legal_article_id(law_title: str, article_number: str, source_filename: str) -> str:
    """Generates a deterministic UUID string for a specific legal article."""
    norm_title = law_title.strip().lower()
    norm_article_num = article_number.strip().lower().replace(" ", "")
    norm_filename = source_filename.strip()
    identifier = f"legal_article::law_title:{norm_title}::article_number:{norm_article_num}::source_file:{norm_filename}"
    return generate_uuid_from_string(identifier)

def generate_graph_node_id(entity_name: str, entity_type: str, source_id: str) -> str:
    """Generates a deterministic UUID string for a graph node (entity)."""
    norm_name = entity_name.strip().lower()
    norm_type = entity_type.strip().lower()
    norm_source = source_id.strip()
    identifier = f"graph_node::name:{norm_name}::type:{norm_type}::source_document_id:{norm_source}"
    return generate_uuid_from_string(identifier)

def generate_graph_edge_id(source_entity_name: str,
                           target_entity_name: str,
                           relationship_type: str,
                           source_id: str) -> str: # source_id is kept for signature compatibility, not used in ID generation
    """
    Creates a canonical/deterministic UUID string so that identical edges found across
    different documents will collapse onto the same Weaviate object.
    """
    # Canonical ordering treats the edge as undirected for identity purposes.
    src = source_entity_name.strip().lower()
    tgt = target_entity_name.strip().lower()
    if src <= tgt:
        left, right = src, tgt
    else:
        left, right = tgt, src

    rel = relationship_type.strip().lower().replace(" ", "_")
    
    # The identifier intentionally excludes source_id to ensure deduplication.
    canonical_identifier = f"graph_edge::from:{left}::to:{right}::rel:{rel}"
    return generate_uuid_from_string(canonical_identifier)

# --- JSON Handling ---
def extract_json_block(text: str) -> Optional[str]:
    """Extract first balanced JSON object/array from text (after removing code fences)."""
    if not isinstance(text, str):
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    start_idx = None
    start_char = ""
    for idx, char in enumerate(cleaned):
        if char in "[{":
            start_idx = idx
            start_char = char
            break

    if start_idx is None:
        return None

    end_char = "}" if start_char == "{" else "]"
    depth = 0
    in_string = False
    escape = False

    for idx in range(start_idx, len(cleaned)):
        char = cleaned[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                return cleaned[start_idx:idx + 1]

    return None


def safe_json_loads(json_str: str, source_info: str = "Unknown source") -> Union[Dict[str, Any], List[Any]]:
    """Safely loads a JSON string with layered fallbacks for wrapped/malformed model output."""
    if not json_str or not json_str.strip():
        logger.warning(f"Received empty or whitespace-only JSON string from {source_info}. Cannot parse.")
        return {}

    cleaned_json_str = json_str.strip()
    # Remove markdown fences early
    if cleaned_json_str.startswith("```"):
        cleaned_json_str = re.sub(r"^```(?:json)?\s*", "", cleaned_json_str, flags=re.IGNORECASE)
        cleaned_json_str = re.sub(r"\s*```$", "", cleaned_json_str).strip()

    try:
        return json.loads(cleaned_json_str)
    except json.JSONDecodeError as e:
        extracted_json_str = extract_json_block(cleaned_json_str)
        if extracted_json_str:
            try:
                return json.loads(extracted_json_str)
            except json.JSONDecodeError:
                pass

        # object substring fallback
        obj_start, obj_end = cleaned_json_str.find("{"), cleaned_json_str.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            try:
                return json.loads(cleaned_json_str[obj_start:obj_end + 1])
            except json.JSONDecodeError:
                pass

        # array substring fallback
        arr_start, arr_end = cleaned_json_str.find("["), cleaned_json_str.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            try:
                return json.loads(cleaned_json_str[arr_start:arr_end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning(f"Standard JSON parsing failed for {source_info}: {e}. Attempting to repair with json_repair.")
        try:
            repair_target = extracted_json_str or cleaned_json_str
            repaired_json_str = repair_json(repair_target)
            if isinstance(repaired_json_str, str):
                parsed = json.loads(repaired_json_str)
                logger.info(f"JSON parsing for {source_info} succeeded after json_repair.")
                return parsed
            logger.error(f"JSON repair for {source_info} did not return a string. Type: {type(repaired_json_str)}")
            return {}
        except Exception as repair_e:
            logger.error(f"JSON repair also failed for {source_info}. Original decode error: {e}, Repair error: {repair_e}", exc_info=True)
            logger.debug(f"Original problematic JSON string from {source_info} (first 500 chars):\n{json_str.strip()[:500]}")
            return {}

# --- File Hashing ---
def get_file_hash(file_path: Path, algorithm: str = 'md5') -> Optional[str]:
    """Computes a hash for a given file."""
    if not isinstance(file_path, Path):
        try:
            file_path = Path(file_path)
        except TypeError:
            logger.error(f"Invalid file_path type for hashing: {file_path}. Must be Path or string.")
            return None

    if not file_path.is_file():
        logger.error(f"Cannot compute hash: File not found or is not a regular file at '{file_path}'.")
        return None

    try:
        hasher = hashlib.new(algorithm)
        with file_path.open('rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        hex_digest = hasher.hexdigest()
        return hex_digest
    except Exception as e:
        logger.error(f"Error computing {algorithm} hash for file '{file_path.name}': {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Setup basic logging for self-testing
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s (%(module)s.%(funcName)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()]
    )
    logger.info("--- Starting utils.py self-test ---")

    # Test text cleaning
    test_text_cyrillic = "Ово је тест са ћирилицом и Unicode replacement " + chr(0xFFFD) + " карактером."
    test_text_email_url = "Kontakt: test@example.com, vidi [www.example.com](https://www.example.com)."

    logger.info(f"Original Cyrillic: '{test_text_cyrillic}'")
    cleaned_cyrillic_final = clean_text(test_text_cyrillic, is_pre_splitting=False)
    logger.info(f"Cleaned Cyrillic (final): '{cleaned_cyrillic_final}'")
    
    logger.info(f"Original Email/URL: '{test_text_email_url}'")
    cleaned_email_url_final = clean_text(test_text_email_url, is_pre_splitting=False)
    logger.info(f"Cleaned Email/URL (final): '{cleaned_email_url_final}'")
    
    # Test generate_graph_edge_id
    logger.info("\n--- Testing generate_graph_edge_id ---")
    edge1_uuid = generate_graph_edge_id("Zakon A", "Zakon B", "CITES", "doc1")
    edge2_uuid = generate_graph_edge_id("Zakon B", "Zakon A", "CITES", "doc2") # Reversed, different doc
    edge3_uuid = generate_graph_edge_id("zakon a", "zakon b", "cites", "doc3") # Lowercase
    
    logger.info(f"Edge 1 ('Zakon A' -> 'Zakon B' in doc1): {edge1_uuid}")
    logger.info(f"Edge 2 ('Zakon B' -> 'Zakon A' in doc2): {edge2_uuid}")
    logger.info(f"Edge 3 ('zakon a' -> 'zakon b' in doc3): {edge3_uuid}")
    logger.info(f"Are all three UUIDs the same? {'YES' if edge1_uuid == edge2_uuid == edge3_uuid else 'NO'}")

    # Test safe_json_loads
    logger.info("\n--- Testing safe_json_loads ---")
    json_tests = [
        ('{"key": "value"}', "Valid JSON"),
        ('```json\n{\n  "name": "Test",\n  "type": "JSON"\n}\n```', "Fenced JSON"),
        ('```JSON\n{\n  "name": "Test",\n  "type": "JSON"\n}\n```', "Fenced JSON (uppercase)"),
        ('```\n{\n  "name": "Test",\n  "type": "JSON"\n}\n```', "Generic Fenced JSON"),
        ('{\n  "name": "Test", // with a comment\n  "type": "JSON"\n}', "JSON with comment (needs repair)"),
        ('{"name": "Test", "type": "JSON",}', "JSON with trailing comma (needs repair)"),
        ('Not JSON', "Invalid JSON string")
    ]
    for json_string, desc in json_tests:
        logger.info(f"Testing {desc}: '{json_string[:50]}...'")
        parsed = safe_json_loads(json_string, source_info=desc)
        if parsed:
            logger.info(f"Parsed successfully: {parsed}")
        else:
            logger.warning(f"Failed to parse: '{json_string[:50]}...'")


    logger.info("\n--- utils.py self-test finished ---")
