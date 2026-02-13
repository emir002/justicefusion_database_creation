#!/usr/bin/env python3
# text_processing.py

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time

# Third-party libraries
from pdfminer.high_level import extract_pages # Changed from extract_text for more granular control
from pdfminer.layout import LAParams, LTTextContainer # LTTextContainer for iterating text elements
from docx import Document as DocxDocument
import stanza
import re
import nltk

try:
    from langdetect import detect, LangDetectException, DetectorFactory
except ImportError:
    logging.getLogger(__name__).warning(
        "langdetect library not found. Please install it (`pip install langdetect`). "
        "Language detection for lemmatization will rely on heuristics."
    )
    detect = None
    LangDetectException = Exception # Define for try-except block to function
    DetectorFactory = None

# Local imports
import config
import utils # Uses the modified utils.py
import metrics

logger = logging.getLogger(__name__)

if DetectorFactory is not None:
    DetectorFactory.seed = 0  # deterministic langdetect behavior

# This dictionary will hold initialized Stanza pipelines {lang_code: pipeline_object}
stanza_pipelines: Dict[str, stanza.Pipeline] = {}

def initialize_stanza():
    """
    Downloads Stanza models specified in config.STANZA_LANGUAGES if not already present
    and initializes the Stanza pipelines for 'tokenize,pos,lemma'.
    Also ensures NLTK 'punkt' tokenizer is available.
    """
    global stanza_pipelines
    if stanza_pipelines: # Avoid re-initialization
        logger.debug("Stanza pipelines already initialized. Skipping.")
        return

    logger.info("Initializing Stanza NLP pipelines...")
    start_time_total = time.time()

    # Configure Stanza's own logger to be less verbose during download/load
    stanza_internal_logger = logging.getLogger('stanza')
    original_stanza_level = stanza_internal_logger.level
    stanza_internal_logger.setLevel(logging.WARNING) # Reduce verbosity for setup

    for lang_code in config.STANZA_LANGUAGES:
        if lang_code not in stanza_pipelines:
            logger.info(f"Setting up Stanza pipeline for language: '{lang_code}'...")
            start_time_lang = time.time()
            try:
                logger.info(f"Checking/downloading Stanza model for '{lang_code}' (processors: tokenize,pos,lemma)...")
                stanza.download(
                    lang=lang_code,
                    processors='tokenize,pos,lemma',
                    verbose=False, # Stanza's internal verbosity
                    logging_level='WARN' # Stanza's internal logging level for download
                )
                logger.info(f"Initializing Stanza pipeline for '{lang_code}' on device: {'GPU' if config.USE_GPU else 'CPU'}...")
                stanza_pipelines[lang_code] = stanza.Pipeline(
                    lang=lang_code,
                    processors='tokenize,pos,lemma',
                    use_gpu=config.USE_GPU,
                    logging_level='WARN'
                )
                duration_lang = time.time() - start_time_lang
                logger.info(f"Stanza pipeline for '{lang_code}' initialized successfully in {duration_lang:.2f}s.")
            except Exception as e:
                logger.error(f"Failed to initialize Stanza pipeline for '{lang_code}': {e}", exc_info=True)
    
    stanza_internal_logger.setLevel(original_stanza_level) # Reset Stanza logger

    # Download NLTK sentence tokenizer model ('punkt') if not present
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK 'punkt' tokenizer already available.")
    except LookupError:
        logger.info("NLTK 'punkt' tokenizer not found. Attempting to download...")
        try:
            nltk.download('punkt', quiet=True)
            logger.info("NLTK 'punkt' tokenizer downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download NLTK 'punkt' tokenizer: {e}", exc_info=True)

    duration_total = time.time() - start_time_total
    if not config.STANZA_LANGUAGES:
        logger.warning("No Stanza languages specified in config. STANZA_LANGUAGES is empty. Lemmatization will be skipped.")
    elif not stanza_pipelines:
        logger.warning("Stanza pipelines dictionary is empty after initialization attempts. Lemmatization may not work.")
    else:
        logger.info(f"Stanza NLP & NLTK setup completed. Loaded Stanza for: {list(stanza_pipelines.keys())}. Total time: {duration_total:.2f}s.")

initialize_stanza() # Call initialization when module is loaded


class FileLoader:
    """Handles loading text content from various file formats."""

    @staticmethod
    def load_text(file_path: Path) -> Optional[Tuple[str, str]]:
        if not file_path.is_file():
            logger.error(f"File not found or is not a file: {file_path}")
            return None

        suffix = file_path.suffix.lower()
        logger.info(f"Attempting to load text from file: '{file_path.name}' (detected type: '{suffix}')")
        content: Optional[str] = None
        
        try:
            start_time = time.time()
            if suffix == '.txt':
                content = file_path.read_text(encoding='utf-8', errors='ignore')
            elif suffix == '.pdf':
                # Using more granular PDF extraction similar to database3.py
                text_content_parts = []
                # laparams can be tuned if PDF extraction quality is an issue.
                laparams = LAParams(line_margin=0.4, boxes_flow=0.5, char_margin=2.0, detect_vertical=False)
                for page_layout in extract_pages(str(file_path), laparams=laparams):
                    for element in page_layout:
                        if isinstance(element, LTTextContainer):
                            text_content_parts.append(element.get_text())
                content = "".join(text_content_parts) if text_content_parts else ""
            elif suffix == '.docx':
                doc = DocxDocument(str(file_path))
                content = "\n".join([para.text for para in doc.paragraphs if para.text and para.text.strip()])
            else:
                logger.warning(f"Unsupported file type '{suffix}' for file: {file_path.name}. Cannot load.")
                return None

            duration = time.time() - start_time
            if content is not None:
                logger.info(f"Successfully loaded {len(content)} characters from '{file_path.name}' (type: {suffix}) in {duration:.2f}s.")
                return content, suffix
            else: # Should only happen if read operation itself returned None (e.g. empty PDF after parsing)
                logger.warning(f"Loading {suffix} file '{file_path.name}' resulted in no text content.")
                return "", suffix # Return empty string and type if content is None but type was supported

        except Exception as e:
            logger.error(f"Error loading file '{file_path.name}' (type: {suffix}): {e}", exc_info=True)
            return None


class TextProcessor:
    """Handles cleaning, refining, splitting, and structuring text content."""

    @staticmethod
    def refine_and_lemmatize(text: str, source_doc_type_for_logging: str = "unknown") -> str:
        """
        Applies final cleaning (using utils.clean_text with is_pre_splitting=False)
        and then lemmatization using Stanza, with language detection.
        """
        # logger.debug(f"Refining and lemmatizing text (len: {len(text)}) from '{source_doc_type_for_logging}'.")
        
        original_len_for_refine = len(text)
        # Call utils.clean_text with is_pre_splitting=False for full cleaning rules on the chunk/article.
        cleaned_text = utils.clean_text(text, is_pre_splitting=False) 
        
        if not cleaned_text.strip():
            logger.warning(f"Text for '{source_doc_type_for_logging}' became empty after final cleaning in refine_and_lemmatize. Returning empty.")
            return ""
        # This log is helpful to see if the final clean changes length significantly
        # logger.info(f"Final cleaning in refine_and_lemmatize for '{source_doc_type_for_logging}'. Length: {original_len_for_refine} -> {len(cleaned_text)}.")

        detected_lang_code = None
        if detect: # Check if langdetect was imported successfully
            try:
                # Use a substantial snippet for detection, but not necessarily the whole chunk if it's huge
                snippet_for_lang_detect = cleaned_text[:min(2000, len(cleaned_text))] 
                if snippet_for_lang_detect.strip(): # Ensure snippet is not just whitespace
                    detected_lang_code = detect(snippet_for_lang_detect)
                    logger.info(f"Langdetect identified language: '{detected_lang_code}' for '{source_doc_type_for_logging}'.")
                else:
                    logger.warning(f"Snippet for langdetect is empty for '{source_doc_type_for_logging}'. Cannot detect language.")
            except LangDetectException as lde: # Specific exception for langdetect
                logger.warning(f"Language detection by langdetect failed for '{source_doc_type_for_logging}' (text might be too short/ambiguous): {lde}.")
            except Exception as e_detect: # Other unexpected errors from detect()
                logger.error(f"Unexpected error during langdetect for '{source_doc_type_for_logging}': {e_detect}", exc_info=True)
        else:
            logger.warning(f"langdetect not available for '{source_doc_type_for_logging}'. Using Stanza configured languages / heuristics.")

        stanza_lang_to_use = None
        # Prioritize langdetect if successful and its result is a configured Stanza language
        if detected_lang_code and detected_lang_code in config.STANZA_LANGUAGES and detected_lang_code in stanza_pipelines:
            stanza_lang_to_use = detected_lang_code
        else: 
            if detected_lang_code: # Log if detected lang wasn't usable
                 logger.info(
                     f"Langdetect code '{detected_lang_code}' for '{source_doc_type_for_logging}' not in configured/available Stanza "
                     f"pipelines ({list(stanza_pipelines.keys())}). Applying fallback heuristics."
                )
            # Fallback heuristic (similar to your original logic)
            cyrillic_present_in_snippet = utils.contains_cyrillic(cleaned_text[:500])
            if 'sr' in config.STANZA_LANGUAGES and 'sr' in stanza_pipelines and cyrillic_present_in_snippet:
                stanza_lang_to_use = 'sr'
            elif 'hr' in config.STANZA_LANGUAGES and 'hr' in stanza_pipelines: # Prefer 'hr' if available and not clearly Serbian Cyrillic
                stanza_lang_to_use = 'hr'
            elif 'sr' in config.STANZA_LANGUAGES and 'sr' in stanza_pipelines: # Fallback to 'sr' (e.g., for Latin Serbian)
                stanza_lang_to_use = 'sr'
        
        if stanza_lang_to_use and stanza_lang_to_use in stanza_pipelines:
            logger.info(f"Attempting Stanza lemmatization for '{source_doc_type_for_logging}' using pipeline: '{stanza_lang_to_use}'.")
            start_time_lemma = time.time()
            try:
                pipeline = stanza_pipelines[stanza_lang_to_use]
                doc = pipeline(cleaned_text) 
                lemmatized_words = [
                    word.lemma for sentence_obj in doc.sentences 
                    for word in sentence_obj.words 
                    if word.lemma and word.lemma.strip() # Ensure lemma is not None or empty/whitespace
                ]
                final_text = ' '.join(lemmatized_words)
                
                duration_lemma = time.time() - start_time_lemma
                metrics.TEXT_PROCESSING_STAGES_LATENCY.labels(stage_name='lemmatization').observe(duration_lemma)
                logger.info(
                    f"Stanza lemmatization for '{source_doc_type_for_logging}' (lang '{stanza_lang_to_use}') "
                    f"done in {duration_lemma:.2f}s. Length: {len(cleaned_text)} -> {len(final_text)}."
                )
                return final_text.strip() # Ensure final result is stripped
            except Exception as e_lemma:
                logger.error(
                    f"Stanza lemmatization failed for '{source_doc_type_for_logging}' (lang '{stanza_lang_to_use}') on text "
                    f"(first 100 chars): '{cleaned_text[:100]}...': {e_lemma}", exc_info=True
                )
                return cleaned_text # Fallback to cleaned (non-lemmatized) text
        else:
            logger.warning(
                f"No suitable Stanza pipeline identified for lemmatization for '{source_doc_type_for_logging}' "
                f"(detected lang: {detected_lang_code}, configured Stanza langs: {config.STANZA_LANGUAGES}). "
                f"Returning cleaned (non-lemmatized) text."
            )
            return cleaned_text

    @staticmethod
    def split_into_chunks(text: str, max_chars: int = config.CHUNK_MAX_CHARS, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
        """
        Splits large text into smaller, overlapping chunks based on sentences.
        Logic adapted from database3.py's TextProcessor.split_large_text for robustness.
        Assumes 'text' has been pre-cleaned with utils.clean_text(text, is_pre_splitting=True).
        """
        # logger.info(f"Splitting text (length {len(text)}) into chunks. Max_chars={max_chars}, Target_char_overlap={overlap}")
        start_time_chunk = time.time()

        if not text or not text.strip():
            logger.warning("Input text for chunking is empty. Returning empty list.")
            return []

        try:
            sentences = nltk.sent_tokenize(text)
            # logger.info(f"NLTK tokenized into {len(sentences)} sentences for chunking.")
        except Exception as e_nltk:
            logger.error(f"NLTK sentence tokenization failed: {e_nltk}. Using simple newline split as fallback.", exc_info=True)
            sentences = [s.strip() for s in text.splitlines() if s.strip()]
            if not sentences:
                logger.error("Fallback newline split also yielded no sentences. Cannot chunk text.")
                return [text] if text.strip() else [] # Return original text if it has content

        if not sentences: 
            logger.warning("No sentences found after tokenization attempts. Returning original text as one chunk if not empty.")
            return [text] if text.strip() else []

        chunks: List[str] = []
        current_chunk_sentences: List[str] = []
        current_length_chars = 0

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)
            # Length if this sentence is added (plus 1 for space if not the first sentence in current_chunk_sentences)
            len_if_added = current_length_chars + sentence_len + (1 if current_chunk_sentences else 0)

            if len_if_added <= max_chars or not current_chunk_sentences:
                # Add sentence if it fits, OR if it's the first sentence for this potential chunk
                current_chunk_sentences.append(sentence)
                current_length_chars = len_if_added
                if not current_chunk_sentences and len_if_added > max_chars : # This condition actually means current_chunk_sentences *was* empty
                     logger.warning(
                         f"Single sentence (len {sentence_len}) starting a chunk exceeds max_chars {max_chars}. "
                         f"Sentence fragment: '{sentence[:70]}...'"
                    )
            else:
                # Current sentence does not fit (and current_chunk_sentences is not empty).
                # Finalize the current_chunk_sentences.
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunks.append(chunk_text.strip())
                    # logger.debug(f"Formed chunk {len(chunks)}: len={len(chunk_text)}, sentences={len(current_chunk_sentences)}")

                    # Prepare for the next chunk with overlap (logic from database3.py)
                    temp_overlap_sentences: List[str] = []
                    temp_overlap_len_chars = 0
                    # Iterate backward over sentences of the chunk just made to build overlap
                    for s_rev in reversed(current_chunk_sentences):
                        s_rev_len_with_space_for_overlap = len(s_rev) + (1 if temp_overlap_sentences else 0)
                        
                        # Ensure at least one sentence for overlap if overlap > 0 and temp_overlap is currently empty
                        if not temp_overlap_sentences and overlap > 0 :
                            temp_overlap_sentences.insert(0, s_rev) # Add at the beginning of the list
                            temp_overlap_len_chars += s_rev_len_with_space_for_overlap
                        # If already have some overlap, add more if it fits within target 'overlap'
                        elif temp_overlap_len_chars + s_rev_len_with_space_for_overlap <= overlap:
                            temp_overlap_sentences.insert(0, s_rev)
                            temp_overlap_len_chars += s_rev_len_with_space_for_overlap
                        else:
                            # Adding this sentence would make overlap too big, or no overlap desired and already have one.
                            break 
                    
                    current_chunk_sentences = temp_overlap_sentences
                    # Add the current sentence (that caused the overflow) to start the new chunk (after overlap)
                    current_chunk_sentences.append(sentence)
                    # Recalculate current_length_chars for the new current_chunk_sentences
                    current_length_chars = sum(len(s) for s in current_chunk_sentences) + max(0, len(current_chunk_sentences) - 1)
                    
                    # logger.debug(f"Prepared next chunk starting with overlap. Has {len(current_chunk_sentences)} sentence(s), new char_len: {current_length_chars}.")
                else:
                    # This case (current_chunk_sentences is empty here) means the current 'sentence'
                    # itself was too long from the start of processing (should be rare if max_chars is reasonable).
                    chunks.append(sentence.strip()) # Add the long sentence as its own chunk
                    logger.warning(f"Formed chunk {len(chunks)} (single oversized sentence directly): len={sentence_len}")
                    current_chunk_sentences = [] # Reset for next potential sentences
                    current_length_chars = 0
        
        # Add any remaining sentences in current_chunk_sentences as the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(chunk_text.strip())
            # logger.debug(f"Formed final chunk {len(chunks)}: len={len(chunk_text)}, sentences={len(current_chunk_sentences)}")

        duration_chunk = time.time() - start_time_chunk
        metrics.TEXT_PROCESSING_STAGES_LATENCY.labels(stage_name='chunking').observe(duration_chunk)
        logger.info(f"Text (length {len(text)}) split into {len(chunks)} chunks in {duration_chunk:.2f}s.")
        return chunks

    @staticmethod
    def split_law_articles(text: str) -> List[Tuple[str, str]]:
        """
        Splits legal texts into articles based on common law article markers (e.g., "Član 1.", "Article 5.").
        Uses the finditer approach.
        Ensure text is pre-cleaned with utils.clean_text(text, is_pre_splitting=True).
        """
        # logger.info(f"Splitting law text (length {len(text)}) into articles using finditer.")
        start_time_split = time.time()
        
        # Using the robust pattern from your main_indexer.py
        pattern = re.compile(
            r'^\s*(?P<marker_word>Član(?:ak)?|Article)\s+(?P<number>\d+[a-zA-Z]*)[\s\.]*(?P<title_on_marker_line>.*?)\s*$',
            re.IGNORECASE | re.MULTILINE
        )
        
        matches = list(pattern.finditer(text))
        articles: List[Tuple[str, str]] = []

        if not matches:
            logger.warning("No law article markers found using the finditer regex pattern for article splitting.")
            return [] # Return empty list if no articles found

        # logger.debug(f"Found {len(matches)} potential article markers (using finditer).")
        for i, current_match in enumerate(matches):
            match_dict = current_match.groupdict()
            # Construct the marker identifier, e.g., "Član 1", "Article 5a"
            article_marker_identifier = f"{match_dict['marker_word'].strip().capitalize()} {match_dict['number'].strip()}"
            title_from_marker_line = match_dict['title_on_marker_line'].strip()
            
            article_body_start_index = current_match.end() # Text of the article starts after the marker line
            article_body_end_index = matches[i+1].start() if (i + 1) < len(matches) else len(text)
            
            article_body_text = text[article_body_start_index:article_body_end_index].strip()

            # Prepend title from marker line to the body text if it exists and is not already part of body
            # This logic helps capture titles that are on the same line as "Član X."
            full_article_content = article_body_text
            if title_from_marker_line:
                # A simple check; might need refinement if titles can be subtly different in casing or spacing
                if not article_body_text.lower().strip().startswith(title_from_marker_line.lower().strip()):
                    full_article_content = f"{title_from_marker_line}\n{article_body_text}".strip()
                # else:
                #     logger.debug(f"Article '{article_marker_identifier}' body appears to already include title part '{title_from_marker_line}'. Not prepending.")

            if not full_article_content.strip(): # If, after all, the content is empty
                logger.warning(f"Extracted empty content for article marker '{article_marker_identifier}'. Title on line was: '{title_from_marker_line}'. Skipping this article.")
                continue

            articles.append((article_marker_identifier, full_article_content))
            # logger.debug(f"Extracted article: Marker='{article_marker_identifier}', Body Length={len(full_article_content)}")

        duration_split = time.time() - start_time_split
        metrics.TEXT_PROCESSING_STAGES_LATENCY.labels(stage_name='article_splitting').observe(duration_split)
        logger.info(f"Successfully split text (length {len(text)}) into {len(articles)} law articles (using finditer) in {duration_split:.2f}s.")
        return articles

if __name__ == "__main__":
    # Setup basic logging JUST for this self-test
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s (%(module)s.%(funcName)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()] # Output to console for test
    )
    logger.info("--- Starting text_processing.py self-test ---")

    # Sample text for chunking
    sample_chunk_text = (
        "Ovo je prva rečenica. Ovo je druga, nešto duža rečenica koja se nastavlja. "
        "Treća rečenica je kratka. Četvrta rečenica također postoji ovdje. Peta rečenica može biti malo duža. "
        "Šesta rečenica služi za popunjavanje. Sedma rečenica. Osma. Deveta je tu. I deseta također."
    )
    logger.info(f"\nOriginal text for chunking (len {len(sample_chunk_text)}): '{sample_chunk_text}'")
    # Pre-clean for splitting
    pre_cleaned_for_chunking = utils.clean_text(sample_chunk_text, is_pre_splitting=True)
    logger.info(f"Pre-cleaned text for chunking (len {len(pre_cleaned_for_chunking)}): '{pre_cleaned_for_chunking}'")
    
    chunks_test = TextProcessor.split_into_chunks(pre_cleaned_for_chunking, max_chars=70, overlap=20)
    logger.info(f"Generated {len(chunks_test)} chunks:")
    for i, chunk in enumerate(chunks_test):
        logger.info(f"  Chunk {i+1} (len {len(chunk)}): '{chunk}'")

    # Sample text for law article splitting
    sample_law_text = """
    ZAKON O RADU

    Član 1. Osnovne odredbe
    Ovim zakonom uređuju se prava i obaveze.
    Članak 2. Definicije
    Poslodavac je pravno ili fizičko lice.
    Radnik je fizičko lice.
    Article 3 Ugovor o radu
    Ugovor se zaključuje u pisanom obliku.
    Član 4. Prestanak
    Završava se otkazom.
    """
    logger.info(f"\nOriginal text for law splitting (len {len(sample_law_text)}): '{sample_law_text}'")
    pre_cleaned_for_law_splitting = utils.clean_text(sample_law_text, is_pre_splitting=True)
    logger.info(f"Pre-cleaned text for law splitting (len {len(pre_cleaned_for_law_splitting)}): '{pre_cleaned_for_law_splitting}'")

    articles_test = TextProcessor.split_law_articles(pre_cleaned_for_law_splitting)
    logger.info(f"Generated {len(articles_test)} articles:")
    for i, (marker, body) in enumerate(articles_test):
        logger.info(f"  Article {i+1}: Marker='{marker}', Body (first 50 chars)='{body[:50].replace(chr(10), ' ')}...'")

    # Sample text for lemmatization
    sample_lemma_text_hr = "Dječaci su trčali brzo prema kućama svojih prijatelja."
    logger.info(f"\nOriginal text for HR lemmatization: '{sample_lemma_text_hr}'")
    lemmatized_hr = TextProcessor.refine_and_lemmatize(sample_lemma_text_hr, "HR_Test")
    logger.info(f"Lemmatized HR: '{lemmatized_hr}'")
    
    sample_lemma_text_sr_cyr = "Дечаци су трчали брзо према кућама својих пријатеља."
    logger.info(f"\nOriginal text for SR CYR lemmatization: '{sample_lemma_text_sr_cyr}'")
    lemmatized_sr_cyr = TextProcessor.refine_and_lemmatize(sample_lemma_text_sr_cyr, "SR_CYR_Test")
    logger.info(f"Lemmatized SR CYR: '{lemmatized_sr_cyr}'")


    logger.info("--- text_processing.py self-test finished ---")