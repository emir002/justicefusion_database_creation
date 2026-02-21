#!/usr/bin/env python3
# llm_interaction.py

import time
import json
import logging  # Standard library import
import threading
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib  # For hashing API keys
import requests, textwrap  # NEW  (place after the existing imports)

# Initialize logger early
logger = logging.getLogger(__name__)

# Third-party libraries
# NOTE: In **local-only** mode we should be able to run without Google SDKs installed.
try:
    # ──► prebacujemo se na novi Google GenAI SDK (google-genai ≥ 1.0.0)
    from google import genai  # type: ignore
    from google.genai.errors import ClientError  # type: ignore

    # Attempt to import FinishReason and GenerateContentConfig directly from the new SDK's typical location
    try:
        from google.genai.types import FinishReason, GenerateContentConfig  # type: ignore
        logger.info("Successfully imported FinishReason and GenerateContentConfig from google.genai.types")
    except ImportError:
        FinishReason = None  # type: ignore
        GenerateContentConfig = None  # type: ignore
        logger.error(
            "Failed to import FinishReason or GenerateContentConfig from google.genai.types. "
            "Ensure the 'google-genai' SDK (version 1.0.0 or later) is correctly installed and up to date. "
            "The script will attempt to proceed but may have compatibility issues."
        )
        if GenerateContentConfig is None and hasattr(genai, 'types') and hasattr(genai.types, 'GenerateContentConfig'):
            GenerateContentConfig = genai.types.GenerateContentConfig  # type: ignore
            logger.info("Fallback: obtained GenerateContentConfig via genai.types.GenerateContentConfig")

    from google.api_core.exceptions import (  # type: ignore
        ResourceExhausted, ServiceUnavailable, GoogleAPIError,
        InternalServerError, DeadlineExceeded, Aborted
    )

except ImportError as e_google:
    # Google SDK is unavailable. This is OK in local mode.
    genai = None  # type: ignore
    ClientError = Exception  # type: ignore
    FinishReason = None  # type: ignore
    GenerateContentConfig = None  # type: ignore

    # Define placeholder exception types so retry config / isinstance checks still work.
    ResourceExhausted = ServiceUnavailable = GoogleAPIError = InternalServerError = DeadlineExceeded = Aborted = Exception  # type: ignore

    logger.warning(
        f"Google GenAI SDK not available ({type(e_google).__name__}: {e_google}). "
        "External Gemini calls will be disabled; local-only mode can still run."
    )
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log, retry_if_exception
)
from filelock import FileLock, Timeout  # For inter-process file locking

# Local imports
import config  # For API keys, model name, rate limits
import utils   # For safe_json_loads
import metrics # For LLM call metrics

# --- Constants ---
LLM_RETRYABLE_ERRORS = (
    ResourceExhausted,
    ServiceUnavailable,
    InternalServerError,
    DeadlineExceeded,
    Aborted,
    ClientError,
)

# Constants for enhanced FinishReason handling
SAFETY_RELATED_ENUM_MEMBER_NAMES = [
    "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "RECITATION", "SPII"
    # Add other known safety-related enum *names* if necessary
]
SAFETY_RELATED_STRINGS = {
    "SAFETY", "BLOCKED", "PROHIBITED_CONTENT", "BLOCKLIST", "IMAGE_SAFETY", "SPII_BLOCK", "RECITATION_ACTION"
}  # Checked case-insensitively
KNOWN_SAFETY_INTEGER_CODES = {
    3: "SAFETY",
    4: "RECITATION",
    # Add other known integer codes that signify a safety block if identified.
    # These are less likely with the new SDK if FinishReason enum is correctly used by the SDK.
}


# --- LLM Rate Limit State Management ---
class RateLimitState:
    """
    Manages the rate limit state for API keys, persisting to a file.
    Uses FileLock for process-safety and RLock for thread-safety of in-memory state.
    API keys are hashed for storage.
    """
    def __init__(self, state_file_path: str):
        self.state_file = Path(state_file_path)
        self.file_lock_path = self.state_file.with_suffix(self.state_file.suffix + '.lock')
        self.file_lock = FileLock(self.file_lock_path, timeout=10)
        self.memory_lock = threading.RLock()
        self.state: Dict[str, Dict[str, Any]] = self._load_state()
        logger.info(f"RateLimitState initialized. Loaded state for {len(self.state)} keys from '{self.state_file}'.")

    def _get_key_hash(self, api_key: str) -> str:
        return hashlib.sha256(f"{api_key}:{len(api_key)}".encode('utf-8')).hexdigest()

    def _load_state(self) -> Dict[str, Dict[str, Any]]:
        logger.debug(f"Attempting to load rate limit state from: {self.state_file}")
        try:
            with self.file_lock:
                if self.state_file.exists():
                    with self.state_file.open("r", encoding='utf-8') as f:
                        loaded_state = json.load(f)
                    logger.info(f"Successfully loaded rate limit state from {self.state_file} for {len(loaded_state)} keys.")
                    return loaded_state
                else:
                    logger.warning(f"Rate limit state file '{self.state_file}' not found. Starting fresh.")
                    return {}
        except Timeout:
            logger.error(f"Timeout acquiring file lock for loading state from {self.state_file}. Returning empty state.")
            return {}
        except Exception as e:
            logger.error(f"Error loading state from {self.state_file}: {e}. Starting fresh.", exc_info=True)
            return {}

    def _save_state(self):
        logger.debug(f"Attempting to save rate limit state to: {self.state_file} for {len(self.state)} keys.")
        try:
            with self.file_lock:
                with self.state_file.open("w", encoding='utf-8') as f:
                    json.dump(self.state, f, indent=4)
        except Timeout:
            logger.error(f"Timeout acquiring file lock for saving state to {self.state_file}.")
        except Exception as e:
            logger.error(f"Error saving state to {self.state_file}: {e}", exc_info=True)

    def _ensure_and_refresh_key_state(self, key_hash: str, now: float):
        if key_hash not in self.state:
            logger.info(f"Initializing new rate limit state for API key hash: {key_hash[:12]}...")
            self.state[key_hash] = {
                "minute_window_start": now,
                "requests_in_current_minute": 0,
                "token_minute_window_start": now,
                "tokens_in_current_minute": 0,
                "total_requests_today": 0,
                "last_day_reset": now,
                "blocked_until": 0.0
            }
        key_state_data = self.state[key_hash]
        key_state_data.setdefault("token_minute_window_start", now)
        key_state_data.setdefault("tokens_in_current_minute", 0)
        key_state_data.setdefault("minute_window_start", now)
        key_state_data.setdefault("requests_in_current_minute", 0)
        key_state_data.setdefault("total_requests_today", 0)
        key_state_data.setdefault("last_day_reset", now)
        if now - key_state_data.get("last_day_reset", 0.0) >= 24 * 3600:
            logger.info(f"Refreshing daily window for key hash {key_hash[:12]}.")
            key_state_data["total_requests_today"] = 0
            key_state_data["last_day_reset"] = now
            key_state_data["requests_in_current_minute"] = 0
            key_state_data["minute_window_start"] = now
            key_state_data["tokens_in_current_minute"] = 0
            key_state_data["token_minute_window_start"] = now
        elif now - key_state_data.get("minute_window_start", 0.0) >= 60:
            logger.debug(f"Refreshing minute window for key hash {key_hash[:12]}.")
            key_state_data["requests_in_current_minute"] = 0
            key_state_data["minute_window_start"] = now
        if now - key_state_data.get("token_minute_window_start", 0.0) >= 60:
            logger.debug(f"Refreshing token minute window for key hash {key_hash[:12]}.")
            key_state_data["tokens_in_current_minute"] = 0
            key_state_data["token_minute_window_start"] = now

    def get_key_state(self, api_key: str) -> Dict[str, Any]:
        key_hash = self._get_key_hash(api_key)
        with self.memory_lock:
            now = time.time()
            self._ensure_and_refresh_key_state(key_hash, now)
            return dict(self.state[key_hash])

    def update_key_state(self, api_key: str, new_values: Dict[str, Any]):
        key_hash = self._get_key_hash(api_key)
        with self.memory_lock:
            if key_hash not in self.state:
                now = time.time()
                self._ensure_and_refresh_key_state(key_hash, now)
            self.state[key_hash].update(new_values)
            self._save_state()

    def increment(self, api_key: str):
        self.record_usage(api_key, 0)

    def record_usage(self, api_key: str, tokens_used: int):
        with self.memory_lock:
            key_hash = self._get_key_hash(api_key)
            log_key_hash = key_hash[:12]
            now = time.time()
            self._ensure_and_refresh_key_state(key_hash, now)
            key_state = self.state[key_hash]
            key_state["requests_in_current_minute"] += 1
            key_state["total_requests_today"] += 1
            if tokens_used:
                key_state["tokens_in_current_minute"] = key_state.get("tokens_in_current_minute", 0) + max(0, int(tokens_used))
            self._save_state()
            logger.debug(
                f"Key hash {log_key_hash}: RPM {key_state['requests_in_current_minute']}/{config.LLM_RATE_LIMIT_PER_MINUTE}, "
                f"TPM {key_state.get('tokens_in_current_minute', 0)}/{getattr(config, 'LLM_TOKEN_LIMIT_PER_MINUTE', '∞')}, "
                f"RPD {key_state['total_requests_today']}/{config.LLM_DAILY_LIMIT_PER_KEY}"
            )

rate_limit_manager = RateLimitState(str(config.RATE_LIMIT_STATE_FILE))

class GlobalQuotaExceeded(Exception):
    """Raised when the project-wide daily quota for a model is exhausted."""


# --- Tenacity Retry Helper ---
def _is_project_quota(exc: Exception) -> bool:
    return isinstance(exc, GlobalQuotaExceeded)


class LLMManager:
    def __init__(self, api_keys: List[str], model_name: str):
        # NEW: Support local-only mode via config.LLM_MODE.
        # In local mode, we do NOT require API keys and we route all calls to Ollama.
        self.llm_mode = getattr(config, "LLM_MODE", "external").strip().lower()

        logger.info(
            f"Initializing LLMManager | mode={self.llm_mode} | external_model={model_name} | keys={len(api_keys)}"
        )

        self.api_keys = api_keys or []
        self.model_name = model_name

        if self.llm_mode != "local":
            if genai is None:
                logger.error(
                    "LLMManager is in external mode but Google GenAI SDK is not installed/available. "
                    "Install 'google-genai' (and dependencies) or switch config.LLM_MODE='local'."
                )
                raise ImportError("google-genai SDK is required for external LLM mode")
            if not self.api_keys:
                logger.error("No Gemini API keys provided (external mode). LLMManager cannot function.")
                raise ValueError("No Gemini API keys provided in configuration.")
        self.key_index = 0
        self.key_rotation_lock = threading.Lock()
        # ▶ Cache za Client instance (jedna po API key-u)
        self._clients: Dict[str, "genai.Client"] = {}
        # model-name  ->  unix epoch (float) until which *all* keys are blocked
        self._model_blocked_until: Dict[str, float] = {}
        # Circuit-breaker cool-down for Gemini
        self._primary_unavailable_until = 0.0
        self._last_ollama_failure_reason = "unknown"
        if self.llm_mode != "local" and GenerateContentConfig is None:  # Only relevant in external mode
            logger.critical("GenerateContentConfig from google.genai.types could not be imported. LLM calls may fail.")
        logger.info(f"LLMManager initialized for model '{self.model_name}'. Key rotation starts with index 0.")

        self._daily_limit = getattr(config, "LLM_DAILY_LIMIT_PER_KEY", None)
        self._token_limit = getattr(config, "LLM_TOKEN_LIMIT_PER_MINUTE", None)
        self._daily_window_seconds = 24 * 3600
        if self._daily_limit:
            logger.info(
                f"Per-key rolling 24h limit enabled: {self._daily_limit} calls per key."
            )
        if self._token_limit:
            logger.info(
                f"Per-key input token limit enabled: {self._token_limit} tokens/minute per key."
            )

    def _estimate_input_tokens(self, prompt: str) -> int:
        """Rough heuristic for counting prompt tokens for rate tracking."""
        if not prompt:
            return 0
        # Gemini tokens roughly align to ~4 characters per token for mixed text.
        estimated = max(1, int(len(prompt) / 4))
        # account for whitespace-heavy prompts by adding newline count bonus
        estimated += prompt.count("\n")
        return estimated

    def _get_key_hash_for_logging(self, api_key: str) -> str:
        return rate_limit_manager._get_key_hash(api_key)[:12]

    def _get_next_available_key(self, model: str) -> Optional[str]:
        # --- NEW global breaker -------------------------------
        block_until = self._model_blocked_until.get(model, 0)
        if time.time() < block_until:
            logger.debug(
                "Model %s globally blocked for %.0fs", model, block_until - time.time()
            )
            return None
        # ------------------------------------------------------
        with self.key_rotation_lock:
            num_keys = len(self.api_keys)
            if num_keys == 0:
                return None
            start_idx = self.key_index
            for i in range(num_keys):
                current_key_to_try_idx = (start_idx + i) % num_keys
                api_key = self.api_keys[current_key_to_try_idx]
                log_key_hash = self._get_key_hash_for_logging(api_key)

                key_state = rate_limit_manager.get_key_state(api_key)
                now = time.time()
                if key_state.get("blocked_until", 0.0) > now:
                    logger.debug(f"Key hash {log_key_hash} is blocked for another {key_state['blocked_until'] - now:.1f}s. Trying next.")
                    continue
                if key_state.get("requests_in_current_minute", 0) >= config.LLM_RATE_LIMIT_PER_MINUTE:
                    wait_time = 60.0 - (now - key_state.get("minute_window_start", now))
                    new_blocked_until = now + max(0, wait_time) + 1
                    logger.warning(f"Minute limit for key hash {log_key_hash}. Blocking for {new_blocked_until - now:.1f}s.")
                    rate_limit_manager.update_key_state(api_key, {"blocked_until": new_blocked_until})
                    continue
                if self._token_limit and key_state.get("tokens_in_current_minute", 0) >= self._token_limit:
                    wait_time = 60.0 - (now - key_state.get("token_minute_window_start", now))
                    new_blocked_until = now + max(0, wait_time) + 1
                    logger.warning(
                        f"Token limit for key hash {log_key_hash}. Blocking for {new_blocked_until - now:.1f}s."
                    )
                    rate_limit_manager.update_key_state(api_key, {"blocked_until": new_blocked_until})
                    continue
                if self._daily_limit and key_state.get("total_requests_today", 0) >= self._daily_limit:
                    reset_at = key_state.get("last_day_reset", now) + self._daily_window_seconds
                    wait_time = max(0.0, reset_at - now)
                    logger.warning(
                        f"Daily limit reached for key hash {log_key_hash}. Blocking for {wait_time:.0f}s until window resets."
                    )
                    rate_limit_manager.update_key_state(api_key, {"blocked_until": reset_at})
                    continue
                self.key_index = current_key_to_try_idx
                logger.debug(f"Selected API key index {self.key_index} (hash: {log_key_hash}) for LLM call.")
                return api_key
            logger.error("All API keys are currently rate-limited or blocked after checking all keys.")
            return None

    def _increment_counters(self, api_key: str, tokens_used: int):
        rate_limit_manager.record_usage(api_key, tokens_used)

    # ----------------------------------------------------------
    # centralised error parser that can trip per-key blocks
    # ----------------------------------------------------------
    def _handle_client_error(self, err: ClientError, model: str, api_key: str) -> bool:
        """
        Returns True if the error was a *project-level per-model daily quota* hit for this key.
        In that case, we block just this key for 24 hours and try another key.
        """
        if getattr(err, "status_code", None) != 429:
            return False

        # New SDK (>=1.0) – the JSON is in err.response
        payload = getattr(err, "response", None)
        if not payload:
            payload = err.args[1] if len(err.args) > 1 else None
        if not isinstance(payload, dict):
            return False  # can't parse → give up

        try:
            for d in payload.get("error", {}).get("details", []):
                if d.get("@type", "").endswith("QuotaFailure"):
                    for v in d.get("violations", []):
                        if v.get("quotaId", "").startswith(
                            "GenerateRequestsPerDayPerProjectPerModel"
                        ):
                            # This key's project is exhausted for this model for today.
                            # Block ONLY this key until the rolling 24h window expires and signal caller to try another key.
                            cooldown = time.time() + self._daily_window_seconds
                            rate_limit_manager.update_key_state(api_key, {"blocked_until": cooldown})

                            log_key_hash = self._get_key_hash_for_logging(api_key)
                            logger.warning(
                                "Per-project per-model daily quota exhausted for key hash %s; "
                                "blocking this key for 24h.", log_key_hash
                            )
                            return True
        except (IndexError, AttributeError, KeyError, TypeError):
            logger.debug("Could not parse project quota details from ClientError.", exc_info=True)

        return False


    @retry(
        stop=stop_after_attempt(config.LLM_RETRY_ATTEMPTS),
        wait=wait_exponential(
            multiplier=config.LLM_RETRY_WAIT_MULTIPLIER,
            min=config.LLM_RETRY_WAIT_INITIAL,
            max=config.LLM_RETRY_WAIT_MAX,
        ),
        retry=retry_if_exception(
            lambda e: not _is_project_quota(e)                # ① skip GlobalQuotaExceeded
            and isinstance(e, LLM_RETRYABLE_ERRORS)           # ② retry only for allowed errors
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _generate_content_with_retry(self, prompt: str,
                                     temperature: float = 0.5,
                                     max_output_tokens: Optional[int] = None,
                                     validator=None,
                                     json_mode: bool = False,
                                     system_prompt: Optional[str] = None) -> Optional[str]:

            # -----------------------------------------------------------------
            # LOCAL-ONLY MODE: always use Ollama
            # -----------------------------------------------------------------
            if self.llm_mode == "local":
                local_temp = temperature if temperature is not None else getattr(config, "LOCAL_LLM_TEMPERATURE", 0.5)
                max_retries = getattr(config, "LLM_MAX_RETRIES", getattr(config, "LLM_MAX_RETRIES_PER_ITEM", 3))
                backoff_base = getattr(config, "LLM_BACKOFF_BASE_SECONDS", 0.8)
                model_name = getattr(config, "LOCAL_LLM_MODEL_NAME", getattr(config, "LIGHTRAG_MODEL_NAME", "")).lower()
                reasoning_prefixes = [p.lower() for p in getattr(config, "REASONING_MODEL_PREFIXES", ["gpt-oss"]) if p]
                is_reasoning_model = any(model_name.startswith(prefix) for prefix in reasoning_prefixes)
                min_reasoning_tokens = int(getattr(config, "MIN_LOCAL_MAX_OUTPUT_TOKENS_FOR_REASONING", 256))

                requested_tokens = max_output_tokens or getattr(config, "LOCAL_LLM_MAX_OUTPUT_TOKENS", 512)
                if is_reasoning_model:
                    requested_tokens = max(requested_tokens, min_reasoning_tokens)

                last_reason = "unknown"
                for attempt in range(1, max_retries + 1):
                    attempt_tokens = requested_tokens + ((attempt - 1) * max(64, requested_tokens // 4))
                    result = self._call_ollama(
                        prompt,
                        local_temp,
                        attempt_tokens,
                        response_format="json" if json_mode else None,
                        system_prompt=system_prompt,
                    )
                    if result is not None:
                        stripped = self._strip_markdown_wrappers(result)
                        if not self._is_effectively_empty(stripped) and not stripped.startswith("Error:") and (validator is None or validator(stripped)):
                            return stripped
                        last_reason = "validator rejected response" if validator and not validator(stripped) else "empty response"
                    else:
                        last_reason = self._last_ollama_failure_reason

                    if attempt < max_retries:
                        sleep_s = min(backoff_base * (2 ** (attempt - 1)) + random.uniform(0.0, 0.2), 10.0)
                        time.sleep(sleep_s)

                return f"Error: Ollama request failed after {max_retries} attempts ({last_reason})."

            now = time.time()

            # Are we in the cool-down window?
            if now < self._primary_unavailable_until:
                # -> Directly use Ollama
                logger.info("Gemini is still cooling-down; using Ollama fallback.")
                return self._call_ollama(prompt, temperature, max_output_tokens)

            # Try Gemini keys; if none available, use Ollama
            tried_any_key = False
            while True:
                api_key = self._get_next_available_key(self.model_name)
                if not api_key:
                    logger.warning("No Gemini keys available – switching to Ollama.")
                    # Cool down briefly so minute windows can reset quickly
                    self._primary_unavailable_until = time.time() + 65.0
                    return self._call_ollama(prompt, temperature, max_output_tokens)

                tried_any_key = True
                log_key_hash = self._get_key_hash_for_logging(api_key)
                logger.info(f"Attempting LLM call with key hash {log_key_hash} for model '{self.model_name}'. Prompt (first 70 chars): '{prompt[:70].replace(chr(10), ' ')}...'")

                if GenerateContentConfig is None:
                    logger.error("GenerateContentConfig is not available (failed import). Cannot make LLM call.")
                    return "Error: System configuration issue (GenerateContentConfig missing)."

                try:
                    # ──► Dohvati ili kreiraj client za dani ključ
                    if api_key not in self._clients:
                        self._clients[api_key] = genai.Client(api_key=api_key)
                    client = self._clients[api_key]

                    cfg_kwargs = {"temperature": temperature}
                    if max_output_tokens is not None:
                        cfg_kwargs["max_output_tokens"] = max_output_tokens

                    safety_settings_list = [
                        genai.types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_MEDIUM_AND_ABOVE",
                        ),
                        genai.types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_MEDIUM_AND_ABOVE",
                        ),
                        genai.types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_MEDIUM_AND_ABOVE",
                        ),
                        genai.types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_MEDIUM_AND_ABOVE",
                        ),
                    ]

                    # ➊ GenerateContentConfig – jedini „dozvoljen” način slanja parametara ​i​ safety_settings-a
                    current_generation_config = (
                        GenerateContentConfig(**cfg_kwargs, safety_settings=safety_settings_list)
                        if GenerateContentConfig
                        else {**cfg_kwargs, "safety_settings": safety_settings_list}
                    )

                    # ---- bump counters *before* we hit the network so 429s are tracked ----
                    estimated_input_tokens = self._estimate_input_tokens(prompt)
                    if max_output_tokens:
                        estimated_input_tokens += max_output_tokens
                    self._increment_counters(api_key, estimated_input_tokens)
                    start_time = time.time()
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=[prompt],
                        config=current_generation_config,
                    )
                    latency = time.time() - start_time
                    metrics.LLM_LATENCY.labels(endpoint=self.model_name).observe(latency)
                    logger.info(f"LLM call with key hash {log_key_hash} completed in {latency:.2f}s.")

                    generated_text_content = None
                    if response.candidates:
                        for candidate_idx, candidate in enumerate(response.candidates):
                            finish = getattr(candidate, "finish_reason", None)
                            is_safety_block = False
                            finish_value_for_comparison = None
                            finish_name_for_comparison = None

                            if finish is not None:
                                if hasattr(finish, 'value') and isinstance(finish.value, int):
                                    finish_value_for_comparison = finish.value
                                if hasattr(finish, 'name') and isinstance(finish.name, str):
                                    finish_name_for_comparison = finish.name.upper()  # Compare uppercase

                                if FinishReason is None:  # Log raw details only if our main Enum import failed
                                    logger.info(
                                        f"FinishReason enum not imported. Candidate {candidate_idx} raw finish_reason: {finish} "
                                        f"(name: {finish_name_for_comparison}, value: {finish_value_for_comparison}, type: {type(finish)})"
                                    )

                                # Check for safety block
                                if FinishReason and isinstance(finish, FinishReason):  # Primary check with imported Enum
                                    if hasattr(finish, 'name') and finish.name in SAFETY_RELATED_ENUM_MEMBER_NAMES:
                                        is_safety_block = True
                                elif finish_name_for_comparison and finish_name_for_comparison in SAFETY_RELATED_STRINGS:  # Fallback to string name
                                    is_safety_block = True
                                elif finish_value_for_comparison is not None and finish_value_for_comparison in KNOWN_SAFETY_INTEGER_CODES:  # Fallback to int value
                                    is_safety_block = True
                                    logger.info(f"Candidate {candidate_idx}: Interpreted integer finish_reason value {finish_value_for_comparison} as safety-related: {KNOWN_SAFETY_INTEGER_CODES[finish_value_for_comparison]}")

                            if is_safety_block:
                                safety_ratings_str = str(getattr(candidate, 'safety_ratings', 'N/A'))
                                logger.warning(f"Candidate {candidate_idx} for key hash {log_key_hash} blocked due to safety. Finish Reason: {finish}. Ratings: {safety_ratings_str}")
                                continue

                            if candidate.content and candidate.content.parts:
                                try:
                                    part_text = candidate.content.parts[0].text
                                    if isinstance(part_text, str):
                                        generated_text_content = part_text
                                        logger.debug(f"Using text from candidate {candidate_idx} for key hash {log_key_hash}.")
                                        break
                                    else:
                                        logger.warning(f"Candidate {candidate_idx} part 0 text is not a string: {type(part_text)}")
                                except IndexError:
                                    logger.warning(f"Candidate {candidate_idx} content.parts was empty for key hash {log_key_hash}.")
                                except AttributeError:
                                    logger.warning(f"Candidate {candidate_idx} content.parts[0] did not have 'text' attribute for key hash {log_key_hash}.")
                            else:
                                log_name = 'N/A'
                                if finish is not None:
                                    if FinishReason and isinstance(finish, FinishReason) and hasattr(finish, 'name'):
                                        log_name = finish.name
                                    elif finish_name_for_comparison:  # From string name
                                        log_name = finish_name_for_comparison
                                    elif finish_value_for_comparison is not None:  # From int value
                                        log_name = f"VALUE_{finish_value_for_comparison}"
                                    else:  # Fallback to raw string of finish object
                                        log_name = str(finish)
                                logger.warning(f"Candidate {candidate_idx} for key hash {log_key_hash} has no content/parts. Finish Reason: {log_name}")

                    if generated_text_content is not None:
                        result_text = generated_text_content.strip()
                        metrics.LLM_CALLS.labels(endpoint=self.model_name, status='success').inc()
                        if validator and not validator(result_text):
                            return "Error: Validator rejected response."
                        return result_text
                    else:
                        block_reason_msg = "No usable text from any candidate"
                        if response.prompt_feedback and response.prompt_feedback.block_reason:
                            block_reason_val = response.prompt_feedback.block_reason
                            block_reason_msg = f"Prompt blocked due to: {block_reason_val.name if hasattr(block_reason_val, 'name') else block_reason_val}"
                            if response.prompt_feedback.block_reason_message:
                                block_reason_msg += f" - {response.prompt_feedback.block_reason_message}"
                        elif response.candidates and getattr(response.candidates[0], "finish_reason", None) is not None:
                            first_cand = response.candidates[0]
                            finish_val_cand = getattr(first_cand, "finish_reason", None)

                            log_name_cand = 'N/A'
                            if finish_val_cand is not None:
                                if FinishReason and isinstance(finish_val_cand, FinishReason) and hasattr(finish_val_cand, 'name'):
                                    log_name_cand = finish_val_cand.name
                                elif hasattr(finish_val_cand, 'name') and isinstance(finish_val_cand.name, str):  # Check if it has .name directly
                                    log_name_cand = finish_val_cand.name
                                elif hasattr(finish_val_cand, 'value') and isinstance(finish_val_cand.value, int):
                                    log_name_cand = f"VALUE_{finish_val_cand.value}"
                                else:
                                    log_name_cand = str(finish_val_cand)
                            block_reason_msg = f"First candidate finish_reason: {log_name_cand}"
                            safety_ratings_str = str(getattr(first_cand, 'safety_ratings', 'N/A'))
                            if getattr(first_cand, 'safety_ratings', None):
                                block_reason_msg += f", SafetyRatings: {safety_ratings_str}"

                        logger.warning(f"LLM response (key hash {log_key_hash}) provided no usable text. {block_reason_msg}. Prompt: '{prompt[:70]}...'")
                        metrics.LLM_CALLS.labels(endpoint=self.model_name, status='blocked_or_empty').inc()
                        return f"Error: Content generation failed. Reason: {block_reason_msg}"

                except ClientError as e_client:
                    # If this was a per-project per-model daily quota for THIS KEY, block it and try next key.
                    quota_exhausted_for_key = self._handle_client_error(e_client, self.model_name, api_key)

                    # Mark the specific key as rate-limited for a short duration for generic 429/RESOURCE_EXHAUSTED
                    if getattr(e_client, "status_code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e_client):
                        rate_limit_manager.update_key_state(
                            api_key, {"blocked_until": time.time() + 65.0}
                        )

                    if quota_exhausted_for_key:
                        logger.info("Key exhausted its project-level per-model daily quota; trying another key.")
                        # Try another key immediately (no long global cool-down)
                        continue

                    logger.warning(f"Gemini ClientError, falling back to Ollama. Error: {e_client}")
                    self._primary_unavailable_until = time.time() + config.PRIMARY_LLM_COOLDOWN
                    # Do not re-raise; fall through to Ollama
                    return self._call_ollama(prompt, temperature, max_output_tokens)

                except LLM_RETRYABLE_ERRORS as e_retry:
                    metrics.LLM_CALLS.labels(endpoint=self.model_name, status='retryable_error').inc()
                    logger.warning(f"LLM call (key hash {log_key_hash}) failed with retryable error: {type(e_retry).__name__} - {e_retry}. Falling back to Ollama.")
                    if isinstance(e_retry, ResourceExhausted):
                        rate_limit_manager.update_key_state(api_key, {"blocked_until": time.time() + 65.0})
                    self._primary_unavailable_until = time.time() + config.PRIMARY_LLM_COOLDOWN
                    # Do not re-raise; fall through to Ollama
                    return self._call_ollama(prompt, temperature, max_output_tokens)

                except GoogleAPIError as e_google:
                    metrics.LLM_CALLS.labels(endpoint=self.model_name, status='api_error').inc()
                    logger.error(f"LLM call (key hash {log_key_hash}) failed: Non-retryable GoogleAPIError: {type(e_google).__name__} - {e_google}. Falling back to Ollama.", exc_info=True)
                    self._primary_unavailable_until = time.time() + config.PRIMARY_LLM_COOLDOWN
                    # Do not return an error string; fall through to Ollama
                    return self._call_ollama(prompt, temperature, max_output_tokens)

                except Exception as e_unexpected:
                    metrics.LLM_CALLS.labels(endpoint=self.model_name, status='other_error').inc()
                    logger.error(f"LLM call (key hash {log_key_hash}) failed: Unexpected error: {type(e_unexpected).__name__} - {e_unexpected}", exc_info=True)
                    return f"Error: Unexpected error during LLM call: {type(e_unexpected).__name__}"

    # ─── LOCAL OLLAMA FALLBACK ────────────────────────────────────────────────────
    def _call_ollama(self, prompt: str,
                     temperature: float = config.LIGHTRAG_TEMPERATURE,
                     max_output_tokens: int | None = None,
                     response_format: Optional[str] = None,
                     system_prompt: Optional[str] = None,
                     use_chat: Optional[bool] = None) -> str | None:
        """
        Fire a single request to the local Ollama endpoint. Supports both /api/generate and /api/chat.
        """
        local_model = getattr(config, "LOCAL_LLM_MODEL_NAME", getattr(config, "LIGHTRAG_MODEL_NAME", ""))
        generate_url = getattr(config, "LIGHTRAG_OLLAMA_URL", None)
        if hasattr(config, "OLLAMA_BASE_URL") and hasattr(config, "OLLAMA_GENERATE_ENDPOINT"):
            generate_url = f"{config.OLLAMA_BASE_URL}{config.OLLAMA_GENERATE_ENDPOINT}"
        chat_url = getattr(config, "OLLAMA_CHAT_ENDPOINT", f"{getattr(config, 'OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat")
    
        options: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05,
        }
        if max_output_tokens:
            options["num_predict"] = int(max_output_tokens)
    
        auto_use_chat = bool(getattr(config, "OLLAMA_USE_CHAT", True))
        endpoint_type = "chat" if (auto_use_chat if use_chat is None else use_chat) else "generate"
        ollama_url = chat_url if endpoint_type == "chat" else generate_url
    
        if endpoint_type == "chat":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload: Dict[str, Any] = {
                "model": local_model,
                "messages": messages,
                "stream": False,
                "options": options,
            }
        else:
            final_prompt = prompt if not system_prompt else f"System:\n{system_prompt}\n\nUser:\n{prompt}"
            payload = {
                "model": local_model,
                "prompt": final_prompt,
                "stream": False,
                "options": options,
            }
    
        if response_format == "json":
            payload["format"] = "json"
    
        prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
        self._last_ollama_failure_reason = "unknown"
        try:
            r = requests.post(ollama_url, json=payload, timeout=getattr(config, "OLLAMA_TIMEOUT", 60))
            r.raise_for_status()
            data = r.json()
    
            if isinstance(data, dict) and data.get("error"):
                self._last_ollama_failure_reason = f"error field: {str(data.get('error'))[:120]}"
                metrics.LLM_CALLS.labels(endpoint="ollama", status="failure").inc()
                return None
    
            response_text = ""
            if isinstance(data, dict):
                response_text = str(data.get("response", "") or "")
                if self._is_effectively_empty(response_text):
                    msg = data.get("message")
                    if isinstance(msg, dict):
                        response_text = str(msg.get("content", "") or "")
                if self._is_effectively_empty(response_text):
                    choices = data.get("choices")
                    if isinstance(choices, list) and choices:
                        first = choices[0] if isinstance(choices[0], dict) else {}
                        message = first.get("message") if isinstance(first, dict) else None
                        if isinstance(message, dict):
                            response_text = str(message.get("content", "") or "")
    
            response_text = self._strip_markdown_wrappers(textwrap.dedent(response_text).strip())
            if self._is_effectively_empty(response_text):
                keys = list(data.keys()) if isinstance(data, dict) else []
                self._last_ollama_failure_reason = "empty response"
                logger.debug(
                    "Empty Ollama response model=%s endpoint_type=%s keys=%s prompt_len=%s prompt_sha1=%s",
                    local_model,
                    endpoint_type,
                    keys,
                    len(prompt),
                    prompt_hash,
                )
                metrics.LLM_CALLS.labels(endpoint="ollama", status="failure").inc()
                return None
    
            metrics.LLM_CALLS.labels(endpoint="ollama", status="success").inc()
            return response_text
        except Exception as e:
            self._last_ollama_failure_reason = f"{type(e).__name__}: {e}"
            logger.error(f"Ollama request failed: {type(e).__name__}: {e}")
            metrics.LLM_CALLS.labels(endpoint="ollama", status="failure").inc()
            return None
    
    @staticmethod
    def _is_effectively_empty(text: Optional[str]) -> bool:
        if text is None:
            return True
        return str(text).strip().lower() in {"", "null", "none", "n/a", "na"}

    @staticmethod
    def _strip_markdown_wrappers(text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip("` \n")

    @staticmethod
    def _normalize_simple_answer(text: str) -> str:
        if not text:
            return ""
        normalized_text = text.strip()
        if normalized_text.upper().startswith("ANSWER:"):
            normalized_text = normalized_text[len("ANSWER:"):].strip()
        return normalized_text.strip('"\':()').upper()

    def generate_summary(self, text: str, max_length: int = config.LLM_SUMMARY_MAX_INPUT_LENGTH) -> str:
        logger.info(f"Request to generate summary (input original length {len(text)}, processing max_length {max_length}).")
        truncated_text = text[:max_length]
        if len(text) > max_length:
            logger.warning(f"Input text for summary was truncated: {len(text)} -> {max_length} chars.")
        prompt = f"Summarize the following document in one concise paragraph:\n\nDocument Text:\n\"\"\"\n{truncated_text}\n\"\"\"\n\nSummary:"
        summary = self._generate_content_with_retry(prompt, temperature=0.4, max_output_tokens=512)
        if summary and not summary.startswith("Error:"):
            cleaned_summary = summary.strip('\"`\'')
            logger.info(f"Summary generated successfully. Length: {len(cleaned_summary)} chars.")
            return cleaned_summary
        else:
            logger.error(f"Failed to generate summary. LLM response: {summary}")
            return summary if summary else "Error: Failed to generate summary (empty response)."

    def extract_document_title(self, text: str, snippet_length: int = config.LLM_TITLE_SNIPPET_MAX_LEN, original_filename: Optional[str] = None) -> str:
        logger.info(f"Request to extract title (snippet {snippet_length}, original_text_len {len(text)}).")
        snippet = text[:snippet_length].strip()

        def _clean_title(raw_title: str) -> str:
            cleaned = self._strip_markdown_wrappers(raw_title)
            cleaned = re.sub(r"(\*\*|__|\*|_)", "", cleaned)
            cleaned = re.sub(r"^[\-*•\d\.)\s]+", "", cleaned)
            cleaned = cleaned.strip("\"'`“”‘’ ")
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:")
            if len(cleaned) > 140:
                cleaned = cleaned[:140].rsplit(" ", 1)[0].strip()
            return cleaned if len(cleaned) >= 5 else ""

        def _filename_fallback() -> str:
            stem = Path(original_filename or "Untitled Document").stem
            stem = re.sub(r"[_\-]+", " ", stem)
            stem = re.sub(r"\s+", " ", stem).strip()
            first_line = ""
            for line in text.splitlines():
                if line.strip():
                    first_line = line.strip()
                    break
            fallback = " - ".join([part for part in [stem, first_line] if part]).strip()
            return _clean_title(fallback) or _clean_title(stem) or "Untitled Document"

        if not snippet:
            logger.warning("Cannot extract title from empty text snippet. Using filename fallback.")
            return _filename_fallback()

        prompt = f"""Based on the following text snippet, provide a concise and descriptive document title.
If an official title is clearly present at the beginning (e.g., "ZAKON O...", "PRAVILNIK O...", "ODLUKA o..."), return that.
Otherwise, create a suitable title.

Snippet:
---
{snippet}
---

Title:"""
        title = self._generate_content_with_retry(prompt, temperature=0.2, max_output_tokens=128)
        cleaned_title = _clean_title(title or "") if title and not title.startswith("Error:") else ""

        if not cleaned_title:
            retry_prompt = f"Return ONLY a concise title (no quotes, no markdown):\n{snippet}"
            retry_title = self._generate_content_with_retry(retry_prompt, temperature=0.1, max_output_tokens=128)
            cleaned_title = _clean_title(retry_title or "") if retry_title and not retry_title.startswith("Error:") else ""

        if cleaned_title:
            logger.info(f"Document title extracted/generated: '{cleaned_title}'")
            return cleaned_title

        fallback = _filename_fallback()
        logger.warning(f"Failed to extract/generate title from LLM; fallback='{fallback}'")
        return fallback

    def normalize_graph_payload(self, payload: Dict[str, Any], source_id: str, source_filename: str = "") -> Dict[str, List[Dict[str, str]]]:
        """Normalize heterogeneous LLM graph output to canonical node/edge schemas."""
        if not isinstance(payload, dict):
            return {"nodes": [], "edges": []}

        raw_nodes = payload.get("nodes")
        raw_edges = payload.get("edges")
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            return {"nodes": [], "edges": []}

        normalized_nodes: List[Dict[str, str]] = []
        for idx, node in enumerate(raw_nodes):
            if not isinstance(node, dict):
                logger.warning(f"Skipping non-dict node at index {idx} for source_id={source_id}.")
                continue
            node_id = str(node.get("node_id") or node.get("id") or node.get("nodeId") or "").strip()
            node_name = str(node.get("node_name") or node.get("entity_name") or node.get("name") or node.get("label") or "").strip()
            node_type = str(node.get("node_type") or node.get("entity_type") or node.get("type") or node.get("category") or "").strip()
            description = str(
                node.get("description")
                or node.get("node_description")
                or node.get("desc")
                or node.get("summary")
                or node.get("definition")
                or node.get("details")
                or ""
            ).strip()
            if not node_id:
                node_id = f"n{idx + 1}"
            if not description:
                if node_name:
                    description = f"{node_name} ({node_type})" if node_type else node_name
                else:
                    description = "Entity"
            normalized_nodes.append({
                "node_id": node_id,
                "node_name": node_name,
                "node_type": node_type,
                "description": description,
                "source_id": source_id,
                "source_filename": source_filename,
            })

        normalized_edges: List[Dict[str, str]] = []
        for idx, edge in enumerate(raw_edges):
            if not isinstance(edge, dict):
                logger.warning(f"Skipping non-dict edge at index {idx} for source_id={source_id}.")
                continue
            source_entity = str(edge.get("source_entity") or edge.get("source") or edge.get("from") or "").strip()
            target_entity = str(edge.get("target_entity") or edge.get("target") or edge.get("to") or "").strip()
            relationship_type = str(edge.get("relationship_type") or edge.get("relation") or edge.get("predicate") or "").strip()
            edge_source_id = str(edge.get("source_id") or edge.get("sourceId") or edge.get("sourceID") or source_id).strip() or source_id
            if not source_entity or not target_entity:
                logger.warning(f"Skipping edge missing source/target at index {idx} for source_id={source_id}.")
                continue
            normalized_edges.append({
                "source_entity": source_entity,
                "target_entity": target_entity,
                "relationship_type": relationship_type,
                "source_id": edge_source_id,
                "source_filename": source_filename,
            })

        return {"nodes": normalized_nodes, "edges": normalized_edges}

    def extract_entities_and_relationships(self, text_content: str, source_id: str, source_filename: str = "", max_text_length: int = config.LLM_ENTITY_MAX_INPUT_LENGTH) -> Dict[str, List[Dict]]:
        logger.info(f"Request to extract entities/relationships for source_id: '{source_id}' (original len {len(text_content)}, processing max {max_text_length}).")
        truncated_text = text_content[:max_text_length]
        if len(text_content) > max_text_length:
            logger.warning(f"Input for entity extraction for '{source_id}' truncated: {len(text_content)} -> {max_text_length} chars.")
        if not truncated_text.strip():
            logger.warning(f"Cannot extract entities from empty text (source: '{source_id}').")
            return {"nodes": [], "edges": []}
        prompt = f"""Task: Analyze the following legal text snippet. Extract key legal entities (nodes) and their relationships (edges).

Source ID (for all extracted items): "{source_id}"

Text Snippet:
---
{truncated_text}
---

Return ONLY valid JSON with keys "nodes" and "edges".

Each node object MUST contain: "node_id", "node_name", "node_type", "description".
"description" must be a short legal meaning/context sentence or short phrase.

Expected JSON shape:
{{
  "nodes": [
    {{
      "node_id": "n1",
      "node_name": "...",
      "node_type": "...",
      "description": "..."
    }}
  ],
  "edges": [
    {{
      "source_entity": "...",
      "target_entity": "...",
      "relationship_type": "..."
    }}
  ]
}}"""

        json_mode = self.llm_mode == "local" and getattr(config, "OLLAMA_JSON_MODE_FOR_ENTITY_EXTRACTION", True)
        response_text = self._generate_content_with_retry(prompt, temperature=0.3, max_output_tokens=2048, json_mode=json_mode)
        if self._is_effectively_empty(response_text) or (response_text and response_text.startswith("Error:")):
            response_text = self._generate_content_with_retry(
                prompt + "\nReturn only JSON.",
                temperature=0.2,
                max_output_tokens=3072,
                json_mode=True,
            )

        def _parse_candidate(raw_text: str) -> Optional[Dict[str, List[Dict]]]:
            candidate_text = self._strip_markdown_wrappers(raw_text)
            candidate_text = utils.extract_json_block(candidate_text) or candidate_text
            parsed = utils.safe_json_loads(candidate_text, source_info=f"entity extraction for '{source_id}'")
            if isinstance(parsed, dict) and isinstance(parsed.get("nodes"), list) and isinstance(parsed.get("edges"), list):
                return parsed
            return None

        parsed_json = _parse_candidate(response_text) if response_text and not response_text.startswith("Error:") else None
        if not parsed_json:
            strict_prompt = prompt + "\nRespond with JSON only. No prose."
            retry_text = self._generate_content_with_retry(strict_prompt, temperature=0.2, max_output_tokens=2048, json_mode=json_mode)
            parsed_json = _parse_candidate(retry_text) if retry_text and not retry_text.startswith("Error:") else None

        if not parsed_json:
            logger.warning(f"Invalid entity JSON for '{source_id}'. Returning empty payload.")
            return {"nodes": [], "edges": []}

        normalized = self.normalize_graph_payload(parsed_json, source_id=source_id, source_filename=source_filename)
        logger.info(f"Entity extraction success for '{source_id}'. Found {len(normalized['nodes'])} nodes, {len(normalized['edges'])} edges.")
        return normalized

    def classify_document(self, text_snippet: str) -> str:
        if not text_snippet.strip():
            logger.warning("Empty snippet for classification – defaulting to OTHER.")
            return "OTHER"

        def _extract_label(value: str) -> str:
            if not value:
                return ""
            upper = self._normalize_simple_answer(value)
            match = re.search(r"\b(LAW|MIXED|OTHER)\b", upper)
            return match.group(1) if match else ""

        prompt = f"""Analyze the following text snippet and classify it into one of three categories: LAW, MIXED, or OTHER.

- **LAW**: The text is purely legislative or regulatory content (e.g., a law, regulation, statute). It consists of articles, sections, and formal legal language.
- **MIXED**: The text contains both legislative/regulatory content AND additional commentary, analysis, news, or explanations. For example, a news article that quotes several articles of a law.
- **OTHER**: The text is not a legal document. Examples include news reports, court summaries (not the full text of the decision), academic papers, or general web content.

**TEXT SNIPPET:**
---
{text_snippet}
---

**OUTPUT:**
Return only a single word: LAW, MIXED, or OTHER."""
        reply = self._generate_content_with_retry(prompt, temperature=0.0, max_output_tokens=64, validator=lambda x: bool(_extract_label(x)))
        label = _extract_label(reply or "") if reply and not reply.startswith("Error:") else ""

        if not label:
            hardened_prompt = (
                "Respond with ONLY one token: LAW or MIXED or OTHER. No punctuation. No explanation.\n\n"
                f"TEXT:\n{text_snippet}"
            )
            reply = self._generate_content_with_retry(hardened_prompt, temperature=0.0, max_output_tokens=64, validator=lambda x: bool(_extract_label(x)))
            label = _extract_label(reply or "") if reply and not reply.startswith("Error:") else ""

        if label:
            logger.info(f"Document snippet classified as '{label}'.")
            return label

        logger.error(f"Classifier LLM failed after retry: {reply}. Defaulting to OTHER.")
        return "OTHER"

    def is_law_document(self, text_snippet: str) -> bool:
        """
        Convenience helper – returns **True** when the snippet is classified
        as purely legislative/regulatory text (LAW), otherwise **False**.
        """
        return self.classify_document(text_snippet) == "LAW"
