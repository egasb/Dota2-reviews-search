import os

# 1. Принудительно отключаем сеть и настраиваем параллелизм ДО импорта библиотек
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gc
import re
from pathlib import Path
from typing import Any, Self

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.core.config import settings
from src.utils.io import read_json, read_jsonl, write_json

type Review = dict[str, Any]
type ValidationItem = dict[str, str]


class LLMQueryGenerator:
    """Manages LLM loading, batch prompt formatting, and GPU inference."""

    def __init__(self, model_name: str, hf_token: str | None = None) -> None:
        logger.info("Initializing LLMQueryGenerator class...")

        # Если токен не передан, принудительно ставим False во избежание автопоиска в env
        actual_token = False if hf_token is None else hf_token

        logger.debug("Checking CUDA availability via torch...")
        cuda_available = torch.cuda.is_available()
        logger.debug(f"CUDA availability result: {cuda_available}")

        self.device = "cuda" if cuda_available else "cpu"
        logger.info(f"Initializing GemmaQueryGenerator on device: {self.device}")

        logger.info(f"Loading tokenizer for {model_name}...")
        logger.debug(
            f"Calling AutoTokenizer.from_pretrained with model_name={model_name}, token={actual_token}..."
        )

        try:
            # Загружаем токенизатор строго из локального кэша
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, token=actual_token, local_files_only=True
            )
            logger.success("Tokenizer successfully loaded!")
        except Exception as e:
            logger.error(f"Failed to load tokenizer from local cache: {e}")
            raise e

        logger.debug("Configuring tokenizer properties...")
        self.tokenizer.padding_side = "left"
        logger.debug(f"Padding side set to: {self.tokenizer.padding_side}")

        if self.tokenizer.pad_token is None:
            logger.debug("pad_token is None, setting to eos_token...")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.debug(f"Tokenizer pad_token current state: {self.tokenizer.pad_token}")

        logger.info(f"Loading model weights for {model_name}...")

        logger.debug("Checking if GPU supports bfloat16...")
        bf16_supported = (
            torch.cuda.is_bf16_supported() if self.device == "cuda" else False
        )
        logger.debug(f"bfloat16 support result: {bf16_supported}")

        dtype = torch.bfloat16 if bf16_supported else torch.float16
        logger.debug(f"Calculated torch calculation dtype: {dtype}")

        quantization_config = None
        if settings.load_in_4bit and self.device == "cuda":
            logger.info("NF4 Quantization (4-bit) is enabled.")
            logger.debug("Initializing BitsAndBytesConfig object...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.debug("BitsAndBytesConfig successfully created.")

        logger.info(
            f"Initiating AutoModelForCausalLM.from_pretrained for {model_name}..."
        )

        # Принудительно загружаем всё на нулевую GPU без разделения слоёв
        device_map = {"": 0} if self.device == "cuda" else None

        # Используем стабильный eager вместо зависающего sdpa
        attn_implementation = "eager" if self.device == "cuda" else None

        logger.debug(
            f"Passing parameters: device_map={device_map}, "
            f"attn_implementation={attn_implementation}, "
            f"quantization_config_present={quantization_config is not None}"
        )

        try:
            # Загружаем модель строго из локального кэша
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device_map,
                token=actual_token,
                attn_implementation=attn_implementation,
                quantization_config=quantization_config,
                local_files_only=True,
            )
            logger.success("Model weights successfully loaded from local cache!")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise e

        logger.debug("Setting model to evaluation mode (model.eval())...")
        self.model.eval()
        logger.success("Model evaluation state set up successfully.")

    def __enter__(self) -> Self:
        logger.debug("Entering LLMQueryGenerator context manager block.")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        logger.debug("Exiting LLMQueryGenerator context manager block.")
        if exc_type is not None:
            logger.error(f"An exception occurred inside the context block: {exc_val}")
        self.close()

    def _build_prompt(self, review_text: str) -> str:
        """Construct the system instruction asking for exactly 3 distinct search queries."""
        logger.debug(
            f"Constructing system prompt for review of length: {len(review_text)} chars..."
        )

        # Промпт на английском (Gamer Persona) для лучшего рассуждения Gemma 4 E4B
        messages = [
            {
                "role": "user",
                "content": (
                    "You are a cynical, seasoned Dota 2 gamer who knows all the memes, slang, and exactly how players search for reviews on Steam.\n"
                    "Your task is to write exactly THREE distinct, realistic search queries IN RUSSIAN that a real gamer would type to find the provided review.\n\n"
                    "CRITICAL RULES (NO BULLSHIT):\n"
                    "1. NO CORPORATE/ACADEMIC SPEAK: Never use dry, clinical, or artificial terms (e.g., do not write 'временный контент', 'продолжительность игры', 'джокерская'). Real players search using raw, simple, emotional, or slang terms (e.g., 'скачал удалил кал', 'игра на один раз', 'дота на пару вечеров').\n"
                    "2. NO HEART SYMBOLS: Completely ignore Steam censors (♥♥♥) and never put them in queries.\n"
                    "3. LOWERCASE & NO PUNCTUATION: Use strictly lowercase and no punctuation (no periods, commas, or question marks).\n"
                    "4. GUARANTEE EXACTLY THREE QUERIES: You must always output exactly 3 queries. If the review is short, do not freeze. Just think of 3 different natural ways a gamer would search for this vibe (e.g., one short keyword query, one slang query, one reaction/meme query).\n\n"
                    "EXAMPLE ANALYSIS:\n"
                    "Review: 'скачаеш потом ♥♥♥ удалиш...'\n"
                    "Good queries: 'скачал потом удалил дота 2', 'дота игра на один раз', 'скачал удалил кал'\n"
                    "Bad queries: 'временный контент дота 2' (Too clinical/artificial!), '♥♥f удалиш' (Contains symbols!)\n\n"
                    "Output format:\n"
                    "<thinking>\n"
                    "[Analyze the review's gamer vibe and plan 3 simple queries]\n"
                    "</thinking>\n"
                    "<query>query 1</query>\n"
                    "<query>query 2</query>\n"
                    "<query>query 3</query>\n\n"
                    f'Review: "{review_text}"'
                ),
            }
        ]
        logger.debug("Applying tokenizer chat template to messages...")
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def _extract_queries(self, raw_text: str) -> list[str]:
        """Extract a query enclosed in XML-like tags using regular expressions."""
        logger.debug(
            f"Extracting queries from model response (response length: {len(raw_text)} chars)..."
        )
        matches = re.findall(
            r"<query>(.*?)</query>", raw_text, re.DOTALL | re.IGNORECASE
        )
        queries = [q.strip() for q in matches if q.strip()]
        logger.debug(f"Extracted {len(queries)} matching queries from tags.")
        return queries

    def generate_batch(self, reviews: list[Review]) -> list[list[str]]:
        """Perform batched generation returning a list of list of queries for each review."""
        logger.info(f"Initiating batch generation for {len(reviews)} reviews...")

        logger.debug("Building system prompts for the batch...")
        prompts = [self._build_prompt(doc["text"]) for doc in reviews]

        logger.debug("Tokenizing prompts batch...")
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        logger.debug(
            f"Tokenization complete. Inputs tensor shape: {inputs.input_ids.shape}"
        )

        logger.debug("Executing model.generate inside torch.no_grad() block...")
        with torch.no_grad():
            outputs = self.model.generate(  # type: ignore
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        logger.debug(
            f"Model generation finished. Outputs tensor shape: {outputs.shape}"
        )

        queries_batch: list[list[str]] = []
        logger.debug("Iterating over generated tokens to decode sequences...")
        for idx, _ in enumerate(reviews):
            prompt_len = inputs.input_ids[idx].shape[0]
            generated_tokens = outputs[idx][prompt_len:]

            logger.debug(
                f"Decoding sequence for item {idx} (tokens length: {len(generated_tokens)})..."
            )
            raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            extracted = self._extract_queries(raw_text)
            queries_batch.append(extracted)

        logger.info(f"Batch generation pipeline finished for {len(reviews)} items.")
        return queries_batch

    def close(self) -> None:
        """Explicitly release model from GPU VRAM and run garbage collection."""
        logger.info("Releasing GemmaQueryGenerator GPU resources...")

        if hasattr(self, "model"):
            logger.debug("Deleting self.model reference...")
            del self.model
        else:
            logger.debug("self.model reference was not initialized or already deleted.")

        if hasattr(self, "tokenizer"):
            logger.debug("Deleting self.tokenizer reference...")
            del self.tokenizer
        else:
            logger.debug(
                "self.tokenizer reference was not initialized or already deleted."
            )

        logger.debug("Calling garbage collector...")
        gc.collect()
        logger.debug("Garbage collection finished.")

        if self.device == "cuda":
            logger.debug("Emptying PyTorch CUDA memory cache...")
            torch.cuda.empty_cache()
            logger.debug("CUDA memory cache cleared.")
        logger.success("GPU memory cleared successfully!")


def load_reviews_sample(filepath: Path, sample_size: int) -> list[Review]:
    """Load filtered reviews from disk and return a deterministic random sample."""
    logger.info(f"Requesting review samples from path: {filepath}")

    logger.debug(f"Calling read_jsonl on {filepath}...")
    all_reviews = read_jsonl(filepath)

    # ФИЛЬТРАЦИЯ: Отрезаем мусорные отзывы короче 70 символов
    reviews = [r for r in all_reviews if len(r.get("text", "")) >= 70]
    logger.debug(
        f"Loaded {len(all_reviews)} total, kept {len(reviews)} reviews after length filter (>= 70 chars)."
    )

    if len(reviews) > sample_size:
        logger.info(
            f"Sampling {sample_size} random reviews out of {len(reviews)} total."
        )
        sampled = random.sample(reviews, sample_size)
        logger.debug("Sampling completed.")
        return sampled

    logger.debug(
        "Number of reviews is less than sample size, returning all reviews without sampling."
    )
    return reviews


def load_existing_progress(
    filepath: Path,
) -> tuple[list[ValidationItem], set[str]]:
    """Load existing progress from checkpoint file if it exists."""
    logger.info(f"Checking for existing checkpoint progress at: {filepath}")

    if not filepath.exists():
        logger.info(
            "No checkpoint file found at destination. Progress starts from scratch."
        )
        return [], set()

    logger.debug("Checkpoint file detected. Attempting to parse...")
    try:
        data: list[ValidationItem] = read_json(filepath)
        processed_ids = {item["relevant_doc_id"] for item in data}
        logger.info(
            f"Found existing progress. Already processed: {len(processed_ids)} items."
        )
        return data, processed_ids
    except Exception as e:
        logger.warning(
            f"Checkpoint file read failed or file is corrupted: {e}. Overwriting."
        )
        return [], set()


def generate_validation_set() -> None:
    """Orchestrate the entire validation set generation pipeline."""
    logger.info("Starting validation set generation process pipeline...")

    logger.debug(f"Interim reviews file: {settings.interim_file}")
    logger.debug(f"Validation destination file: {settings.validation_set_file}")
    logger.debug(f"Target validation set size: {settings.validation_size}")
    logger.debug(f"Configured model identifier: {settings.gemma_model}")

    sample_reviews = load_reviews_sample(
        settings.interim_file, settings.validation_size
    )

    results, processed_ids = load_existing_progress(settings.validation_set_file)

    logger.debug("Filtering reviews list against processed checkpoints...")
    reviews_to_process = [r for r in sample_reviews if r["id"] not in processed_ids]
    logger.info(
        f"Filtered. Total reviews scheduled to process: {len(reviews_to_process)}"
    )

    if not reviews_to_process:
        logger.success("All reviews have already been processed.")
        return

    batch_size = 8
    start_q_idx = len(results)
    logger.debug(
        f"Batch size set to: {batch_size}. Incremental starting index: {start_q_idx}"
    )

    logger.info("Starting LLMQueryGenerator context block...")
    with LLMQueryGenerator(settings.gemma_model, None) as generator:
        logger.debug("Context successfully entered. Starting iteration over batches...")
        for i in tqdm(
            range(0, len(reviews_to_process), batch_size), desc="Generating QGen"
        ):
            batch_docs = reviews_to_process[i : i + batch_size]
            logger.info(f"Processing batch block [{i} : {i + len(batch_docs)}]")

            logger.debug("Calling batch generation method...")
            queries_list = generator.generate_batch(batch_docs)
            logger.debug("Batch queries generated successfully.")

            logger.debug(
                "Post-processing and appending generated queries to local storage..."
            )
            for idx, queries in enumerate(queries_list):
                for query in queries:
                    results.append(
                        {
                            "query_id": f"q_{start_q_idx:04d}",
                            "query": query,
                            "relevant_doc_id": batch_docs[idx]["id"],
                            "source_review_snippet": batch_docs[idx]["text"][:120]
                            + "...",
                        }
                    )
                    start_q_idx += 1

            logger.info(
                f"Saving progress incrementally to: {settings.validation_set_file}"
            )
            write_json(settings.validation_set_file, results)
            logger.debug("Progress write-out complete.")

    logger.success("Validation set generation pipeline executed successfully!")


if __name__ == "__main__":
    import random

    generate_validation_set()
