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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing GemmaQueryGenerator on device: {self.device}")

        logger.info(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading model weights for {model_name}...")
        torch_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        quantization_config = None
        if settings.load_in_4bit and self.device == "cuda":
            logger.info("NF4 Quantization (4-bit) is enabled.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            token=hf_token,
            attn_implementation="sdpa" if self.device == "cuda" else None,
            quantization_config=quantization_config,
        )
        self.model.eval()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _build_prompt(self, review_text: str) -> str:
        """Construct the system instruction and chat template for a single review."""
        messages = [
            {
                "role": "user",
                "content": (
                    "Read the following Dota 2 Steam review and formulate "
                    "ONE short, realistic search query (2 to 5 words) IN RUSSIAN "
                    "that a user would type to find this specific review.\n"
                    "The query must capture the key issue of the review "
                    "(e.g., bug, toxicity, high ping, bad update, emotion).\n\n"
                    "CRITICAL REQUIREMENT: Output the final query wrapped inside "
                    "<query> and </query> tags. Example: <query>ваш запрос здесь</query>.\n"
                    "Do not write any introductory text, notes, or explanation outside the tags.\n\n"
                    f'Review: "{review_text}"'
                ),
            }
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _extract_query(self, raw_text: str) -> str | None:
        """Extract a query enclosed in XML-like tags using regular expressions."""
        match = re.search(r"<query>(.*?)</query>", raw_text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def generate_batch(self, reviews: list[Review]) -> list[str | None]:
        """Perform batched generation for a list of reviews in a single forward pass."""
        prompts = [self._build_prompt(doc["text"]) for doc in reviews]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(  # type: ignore
                **inputs,
                max_new_tokens=24,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        queries: list[str | None] = []
        for idx, _ in enumerate(reviews):
            prompt_len = inputs.input_ids[idx].shape[0]
            generated_tokens = outputs[idx][prompt_len:]
            raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            queries.append(self._extract_query(raw_text))

        return queries

    def close(self) -> None:
        """Explicitly release model from GPU VRAM and run garbage collection."""
        logger.info("Releasing GemmaQueryGenerator GPU resources...")
        del self.model
        del self.tokenizer
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.success("GPU memory cleared successfully!")


def load_reviews_sample(filepath: Path, sample_size: int) -> list[Review]:
    """Load filtered reviews from disk and return a deterministic random sample."""
    reviews = read_jsonl(filepath)
    if len(reviews) > sample_size:
        logger.info(
            f"Sampling {sample_size} random reviews out of {len(reviews)} total."
        )
        return random.sample(reviews, sample_size)
    return reviews


def load_existing_progress(
    filepath: Path,
) -> tuple[list[ValidationItem], set[str]]:
    """Load existing progress from checkpoint file if it exists."""
    if not filepath.exists():
        return [], set()

    try:
        data: list[ValidationItem] = read_json(filepath)
        processed_ids = {item["relevant_doc_id"] for item in data}
        logger.info(f"Found checkpoint. Already processed: {len(processed_ids)}")
        return data, processed_ids
    except Exception:
        logger.warning("Checkpoint file is corrupted. Overwriting.")
        return [], set()


def generate_validation_set() -> None:
    """Orchestrate the entire validation set generation pipeline."""
    sample_reviews = load_reviews_sample(
        settings.interim_file, settings.validation_size
    )

    results, processed_ids = load_existing_progress(settings.validation_set_file)
    reviews_to_process = [r for r in sample_reviews if r["id"] not in processed_ids]

    if not reviews_to_process:
        logger.success("All reviews have already been processed.")
        return

    batch_size = 8
    start_q_idx = len(results)

    with LLMQueryGenerator(settings.gemma_model, settings.hf_token) as generator:
        for i in tqdm(
            range(0, len(reviews_to_process), batch_size), desc="Generating QGen"
        ):
            batch_docs = reviews_to_process[i : i + batch_size]
            queries = generator.generate_batch(batch_docs)

            for idx, query in enumerate(queries):
                if query:
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

            # Save progress incrementally
            write_json(settings.validation_set_file, results)


if __name__ == "__main__":
    import random

    generate_validation_set()
