import json
import re
from src.core.config import settings

CYRILLIC_PATTERN = re.compile(r"[а-яА-ЯёЁ]")
SPAM_CHARS_PATTERN = re.compile(r"(.)\1{4,}")


def filter_steam_reviews(input_file, output_file):
    kept_count = 0
    dropped_count = 0

    with (
        open(input_file, "r", encoding="utf-8") as in_f,
        open(output_file, "w", encoding="utf-8") as out_f,
    ):
        for line in in_f:
            if not line.strip():
                continue

            data = json.loads(line)
            text = data.get("text", "")

            # Используем конфиг вместо захардкоженных чисел
            if len(text) < settings.min_text_len:
                dropped_count += 1
                continue

            cyrillic_chars = CYRILLIC_PATTERN.findall(text)
            if (
                len(cyrillic_chars) / max(len(text), 1)
            ) < settings.cyrillic_ratio_threshold:
                dropped_count += 1
                continue

            if SPAM_CHARS_PATTERN.search(text):
                dropped_count += 1
                continue

            score = data.get("weighted_vote_score", 0.5)
            if score < settings.min_vote_score:
                dropped_count += 1
                continue

            out_f.write(line)
            kept_count += 1

    print(f"Готово! Сохранено: {kept_count}. Отброшено: {dropped_count}.")


if __name__ == "__main__":
    filter_steam_reviews(settings.raw_file, settings.interim_file)
