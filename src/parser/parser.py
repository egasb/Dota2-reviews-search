import json
import re

# Регулярные выражения
CYRILLIC_PATTERN = re.compile(r'[а-яА-ЯёЁ]')
# Ищет 5 и более одинаковых символов подряд (например, "ннннн", "!!!!!", "ааааа")
SPAM_CHARS_PATTERN = re.compile(r'(.)\1{4,}')


def filter_steam_reviews(input_file, output_file):
    kept_count = 0
    dropped_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            if not line.strip():
                continue

            data = json.loads(line)
            text = data.get('text', '')

            # 1. Минимум 20 символов
            if len(text) < 20:
                dropped_count += 1
                continue

            # 2. Минимум 30% кириллицы
            cyrillic_chars = CYRILLIC_PATTERN.findall(text)
            if (len(cyrillic_chars) / len(text)) < 0.3:
                dropped_count += 1
                continue

            # 3. Фильтр спама (залипание клавиш)
            # Отсеет отзывы вроде "коч братаннннннннннннннн"
            if SPAM_CHARS_PATTERN.search(text):
                dropped_count += 1
                continue

            # 4. Фильтр по weighted_vote_score
            # У новых отзывов без оценок score = 0.5.
            # Если score < 0.45, значит отзыв активно минусовали другие пользователи.
            score = data.get('weighted_vote_score', 0.5)
            if score < 0.48:
                dropped_count += 1
                continue

            # Записываем строку, если она прошла все фильтры
            outfile.write(line)
            kept_count += 1

    print(f"Готово! Сохранено отзывов: {kept_count}. Отброшено: {dropped_count}.")


# --- Запуск скрипта ---
# Рекомендую в input подать ваш уже отфильтрованный файл (на 647976 строк),
# чтобы скрипт отработал за пару секунд.
input_filename = '../dataset/dota2_reviews.jsonl'
output_filename = '../dataset/filtered_reviews.jsonl'

filter_steam_reviews(input_filename, output_filename)