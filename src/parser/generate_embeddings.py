import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# В Colab предварительно установите: !pip install sentence-transformers

INPUT_DATASET = '/content/drive/MyDrive/DLS Datasets/filtered_reviews.jsonl'
MODEL_NAME = 'intfloat/multilingual-e5-small'
OUTPUT_VECTORS = 'dota2_vectors.npy'
OUTPUT_PAYLOAD = 'dota2_payload.json'

print("Загрузка модели на GPU...")
# Модель автоматически загрузится на видеокарту, если доступна
model = SentenceTransformer(MODEL_NAME)

texts = []
payloads = []

print("Чтение данных...")
with open(INPUT_DATASET, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        texts.append(f"passage: {data['text']}")
        payloads.append({
            "id": data["id"],
            "text": data["text"],
            "score": data.get("weighted_vote_score", 0)
        })

print("Генерация векторов на GPU...")
# На GPU можно смело ставить батч 1024 или 2048
embeddings = model.encode(texts, batch_size=1024, show_progress_bar=True)

print("Сохранение...")
np.save(OUTPUT_VECTORS, embeddings)
with open(OUTPUT_PAYLOAD, 'w', encoding='utf-8') as f:
    json.dump(payloads, f, ensure_ascii=False)

print("Готово! Теперь скачайте файлы dota2_vectors.npy и dota2_payload.json")