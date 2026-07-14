import json
import numpy as np
from sentence_transformers import SentenceTransformer
from src.core.config import settings

print("Загрузка модели на GPU...")
model = SentenceTransformer(settings.model_name)

texts, payloads = [], []

print("Чтение данных...")
with open(settings.interim_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        texts.append(f"passage: {data['text']}")
        payloads.append(
            {
                "id": data["id"],
                "text": data["text"],
                "score": data.get("weighted_vote_score", 0),
            }
        )

print("Генерация векторов на GPU...")
embeddings = model.encode(texts, batch_size=settings.batch_size, show_progress_bar=True)

print("Сохранение...")
np.save(settings.vectors_file, embeddings)
with open(settings.payload_file, "w", encoding="utf-8") as f:
    json.dump(payloads, f, ensure_ascii=False)

print(f"Готово! Файлы в {settings.vectors_file.parent}")
