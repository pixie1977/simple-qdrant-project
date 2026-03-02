# index_faq.py
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

# Загружаем модель (рекомендуемая компактная версия)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Подключаемся к Qdrant
client = QdrantClient(
    "localhost",
    port=6333,
    prefer_grpc=True,
    timeout=60,
    check_version=False)

# Создаем коллекцию
collection_name = "faq_collection"

# Check if collection exists and delete if it does
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

# Create new collection
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Читаем faq.txt и разбиваем на абзацы
with open("./data/faq.txt", "r", encoding="utf-8") as f:
    paragraphs = [p.strip() for p in f.read().split('\n') if p.strip()]

points = []
for para in paragraphs:
    vector = model.encode(para).tolist()
    point_id = str(uuid.uuid4())
    points.append(
        PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "text": para}))

# Загружаем в Qdrant
client.upsert(collection_name=collection_name, points=points)

print(f"Проиндексировано {len(paragraphs)} абзацев.")
