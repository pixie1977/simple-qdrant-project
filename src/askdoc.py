# askdoc.py
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient("localhost", port=6333, prefer_grpc=True, timeout=60)
collection_name = "faq_collection"


def askdoc(query: str, top_k: int = 1):
    # Генерируем эмбеддинг запроса
    query_vector = model.encode(query).tolist()

    # Поиск ближайших фрагментов
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        score_threshold=0.5  # Минимальная релевантность
    ).points

    if not results:
        print("Не знаю.")
        return

    print("Ответ:")
    for i, hit in enumerate(results, 1):
        print(f"{i}. {hit.payload['text']}")
        print(f"(релевантность: {hit.score:.3f})")



# Тест
if __name__ == "__main__":
    while True:
        q = input("\n/askdoc: ").strip()
        if q.lower() in ["exit", "quit", "выйти"]:
            break
        askdoc(q)
