from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Путь к файлу
FILE_PATH = "../data/faq.txt"

# Загружаем модель (для русского языка — rubert-tiny2)
model = SentenceTransformer('cointegrated/rubert-tiny2')

# Читаем вопросы из файла
def load_faq_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}. Убедитесь, что он существует.")

    with open(filepath, "r", encoding="utf-8") as f:
        # Читаем строки, убираем пробелы и пустые
        questions = [line.strip() for line in f.readlines() if line.strip()]
    return questions

# Загружаем базу вопросов
faq_questions = load_faq_data(FILE_PATH)
print(f"Загружено {len(faq_questions)} вопросов из {FILE_PATH}")

# Кодируем все вопросы в векторы
faq_embeddings = model.encode(faq_questions)

# Функция поиска
def search(query, top_k=3):
    query_embedding = model.encode([query])[0]
    similarities = np.dot(faq_embeddings, query_embedding) / (
        np.linalg.norm(faq_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": faq_questions[idx],
            "score": float(similarities[idx])
        })

    answer = {"text":"не знаю", "score":0}
    for result in results:
        if result["score"] > answer["score"] and \
            result["score"] > 0.45:
            answer=result
    return answer


def main():
   while True:
       q = input("\n/vector: ").strip()
       if q.lower() in ["exit", "quit", "выйти"]:
           break
       res = search(q)
       print(f"📌 Ответ: {res['text']}")
       print(f" схожесть: {res['score']:.3f})")


# Пример использования
if __name__ == "__main__":
    main()