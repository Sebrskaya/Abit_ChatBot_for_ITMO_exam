# Create_vector_db.py
import os
import json
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# import hashlib # hashlib не используется в текущем коде, можно удалить

# Пути к папкам
PROCESSED_DIR = "processed_data"
VECTOR_DB_DIR = "vector_db"

# Создаем папку для векторной базы данных
os.makedirs(VECTOR_DB_DIR, exist_ok=True)



def load_processed_documents() -> List[Dict[str, Any]]:
    """
    Загружает обработанные документы из JSON файла.
    """
    input_file = os.path.join(PROCESSED_DIR, "processed_documents.json")
    print(f"[DEBUG] Попытка загрузить документы из: {input_file}")
    try:
        if not os.path.exists(input_file):
            print(f"[ERROR] Файл {input_file} не существует.")
            return []

        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = f.read()
            print(f"[DEBUG] Содержимое файла (первые 200 символов): {raw_data[:200]}...")

            # Проверка на пустоту или почти пустоту
            if not raw_data.strip() or raw_data.strip() == "[]":
                print(f"[WARNING] Файл {input_file} пуст или содержит пустой массив.")
                return []

            documents = json.loads(raw_data)

        print(f"[DEBUG] Успешно загружено. Тип данных: {type(documents)}")
        if isinstance(documents, list):
            print(f"[DEBUG] Количество документов в списке: {len(documents)}")
            if documents:
                # Проверим структуру первого документа
                first_doc = documents[0]
                print(f"[DEBUG] Структура первого документа: {list(first_doc.keys())}")
                print(
                    f"[DEBUG] Первый документ (первые 200 символов текста): {first_doc.get('text', 'NO TEXT KEY')[:200] if isinstance(first_doc.get('text'), str) else 'TEXT KEY IS NOT A STRING'}...")
        else:
            print(f"[ERROR] Ожидался список, но загружены данные типа: {type(documents)}")
            return []

        print(f"Загружено {len(documents)} документов из {input_file}")
        return documents
    except FileNotFoundError:
        print(f"Файл {input_file} не найден. Убедитесь, что предыдущий шаг выполнен.")
        return []
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON в файле {input_file}: {e}")
        return []
    except Exception as e:
        print(f"Ошибка при загрузке документов: {e}")
        import traceback
        traceback.print_exc()
        return []


def initialize_embedding_model() -> SentenceTransformer:
    """
    Инициализирует модель для создания эмбеддингов.
    """
    try:
        model_name = 'distiluse-base-multilingual-cased-v1'
        print(f"Загрузка модели эмбеддингов: {model_name}")
        model = SentenceTransformer(model_name)
        print("Модель эмбеддингов загружена успешно")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели эмбеддингов: {e}")
        raise e


def initialize_vector_db() -> chromadb.Collection:
    """
    Инициализирует векторную базу данных ChromaDB.
    """
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR, settings=Settings(anonymized_telemetry=False))
        collection_name = "itmo_master_programs"
        collection = client.get_or_create_collection(name=collection_name)
        print(f"Векторная база данных инициализирована. Коллекция: {collection_name}")
        return collection
    except Exception as e:
        print(f"Ошибка при инициализации векторной базы данных: {e}")
        raise e


def add_documents_to_vector_db(
        collection: chromadb.Collection,
        documents: List[Dict[str, Any]],
        model: SentenceTransformer
):
    """
    Добавляет документы в векторную базу данных.
    """
    print("Начало добавления документов в векторную базу данных...")
    print(f"[DEBUG] Получено {len(documents)} документов для обработки.")

    ids = []
    embeddings = []
    metadatas = []
    texts = []

    added_count = 0
    for i, doc in enumerate(documents):
        # Проверяем наличие необходимых ключей
        if 'id' not in doc:
            print(f"[WARNING] Документ индекс {i} не содержит ключ 'id'. Пропущен. Документ: {str(doc)[:100]}...")
            continue
        if 'text' not in doc:
            print(f"[WARNING] Документ индекс {i} не содержит ключ 'text'. Пропущен. Документ: {str(doc)[:100]}...")
            continue
        if 'metadata' not in doc:
            print(f"[WARNING] Документ индекс {i} не содержит ключ 'metadata'. Пропущен. Документ: {str(doc)[:100]}...")
            continue

        # Проверим, что text - это строка
        if not isinstance(doc['text'], str):
            print(
                f"[WARNING] В документе индекс {i} поле 'text' не является строкой. Тип: {type(doc['text'])}. Пропущен.")
            continue
        # Проверим, что metadata - это словарь
        if not isinstance(doc['metadata'], dict):
            print(
                f"[WARNING] В документе индекс {i} поле 'metadata' не является словарем. Тип: {type(doc['metadata'])}. Пропущен.")
            continue

        doc_id = doc['id']
        text = doc['text']
        metadata = doc['metadata']

        # Создаем эмбеддинг для текста
        try:
            embedding = model.encode(text).tolist()
        except Exception as e:
            print(f"Ошибка при создании эмбеддинга для документа {doc_id}: {e}")
            continue

        ids.append(doc_id)
        embeddings.append(embedding)
        metadatas.append(metadata)
        texts.append(text)
        added_count += 1

    print(f"[DEBUG] Подготовлено {added_count} документов для добавления в БД.")

    # Добавляем документы в коллекцию
    if ids:
        try:
            print(f"[DEBUG] Попытка добавить {len(ids)} документов в коллекцию Chroma...")
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            print(f"Успешно добавлено {len(ids)} документов в векторную базу данных")
        except Exception as e:
            print(f"Ошибка при добавлении документов в векторную базу данных: {e}")
            import traceback
            traceback.print_exc()
            raise e
    else:
        print("[WARNING] Нет документов для добавления после фильтрации.")


def verify_vector_db(collection: chromadb.Collection):
    """
    Проверяет содержимое векторной базы данных.
    """
    try:
        count = collection.count()
        print(f"Общее количество документов в коллекции: {count}")

        if count > 0:
            results = collection.get(limit=3)
            print("Примеры документов в векторной базе:")
            # Проверяем, есть ли данные в results
            if results and 'ids' in results and results['ids']:
                for i in range(min(3, len(results['ids']))):
                    print(f"  ID: {results['ids'][i]}")
                    # Проверяем длину documents перед доступом
                    if i < len(results['documents']):
                        print(
                            f"  Текст (первые 100 символов): {results['documents'][i][:100] if results['documents'][i] else 'N/A'}...")
                    else:
                        print(f"  Текст: N/A")
                    # Проверяем длину metadatas перед доступом
                    if i < len(results['metadatas']):
                        print(f"  Метаданные: {results['metadatas'][i]}")
                    else:
                        print(f"  Метаданные: N/A")
                    print("-" * 20)
            else:
                print("  Получен пустой результат или отсутствуют данные.")
    except Exception as e:
        print(f"Ошибка при проверке векторной базы данных: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Основная функция для создания векторной базы знаний.
    """
    print("=" * 40)
    print("Начало создания векторной базы знаний...")
    print("=" * 40)

    documents = load_processed_documents()
    if not documents:
        print("Нет документов для добавления в векторную базу данных. Завершение.")
        return

    print(f"Шаг 1: Загружены {len(documents)} документов.")

    try:
        model = initialize_embedding_model()
        print("Шаг 2: Модель эмбеддингов инициализирована.")
    except Exception as e:
        print(f"Шаг 2: Не удалось инициализировать модель эмбеддингов: {e}")
        return

    try:
        collection = initialize_vector_db()
        print("Шаг 3: Векторная база данных инициализирована.")
    except Exception as e:
        print(f"Шаг 3: Не удалось инициализировать векторную базу данных: {e}")
        return

    try:
        add_documents_to_vector_db(collection, documents, model)
        print("Шаг 4: Документы добавлены в векторную базу данных.")
    except Exception as e:
        print(f"Шаг 4: Не удалось добавить документы в векторную базу данных: {e}")
        return

    print("Шаг 5: Проверка содержимого векторной базы данных...")
    verify_vector_db(collection)

    print("=" * 40)
    print("Создание векторной базы знаний завершено успешно!")
    print("=" * 40)


if __name__ == "__main__":
    main()