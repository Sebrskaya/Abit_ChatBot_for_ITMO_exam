import os
import json
from pathlib import Path
import hashlib
from typing import List, Dict, Any

# Для работы с PDF
import PyPDF2

# Для работы с эмбеддингами и векторной БД
# (эти импорты понадобятся позже, но добавим для полноты картины)
# import chromadb
# from sentence_transformers import SentenceTransformer

# Пути к папкам
DOWNLOADS_DIR = "downloads"
PROCESSED_DIR = "processed_data"

# Создаем папку для обработанных данных
os.makedirs(PROCESSED_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Извлекает текст из PDF файла.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Ошибка при извлечении текста из {pdf_path}: {e}")
        return ""
    return text

def read_text_file(file_path: str) -> str:
    """
    Читает текст из файла.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
        return ""

def get_file_hash(file_path: str) -> str:
    """
    Возвращает хеш файла для идентификации.
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except Exception as e:
        print(f"Ошибка при вычислении хеша файла {file_path}: {e}")
        return ""
    return hash_md5.hexdigest()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Разбивает текст на чанки.
    """
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():  # Проверяем, что чанк не пустой
            chunks.append(chunk)
    return chunks

def process_program_data(program_name: str, content_file: str, plan_info_file: str) -> List[Dict[str, Any]]:
    """
    Обрабатывает данные для одной программы.
    """
    print(f"Обработка данных для программы: {program_name}")
    
    # Чтение текстового контента
    content_text = read_text_file(content_file)
    if not content_text:
        print(f"  Не удалось прочитать текстовый контент для {program_name}")
        return []
    
    # Чтение информации о плане
    plan_info_text = read_text_file(plan_info_file)
    plan_pdf_path = None
    if plan_info_text:
        # Проверяем, есть ли путь к PDF в информации
        if "Скачанный учебный план:" in plan_info_text:
            # Извлекаем путь к PDF
            for line in plan_info_text.split('\n'):
                if line.startswith("Скачанный учебный план:"):
                    plan_pdf_path = line.split(":", 1)[1].strip()
                    break
    
    # Извлечение текста из PDF учебного плана
    plan_text = ""
    if plan_pdf_path and os.path.exists(plan_pdf_path):
        print(f"  Извлечение текста из PDF: {plan_pdf_path}")
        plan_text = extract_text_from_pdf(plan_pdf_path)
    else:
        print(f"  PDF файл учебного плана не найден для {program_name}")
    
    # Подготовка метаданных
    metadata = {
        "program_name": program_name,
        "content_source": "web_content",
        "source_file": content_file
    }
    
    # Разбиение текстового контента на чанки
    content_chunks = chunk_text(content_text)
    print(f"  Создано {len(content_chunks)} чанков из текстового контента")
    
    # Создание документов для текстового контента
    documents = []
    for i, chunk in enumerate(content_chunks):
        doc = {
            "id": f"{program_name}_content_{i}",
            "text": chunk,
            "metadata": {**metadata, "chunk_index": i, "chunk_type": "web_content"}
        }
        documents.append(doc)
    
    # Обработка текста учебного плана, если он есть
    if plan_text:
        plan_metadata = {
            "program_name": program_name,
            "content_source": "study_plan",
            "source_file": plan_pdf_path
        }
        
        # Разбиение текста плана на чанки
        plan_chunks = chunk_text(plan_text)
        print(f"  Создано {len(plan_chunks)} чанков из учебного плана")
        
        # Создание документов для учебного плана
        for i, chunk in enumerate(plan_chunks):
            doc = {
                "id": f"{program_name}_plan_{i}",
                "text": chunk,
                "metadata": {**plan_metadata, "chunk_index": i, "chunk_type": "study_plan"}
            }
            documents.append(doc)
    
    return documents

def main():
    """
    Основная функция для обработки всех данных.
    """
    print("Начало обработки данных...")
    
    all_documents = []
    
    # Получаем список программ из имен файлов контента
    content_files = list(Path(DOWNLOADS_DIR).glob("*_content.txt"))
    
    for content_file in content_files:
        # Определяем имя программы
        program_name = content_file.stem.replace("_content", "")
        
        # Ищем соответствующий файл с информацией о плане
        plan_info_file = os.path.join(DOWNLOADS_DIR, f"{program_name}_plan_info.txt")
        
        if os.path.exists(plan_info_file):
            # Обрабатываем данные для программы
            program_documents = process_program_data(
                program_name, 
                str(content_file), 
                plan_info_file
            )
            all_documents.extend(program_documents)
        else:
            print(f"Файл с информацией о плане не найден для {program_name}")
    
    # Сохранение всех документов в JSON файл
    output_file = os.path.join(PROCESSED_DIR, "processed_documents.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_documents, f, ensure_ascii=False, indent=2)
        print(f"Все документы сохранены в {output_file}")
        print(f"Общее количество документов: {len(all_documents)}")
    except Exception as e:
        print(f"Ошибка при сохранении документов: {e}")
        
    # Вывод статистики
    print("\nСтатистика:")
    program_stats = {}
    for doc in all_documents:
        program = doc['metadata']['program_name']
        chunk_type = doc['metadata']['chunk_type']
        if program not in program_stats:
            program_stats[program] = {"web_content": 0, "study_plan": 0}
        program_stats[program][chunk_type] += 1
    
    for program, stats in program_stats.items():
        print(f"  {program}:")
        print(f"    Веб-контент: {stats['web_content']} чанков")
        print(f"    Учебный план: {stats['study_plan']} чанков")

if __name__ == "__main__":
    main()