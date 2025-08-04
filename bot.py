# bot.py
import os
import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain.chains import RetrievalQA
from local_llm import load_local_llm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from agent import get_agent_executor

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Пути
VECTOR_DB_DIR = "vector_db"
MODEL_NAME = "distiluse-base-multilingual-cased-v1"

# Инициализация векторного хранилища
def load_retriever():
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    try:
        # Явно указываем имя коллекции
        db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embedding_function,
            collection_name="itmo_master_programs" # <-- Добавить это
        )
        return db.as_retriever(search_kwargs={"k": 1})
    except Exception as e:
        logger.error(f"Ошибка загрузки векторной БД: {e}")
        return None

# Инициализация QA-цепочки
def load_qa_chain(retriever):
    llm = load_local_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# Глобальные переменные
qa_chain = None
agent_executor = None

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот-помощник по магистерским программам ИТМО:\n"
        "• Искусственный интеллект\n"
        "• AI Product\n\n"
        "Задавайте вопросы о программах, поступлении, карьере или попросите помочь выбрать дисциплины!"
    )

# Обработка сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    logger.info(f"Пользователь: {user_input}")

    # Проверка, нужно ли использовать агента
    if any(keyword in user_input.lower() for keyword in ["рекомендуй", "подбери", "совет", "какие курсы", "что выбрать", "сравни"]):
        try:
            response = await agent_executor.ainvoke({"input": user_input})
            answer = response["output"]
        except Exception as e:
            logger.error(f"Ошибка агента: {e}")
            answer = "Извините, не удалось обработать ваш запрос."
    else:
        # Обычный QA
        # В bot.py, в handle_message, в блоке try для обычного QA
        try:
            # --- Отладка: Проверка контекста ---
            print(f"[DEBUG] Поиск контекста для вопроса: {user_input}")
            debug_docs = retriever.invoke(user_input)  # retriever доступен в области видимости?
            # Если retriever недоступен, получите его, например, через qa_chain
            # debug_docs = qa_chain.retriever.invoke(user_input)
            # или сохраните retriever в переменной выше
            if hasattr(qa_chain, 'retriever'):
                debug_docs = qa_chain.retriever.invoke(user_input)
                print(f"[DEBUG] Найдено {len(debug_docs)} документов через qa_chain.retriever:")
                for i, doc in enumerate(debug_docs):
                    print(
                        f"[DEBUG]   Документ {i + 1} (источник: {doc.metadata.get('source', 'N/A') or doc.metadata.get('source_file', 'N/A')}):")
                    # Выведем больше текста для проверки
                    print(
                        f"[DEBUG]     Содержимое: {doc.page_content[:500] if hasattr(doc, 'page_content') else doc.page_content[:500] if 'page_content' in doc else 'NO page_content'}...")
                    # Или, если документ - это просто словарь
                    # print(f"[DEBUG]     Содержимое: {doc['document'][:500] if 'document' in doc else 'NO document key'}...")
                    print("-" * 20)
            else:
                # Если retriever напрямую доступен в handle_message
                debug_docs = retriever.invoke(user_input)
            print(f"[DEBUG] Найдено {len(debug_docs)} документов:")
            for i, doc in enumerate(debug_docs):
                print(
                    f"[DEBUG]   Документ {i + 1} (источник: {doc.metadata.get('source', 'N/A')}): {doc.page_content[:200]}...")
            # --- Конец отладки ---

            result = qa_chain.invoke({"query": user_input})  # Используем invoke вместо __call__
            answer = result["result"]
        except Exception as e:
            logger.error(f"Ошибка QA: {e}")
            answer = "Не удалось найти ответ. Попробуйте уточнить вопрос."

    await update.message.reply_text(answer)

def main():
    global qa_chain, agent_executor, retriever

    # Загружаем компоненты
    retriever = load_retriever()
    if not retriever:
        logger.error("Не удалось загрузить retriever. Выход.")
        return

    qa_chain = load_qa_chain(retriever)
    agent_executor = get_agent_executor(retriever)

    # Запуск бота
    app = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()