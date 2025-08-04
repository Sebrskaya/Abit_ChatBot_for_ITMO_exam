# tools.py
from langchain.tools import BaseTool
from pydantic import Field
from typing import Type
import re

class CourseRecommenderTool(BaseTool):
    name: str = "рекомендация_курсов"
    description: str = "Рекомендует выборные дисциплины на основе бэкграунда пользователя: опыт в программировании, данных, продуктах, цели (ML Engineer, Data Analyst и т.д.)"
    background: str = Field(default="")

    def _run(self, query: str) -> str:
        query_lower = query.lower()
        skills = []
        goals = []

        if any(x in query_lower for x in ["программирование", "python", "разработка", "coding"]):
            skills.append("программирование")
        if any(x in query_lower for x in ["данные", "анализ", "data", "sql", "аналитик"]):
            skills.append("работа с данными")
        if any(x in query_lower for x in ["продукт", "product", "менеджмент", "бизнес"]):
            skills.append("продуктовый бэкграунд")
        if any(x in query_lower for x in ["ml", "машинное обучение", "нейросети", "искусственный интеллект"]):
            skills.append("ML/AI")

        if "ml engineer" in query_lower or "мл инженер" in query_lower:
            goals.append("ML Engineer")
        if "data analyst" in query_lower or "аналитик данных" in query_lower:
            goals.append("Data Analyst")
        if "ai product" in query_lower or "продукт" in query_lower:
            goals.append("AI Product Developer")
        if "data engineer" in query_lower or "инженер данных" in query_lower:
            goals.append("Data Engineer")

        recommendations = []

        if "ML Engineer" in goals:
            recommendations.append("- Продвинутый машинный обучение (Advanced ML)")
            recommendations.append("- Глубокое обучение (Deep Learning)")
            recommendations.append("- MLOps и внедрение моделей")
            recommendations.append("- Оптимизация и масштабирование моделей")

        if "Data Analyst" in goals:
            recommendations.append("- Анализ больших данных (Big Data Analytics)")
            recommendations.append("- Визуализация данных")
            recommendations.append("- A/B тестирование и метрики")
            recommendations.append("- Python для анализа данных")

        if "AI Product Developer" in goals:
            recommendations.append("- Управление AI-продуктами")
            recommendations.append("- UX для AI-систем")
            recommendations.append("- Экономика AI и монетизация")
            recommendations.append("- Agile и Scrum в AI-проектах")

        if "Data Engineer" in goals:
            recommendations.append("- Архитектура данных")
            recommendations.append("- ETL/ELT процессы")
            recommendations.append("- Работа с Apache Spark, Kafka")
            recommendations.append("- Облачные платформы (AWS, GCP)")

        return (
            f"На основе вашего бэкграунда ({', '.join(skills or ['не указан'])}) "
            f"и цели ({', '.join(goals or ['не указана'])}), рекомендую следующие дисциплины:\n"
            + "\n".join(recommendations or ["Уточните ваш бэкграунд и цели для персонализированных рекомендаций."])
        )

class ProgramComparatorTool(BaseTool):
    name: str = "сравнение_программ"
    description: str = "Сравнивает программы 'Искусственный интеллект' и 'AI Product' по различным параметрам: фокус, карьера, партнеры, требования."
    retriever: any = None

    def __init__(self, retriever, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever

    def _run(self, query: str) -> str:
        comparison = """
Сравнение программ:

🔹 **Искусственный интеллект (AI)**:
- Фокус: Техническая подготовка ML-инженеров, глубокое обучение, исследования.
- Карьера: ML Engineer, Data Scientist, Researcher.
- Партнёры: Sber AI, X5, Ozon Bank, МТС, Яндекс.
- Подходит тем, кто хочет глубоко погрузиться в ML и разработку моделей.

🔹 **AI Product**:
- Фокус: Разработка AI-продуктов, управление проектами, взаимодействие бизнеса и технологий.
- Карьера: AI Product Manager, Product Developer, Tech Lead.
- Партнёры: Napoleon IT, Raft, Genotek, AIRI.
- Подходит тем, кто хочет создавать продукты на основе ИИ, а не только модели.

💡 Совет: Выберите 'AI', если хотите быть инженером. Выберите 'AI Product', если хотите управлять продуктами или командами.
        """
        return comparison