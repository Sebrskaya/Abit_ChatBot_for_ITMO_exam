# local_llm.py
import os
from llama_cpp import Llama
from langchain_core.runnables import Runnable
from typing import Any, List, Optional, Union, Dict

# Глобальная переменная для хранения модели
_LOCAL_MODEL = None
# Путь к локальной модели
MODEL_PATH = os.path.join("models", "saiga2_7b.gguf")
# Или используем repo_id и filename для автоматической загрузки
MODEL_REPO_ID = "IlyaGusev/saiga2_7b_gguf"
MODEL_FILENAME = "model-q4_K.gguf"  # Вы можете изменить на model-q5_K.gguf и т.д.


def get_local_model():
    """Ленивая загрузка локальной модели через llama-cpp-python."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        print("Загрузка локальной модели IlyaGusev/saiga2_7b_gguf через llama-cpp-python...")
        try:
            # Проверяем, существует ли локальный файл
            if os.path.exists(MODEL_PATH):
                print(f"Загрузка модели из локального файла: {MODEL_PATH}")
                _LOCAL_MODEL = Llama(
                    model_path=MODEL_PATH,
                    n_ctx=4096,  # Контекстное окно
                    n_threads=8,  # Количество потоков CPU
                    n_batch=512,  # Размер батча
                    verbose=False  # Отключить подробный лог llama.cpp, если нужно
                    # n_gpu_layers=35 # Раскомментируйте, если у вас подходящая GPU и установлен llama-cpp-python с поддержкой GPU
                )
            else:
                print(f"Локальный файл {MODEL_PATH} не найден. Попытка загрузки с HuggingFace...")
                # llama-cpp-python может загрузить модель напрямую из HuggingFace
                _LOCAL_MODEL = Llama.from_pretrained(
                    repo_id=MODEL_REPO_ID,
                    filename=MODEL_FILENAME,
                    n_ctx=4096,
                    n_threads=8,
                    n_batch=512,
                    verbose=False
                    # n_gpu_layers=35 # Раскомментируйте для GPU
                )
                # После загрузки модель будет кэширована huggingface_hub
            print("Локальная модель загружена успешно.")
        except Exception as e:
            print(f"Ошибка при загрузке локальной модели: {e}")
            import traceback
            traceback.print_exc()
            raise e
    return _LOCAL_MODEL


class LocalLLMWrapper(Runnable):
    """Обертка для локальной LLM через llama-cpp-python."""

    def __init__(self, **kwargs):
        # Сохраняем параметры по умолчанию для генерации
        # Эти параметры будут использоваться как базовые, но могут быть переопределены при вызове
        self.default_generation_config = {
            "max_tokens": kwargs.get("max_tokens", 200),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 40),
            "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
            "stop": kwargs.get("stop", ["</s>", "<|user|>", "<|assistant|>"]),  # Стандартные стоп-слова для Saiga2
            "echo": False,  # Не возвращать промпт в ответе
        }
        # Параметры, которые не должны передаваться в create_completion напрямую
        self.invalid_completion_params = {
            'return_exceptions', 'config', 'kwargs'  # Добавим 'kwargs' на случай распаковки пустого словаря
            # Добавьте сюда другие, если обнаружите
        }

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            model = get_local_model()

            # Формируем промпт в формате, ожидаемом моделью Saiga2
            full_prompt = (
                f"<|system|>Ты помощник абитуриента ИТМО. Отвечай точно, кратко и только на основе предоставленной информации. "
                f"Отвечай на русском языке. Не добавляй фразы вроде 'Question:' или 'Helpful Answer:'.</|system|>\n"
                f"<|user|>{prompt}</|user|>\n"
                f"<|assistant|>"
            )

            # Начинаем с базовой конфигурации
            generation_kwargs = self.default_generation_config.copy()
            # Обновляем параметрами из kwargs вызова _call
            # Это позволяет передавать специфичные параметры для каждого вызова
            generation_kwargs.update(kwargs)

            # Если stop передан напрямую в _call, используем его (имеет наивысший приоритет)
            if stop is not None:
                generation_kwargs['stop'] = stop

            # Фильтруем параметры, убирая те, которые не поддерживаются create_completion
            filtered_kwargs = {
                k: v for k, v in generation_kwargs.items()
                if k not in self.invalid_completion_params
            }

            # --- Отладка (раскомментируйте при необходимости) ---
            # print(f"[DEBUG] Full Prompt: {repr(full_prompt)}")
            # print(f"[DEBUG] Filtered Generation Kwargs: {filtered_kwargs}")
            # ---

            # Генерируем ответ
            # llama-cpp-python возвращает словарь с результатами
            response = model.create_completion(
                prompt=full_prompt,
                **filtered_kwargs
            )

            # --- Отладка (раскомментируйте при необходимости) ---
            # print(f"[DEBUG] Raw Response: {response}")
            # ---

            # Извлекаем текст ответа
            answer = response['choices'][0]['text'].strip()

            # Финальная обрезка по стоп-словам на случай, если модель не обработала их полностью
            # (create_completion должен это делать, но на всякий случай)
            final_stop = filtered_kwargs.get('stop', [])
            if final_stop:
                min_idx = len(answer)  # Начинаем с конца
                for s in final_stop:
                    idx = answer.find(s)
                    if idx != -1 and idx < min_idx:
                        min_idx = idx
                if min_idx < len(answer):
                    answer = answer[:min_idx]

            return answer

        except Exception as e:
            print(f"Ошибка при генерации текста локальной моделью: {e}")
            import traceback
            traceback.print_exc()
            return "Извините, не удалось получить ответ от локальной модели."

    # Реализация обязательных методов Runnable
    def invoke(self, input: Union[str, Dict], config=None, **kwargs) -> str:
        if isinstance(input, str):
            prompt = input
        elif isinstance(input, Dict) and "prompt" in input:
            prompt = input["prompt"]
        elif isinstance(input, Dict) and "text" in input:
            prompt = input["text"]
        else:
            prompt = str(input)
        return self._call(prompt, **kwargs)

    async def ainvoke(self, input: Union[str, Dict], config=None, **kwargs) -> str:
        # Для простоты асинхронная версия вызывает синхронный метод
        # В реальном асинхронном коде тут должна быть awaitable операция
        return self.invoke(input, config=config, **kwargs)

    def batch(self, inputs: List[Union[str, Dict]], config=None, *, return_exceptions: bool = False, **kwargs) -> List[
        str]:
        results = []
        for inp in inputs:
            try:
                res = self.invoke(inp, config=config, **kwargs)
                results.append(res)
            except Exception as e:
                if return_exceptions:
                    results.append(e)
                else:
                    # Если return_exceptions=False, пробрасываем исключение
                    raise
        return results

    async def abatch(self, inputs: List[Union[str, Dict]], config=None, *, return_exceptions: bool = False, **kwargs) -> \
    List[str]:
        # Асинхронная версия batch
        # Для простоты реализуем как синхронный batch
        # В реальном асинхронном коде тут должна быть awaitable логика
        return self.batch(inputs, config=config, return_exceptions=return_exceptions, **kwargs)


def load_local_llm():
    """
    Загружает более качественную локальную LLM IlyaGusev/saiga2_7b_gguf через llama-cpp-python.
    Модель будет загружена из локального файла 'models/saiga2_7b.gguf' или скачана с HuggingFace.
    """
    llm = LocalLLMWrapper(
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.1,
        stop=["</s>", "<|user|>", "<|assistant|>"]
        # n_gpu_layers=35 # Раскомментируйте, если у вас подходящая GPU и установлен llama-cpp-python с поддержкой GPU
    )
    return llm