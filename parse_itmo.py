import time
import os
import glob
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager

# Список URL программ
PROGRAM_URLS = [
    "https://abit.itmo.ru/program/master/ai",
    "https://abit.itmo.ru/program/master/ai_product"
]

# Директория для сохранения данных
DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def setup_driver():
    """Настраивает и возвращает экземпляр WebDriver с настройками для скачивания."""
    chrome_options = Options()
    
    # Настройки для автоматического скачивания
    prefs = {
        "download.default_directory": DOWNLOAD_DIR, # Указываем папку для скачивания
        "download.prompt_for_download": False, # Отключаем диалоговое окно скачивания
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "plugins.always_open_pdf_externally": True # Открывать PDF во внешнем приложении (скачивать), а не в браузере
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    # Прочие настройки
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    try:
        service = Service(ChromeDriverManager().install())
        print("ChromeDriver установлен через webdriver-manager")
    except Exception as e:
        print(f"Ошибка при установке ChromeDriver через webdriver-manager: {e}")
        raise e
        
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        print(f"Настроена папка для скачивания: {DOWNLOAD_DIR}")
        return driver
    except Exception as e:
        print(f"Ошибка при создании драйвера: {e}")
        raise e

# В начале файла, после импортов и констант
# Храним список файлов, существовавших до запуска скрипта
INITIAL_PDF_FILES = set()

def get_program_data_selenium(driver, url):
    """
    Получает данные о программе с использованием Selenium и скачивает учебный план.
    """
    global INITIAL_PDF_FILES
    print(f"Парсинг (Selenium) {url}...")
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 20)
        
        # Ждем загрузки основного контента
        try:
            wait.until(EC.presence_of_element_located((By.ID, "study-plan")))
            print("  Элемент с id='study-plan' загружен")
        except TimeoutException:
            print("  Элемент с id='study-plan' не найден в течение 20 секунд, продолжаем...")
        
        # Получаем весь текст страницы
        body_element = driver.find_element(By.TAG_NAME, "body")
        text_content = body_element.text
        
        plan_url = None
        downloaded_plan_path = None
        
        # Ищем кнопку "Скачать учебный план"
        try:
            download_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Скачать учебный план')]"))
            )
            print("  Найдена кликабельная кнопка 'Скачать учебный план'")
            
            # Прокручиваем страницу так, чтобы кнопка оказалась в центре экрана
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", download_button)
            print("    Выполнена прокрутка к кнопке")
            time.sleep(1)
            
            # Получаем список PDF файлов в папке ДО клика
            # Это поможет нам определить, какой файл был скачан
            pdf_files_before = set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.pdf")))
            print(f"    Файлы PDF до клика: {pdf_files_before}")
            
            # Попытка клика
            print("    Попытка клика по кнопке...")
            try:
                download_button.click()
            except ElementClickInterceptedException:
                print("    Прямой клик не удался, пробуем ActionChains...")
                actions = ActionChains(driver)
                actions.move_to_element(download_button).click().perform()
                
            # Ждем скачивания файла
            print("    Ожидаем скачивания PDF файла...")
            downloaded_file = None
            for i in range(40):
                time.sleep(1)
                pdf_files_after = set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.pdf")))
                new_files = pdf_files_after - pdf_files_before
                if new_files:
                    # Найден новый файл
                    downloaded_file = list(new_files)[0]
                    print(f"    Новый файл скачан: {downloaded_file}")
                    downloaded_plan_path = os.path.relpath(downloaded_file, os.getcwd())
                    break
            else:
                print("    Новый файл PDF не был скачан в течение 40 секунд")
                # Проверим, может быть файл скачался, но мы его не заметили
                pdf_files_final = set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.pdf")))
                all_files_during_session = pdf_files_final - INITIAL_PDF_FILES
                if all_files_during_session:
                    downloaded_file = list(all_files_during_session)[0]
                    print(f"    Возможно, файл скачался: {downloaded_file} (определяем по общей сессии)")
                    downloaded_plan_path = os.path.relpath(downloaded_file, os.getcwd())
                
        except TimeoutException:
            print("  Кнопка 'Скачать учебный план' не найдена или не стала кликабельной")
        except Exception as e:
            print(f"    Ошибка при работе с кнопкой: {e}")
            import traceback
            traceback.print_exc()

        return {
            'url': url,
            'text_content': text_content,
            'plan_url': plan_url,
            'downloaded_plan_path': downloaded_plan_path
        }
        
    except Exception as e:
        print(f"Ошибка при парсинге {url} с Selenium: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_data(program_data):
    """
    Сохраняет текстовый контент и информацию о плане в файлы.
    """
    if not program_data:
        return
        
    parsed_url = urlparse(program_data['url'])
    program_name = parsed_url.path.strip('/').replace('/', '_')
    
    # Сохраняем текстовый контент
    text_filename = os.path.join(DOWNLOAD_DIR, f"{program_name}_content.txt")
    with open(text_filename, 'w', encoding='utf-8') as f:
        f.write(program_data['text_content'])
    print(f"Текстовый контент сохранен в {text_filename}")
    
    # Сохраняем информацию о ссылке на план и/или скачанном файле
    plan_info_filename = os.path.join(DOWNLOAD_DIR, f"{program_name}_plan_info.txt")
    with open(plan_info_filename, 'w', encoding='utf-8') as f:
        info_lines = []
        if program_data['plan_url']:
            info_lines.append(f"Ссылка на учебный план: {program_data['plan_url']}")
        if program_data['downloaded_plan_path']:
            info_lines.append(f"Скачанный учебный план: {program_data['downloaded_plan_path']}")
        if not program_data['plan_url'] and not program_data['downloaded_plan_path']:
            info_lines.append("Ссылка на учебный план не найдена и файл не был скачан.")
            
        f.write("\n".join(info_lines))
        print(f"Информация о учебном плане сохранена в {plan_info_filename}")

# Обновим функцию main, чтобы инициализировать INITIAL_PDF_FILES
def main():
    """
    Основная функция для парсинга всех программ с использованием Selenium.
    """
    global INITIAL_PDF_FILES
    driver = None
    try:
        # Сохраняем начальное состояние папки downloads
        INITIAL_PDF_FILES = set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.pdf")))
        print(f"Начальные PDF файлы в папке: {INITIAL_PDF_FILES}")
        
        driver = setup_driver()
        for url in PROGRAM_URLS:
            data = get_program_data_selenium(driver, url)
            save_data(data)
            print("-" * 40)
    finally:
        if driver:
            driver.quit()
            print("WebDriver закрыт.")

if __name__ == "__main__":
    main()