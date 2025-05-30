"""
Конфігураційний файл для системи автоматизації воріт
"""

import os
from pathlib import Path

# Базові шляхи
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Створюємо необхідні директорії
for dir_path in [CONFIG_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# GPIO піни
GPIO_CONFIG = {
    "OPEN_RELAY": 17,      # Реле відкриття воріт
    "CLOSE_RELAY": 27,     # Реле закриття воріт
    "MAGNETIC_SENSOR": 22,  # Герконовий датчик
    "ULTRASONIC_TRIG": 23,  # JSN-SR04T Trigger
    "ULTRASONIC_ECHO": 24   # JSN-SR04T Echo
}

# Камери
CAMERA_CONFIG = {
    "ENTRANCE": {
        "name": "imx708",
        "resolution": (1920, 1080),
        "fps": 30,
        "roi_file": CONFIG_DIR / "roi_entrance.json"
    },
    "EXIT": {
        "name": "imx219",
        "resolution": (1920, 1080),
        "fps": 30,
        "roi_file": CONFIG_DIR / "roi_exit.json"
    }
}

# Моделі
MODEL_CONFIG = {
    "VEHICLE_DETECTION": {
        "path": MODELS_DIR / "mobilenet_ssdv1.onnx",
        "confidence_threshold": 0.7,
        "classes": ["car", "truck", "bus"]  # Класи для детекції
    },
    "LICENSE_PLATE_DETECTION": {
        "path": MODELS_DIR / "license.onnx",
        "confidence_threshold": 0.6,
        "input_shape": (448, 448),
        "input_dtype": "float32"
    },
    "OCR": {
        "path": MODELS_DIR / "ocr.pt",
        "confidence_threshold": 0.4,
        "img_size": 320
    }
}

# Google Sheets
GOOGLE_SHEETS_CONFIG = {
    "credentials_file": BASE_DIR / "credentials.json",
    "spreadsheet_id": "1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4",
    "scopes": ["https://www.googleapis.com/auth/spreadsheets"],
    "ranges": {
        "allowed_plates": "A3:A",      # Дозволені номери
        "last_seen": "B3:B",           # Останній час розпізнавання
        "denied_plates": "D3:D",       # Заборонені номери
        "denied_times": "E3:E",        # Час спроби заборонених
        "exit_plates": "G3:G",         # Номери на виїзд
        "exit_times": "H3:H"           # Час виїзду
    }
}

# Ультразвуковий датчик
ULTRASONIC_CONFIG = {
    "normal_distance": 100,      # Нормальна відстань в см
    "detection_threshold": 80,   # Поріг детекції машини в см
    "measurement_timeout": 0.1,  # Таймаут вимірювання в секундах
    "samples": 3                 # Кількість вимірювань для усереднення
}

# Таймери та затримки
TIMING_CONFIG = {
    "gate_close_delay": 5,       # Затримка закриття воріт після проїзду (сек)
    "relay_pulse_duration": 0.5, # Тривалість імпульсу реле (сек)
    "sensor_timeout": 30,        # Таймаут для датчиків (сек)
    "duplicate_timeout": 60,     # Час ігнорування дублікатів номерів (сек)
    "gate_operation_timeout": 5  # Час очікування зміни стану воріт (сек)
}

# Логування
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "gate_system.log",
    "max_bytes": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5
}

# Налаштування системи
SYSTEM_CONFIG = {
    "debug_mode": False,
    "save_images": True,         # Зберігати зображення для налагодження
    "images_dir": BASE_DIR / "captured_images",
    "max_images": 1000,          # Максимальна кількість збережених зображень
    "cleanup_interval": 3600     # Інтервал очищення старих зображень (сек)
}

# Створюємо директорію для зображень
if SYSTEM_CONFIG["save_images"]:
    SYSTEM_CONFIG["images_dir"].mkdir(exist_ok=True)