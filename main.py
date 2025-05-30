#!/usr/bin/env python3
"""
Головний файл системи автоматизації воріт
"""

import sys
import signal
import logging
import time
import argparse
from pathlib import Path
from logging.handlers import RotatingFileHandler

from hardware import HardwareManager
from gate_controller import GateController, SafetyMonitor
from detection import DetectionPipeline
from google_sheets import GoogleSheetsManager
from processes import ProcessManager
from config import LOGGING_CONFIG, SYSTEM_CONFIG


# Налаштування логування
def setup_logging():
    """Налаштувати систему логування"""
    # Створюємо форматер
    formatter = logging.Formatter(LOGGING_CONFIG["format"])

    # Корневий логер
    root_logger = logging.getLogger()
    root_logger.setLevel(LOGGING_CONFIG["level"])

    # Консольний обробник
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Файловий обробник з ротацією
    file_handler = RotatingFileHandler(
        LOGGING_CONFIG["file"],
        maxBytes=LOGGING_CONFIG["max_bytes"],
        backupCount=LOGGING_CONFIG["backup_count"]
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Зменшуємо рівень логування для деяких бібліотек
    logging.getLogger("googleapiclient").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


class GateAutomationSystem:
    """Основна система автоматизації воріт"""

    def __init__(self):
        """Ініціалізація системи"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 50)
        self.logger.info("Запуск системи автоматизації воріт")
        self.logger.info("=" * 50)

        # Компоненти системи
        self.hardware = None
        self.gate_controller = None
        self.safety_monitor = None
        self.detection_pipeline = None
        self.sheets_manager = None
        self.process_manager = None

        # Стан системи
        self.running = False
        self._shutdown_requested = False

        # Реєструємо обробник сигналів
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Обробник сигналів для коректного завершення"""
        self.logger.info(f"Отримано сигнал {signum}, запит на завершення...")
        self._shutdown_requested = True

    def initialize(self) -> bool:
        """Ініціалізувати всі компоненти"""
        try:
            self.logger.info("Ініціалізація компонентів...")

            # Апаратне забезпечення
            self.logger.info("Ініціалізація апаратного забезпечення...")
            self.hardware = HardwareManager()

            # Контролер воріт
            self.logger.info("Ініціалізація контролера воріт...")
            self.gate_controller = GateController(self.hardware)

            # Монітор безпеки
            self.logger.info("Ініціалізація монітора безпеки...")
            self.safety_monitor = SafetyMonitor(self.hardware, self.gate_controller)

            # Пайплайн детекції
            self.logger.info("Ініціалізація системи детекції...")
            self.detection_pipeline = DetectionPipeline()

            # Менеджер Google Sheets
            self.logger.info("Ініціалізація Google Sheets...")
            self.sheets_manager = GoogleSheetsManager()

            # Менеджер процесів
            self.logger.info("Ініціалізація менеджера процесів...")
            self.process_manager = ProcessManager(
                self.hardware,
                self.gate_controller,
                self.detection_pipeline,
                self.sheets_manager
            )

            # Реєструємо обробники подій
            self._register_event_handlers()

            self.logger.info("Ініціалізація завершена успішно")
            return True

        except Exception as e:
            self.logger.error(f"Помилка ініціалізації: {e}", exc_info=True)
            return False

    def _register_event_handlers(self):
        """Зареєструвати обробники подій"""
        # Обробники подій воріт
        self.gate_controller.register_handler('gate_opened', self._on_gate_opened)
        self.gate_controller.register_handler('gate_closed', self._on_gate_closed)
        self.gate_controller.register_handler('vehicle_passed', self._on_vehicle_passed)
        self.gate_controller.register_handler('error', self._on_error)

    def _on_gate_opened(self):
        """Обробник відкриття воріт"""
        self.logger.info("Подія: Ворота відкриті")

    def _on_gate_closed(self):
        """Обробник закриття воріт"""
        self.logger.info("Подія: Ворота закриті")

    def _on_vehicle_passed(self):
        """Обробник проїзду автомобіля"""
        self.logger.info("Подія: Автомобіль проїхав")

    def _on_error(self, error_msg: str):
        """Обробник помилки"""
        self.logger.error(f"Подія помилки: {error_msg}")

    def start(self):
        """Запустити систему"""
        if not self.initialize():
            self.logger.error("Не вдалося ініціалізувати систему")
            return False

        try:
            self.logger.info("Запуск системи...")
            self.running = True

            # Запускаємо монітор безпеки
            self.safety_monitor.start()

            # Запускаємо обробку відео
            self.process_manager.start()

            self.logger.info("Система запущена успішно")
            self.logger.info("Натисніть Ctrl+C для завершення")

            # Основний цикл
            self._main_loop()

        except Exception as e:
            self.logger.error(f"Критична помилка: {e}", exc_info=True)

        finally:
            self.stop()

    def _main_loop(self):
        """Основний цикл програми"""
        stats_interval = 60  # Інтервал виводу статистики (секунд)
        last_stats_time = time.time()

        while self.running and not self._shutdown_requested:
            try:
                # Виводимо статистику періодично
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    self._print_stats()
                    last_stats_time = current_time

                # Перевіряємо стан системи
                if self.gate_controller.get_state().value == "error":
                    self.logger.error("Система в стані помилки!")
                    break

                # Невелика затримка
                time.sleep(1)

            except KeyboardInterrupt:
                self.logger.info("Отримано запит на завершення")
                break

    def _print_stats(self):
        """Вивести статистику роботи"""
        stats = self.process_manager.get_stats()

        self.logger.info("=" * 50)
        self.logger.info("СТАТИСТИКА СИСТЕМИ:")
        self.logger.info(f"Стан воріт: {stats['gate_state']}")

        self.logger.info("В'їзд:")
        for key, value in stats['entrance'].items():
            self.logger.info(f"  {key}: {value}")

        self.logger.info("Виїзд:")
        for key, value in stats['exit'].items():
            self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 50)

    def stop(self):
        """Зупинити систему"""
        if not self.running:
            return

        self.logger.info("Зупинка системи...")
        self.running = False

        try:
            # Зупиняємо компоненти в зворотному порядку
            if self.process_manager:
                self.logger.info("Зупинка обробки...")
                self.process_manager.stop()

            if self.safety_monitor:
                self.logger.info("Зупинка монітора безпеки...")
                self.safety_monitor.stop()

            if self.gate_controller:
                # Переконуємося, що ворота закриті
                if self.gate_controller.get_state() == GateState.OPEN:
                    self.logger.info("Закриття воріт перед завершенням...")
                    self.gate_controller.close_gate()

            if self.hardware:
                self.logger.info("Очищення апаратних ресурсів...")
                self.hardware.cleanup()

            self.logger.info("Система зупинена успішно")

        except Exception as e:
            self.logger.error(f"Помилка під час зупинки: {e}", exc_info=True)


def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(
        description="Система автоматизації воріт для ЖК"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Увімкнути режим налагодження"
    )

    parser.add_argument(
        "--test-hardware",
        action="store_true",
        help="Тестувати апаратне забезпечення"
    )

    args = parser.parse_args()

    # Налаштування логування
    setup_logging()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        SYSTEM_CONFIG["debug_mode"] = True

    # Тестування апаратного забезпечення
    if args.test_hardware:
        test_hardware()
        return

    # Запуск основної системи
    system = GateAutomationSystem()
    system.start()


def test_hardware():
    """Тестування апаратного забезпечення"""
    logger = logging.getLogger(__name__)
    logger.info("Тестування апаратного забезпечення...")

    try:
        # Ініціалізація
        hardware = HardwareManager()

        # Тест GPIO
        logger.info("\n--- Тест GPIO ---")
        logger.info(f"Стан воріт: {'Відкриті' if hardware.gpio.is_gate_open() else 'Закриті'}")

        # Тест реле
        response = input("Тестувати реле? (y/n): ")
        if response.lower() == 'y':
            logger.info("Тест реле OPEN...")
            hardware.gpio.pulse_relay(GPIO_CONFIG["OPEN_RELAY"])
            time.sleep(1)

            logger.info("Тест реле CLOSE...")
            hardware.gpio.pulse_relay(GPIO_CONFIG["CLOSE_RELAY"])
            time.sleep(1)

        # Тест ультразвукового датчика
        logger.info("\n--- Тест ультразвукового датчика ---")
        for i in range(5):
            distance = hardware.ultrasonic.get_distance()
            logger.info(f"Відстань: {distance} см")
            time.sleep(0.5)

        # Тест камер
        logger.info("\n--- Тест камер ---")
        for camera_type in ["ENTRANCE", "EXIT"]:
            frame = hardware.camera.capture_frame(camera_type)
            if frame is not None:
                logger.info(f"Камера {camera_type}: OK, розмір {frame.shape}")
            else:
                logger.error(f"Камера {camera_type}: ПОМИЛКА")

        logger.info("\nТестування завершено")

    except Exception as e:
        logger.error(f"Помилка тестування: {e}", exc_info=True)

    finally:
        if 'hardware' in locals():
            hardware.cleanup()


if __name__ == "__main__":
    main()