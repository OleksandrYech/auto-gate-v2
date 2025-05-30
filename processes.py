"""
Процеси обробки в'їзду та виїзду автомобілів
"""

import time
import logging
import threading
from typing import Optional
from queue import Queue, Empty

from hardware import HardwareManager
from gate_controller import GateController
from detection import DetectionPipeline
from google_sheets import GoogleSheetsManager
from config import TIMING_CONFIG

logger = logging.getLogger(__name__)


class VehicleProcessor:
    """Базовий клас для обробки автомобілів"""

    def __init__(self,
                 hardware: HardwareManager,
                 gate_controller: GateController,
                 detection_pipeline: DetectionPipeline,
                 sheets_manager: GoogleSheetsManager,
                 camera_type: str):
        """Ініціалізація процесора"""
        self.hardware = hardware
        self.gate_controller = gate_controller
        self.detection = detection_pipeline
        self.sheets = sheets_manager
        self.camera_type = camera_type

        # Черга для обробки
        self.frame_queue = Queue(maxsize=10)
        self._processing = False
        self._process_thread = None
        self._capture_thread = None
        self._stop_event = threading.Event()

        # Статистика
        self.stats = {
            'vehicles_detected': 0,
            'plates_recognized': 0,
            'access_granted': 0,
            'access_denied': 0
        }

        logger.info(f"Процесор {camera_type} ініціалізовано")

    def start(self):
        """Запустити обробку"""
        if self._processing:
            logger.warning(f"Обробка {self.camera_type} вже запущена")
            return

        self._processing = True
        self._stop_event.clear()

        # Запускаємо потоки
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"{self.camera_type}_capture"
        )
        self._capture_thread.daemon = True
        self._capture_thread.start()

        self._process_thread = threading.Thread(
            target=self._process_loop,
            name=f"{self.camera_type}_process"
        )
        self._process_thread.daemon = True
        self._process_thread.start()

        logger.info(f"Обробка {self.camera_type} запущена")

    def stop(self):
        """Зупинити обробку"""
        if not self._processing:
            return

        logger.info(f"Зупинка обробки {self.camera_type}")
        self._processing = False
        self._stop_event.set()

        # Чекаємо завершення потоків
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        if self._process_thread:
            self._process_thread.join(timeout=2)

        # Очищаємо чергу
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break

        logger.info(f"Обробка {self.camera_type} зупинена")

    def _capture_loop(self):
        """Цикл захоплення кадрів"""
        frame_interval = 0.1  # 10 FPS для обробки

        while self._processing and not self._stop_event.is_set():
            try:
                # Захоплюємо кадр
                frame = self.hardware.camera.capture_frame(self.camera_type)

                if frame is not None:
                    # Додаємо в чергу якщо є місце
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)

                time.sleep(frame_interval)

            except Exception as e:
                logger.error(f"Помилка захоплення кадру {self.camera_type}: {e}")
                time.sleep(1)

    def _process_loop(self):
        """Основний цикл обробки"""
        while self._processing and not self._stop_event.is_set():
            try:
                # Отримуємо кадр з черги
                try:
                    frame = self.frame_queue.get(timeout=1)
                except Empty:
                    continue

                # Обробляємо кадр
                self.process_frame(frame)

            except Exception as e:
                logger.error(f"Помилка обробки {self.camera_type}: {e}")
                time.sleep(0.5)

    def process_frame(self, frame):
        """Обробити кадр - має бути перевизначено в підкласах"""
        raise NotImplementedError

    def get_stats(self) -> dict:
        """Отримати статистику"""
        return self.stats.copy()


class EntranceProcessor(VehicleProcessor):
    """Процесор для в'їзду"""

    def __init__(self, *args, **kwargs):
        """Ініціалізація процесора в'їзду"""
        super().__init__(*args, **kwargs)

        # Для відстеження обробленого автомобіля
        self._last_processed_time = 0
        self._processing_cooldown = 5  # секунд між обробками

        logger.info("Процесор в'їзду готовий")

    def process_frame(self, frame):
        """Обробити кадр для в'їзду"""
        # Перевіряємо чи не обробляли нещодавно
        current_time = time.time()
        if current_time - self._last_processed_time < self._processing_cooldown:
            return

        # Детекція
        results = self.detection.process_frame(frame, self.camera_type)

        # Якщо немає автомобілів - виходимо
        if not results['vehicles']:
            return

        self.stats['vehicles_detected'] += 1

        # Якщо немає розпізнаних номерів - виходимо
        if not results['recognized_numbers']:
            return

        # Беремо перший розпізнаний номер
        plate_number = results['recognized_numbers'][0]
        self.stats['plates_recognized'] += 1

        logger.info(f"В'їзд: розпізнано номер {plate_number}")

        # Перевіряємо доступ
        if self.sheets.check_plate_access(plate_number):
            logger.info(f"В'їзд: доступ дозволено для {plate_number}")
            self.stats['access_granted'] += 1

            # Відкриваємо ворота
            if self.gate_controller.open_gate():
                # Оновлюємо час в таблиці
                self.sheets.update_last_seen(plate_number)

                # Оновлюємо час обробки
                self._last_processed_time = current_time
            else:
                logger.error("Не вдалося відкрити ворота")
        else:
            logger.warning(f"В'їзд: доступ заборонено для {plate_number}")
            self.stats['access_denied'] += 1

            # Записуємо спробу
            self.sheets.log_denied_access(plate_number)

            # Оновлюємо час обробки
            self._last_processed_time = current_time


class ExitProcessor(VehicleProcessor):
    """Процесор для виїзду"""

    def __init__(self, *args, **kwargs):
        """Ініціалізація процесора виїзду"""
        super().__init__(*args, **kwargs)

        # Для відстеження обробленого автомобіля
        self._last_processed_time = 0
        self._processing_cooldown = 5  # секунд між обробками
        self._no_plate_timeout = 3  # час очікування якщо немає номера
        self._vehicle_first_seen = None

        logger.info("Процесор виїзду готовий")

    def process_frame(self, frame):
        """Обробити кадр для виїзду"""
        # Детекція
        results = self.detection.process_frame(frame, self.camera_type)

        # Якщо немає автомобілів
        if not results['vehicles']:
            self._vehicle_first_seen = None
            return

        current_time = time.time()

        # Якщо це новий автомобіль
        if self._vehicle_first_seen is None:
            self._vehicle_first_seen = current_time
            self.stats['vehicles_detected'] += 1

        # Перевіряємо чи не обробляли нещодавно
        if current_time - self._last_processed_time < self._processing_cooldown:
            return

        # Якщо є розпізнаний номер
        if results['recognized_numbers']:
            plate_number = results['recognized_numbers'][0]
            self.stats['plates_recognized'] += 1

            logger.info(f"Виїзд: розпізнано номер {plate_number}")

            # Записуємо виїзд
            self.sheets.log_exit(plate_number)

            # Відкриваємо ворота
            self._open_gate_for_exit()

        else:
            # Якщо немає номера, але автомобіль чекає достатньо довго
            if current_time - self._vehicle_first_seen >= self._no_plate_timeout:
                logger.info("Виїзд: відкриваємо ворота без розпізнавання номера")

                # Записуємо виїзд без номера
                self.sheets.log_exit(None)

                # Відкриваємо ворота
                self._open_gate_for_exit()

    def _open_gate_for_exit(self):
        """Відкрити ворота для виїзду"""
        if self.gate_controller.open_gate():
            self._last_processed_time = time.time()
            self._vehicle_first_seen = None
            self.stats['access_granted'] += 1
        else:
            logger.error("Не вдалося відкрити ворота для виїзду")


class ProcessManager:
    """Менеджер процесів обробки"""

    def __init__(self,
                 hardware: HardwareManager,
                 gate_controller: GateController,
                 detection_pipeline: DetectionPipeline,
                 sheets_manager: GoogleSheetsManager):
        """Ініціалізація менеджера"""
        self.hardware = hardware
        self.gate_controller = gate_controller
        self.detection = detection_pipeline
        self.sheets = sheets_manager

        # Створюємо процесори
        self.entrance_processor = EntranceProcessor(
            hardware, gate_controller, detection_pipeline,
            sheets_manager, "ENTRANCE"
        )

        self.exit_processor = ExitProcessor(
            hardware, gate_controller, detection_pipeline,
            sheets_manager, "EXIT"
        )

        logger.info("Менеджер процесів ініціалізовано")

    def start(self):
        """Запустити всі процеси"""
        logger.info("Запуск процесів обробки")

        self.entrance_processor.start()
        self.exit_processor.start()

        logger.info("Процеси обробки запущені")

    def stop(self):
        """Зупинити всі процеси"""
        logger.info("Зупинка процесів обробки")

        self.entrance_processor.stop()
        self.exit_processor.stop()

        logger.info("Процеси обробки зупинені")

    def get_stats(self) -> dict:
        """Отримати загальну статистику"""
        return {
            'entrance': self.entrance_processor.get_stats(),
            'exit': self.exit_processor.get_stats(),
            'gate_state': self.gate_controller.get_state().value
        }