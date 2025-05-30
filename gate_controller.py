"""
Модуль для управління логікою роботи воріт
"""

import time
import logging
import threading
from enum import Enum
from typing import Optional, Callable
from datetime import datetime, timedelta

from hardware import HardwareManager
from config import TIMING_CONFIG, ULTRASONIC_CONFIG

logger = logging.getLogger(__name__)


class GateState(Enum):
    """Стани воріт"""
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    ERROR = "error"


class VehiclePassageDetector:
    """Детектор проїзду автомобіля через ворота"""

    def __init__(self, ultrasonic_sensor):
        """Ініціалізація детектора"""
        self.sensor = ultrasonic_sensor
        self.baseline_distance = ULTRASONIC_CONFIG["normal_distance"]
        self.threshold = ULTRASONIC_CONFIG["detection_threshold"]
        self._monitoring = False
        self._passage_callback = None
        self._monitor_thread = None
        self._stop_event = threading.Event()

        # Стан детекції
        self._vehicle_detected = False
        self._last_detection_time = None

        logger.info("Детектор проїзду ініціалізовано")

    def start_monitoring(self, callback: Callable):
        """Почати моніторинг проїзду"""
        if self._monitoring:
            logger.warning("Моніторинг вже запущено")
            return

        self._monitoring = True
        self._passage_callback = callback
        self._stop_event.clear()
        self._vehicle_detected = False

        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        logger.info("Моніторинг проїзду розпочато")

    def stop_monitoring(self):
        """Зупинити моніторинг"""
        if not self._monitoring:
            return

        self._monitoring = False
        self._stop_event.set()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)

        logger.info("Моніторинг проїзду зупинено")

    def _monitor_loop(self):
        """Основний цикл моніторингу"""
        while self._monitoring and not self._stop_event.is_set():
            try:
                distance = self.sensor.get_distance()

                if distance is None:
                    logger.warning("Не вдалося отримати відстань")
                    time.sleep(0.1)
                    continue

                # Детекція автомобіля
                vehicle_present = distance < self.threshold

                # Логіка детекції проїзду
                if vehicle_present and not self._vehicle_detected:
                    # Автомобіль з'явився
                    self._vehicle_detected = True
                    self._last_detection_time = time.time()
                    logger.info(f"Автомобіль виявлено на відстані {distance} см")

                elif not vehicle_present and self._vehicle_detected:
                    # Автомобіль проїхав
                    self._vehicle_detected = False
                    passage_time = time.time() - self._last_detection_time
                    logger.info(f"Автомобіль проїхав за {passage_time:.1f} сек")

                    if self._passage_callback:
                        self._passage_callback()

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Помилка в циклі моніторингу: {e}")
                time.sleep(0.5)

    def is_vehicle_present(self) -> bool:
        """Перевірити чи є автомобіль в зоні датчика"""
        return self._vehicle_detected


class GateController:
    """Основний контролер воріт"""

    def __init__(self, hardware: HardwareManager):
        """Ініціалізація контролера"""
        self.hardware = hardware
        self.state = GateState.CLOSED
        self._state_lock = threading.Lock()

        # Детектор проїзду
        self.passage_detector = VehiclePassageDetector(hardware.ultrasonic)

        # Таймер закриття
        self._close_timer = None
        self._timer_lock = threading.Lock()

        # Обробники подій
        self._event_handlers = {
            'gate_opened': [],
            'gate_closed': [],
            'vehicle_passed': [],
            'error': []
        }

        # Ініціалізація стану
        self._update_state()

        logger.info(f"Контролер воріт ініціалізовано. Стан: {self.state.value}")

    def _update_state(self):
        """Оновити стан воріт на основі датчиків"""
        try:
            if self.hardware.gpio.is_gate_open():
                self.state = GateState.OPEN
            else:
                self.state = GateState.CLOSED
        except Exception as e:
            logger.error(f"Помилка оновлення стану: {e}")
            self.state = GateState.ERROR

    def open_gate(self, wait_for_open: bool = True) -> bool:
        """Відкрити ворота"""
        with self._state_lock:
            if self.state == GateState.OPEN:
                logger.info("Ворота вже відкриті")
                return True

            if self.state == GateState.OPENING:
                logger.info("Ворота вже відкриваються")
                return True

            try:
                # Скасовуємо таймер закриття якщо є
                self._cancel_close_timer()

                # Змінюємо стан
                self.state = GateState.OPENING
                logger.info("Відкриття воріт...")

                # Подаємо сигнал
                self.hardware.gpio.open_gate()

                if wait_for_open:
                    # Чекаємо відкриття
                    start_time = time.time()
                    timeout = TIMING_CONFIG["gate_operation_timeout"]

                    while time.time() - start_time < timeout:
                        if self.hardware.gpio.is_gate_open():
                            self.state = GateState.OPEN
                            logger.info("Ворота відкриті")
                            self._trigger_event('gate_opened')

                            # Запускаємо моніторинг проїзду
                            self.passage_detector.start_monitoring(self._on_vehicle_passed)
                            return True

                        time.sleep(0.1)

                    logger.error("Таймаут відкриття воріт")
                    self.state = GateState.ERROR
                    self._trigger_event('error', "Таймаут відкриття")
                    return False

                return True

            except Exception as e:
                logger.error(f"Помилка відкриття воріт: {e}")
                self.state = GateState.ERROR
                self._trigger_event('error', str(e))
                return False

    def close_gate(self, wait_for_close: bool = True) -> bool:
        """Закрити ворота"""
        with self._state_lock:
            if self.state == GateState.CLOSED:
                logger.info("Ворота вже закриті")
                return True

            if self.state == GateState.CLOSING:
                logger.info("Ворота вже закриваються")
                return True

            try:
                # Зупиняємо моніторинг проїзду
                self.passage_detector.stop_monitoring()

                # Змінюємо стан
                self.state = GateState.CLOSING
                logger.info("Закриття воріт...")

                # Подаємо сигнал
                self.hardware.gpio.close_gate()

                if wait_for_close:
                    # Чекаємо закриття
                    start_time = time.time()
                    timeout = TIMING_CONFIG["gate_operation_timeout"]

                    while time.time() - start_time < timeout:
                        if not self.hardware.gpio.is_gate_open():
                            self.state = GateState.CLOSED
                            logger.info("Ворота закриті")
                            self._trigger_event('gate_closed')
                            return True

                        time.sleep(0.1)

                    logger.error("Таймаут закриття воріт")
                    self.state = GateState.ERROR
                    self._trigger_event('error', "Таймаут закриття")
                    return False

                return True

            except Exception as e:
                logger.error(f"Помилка закриття воріт: {e}")
                self.state = GateState.ERROR
                self._trigger_event('error', str(e))
                return False

    def _on_vehicle_passed(self):
        """Обробник проїзду автомобіля"""
        logger.info("Автомобіль проїхав через ворота")
        self._trigger_event('vehicle_passed')

        # Запускаємо таймер закриття
        self._start_close_timer()

    def _start_close_timer(self):
        """Запустити таймер закриття воріт"""
        with self._timer_lock:
            # Скасовуємо попередній таймер
            self._cancel_close_timer()

            # Створюємо новий
            delay = TIMING_CONFIG["gate_close_delay"]
            self._close_timer = threading.Timer(delay, self._on_close_timer)
            self._close_timer.start()

            logger.info(f"Таймер закриття запущено на {delay} сек")

    def _cancel_close_timer(self):
        """Скасувати таймер закриття"""
        with self._timer_lock:
            if self._close_timer and self._close_timer.is_alive():
                self._close_timer.cancel()
                logger.info("Таймер закриття скасовано")
            self._close_timer = None

    def _on_close_timer(self):
        """Обробник спрацювання таймера закриття"""
        # Перевіряємо чи немає автомобіля в зоні
        if self.passage_detector.is_vehicle_present():
            logger.warning("Автомобіль в зоні воріт, відкладаємо закриття")
            # Чекаємо поки автомобіль проїде
            self._start_close_timer()
        else:
            logger.info("Закриття воріт за таймером")
            self.close_gate()

    def emergency_stop(self):
        """Аварійна зупинка"""
        logger.warning("Аварійна зупинка системи")

        # Скасовуємо всі таймери
        self._cancel_close_timer()

        # Зупиняємо моніторинг
        self.passage_detector.stop_monitoring()

        # Змінюємо стан
        with self._state_lock:
            self.state = GateState.ERROR

        self._trigger_event('error', "Аварійна зупинка")

    def register_handler(self, event: str, handler: Callable):
        """Зареєструвати обробник події"""
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)
            logger.debug(f"Зареєстровано обробник для події {event}")

    def _trigger_event(self, event: str, *args, **kwargs):
        """Викликати обробники події"""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Помилка в обробнику події {event}: {e}")

    def get_state(self) -> GateState:
        """Отримати поточний стан воріт"""
        return self.state

    def is_safe_to_operate(self) -> bool:
        """Перевірити чи безпечно керувати воротами"""
        # Перевіряємо стан
        if self.state == GateState.ERROR:
            return False

        # Перевіряємо чи немає автомобіля в зоні
        if self.passage_detector.is_vehicle_present():
            logger.warning("Автомобіль в зоні воріт")
            return False

        return True


class SafetyMonitor:
    """Монітор безпеки системи"""

    def __init__(self, hardware: HardwareManager, gate_controller: GateController):
        """Ініціалізація монітора"""
        self.hardware = hardware
        self.gate_controller = gate_controller
        self._monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()

        # Лічильники для таймаутів
        self._sensor_timeouts = {
            'ultrasonic': 0,
            'magnetic': 0,
            'camera': 0
        }

        self._max_timeouts = 3
        self._check_interval = 5  # секунд

        logger.info("Монітор безпеки ініціалізовано")

    def start(self):
        """Запустити моніторинг"""
        if self._monitoring:
            return

        self._monitoring = True
        self._stop_event.clear()

        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        logger.info("Моніторинг безпеки запущено")

    def stop(self):
        """Зупинити моніторинг"""
        if not self._monitoring:
            return

        self._monitoring = False
        self._stop_event.set()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)

        logger.info("Моніторинг безпеки зупинено")

    def _monitor_loop(self):
        """Основний цикл моніторингу"""
        while self._monitoring and not self._stop_event.is_set():
            try:
                # Перевіряємо датчики
                self._check_sensors()

                # Перевіряємо критичні таймаути
                if any(count >= self._max_timeouts for count in self._sensor_timeouts.values()):
                    logger.critical("Критичний збій датчиків!")
                    self.gate_controller.emergency_stop()
                    self._monitoring = False
                    break

                self._stop_event.wait(self._check_interval)

            except Exception as e:
                logger.error(f"Помилка в моніторі безпеки: {e}")

    def _check_sensors(self):
        """Перевірити роботу датчиків"""
        # Перевірка ультразвукового датчика
        try:
            distance = self.hardware.ultrasonic.get_distance()
            if distance is None:
                self._sensor_timeouts['ultrasonic'] += 1
                logger.warning(f"Таймаут ультразвукового датчика ({self._sensor_timeouts['ultrasonic']})")
            else:
                self._sensor_timeouts['ultrasonic'] = 0
        except Exception as e:
            self._sensor_timeouts['ultrasonic'] += 1
            logger.error(f"Помилка ультразвукового датчика: {e}")

        # Перевірка магнітного датчика
        try:
            _ = self.hardware.gpio.is_gate_open()
            self._sensor_timeouts['magnetic'] = 0
        except Exception as e:
            self._sensor_timeouts['magnetic'] += 1
            logger.error(f"Помилка магнітного датчика: {e}")

        # Перевірка камер
        try:
            for camera_type in ['ENTRANCE', 'EXIT']:
                frame = self.hardware.camera.capture_frame(camera_type)
                if frame is None:
                    self._sensor_timeouts['camera'] += 1
                    logger.warning(f"Таймаут камери {camera_type}")
                else:
                    self._sensor_timeouts['camera'] = 0
        except Exception as e:
            self._sensor_timeouts['camera'] += 1
            logger.error(f"Помилка камер: {e}")