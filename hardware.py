"""
Модуль для роботи з апаратним забезпеченням:
- GPIO (реле, герконовий датчик)
- Ультразвуковий датчик JSN-SR04T
- Камери
"""

import time
import logging
import threading
from typing import Optional, Tuple
import numpy as np
import lgpio
from picamera2 import Picamera2
from libcamera import controls

from config import GPIO_CONFIG, ULTRASONIC_CONFIG, TIMING_CONFIG, CAMERA_CONFIG

logger = logging.getLogger(__name__)


class GPIOController:
    """Контролер для роботи з GPIO пінами"""

    def __init__(self):
        """Ініціалізація GPIO"""
        try:
            # Відкриваємо чіп GPIO
            self.chip = lgpio.gpiochip_open(0)

            # Налаштування пінів реле як виходи
            lgpio.gpio_claim_output(self.chip, GPIO_CONFIG["OPEN_RELAY"], lgpio.LOW)
            lgpio.gpio_claim_output(self.chip, GPIO_CONFIG["CLOSE_RELAY"], lgpio.LOW)

            # Налаштування герконового датчика як вхід з підтягуючим резистором
            lgpio.gpio_claim_input(self.chip, GPIO_CONFIG["MAGNETIC_SENSOR"], lgpio.SET_PULL_UP)

            logger.info("GPIO ініціалізовано")

        except Exception as e:
            logger.error(f"Помилка ініціалізації GPIO: {e}")
            raise

    def pulse_relay(self, relay_type: str, duration: float = None):
        """Подати імпульс на реле"""
        if duration is None:
            duration = TIMING_CONFIG["relay_pulse_duration"]

        try:
            if relay_type == "open":
                pin = GPIO_CONFIG["OPEN_RELAY"]
            elif relay_type == "close":
                pin = GPIO_CONFIG["CLOSE_RELAY"]
            else:
                raise ValueError(f"Невідомий тип реле: {relay_type}")

            lgpio.gpio_write(self.chip, pin, lgpio.HIGH)
            logger.debug(f"Реле {relay_type} увімкнено")
            time.sleep(duration)
            lgpio.gpio_write(self.chip, pin, lgpio.LOW)
            logger.debug(f"Реле {relay_type} вимкнено")

        except Exception as e:
            logger.error(f"Помилка керування реле {relay_type}: {e}")
            raise

    def open_gate(self):
        """Відкрити ворота"""
        logger.info("Відкриття воріт")
        self.pulse_relay("open")

    def close_gate(self):
        """Закрити ворота"""
        logger.info("Закриття воріт")
        self.pulse_relay("close")

    def is_gate_open(self) -> bool:
        """Перевірити чи відкриті ворота"""
        # Магніт зімкнутий (LOW) = ворота відкриті
        state = lgpio.gpio_read(self.chip, GPIO_CONFIG["MAGNETIC_SENSOR"])
        return state == lgpio.LOW

    def cleanup(self):
        """Очистити GPIO"""
        try:
            # Встановлюємо всі виходи в LOW
            lgpio.gpio_write(self.chip, GPIO_CONFIG["OPEN_RELAY"], lgpio.LOW)
            lgpio.gpio_write(self.chip, GPIO_CONFIG["CLOSE_RELAY"], lgpio.LOW)

            # Звільняємо піни
            lgpio.gpio_free(self.chip, GPIO_CONFIG["OPEN_RELAY"])
            lgpio.gpio_free(self.chip, GPIO_CONFIG["CLOSE_RELAY"])
            lgpio.gpio_free(self.chip, GPIO_CONFIG["MAGNETIC_SENSOR"])

            # Закриваємо чіп
            lgpio.gpiochip_close(self.chip)

            logger.info("GPIO очищено")
        except Exception as e:
            logger.error(f"Помилка очищення GPIO: {e}")


class UltrasonicSensor:
    """Клас для роботи з ультразвуковим датчиком JSN-SR04T"""

    def __init__(self):
        """Ініціалізація датчика"""
        self.trig_pin = GPIO_CONFIG["ULTRASONIC_TRIG"]
        self.echo_pin = GPIO_CONFIG["ULTRASONIC_ECHO"]
        self.timeout = ULTRASONIC_CONFIG["measurement_timeout"]
        self.samples = ULTRASONIC_CONFIG["samples"]
        self._lock = threading.Lock()

        try:
            # Відкриваємо чіп GPIO (якщо ще не відкритий)
            self.chip = lgpio.gpiochip_open(0)

            # Налаштування пінів
            lgpio.gpio_claim_output(self.chip, self.trig_pin, lgpio.LOW)
            lgpio.gpio_claim_input(self.chip, self.echo_pin)

            logger.info("Ультразвуковий датчик ініціалізовано")
        except Exception as e:
            logger.error(f"Помилка ініціалізації ультразвукового датчика: {e}")
            raise

    def _measure_once(self) -> Optional[float]:
        """Одне вимірювання відстані"""
        try:
            # Очищаємо тригер
            lgpio.gpio_write(self.chip, self.trig_pin, lgpio.LOW)
            time.sleep(0.002)

            # Генеруємо імпульс
            lgpio.gpio_write(self.chip, self.trig_pin, lgpio.HIGH)
            time.sleep(0.00001)  # 10 мкс
            lgpio.gpio_write(self.chip, self.trig_pin, lgpio.LOW)

            # Чекаємо на відповідь
            pulse_start = time.time()
            timeout_start = pulse_start

            while lgpio.gpio_read(self.chip, self.echo_pin) == lgpio.LOW:
                pulse_start = time.time()
                if pulse_start - timeout_start > self.timeout:
                    return None

            pulse_end = time.time()
            timeout_start = pulse_end

            while lgpio.gpio_read(self.chip, self.echo_pin) == lgpio.HIGH:
                pulse_end = time.time()
                if pulse_end - timeout_start > self.timeout:
                    return None

            # Обчислюємо відстань
            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 17150  # Швидкість звуку / 2

            return round(distance, 2)

        except Exception as e:
            logger.error(f"Помилка вимірювання відстані: {e}")
            return None

    def get_distance(self) -> Optional[float]:
        """Отримати усереднену відстань в сантиметрах"""
        with self._lock:
            distances = []

            for _ in range(self.samples):
                distance = self._measure_once()
                if distance is not None and 2 < distance < 400:  # Діапазон JSN-SR04T
                    distances.append(distance)
                time.sleep(0.05)  # Затримка між вимірюваннями

            if not distances:
                return None

            # Видаляємо викиди та усереднюємо
            distances.sort()
            if len(distances) > 2:
                distances = distances[1:-1]  # Видаляємо мін та макс

            avg_distance = sum(distances) / len(distances)
            return round(avg_distance, 2)

    def is_vehicle_detected(self) -> bool:
        """Перевірити чи є машина в зоні датчика"""
        distance = self.get_distance()
        if distance is None:
            return False

        return distance < ULTRASONIC_CONFIG["detection_threshold"]

    def cleanup(self):
        """Очистити ресурси"""
        try:
            lgpio.gpio_write(self.chip, self.trig_pin, lgpio.LOW)
            lgpio.gpio_free(self.chip, self.trig_pin)
            lgpio.gpio_free(self.chip, self.echo_pin)
            lgpio.gpiochip_close(self.chip)
        except Exception as e:
            logger.error(f"Помилка очищення ультразвукового датчика: {e}")


class CameraController:
    """Контролер для роботи з камерами"""

    def __init__(self):
        """Ініціалізація камер"""
        self.cameras = {}
        self._init_cameras()

    def _init_cameras(self):
        """Ініціалізація всіх камер"""
        try:
            # Отримуємо список доступних камер
            available_cameras = Picamera2.global_camera_info()
            logger.info(f"Знайдено {len(available_cameras)} камер")

            # Ініціалізуємо камери за іменами
            for cam_info in available_cameras:
                cam_name = cam_info.get('Model', '').lower()

                if 'imx708' in cam_name and "ENTRANCE" not in self.cameras:
                    self._setup_camera("ENTRANCE", cam_info['Num'])
                elif 'imx219' in cam_name and "EXIT" not in self.cameras:
                    self._setup_camera("EXIT", cam_info['Num'])

            logger.info(f"Ініціалізовано камери: {list(self.cameras.keys())}")

        except Exception as e:
            logger.error(f"Помилка ініціалізації камер: {e}")
            raise

    def _setup_camera(self, camera_type: str, camera_num: int):
        """Налаштування окремої камери"""
        try:
            picam2 = Picamera2(camera_num)
            config = CAMERA_CONFIG[camera_type]

            # Конфігурація камери
            camera_config = picam2.create_preview_configuration(
                main={"size": config["resolution"], "format": "RGB888"},
                controls={
                    "FrameRate": config["fps"],
                    "ExposureTime": 20000,  # Автоекспозиція
                    "AwbMode": controls.AwbModeEnum.Auto
                }
            )

            picam2.configure(camera_config)
            picam2.start()

            # Даємо камері час на стабілізацію
            time.sleep(0.5)

            self.cameras[camera_type] = picam2
            logger.info(f"Камера {camera_type} ({camera_num}) налаштована")

        except Exception as e:
            logger.error(f"Помилка налаштування камери {camera_type}: {e}")
            raise

    def capture_frame(self, camera_type: str) -> Optional[np.ndarray]:
        """Захопити кадр з камери"""
        if camera_type not in self.cameras:
            logger.error(f"Камера {camera_type} не знайдена")
            return None

        try:
            frame = self.cameras[camera_type].capture_array()
            return frame
        except Exception as e:
            logger.error(f"Помилка захоплення кадру з {camera_type}: {e}")
            return None

    def cleanup(self):
        """Закрити всі камери"""
        for camera_type, camera in self.cameras.items():
            try:
                camera.stop()
                camera.close()
                logger.info(f"Камера {camera_type} закрита")
            except Exception as e:
                logger.error(f"Помилка закриття камери {camera_type}: {e}")


class HardwareManager:
    """Менеджер для керування всім обладнанням"""

    def __init__(self):
        """Ініціалізація менеджера"""
        self.gpio = GPIOController()
        self.ultrasonic = UltrasonicSensor()
        self.camera = CameraController()

        logger.info("Менеджер обладнання ініціалізовано")

    def cleanup(self):
        """Очистити всі ресурси"""
        logger.info("Очищення ресурсів обладнання")
        self.camera.cleanup()
        self.ultrasonic.cleanup()
        self.gpio.cleanup()