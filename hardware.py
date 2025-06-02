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
from gpiozero import OutputDevice, InputDevice, DistanceSensor
from gpiozero.pins.native import NativeFactory
from picamera2 import Picamera2
from libcamera import controls

from config import GPIO_CONFIG, ULTRASONIC_CONFIG, TIMING_CONFIG, CAMERA_CONFIG

logger = logging.getLogger(__name__)

# Використовуємо нативний драйвер для Raspberry Pi 5
from gpiozero import Device
Device.pin_factory = NativeFactory()


class GPIOController:
    """Контролер для роботи з GPIO пінами"""

    def __init__(self):
        """Ініціалізація GPIO"""
        try:
            # Налаштування пінів реле як виходи
            self.open_relay = OutputDevice(GPIO_CONFIG["OPEN_RELAY"], active_high=True, initial_value=False)
            self.close_relay = OutputDevice(GPIO_CONFIG["CLOSE_RELAY"], active_high=True, initial_value=False)

            # Налаштування герконового датчика як вхід
            # pull_up=True означає що коли контакт розімкнутий, читається HIGH
            self.magnetic_sensor = InputDevice(GPIO_CONFIG["MAGNETIC_SENSOR"], pull_up=True)

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
                relay = self.open_relay
            elif relay_type == "close":
                relay = self.close_relay
            else:
                raise ValueError(f"Невідомий тип реле: {relay_type}")

            relay.on()
            logger.debug(f"Реле {relay_type} увімкнено")
            time.sleep(duration)
            relay.off()
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
        # Магніт зімкнутий (is_active=False через pull_up) = ворота відкриті
        return not self.magnetic_sensor.is_active

    def cleanup(self):
        """Очистити GPIO"""
        try:
            self.open_relay.close()
            self.close_relay.close()
            self.magnetic_sensor.close()
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
            # Використовуємо DistanceSensor з gpiozero
            self.sensor = DistanceSensor(
                echo=self.echo_pin,
                trigger=self.trig_pin,
                max_distance=4,  # максимальна відстань 4 метри
                threshold_distance=ULTRASONIC_CONFIG["detection_threshold"] / 100  # в метрах
            )
            logger.info("Ультразвуковий датчик ініціалізовано")
        except Exception as e:
            logger.error(f"Помилка ініціалізації ультразвукового датчика: {e}")
            raise

    def get_distance(self) -> Optional[float]:
        """Отримати усереднену відстань в сантиметрах"""
        with self._lock:
            distances = []

            for _ in range(self.samples):
                try:
                    # distance повертає значення в метрах
                    distance_m = self.sensor.distance
                    if distance_m is not None:
                        distance_cm = distance_m * 100  # конвертуємо в сантиметри
                        if 2 < distance_cm < 400:  # Діапазон JSN-SR04T
                            distances.append(distance_cm)
                except Exception as e:
                    logger.debug(f"Помилка вимірювання: {e}")

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
            self.sensor.close()
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