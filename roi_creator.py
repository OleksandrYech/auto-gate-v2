#!/usr/bin/env python3
"""
Утиліта для створення зон інтересу (ROI) для камер
"""

import cv2
import json
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import List, Tuple, Optional

from hardware import CameraController
from config import CONFIG_DIR, CAMERA_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ROICreator:
    """Клас для створення ROI"""

    def __init__(self, camera_type: str):
        """Ініціалізація"""
        self.camera_type = camera_type
        self.camera = CameraController()

        # Параметри ROI
        self.points = []
        self.current_frame = None
        self.roi_type = "rectangle"  # rectangle або polygon
        self.drawing = False
        self.roi_complete = False

        # Параметри вікна
        self.window_name = f"ROI Creator - {camera_type}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        logger.info(f"ROI Creator для {camera_type} ініціалізовано")

    def _mouse_callback(self, event, x, y, flags, param):
        """Обробник подій миші"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.roi_type == "rectangle" and len(self.points) == 0:
                # Початок прямокутника
                self.points = [(x, y)]
                self.drawing = True
            elif self.roi_type == "polygon":
                # Додаємо точку полігону
                self.points.append((x, y))
                if len(self.points) > 2:
                    # Перевіряємо чи клікнули біля першої точки (замикання)
                    if self._distance(self.points[0], (x, y)) < 10:
                        self.points[-1] = self.points[0]  # Замикаємо
                        self.roi_complete = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.roi_type == "rectangle" and self.drawing and len(self.points) == 1:
                # Оновлюємо другу точку прямокутника
                if len(self.points) == 2:
                    self.points[1] = (x, y)
                else:
                    self.points.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            if self.roi_type == "rectangle" and self.drawing:
                # Завершуємо прямокутник
                if len(self.points) == 2:
                    self.roi_complete = True
                self.drawing = False

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Обчислити відстань між точками"""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _draw_roi(self, frame: np.ndarray) -> np.ndarray:
        """Намалювати ROI на кадрі"""
        display = frame.copy()

        if self.roi_type == "rectangle" and len(self.points) >= 1:
            if len(self.points) == 2:
                # Малюємо прямокутник
                cv2.rectangle(display, self.points[0], self.points[1], (0, 255, 0), 2)
            elif self.drawing:
                # Малюємо тимчасовий прямокутник під час малювання
                current_pos = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE)

        elif self.roi_type == "polygon" and len(self.points) > 0:
            # Малюємо точки
            for i, point in enumerate(self.points):
                cv2.circle(display, point, 5, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(display, self.points[i - 1], point, (0, 255, 0), 2)

            # Замикаємо полігон якщо завершено
            if self.roi_complete and len(self.points) > 2:
                cv2.line(display, self.points[-1], self.points[0], (0, 255, 0), 2)

        # Додаємо інструкції
        instructions = [
            f"Тип ROI: {self.roi_type} (натисніть 'r' для прямокутника, 'p' для полігону)",
            "Ліва кнопка миші - додати точку",
            "'c' - очистити, 's' - зберегти, 'q' - вийти"
        ]

        for i, text in enumerate(instructions):
            cv2.putText(display, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return display

    def create_roi(self) -> Optional[dict]:
        """Створити ROI"""
        logger.info("Починаємо створення ROI")
        logger.info("Інструкції:")
        logger.info("- 'r' - режим прямокутника")
        logger.info("- 'p' - режим полігону")
        logger.info("- 'c' - очистити поточний ROI")
        logger.info("- 's' - зберегти ROI")
        logger.info("- 'q' - вийти без збереження")

        while True:
            # Захоплюємо кадр
            frame = self.camera.capture_frame(self.camera_type)
            if frame is None:
                logger.error("Не вдалося захопити кадр")
                break

            self.current_frame = frame

            # Малюємо ROI
            display = self._draw_roi(frame)

            # Показуємо
            cv2.imshow(self.window_name, display)

            # Обробка клавіш
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info("Вихід без збереження")
                break

            elif key == ord('r'):
                # Режим прямокутника
                self.roi_type = "rectangle"
                self.points = []
                self.roi_complete = False
                logger.info("Режим прямокутника")

            elif key == ord('p'):
                # Режим полігону
                self.roi_type = "polygon"
                self.points = []
                self.roi_complete = False
                logger.info("Режим полігону")

            elif key == ord('c'):
                # Очистити
                self.points = []
                self.roi_complete = False
                logger.info("ROI очищено")

            elif key == ord('s'):
                # Зберегти
                if self.roi_complete and len(self.points) >= 2:
                    roi_data = self._create_roi_data()
                    if roi_data:
                        self._save_roi(roi_data)
                        logger.info("ROI збережено")
                        cv2.destroyAllWindows()
                        return roi_data
                else:
                    logger.warning("ROI не завершено")

        cv2.destroyAllWindows()
        return None

    def _create_roi_data(self) -> Optional[dict]:
        """Створити дані ROI"""
        if not self.points:
            return None

        roi_data = {
            "camera_type": self.camera_type,
            "type": self.roi_type,
            "frame_size": [self.current_frame.shape[1], self.current_frame.shape[0]]
        }

        if self.roi_type == "rectangle" and len(self.points) == 2:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]

            # Нормалізуємо координати
            roi_data["x1"] = min(x1, x2)
            roi_data["y1"] = min(y1, y2)
            roi_data["x2"] = max(x1, x2)
            roi_data["y2"] = max(y1, y2)

        elif self.roi_type == "polygon" and len(self.points) > 2:
            roi_data["polygon"] = self.points

        else:
            return None

        return roi_data

    def _save_roi(self, roi_data: dict):
        """Зберегти ROI у файл"""
        filename = CONFIG_DIR / f"roi_{self.camera_type.lower()}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(roi_data, f, indent=4)
            logger.info(f"ROI збережено у {filename}")
        except Exception as e:
            logger.error(f"Помилка збереження ROI: {e}")

    def cleanup(self):
        """Очистити ресурси"""
        cv2.destroyAllWindows()
        self.camera.cleanup()


def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(description="Створення ROI для камер")
    parser.add_argument(
        "camera_type",
        choices=["ENTRANCE", "EXIT"],
        help="Тип камери"
    )

    args = parser.parse_args()

    # Створюємо ROI
    creator = ROICreator(args.camera_type)

    try:
        roi_data = creator.create_roi()
        if roi_data:
            print(f"\nROI створено успішно:")
            print(json.dumps(roi_data, indent=2))
        else:
            print("\nROI не створено")

    except KeyboardInterrupt:
        print("\nПерервано користувачем")

    except Exception as e:
        logger.error(f"Помилка: {e}")

    finally:
        creator.cleanup()


if __name__ == "__main__":
    main()