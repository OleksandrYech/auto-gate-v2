"""
Модуль для детекції автомобілів, номерних знаків та розпізнавання символів
"""

import cv2
import numpy as np
import onnxruntime as ort
import torch
import logging
import json
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dataclass import Detection

from config import MODEL_CONFIG, CONFIG_DIR, SYSTEM_CONFIG

logger = logging.getLogger(__name__)

    """Детектор автомобілів на основі MobileNet SSD"""

    def __init__(self):
        """Ініціалізація детектора"""
        self.model_path = MODEL_CONFIG["VEHICLE_DETECTION"]["path"]
        self.confidence_threshold = MODEL_CONFIG["VEHICLE_DETECTION"]["confidence_threshold"]
        self.target_classes = MODEL_CONFIG["VEHICLE_DETECTION"]["classes"]
        self.session = None

        # Класи COCO для MobileNet SSD
        self.coco_classes = {
            2: "car",
            5: "bus",
            7: "truck"
        }

        self._load_model()

    def _load_model(self):
        """Завантажити ONNX модель"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Модель не знайдена: {self.model_path}")

            # Налаштування ONNX Runtime
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)

            # Отримуємо інформацію про входи та виходи
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape

            logger.info(f"Модель детекції авто завантажена: {self.input_shape}")

        except Exception as e:
            logger.error(f"Помилка завантаження моделі детекції: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Підготовка зображення для моделі"""
        # Розмір входу для MobileNet SSD зазвичай 300x300
        input_size = (300, 300)

        # Змінюємо розмір
        resized = cv2.resize(image, input_size)

        # Нормалізація (зазвичай для MobileNet потрібна нормалізація 0-1)
        normalized = resized.astype(np.float32) / 255.0

        # Додаємо batch dimension та змінюємо порядок каналів якщо потрібно
        if len(normalized.shape) == 3:
            # HWC -> CHW
            normalized = np.transpose(normalized, (2, 0, 1))
            # Додаємо batch dimension
            normalized = np.expand_dims(normalized, axis=0)

        return normalized

    def postprocess(self, outputs: List[np.ndarray], image_shape: Tuple[int, int]) -> List[Detection]:
        """Обробка виходів моделі"""
        detections = []
        h, w = image_shape[:2]

        # Розбираємо виходи MobileNet SSD
        # Зазвичай формат: [batch, num_detections, 7]
        # де 7 = [image_id, class_id, confidence, x_min, y_min, x_max, y_max]
        if len(outputs) > 0:
            output = outputs[0]

            if len(output.shape) == 3:
                output = output[0]  # Видаляємо batch dimension

            for detection in output:
                if len(detection) >= 6:
                    class_id = int(detection[1]) if len(detection) > 6 else int(detection[0])
                    confidence = detection[2] if len(detection) > 6 else detection[1]

                    # Перевіряємо чи це потрібний клас
                    if class_id in self.coco_classes and confidence >= self.confidence_threshold:
                        class_name = self.coco_classes[class_id]
                        if class_name in self.target_classes:
                            # Координати bbox (нормалізовані 0-1)
                            if len(detection) > 6:
                                x1 = int(detection[3] * w)
                                y1 = int(detection[4] * h)
                                x2 = int(detection[5] * w)
                                y2 = int(detection[6] * h)
                            else:
                                x1 = int(detection[2] * w)
                                y1 = int(detection[3] * h)
                                x2 = int(detection[4] * w)
                                y2 = int(detection[5] * h)

                            # Перевіряємо коректність координат
                            if x1 < x2 and y1 < y2:
                                detections.append(Detection(
                                    bbox=(x1, y1, x2, y2),
                                    confidence=float(confidence),
                                    class_name=class_name
                                ))

        return detections

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Виявити автомобілі на зображенні"""
        if self.session is None:
            logger.error("Модель не завантажена")
            return []

        try:
            # Підготовка зображення
            input_tensor = self.preprocess(image)

            # Виконуємо інференс
            outputs = self.session.run(None, {self.input_name: input_tensor})

            # Обробка результатів
            detections = self.postprocess(outputs, image.shape[:2])

            # Сортуємо за впевненістю
            detections.sort(key=lambda x: x.confidence, reverse=True)

            return detections

        except Exception as e:
            logger.error(f"Помилка детекції авто: {e}")
            return []


class LicensePlateDetector:
    """Детектор номерних знаків"""

    def __init__(self):
        """Ініціалізація детектора"""
        self.model_path = MODEL_CONFIG["LICENSE_PLATE_DETECTION"]["path"]
        self.confidence_threshold = MODEL_CONFIG["LICENSE_PLATE_DETECTION"]["confidence_threshold"]
        self.input_shape = MODEL_CONFIG["LICENSE_PLATE_DETECTION"]["input_shape"]
        self.session = None

        self._load_model()

    def _load_model(self):
        """Завантажити ONNX модель"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Модель не знайдена: {self.model_path}")

            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)

            self.input_name = self.session.get_inputs()[0].name
            logger.info("Модель детекції номерів завантажена")

        except Exception as e:
            logger.error(f"Помилка завантаження моделі номерів: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Підготовка зображення для моделі"""
        # Змінюємо розмір до вхідного розміру моделі
        resized = cv2.resize(image, self.input_shape)

        # Нормалізація
        normalized = resized.astype(np.float32) / 255.0

        # CHW формат
        if len(normalized.shape) == 3:
            normalized = np.transpose(normalized, (2, 0, 1))

        # Додаємо batch dimension
        normalized = np.expand_dims(normalized, axis=0)

        return normalized

    def detect(self, image: np.ndarray, vehicle_bbox: Optional[Tuple[int, int, int, int]] = None) -> List[Detection]:
        """Виявити номерні знаки"""
        if self.session is None:
            return []

        try:
            # Якщо є bbox авто, обрізаємо зображення
            if vehicle_bbox:
                x1, y1, x2, y2 = vehicle_bbox
                # Додаємо відступи
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)

                cropped = image[y1:y2, x1:x2]
                offset = (x1, y1)
            else:
                cropped = image
                offset = (0, 0)

            # Підготовка та інференс
            input_tensor = self.preprocess(cropped)
            outputs = self.session.run(None, {self.input_name: input_tensor})

            # Обробка результатів
            detections = []
            if outputs and len(outputs) > 0:
                # Припускаємо формат виходу подібний до YOLO
                predictions = outputs[0]

                # Розбираємо детекції
                for pred in predictions[0]:
                    if len(pred) >= 5:
                        confidence = float(pred[4])
                        if confidence >= self.confidence_threshold:
                            # Координати відносно обрізаного зображення
                            cx, cy, w, h = pred[:4]
                            x1 = int((cx - w/2) * cropped.shape[1]) + offset[0]
                            y1 = int((cy - h/2) * cropped.shape[0]) + offset[1]
                            x2 = int((cx + w/2) * cropped.shape[1]) + offset[0]
                            y2 = int((cy + h/2) * cropped.shape[0]) + offset[1]

                            detections.append(Detection(
                                bbox=(x1, y1, x2, y2),
                                confidence=confidence,
                                class_name="license_plate"
                            ))

            return detections

        except Exception as e:
            logger.error(f"Помилка детекції номерів: {e}")
            return []


class OCRRecognizer:
    """Розпізнавач символів на номерних знаках"""

    def __init__(self):
        """Ініціалізація OCR"""
        self.model_path = MODEL_CONFIG["OCR"]["path"]
        self.confidence_threshold = MODEL_CONFIG["OCR"]["confidence_threshold"]
        self.img_size = MODEL_CONFIG["OCR"]["img_size"]
        self.model = None

        # Символи для українських номерів
        self.alphabet = "ABCEHIKMOPTX0123456789"

        self._load_model()

    def _load_model(self):
        """Завантажити PyTorch модель"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Модель не знайдена: {self.model_path}")

            # Завантажуємо модель
            self.model = torch.jit.load(str(self.model_path), map_location='cpu')
            self.model.eval()

            logger.info("OCR модель завантажена")

        except Exception as e:
            logger.error(f"Помилка завантаження OCR моделі: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Підготовка зображення для OCR"""
        # Конвертуємо в відтінки сірого якщо потрібно
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Змінюємо розмір
        resized = cv2.resize(gray, (self.img_size, self.img_size))

        # Нормалізація
        normalized = resized.astype(np.float32) / 255.0

        # Конвертуємо в тензор
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

        return tensor

    def recognize(self, image: np.ndarray) -> Optional[str]:
        """Розпізнати текст на зображенні"""
        if self.model is None:
            return None

        try:
            # Підготовка зображення
            input_tensor = self.preprocess(image)

            # Виконуємо розпізнавання
            with torch.no_grad():
                output = self.model(input_tensor)

            # Декодуємо результат
            text = self._decode_output(output)

            # Валідація номера
            if self._validate_plate(text):
                return text

            return None

        except Exception as e:
            logger.error(f"Помилка OCR: {e}")
            return None

    def _decode_output(self, output: torch.Tensor) -> str:
        """Декодувати вихід моделі в текст"""
        # Тут має бути логіка декодування специфічна для вашої моделі
        # Це приклад для CTC декодування
        _, predicted = output.max(2)
        predicted = predicted.squeeze(0).cpu().numpy()

        # Видаляємо повтори та blank токени
        text = []
        prev_char = None
        for idx in predicted:
            if idx < len(self.alphabet) and idx != prev_char:
                text.append(self.alphabet[idx])
            prev_char = idx

        return ''.join(text)

    def _validate_plate(self, text: str) -> bool:
        """Перевірити валідність номера"""
        if not text or len(text) < 6:
            return False

        # Базова перевірка для українських номерів
        # Приклад: AA1234BB
        if len(text) == 8:
            # Перші 2 - літери, наступні 4 - цифри, останні 2 - літери
            if (text[:2].isalpha() and
                text[2:6].isdigit() and
                text[6:].isalpha()):
                return True

        # Можна додати інші формати номерів
        return False


class DetectionPipeline:
    """Повний пайплайн детекції та розпізнавання"""

    def __init__(self):
        """Ініціалізація пайплайну"""
        self.vehicle_detector = VehicleDetector()
        self.plate_detector = LicensePlateDetector()
        self.ocr = OCRRecognizer()
        self.roi_manager = ROIManager()

        logger.info("Пайплайн детекції ініціалізовано")

    def process_frame(self, frame: np.ndarray, camera_type: str) -> Dict:
        """Обробити кадр та повернути результати"""
        results = {
            "vehicles": [],
            "plates": [],
            "recognized_numbers": []
        }

        try:
            # Застосовуємо ROI якщо є
            roi_frame, roi_offset = self.roi_manager.apply_roi(frame, camera_type)

            # Детекція автомобілів
            vehicles = self.vehicle_detector.detect(roi_frame)

            # Коригуємо координати з урахуванням ROI
            for vehicle in vehicles:
                x1, y1, x2, y2 = vehicle.bbox
                vehicle.bbox = (
                    x1 + roi_offset[0],
                    y1 + roi_offset[1],
                    x2 + roi_offset[0],
                    y2 + roi_offset[1]
                )

            results["vehicles"] = vehicles

            # Для кожного авто шукаємо номерний знак
            for vehicle in vehicles:
                plates = self.plate_detector.detect(frame, vehicle.bbox)

                for plate in plates:
                    results["plates"].append(plate)

                    # Вирізаємо номерний знак
                    x1, y1, x2, y2 = plate.bbox
                    plate_img = frame[y1:y2, x1:x2]

                    # Розпізнаємо текст
                    text = self.ocr.recognize(plate_img)
                    if text:
                        plate.text = text
                        results["recognized_numbers"].append(text)
                        logger.info(f"Розпізнано номер: {text}")

            # Зберігаємо зображення для налагодження
            if SYSTEM_CONFIG["save_images"] and results["recognized_numbers"]:
                self._save_debug_image(frame, results, camera_type)

        except Exception as e:
            logger.error(f"Помилка обробки кадру: {e}")

        return results

    def _save_debug_image(self, frame: np.ndarray, results: Dict, camera_type: str):
        """Зберегти зображення з анотаціями для налагодження"""
        try:
            annotated = frame.copy()

            # Малюємо bbox автомобілів
            for vehicle in results["vehicles"]:
                x1, y1, x2, y2 = vehicle.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{vehicle.class_name} {vehicle.confidence:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Малюємо bbox номерів
            for plate in results["plates"]:
                x1, y1, x2, y2 = plate.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                if plate.text:
                    cv2.putText(annotated, plate.text,
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Зберігаємо
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = SYSTEM_CONFIG["images_dir"] / f"{camera_type}_{timestamp}.jpg"
            cv2.imwrite(str(filename), annotated)

        except Exception as e:
            logger.error(f"Помилка збереження зображення: {e}")es import dataclass

from config import MODEL_CONFIG, CONFIG_DIR, SYSTEM_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Клас для збереження результатів детекції"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str = ""
    text: str = ""


class ROIManager:
    """Менеджер для роботи з зонами інтересу (ROI)"""

    def __init__(self):
        """Ініціалізація менеджера ROI"""
        self.roi_configs = {}
        self._load_roi_configs()

    def _load_roi_configs(self):
        """Завантажити конфігурації ROI"""
        for camera_type in ["ENTRANCE", "EXIT"]:
            roi_file = CONFIG_DIR / f"roi_{camera_type.lower()}.json"
            if roi_file.exists():
                try:
                    with open(roi_file, 'r') as f:
                        self.roi_configs[camera_type] = json.load(f)
                    logger.info(f"ROI для {camera_type} завантажено")
                except Exception as e:
                    logger.error(f"Помилка завантаження ROI для {camera_type}: {e}")
            else:
                logger.warning(f"ROI для {camera_type} не знайдено, використовуємо весь кадр")

    def apply_roi(self, image: np.ndarray, camera_type: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Застосувати ROI до зображення"""
        if camera_type not in self.roi_configs:
            return image, (0, 0)

        roi = self.roi_configs[camera_type]
        x1, y1, x2, y2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']

        # Обрізаємо зображення
        cropped = image[y1:y2, x1:x2]
        return cropped, (x1, y1)

    def get_roi_mask(self, image_shape: Tuple[int, int], camera_type: str) -> Optional[np.ndarray]:
        """Отримати маску ROI"""
        if camera_type not in self.roi_configs:
            return None

        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        roi = self.roi_configs[camera_type]

        # Створюємо маску з полігону або прямокутника
        if 'polygon' in roi:
            pts = np.array(roi['polygon'], np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            x1, y1, x2, y2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']
            mask[y1:y2, x1:x2] = 255

        return mask


class