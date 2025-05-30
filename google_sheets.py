"""
Модуль для роботи з Google Sheets API
"""

import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import threading

from config import GOOGLE_SHEETS_CONFIG, TIMING_CONFIG

logger = logging.getLogger(__name__)


class GoogleSheetsManager:
    """Менеджер для роботи з Google Sheets"""

    def __init__(self):
        """Ініціалізація з'єднання з Google Sheets"""
        self.spreadsheet_id = GOOGLE_SHEETS_CONFIG["spreadsheet_id"]
        self.credentials = None
        self.service = None
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._last_cache_update = 0
        self._cache_lifetime = 300  # 5 хвилин
        self._recent_plates = {}  # Для відстеження дублікатів
        self._recent_plates_lock = threading.Lock()

        self._init_service()

    def _init_service(self):
        """Ініціалізація сервісу Google Sheets"""
        try:
            # Завантажуємо облікові дані
            self.credentials = service_account.Credentials.from_service_account_file(
                str(GOOGLE_SHEETS_CONFIG["credentials_file"]),
                scopes=GOOGLE_SHEETS_CONFIG["scopes"]
            )

            # Створюємо сервіс
            self.service = build('sheets', 'v4', credentials=self.credentials)

            logger.info("Google Sheets сервіс ініціалізовано")

            # Початкове завантаження кешу
            self._update_cache()

        except Exception as e:
            logger.error(f"Помилка ініціалізації Google Sheets: {e}")
            raise

    def _update_cache(self):
        """Оновити кеш дозволених номерів"""
        try:
            with self._cache_lock:
                # Отримуємо дозволені номери
                result = self.service.spreadsheets().values().get(
                    spreadsheetId=self.spreadsheet_id,
                    range=GOOGLE_SHEETS_CONFIG["ranges"]["allowed_plates"]
                ).execute()

                values = result.get('values', [])
                self._cache['allowed_plates'] = [row[0] for row in values if row]
                self._last_cache_update = time.time()

                logger.info(f"Кеш оновлено: {len(self._cache['allowed_plates'])} номерів")

        except HttpError as e:
            logger.error(f"Помилка оновлення кешу: {e}")

    def _is_cache_valid(self) -> bool:
        """Перевірити чи кеш актуальний"""
        return (time.time() - self._last_cache_update) < self._cache_lifetime

    def check_plate_access(self, plate_number: str) -> bool:
        """Перевірити чи має номер доступ"""
        # Оновлюємо кеш якщо потрібно
        if not self._is_cache_valid():
            self._update_cache()

        with self._cache_lock:
            return plate_number in self._cache.get('allowed_plates', [])

    def _is_duplicate(self, plate_number: str, check_type: str) -> bool:
        """Перевірити чи це дублікат номера"""
        with self._recent_plates_lock:
            key = f"{check_type}_{plate_number}"
            current_time = time.time()

            # Очищаємо старі записи
            self._recent_plates = {
                k: v for k, v in self._recent_plates.items()
                if current_time - v < TIMING_CONFIG["duplicate_timeout"]
            }

            # Перевіряємо дублікат
            if key in self._recent_plates:
                return True

            # Додаємо новий запис
            self._recent_plates[key] = current_time
            return False

    def update_last_seen(self, plate_number: str) -> bool:
        """Оновити час останнього розпізнавання"""
        if self._is_duplicate(plate_number, "last_seen"):
            logger.debug(f"Ігноруємо дублікат номера {plate_number}")
            return True

        try:
            # Знаходимо рядок з номером
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=GOOGLE_SHEETS_CONFIG["ranges"]["allowed_plates"]
            ).execute()

            values = result.get('values', [])
            row_index = None

            for i, row in enumerate(values):
                if row and row[0] == plate_number:
                    row_index = i + 3  # +3 бо починаємо з 3-го рядка
                    break

            if row_index:
                # Оновлюємо час у колонці B
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.service.spreadsheets().values().update(
                    spreadsheetId=self.spreadsheet_id,
                    range=f'B{row_index}',
                    valueInputOption='RAW',
                    body={'values': [[timestamp]]}
                ).execute()

                logger.info(f"Оновлено час для {plate_number}: {timestamp}")
                return True

            return False

        except HttpError as e:
            logger.error(f"Помилка оновлення часу: {e}")
            return False

    def log_denied_access(self, plate_number: str) -> bool:
        """Записати спробу несанкціонованого доступу"""
        if self._is_duplicate(plate_number, "denied"):
            return True

        try:
            # Знаходимо перший вільний рядок у колонках D та E
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range='D3:E1000'
            ).execute()

            values = result.get('values', [])
            next_row = len(values) + 3

            # Записуємо номер та час
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=f'D{next_row}:E{next_row}',
                valueInputOption='RAW',
                body={'values': [[plate_number, timestamp]]}
            ).execute()

            logger.warning(f"Записано спробу доступу: {plate_number} о {timestamp}")
            return True

        except HttpError as e:
            logger.error(f"Помилка запису спроби доступу: {e}")
            return False

    def log_exit(self, plate_number: Optional[str]) -> bool:
        """Записати виїзд"""
        # Якщо немає номера, записуємо "UNKNOWN"
        if not plate_number:
            plate_number = "UNKNOWN"

        if self._is_duplicate(plate_number, "exit"):
            return True

        try:
            # Знаходимо перший вільний рядок у колонках G та H
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range='G3:H1000'
            ).execute()

            values = result.get('values', [])
            next_row = len(values) + 3

            # Записуємо номер та час
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=f'G{next_row}:H{next_row}',
                valueInputOption='RAW',
                body={'values': [[plate_number, timestamp]]}
            ).execute()

            logger.info(f"Записано виїзд: {plate_number} о {timestamp}")
            return True

        except HttpError as e:
            logger.error(f"Помилка запису виїзду: {e}")
            return False

    def get_all_allowed_plates(self) -> List[str]:
        """Отримати всі дозволені номери"""
        if not self._is_cache_valid():
            self._update_cache()

        with self._cache_lock:
            return self._cache.get('allowed_plates', []).copy()

    def refresh_cache(self):
        """Примусово оновити кеш"""
        self._update_cache()