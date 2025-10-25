import logging
import os
import sys
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Global configuration
CSV_PATH = os.getenv("CSV_PATH", "books.csv")
SPINE_CLUSTER_THRESHOLD_RATIO = 0.04
STOP_WORDS = {
    "series",
    "collection",
    "classic",
    "classics",
    "bestseller",
    "издательство",
    "серия",
    "коллекция",
    "edition",
    "publishing",
    "publisher",
}

logger = logging.getLogger(__name__)

OCR_ENGINE: Optional[PaddleOCR] = None


def initialize_logging() -> None:
    """Configure root logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_image_and_size(image_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    """Decode image bytes into a BGR numpy array and return its dimensions."""
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось прочитать изображение из переданных данных.")
    height, width = image.shape[:2]
    return image, width, height


def run_ocr(image: np.ndarray) -> List[Dict[str, object]]:
    """Run PaddleOCR on the provided image array and return structured blocks."""
    if OCR_ENGINE is None:
        raise RuntimeError("OCR engine не инициализирован.")

    raw_result = OCR_ENGINE.ocr(image, cls=True)
    lines: List[Dict[str, object]] = []
    for page in raw_result:
        for line in page:
            bbox = np.array(line[0], dtype=float)
            text = line[1][0].strip()
            score = float(line[1][1])
            x_center = float(np.mean(bbox[:, 0]))
            y_center = float(np.mean(bbox[:, 1]))
            lines.append(
                {
                    "text": text,
                    "score": score,
                    "bbox": bbox,
                    "x_center": x_center,
                    "y_center": y_center,
                }
            )
    return lines


def cluster_by_spine(
    lines: Sequence[Dict[str, object]],
    img_width: int,
    x_threshold_ratio: float = SPINE_CLUSTER_THRESHOLD_RATIO,
) -> List[List[Dict[str, object]]]:
    """Group OCR blocks into clusters that correspond to book spines."""
    if not lines:
        return []

    x_threshold = img_width * x_threshold_ratio
    sorted_lines = sorted(lines, key=lambda item: item["x_center"])  # type: ignore[index]

    clusters: List[List[Dict[str, object]]] = [[sorted_lines[0]]]
    previous_line = sorted_lines[0]
    for line in sorted_lines[1:]:
        if abs(line["x_center"] - previous_line["x_center"]) <= x_threshold:  # type: ignore[index]
            clusters[-1].append(line)
        else:
            clusters.append([line])
        previous_line = line
    return clusters


def _contains_stop_word(text: str) -> bool:
    words = [word for word in text.lower().replace("\n", " ").split() if word]
    return any(word in STOP_WORDS for word in words)


def _all_words_stop(text: str) -> bool:
    words = [word for word in text.lower().replace("\n", " ").split() if word]
    if not words:
        return False
    return all(word in STOP_WORDS for word in words)


def pick_author_and_title(
    cluster_lines: Sequence[Dict[str, object]]
) -> Tuple[Optional[str], Optional[str]]:
    """Heuristically choose author and title from lines belonging to one spine."""
    sorted_lines = sorted(cluster_lines, key=lambda item: item["y_center"])  # type: ignore[index]
    texts = [str(line["text"]).strip() for line in sorted_lines if str(line["text"]).strip()]

    if not texts:
        return None, None

    # Identify potential authors
    author_candidates: List[str] = []
    for text in texts:
        words = [word for word in text.split() if word]
        if not words:
            continue
        if len(text) > 30:
            continue
        if not (1 <= len(words) <= 4):
            continue
        if _contains_stop_word(text):
            continue
        author_candidates.append(text)

    author: Optional[str] = None
    if author_candidates:
        author = min(author_candidates, key=len)

    # Identify potential titles
    title_candidates: List[str] = []
    for text in texts:
        if len(text) < 3:
            continue
        if _contains_stop_word(text):
            continue
        if _all_words_stop(text):
            continue
        title_candidates.append(text)

    title: Optional[str] = None
    if title_candidates:
        title = max(title_candidates, key=len)

    return author, title


def process_image_to_records(image_bytes: bytes) -> Tuple[List[Dict[str, str]], List[Dict[str, object]]]:
    """Process image bytes through OCR and heuristics to produce CSV-ready records."""
    image, width, _ = load_image_and_size(image_bytes)
    lines = run_ocr(image)
    if not lines:
        return [], []

    clusters = cluster_by_spine(lines, width)
    records: List[Dict[str, str]] = []
    for cluster in clusters:
        if not cluster:
            continue
        author, title = pick_author_and_title(cluster)
        if not author and not title:
            continue
        records.append({"title": title or "", "author": author or ""})
    return records, list(lines)


def append_records_to_csv(records: List[Dict[str, str]], csv_path: str) -> None:
    """Append records to CSV file, creating it if needed."""
    if not records:
        return

    df_new = pd.DataFrame(records)
    csv_absolute = os.path.abspath(csv_path)
    os.makedirs(os.path.dirname(csv_absolute), exist_ok=True)

    if os.path.exists(csv_absolute):
        try:
            df_existing = pd.read_csv(csv_absolute)
        except Exception:
            logger.exception("Не удалось прочитать существующий CSV, создаю новый.")
            df_existing = pd.DataFrame(columns=["title", "author"])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(csv_absolute, index=False)


def format_records_for_user(records: Sequence[Dict[str, str]]) -> str:
    """Prepare a friendly response summarizing saved records for the user."""
    lines = ["Я добавил такие книги:"]
    for idx, record in enumerate(records, start=1):
        author = record.get("author") or "(не найден)"
        title = record.get("title") or "(не найден)"
        lines.append(f"{idx}. Автор: {author} / Название: {title}")
    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command."""
    if update.message is None:
        return
    await update.message.reply_text(
        "Пришли мне фото книжной полки (видны корешки), я распознаю названия и авторов и сохраню это в базу."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download the photo, run OCR, and respond with extracted records."""
    message = update.message
    if message is None or not message.photo:
        return

    largest_photo = message.photo[-1]
    try:
        file = await context.bot.get_file(largest_photo.file_id)
        buffer = BytesIO()
        await file.download_to_memory(out=buffer)
        image_bytes = buffer.getvalue()
    except Exception as exc:
        logger.exception("Не удалось загрузить файл из Telegram: %s", exc)
        await message.reply_text("Произошла ошибка при обработке этого фото")
        return

    try:
        records, raw_lines = process_image_to_records(image_bytes)
    except Exception as exc:
        logger.exception("Ошибка обработки изображения: %s", exc)
        await message.reply_text("Произошла ошибка при обработке этого фото")
        return

    if not raw_lines:
        await message.reply_text("Не смог распознать текст на фото 😢")
        return

    if not records:
        await message.reply_text("Я не нашёл книг 🤔")
        return

    try:
        append_records_to_csv(records, CSV_PATH)
    except Exception as exc:
        logger.exception("Ошибка при сохранении CSV: %s", exc)
        await message.reply_text("Произошла ошибка при обработке этого фото")
        return

    response_text = format_records_for_user(records)
    await message.reply_text(response_text)


def main() -> None:
    """Entry point for launching the Telegram bot."""
    initialize_logging()

    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        print(
            "Ошибка: переменная окружения TELEGRAM_TOKEN не установлена. Установите её и перезапустите программу.",
            file=sys.stderr,
        )
        sys.exit(1)

    global OCR_ENGINE
    OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Бот запущен. Ожидание сообщений...")
    application.run_polling()


if __name__ == "__main__":
    main()
