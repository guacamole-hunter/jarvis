import os
import re
import sqlite3
import pdfplumber
import pytesseract
import asyncio
from glob import glob
from pdf2image import convert_from_path
import cv2
import numpy as np

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert the image if necessary
    if np.mean(thresh) > 128:
        thresh = cv2.bitwise_not(thresh)

    return thresh

async def pdf_to_text(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()

                if text is None:
                    # If text extraction fails, convert the page to an image and use OCR
                    images = convert_from_path(file_path, dpi=300)
                    current_page_number = page.page_number
                    image = images[current_page_number - 1]

                    # Preprocess the image using OpenCV
                    preprocessed_image = preprocess_image(np.array(image))

                    text = pytesseract.image_to_string(preprocessed_image)

                pages.append(text)

            text_content = "\n".join(pages)
            if text_content.strip():
                return text_content
            else:
                return None

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def create_connect_db():
    conn = sqlite3.connect("service_manuals.db")
    conn.execute("CREATE TABLE IF NOT EXISTS service_manuals (manufacturer TEXT, model TEXT, manual_type TEXT, content TEXT, PRIMARY KEY(manufacturer, model, manual_type))")
    return conn

def is_duplicate_by_filename(conn, manufacturer, model):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM service_manuals WHERE manufacturer=? AND model=?", (manufacturer, model))
    return bool(cursor.fetchone())

def store_data(conn, manufacturer, model, manual_type, content):
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO service_manuals (manufacturer, model, manual_type, content) VALUES (?, ?, ?, ?)", (manufacturer, model, manual_type, content))
    conn.commit()
def is_model_number_valid(model):
    model_number_patterns = [
        r"^[a-zA-Z0-9-_]+$",
        r"^[a-zA-Z]+[\d]+[a-zA-Z]*$",
        r"^[a-zA-Z]+[-\s]?[\d]+[a-zA-Z]*[-\s]?[\d]*$",
        r"^[\d]+[a-zA-Z]+[-\s]?[\d]*$"
    ]

    for pattern in model_number_patterns:
        if re.match(pattern, model):
            return True
    return False

def extract_model_number(text):
    model_number_patterns = [
        r"\b[a-zA-Z]{1,2}\d+\b",  # A single or two letters followed by digits
        r"\b\d+[a-zA-Z]{1,2}\b",  # Digits followed by a single or two letters
        r"\b[a-zA-Z]{1,2}\d+[a-zA-Z]{1,2}\b"  # A single or two letters, digits, and a single or two letters
    ]

    model_numbers = []
    for pattern in model_number_patterns:
        matches = re.findall(pattern, text)
        model_numbers.extend(matches)

    # Prioritize model numbers with more digits
    model_numbers = sorted(model_numbers, key=lambda x: sum(c.isdigit() for c in x), reverse=True)

    for model_number in model_numbers:
        if is_model_number_valid(model_number):
            return model_number
    return None



def find_model_number_in_text(text):
    lines = text.splitlines()
    for line in lines:
        model_number = extract_model_number(line)
        if model_number and is_model_number_valid(model_number):
            return model_number
    return None

async def main():
    conn = create_connect_db()

    pdf_files = [y for x in os.walk("manuals") for y in glob(os.path.join(x[0], "*.pdf"))]

    for pdf_file in pdf_files:
        manufacturer = os.path.basename(os.path.dirname(pdf_file))
        file_basename = os.path.splitext(os.path.basename(pdf_file))[0]

  

        print(f"Processing {pdf_file}...")
        text = await pdf_to_text(pdf_file)

        if text is None:
            print("Skipping invalid PDF.")
            continue

        model = find_model_number_in_text(text)
        if model is None:
            model = extract_model_number(file_basename)

        if model and is_model_number_valid(model):
            print(f"Manufacturer: {manufacturer}, Model: {model}")

            if is_duplicate_by_filename(conn, manufacturer, model):
                print("Duplicate entry found. Skipping.")
                continue

            store_data(conn, manufacturer, model, 'Service Manual', text)
            print("Stored data in the database.")
        else:
            print("Model number not valid or not found in the filename or text.")

    conn.close()

if __name__ == "__main__":
    asyncio.run(main())
