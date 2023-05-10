import os
import re
import sqlite3
import pdfplumber
import pytesseract
import asyncio
from glob import glob

async def pdf_to_text(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            pages = []
            for page in pdf.pages:
                if page.images:
                    # If the page has images, use OCR to extract text
                    im = page.to_image(resolution=300)
                    text = pytesseract.image_to_string(im.image)
                else:
                    text = page.extract_text()
                pages.append(text)
            return "\n".join(pages)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_connect_db():
    conn = sqlite3.connect("service_manuals.db")
    conn.execute("CREATE TABLE IF NOT EXISTS service_manuals (manufacturer TEXT, model TEXT, manual_type TEXT, content TEXT, PRIMARY KEY(manufacturer, model, manual_type))")
    return conn

def is_duplicate(conn, manufacturer, model):
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

async def main():
    conn = create_connect_db()

    pdf_files = [y for x in os.walk("manuals") for y in glob(os.path.join(x[0], "*.pdf"))]

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        text = await pdf_to_text(pdf_file)

        if text is None:
            print("Skipping invalid PDF.")
            continue

        print("Extracted text from PDF.")
        manufacturer = os.path.basename(os.path.dirname(pdf_file))
        model = os.path.splitext(os.path.basename(pdf_file))[0]

        if is_model_number_valid(model):
            print(f"Manufacturer: {manufacturer}, Model: {model}")

            if is_duplicate(conn, manufacturer, model):
                print("Duplicate entry found. Skipping.")
                continue

            store_data(conn, manufacturer, model, 'Service Manual', text)
            print("Stored data in the database.")
        else:
            print("Model number not valid.")

    conn.close()

if __name__ == "__main__":
    asyncio.run(main())
