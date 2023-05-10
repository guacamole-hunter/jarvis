import os
import re
import sqlite3
import pdfplumber
import asyncio
from glob import glob

async def pdf_to_text(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
            return "\n".join(pages)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_connect_db():
    conn = sqlite3.connect("service_manuals.db")
    conn.execute("CREATE TABLE IF NOT EXISTS service_manuals (make TEXT, model TEXT, manual_type TEXT, content TEXT, PRIMARY KEY(make, model, manual_type))")
    return conn

def is_duplicate(conn, make, model):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM service_manuals WHERE make=? AND model=?", (make, model))
    return bool(cursor.fetchone())

def store_data(conn, make, model, manual_type, content):
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO service_manuals (make, model, manual_type, content) VALUES (?, ?, ?, ?)", (make, model, manual_type, content))
    conn.commit()

def extract_model_manual_type(text):
    model_manual_patterns = [
        r"(?i)Model:\s*(\w+)\s*Type:\s*(\w+)",
        r"(?i)Model:\s*(\w+)[\s\S]*?Type:\s*(\w+)",
        r"(?i)Model\s+(\w+)[\s\S]*?Type\s+(\w+)"
    ]

    for pattern in model_manual_patterns:
        match = re.search(pattern, text)
        if match:
            model = match.group(1)
            manual_type = match.group(2)
            return model, manual_type

    return None, None

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
        make = os.path.basename(os.path.dirname(pdf_file))
        model = os.path.splitext(os.path.basename(pdf_file))[0]
        _, manual_type = extract_model_manual_type(text)

        if make and model and manual_type:
            print(f"Make: {make}, Model: {model}, Type: {manual_type}")

            if is_duplicate(conn, make, model):
                print("Duplicate entry found. Skipping.")
                continue

            store_data(conn, make, model, manual_type, text)
            print("Stored data in the database.")
        else:
            print("Manual type not found.")

    conn.close()

if __name__ == "__main__":
    asyncio.run(main())
