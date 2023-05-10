import asyncio
import openai
import sqlite3
import difflib
import re


# Replace 'your_api_key' with your OpenAI API key
openai.api_key = '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'

class Jarvis:
    def __init__(self, database_path):
        self.expert = "I am Jarvis, a test equipment expert with knowledge in troubleshooting, repairing, and calibrating test equipments."
        self.conn = sqlite3.connect(database_path)
        self.cursor = self.conn.cursor()

    def generate_prompt(self, rephrased_question, documents):
        documents_prompt = "\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(documents))
        return f"{self.expert}\n{documents_prompt}\n\nUser: {rephrased_question}\nJarvis:"

    async def rephrase_question(self, question):
        prompt = f"Please rephrase the following question: '{question}'"
        response = await self.query_openai(prompt)
        return response

    async def query_openai(self, prompt):
        messages = [
            {"role": "system", "content": self.expert},
            {"role": "user", "content": prompt},
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=1,
        )
        
        return response.choices[0].message["content"]

    def extract_manufacturer_and_model(self, user_input):
        manufacturer_model_pattern = r"(\w+)\s+(\w+)"
        match = re.search(manufacturer_model_pattern, user_input)
        if match:
            return match.groups()
        else:
            return None, None

    def get_service_manual_by_model(self, manufacturer, model):
        self.cursor.execute("SELECT manufacturer, model, content FROM service_manuals")
        service_manuals = self.cursor.fetchall()

        # Get list of unique manufacturer and model names
        manufacturer_names = list(set([man[0] for man in service_manuals]))
        model_names = list(set([man[1] for man in service_manuals]))

        # Find the closest match for the input manufacturer and model
        closest_manufacturer_match = difflib.get_close_matches(manufacturer, manufacturer_names, n=1, cutoff=0.5)
        closest_model_match = difflib.get_close_matches(model, model_names, n=1, cutoff=0.5)

        if closest_manufacturer_match and closest_model_match:
            manufacturer = closest_manufacturer_match[0]
            model = closest_model_match[0]
        else:
            return []

        # Find the service manual for the matched manufacturer and model
        matched_manuals = [man[2] for man in service_manuals if man[0] == manufacturer and man[1] == model]
        return matched_manuals

    async def handle_input(self, user_input, current_manufacturer, current_model):
        manufacturer, model = self.extract_manufacturer_and_model(user_input)
        if manufacturer and model:
            current_manufacturer = manufacturer
            current_model = model
        service_manual = self.get_service_manual_by_model(current_manufacturer, current_model)
        rephrased_question = await self.rephrase_question(user_input)
        prompt = self.generate_prompt(rephrased_question, service_manual)
        response = await self.query_openai(prompt)
        return response, current_manufacturer, current_model


async def main():
    database_path = "service_manuals.db"  # Update with your SQLite database path
    jarvis = Jarvis(database_path)

    current_make = None
    current_model = None

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        response, current_make, current_model = await jarvis.handle_input(user_input, current_make, current_model)
        print(f"Jarvis: {response}")

if __name__ == "__main__":
    asyncio.run(main())

