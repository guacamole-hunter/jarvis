import os
import asyncio
import openai
import sqlite3
import difflib
import re

# Replace 'your_api_key' with your OpenAI API key
openai.api_key = 'your_api_key'

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
        system_message = "I am Jarvis, a test equipment expert with knowledge in troubleshooting, repairing, and calibrating equipment."
        user_message = f"Please answer the following question with knowledge from the database: '{prompt}'"
        full_prompt = f"{system_message}\n{user_message}"
        
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=full_prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=1,
        )
        return response

    def extract_make_and_model(self, user_input):
        make_model_pattern = r"(\w+)\s+(\w+)"
        match = re.search(make_model_pattern, user_input)
        if match:
            return match.groups()
        else:
            return None, None

    def get_service_manual_by_model(self, make, model):
        self.cursor.execute("SELECT make, model, content FROM service_manuals")
        service_manuals = self.cursor.fetchall()

        # Get list of unique make and model names
        make_names = list(set([man[0] for man in service_manuals]))
        model_names = list(set([man[1] for man in service_manuals]))

        # Find the closest match for the input make and model
        closest_make_match = difflib.get_close_matches(make, make_names, n=1, cutoff=0.5)
        closest_model_match = difflib.get_close_matches(model, model_names, n=1, cutoff=0.5)

        if closest_make_match and closest_model_match:
            make = closest_make_match[0]
            model = closest_model_match[0]
        else:
            return []

        # Find the service manual for the matched make and model
        matched_manuals = [man[2] for man in service_manuals if man[0] == make and man[1] == model]
        return matched_manuals

    async def handle_input(self, user_input, current_make, current_model):
        make, model = self.extract_make_and_model(user_input)
        if make and model:
            current_make = make
            current_model = model
        service_manual = self.get_service_manual_by_model(current_make, current_model)
        rephrased_question = await self.rephrase_question(user_input)
        prompt = self.generate_prompt(rephrased_question, service_manual)
        response = await self.query_openai(prompt)
        return response, current_make, current_model

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

