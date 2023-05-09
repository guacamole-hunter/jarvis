import asyncio
import openai
import sqlite3

# Replace 'your_api_key' with your OpenAI API key
openai.api_key = "your_api_key"

class Jarvis:
    def __init__(self, database_path):
        self.expert = "I am Jarvis, a test equipment expert with knowledge in troubleshooting, repairing, and calibrating equipment."
        self.conn = sqlite3.connect(database_path)
        self.cursor = self.conn.cursor()

    def generate_prompt(self, user_input, documents):
        documents_prompt = "\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(documents))
        return f"{self.expert}\n{documents_prompt}\n\nUser: {user_input}\nJarvis:"

    async def query_openai(self, prompt):
        response = await openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].text.strip()

    def get_documents_by_model_number(self, model_number):
        self.cursor.execute("SELECT content FROM documents WHERE model_number=?", (model_number,))
        documents = self.cursor.fetchall()
        return [doc[0] for doc in documents]

    async def handle_input(self, user_input):
        model_number = input("Enter the model number: ")
        documents = self.get_documents_by_model_number(model_number)
        prompt = self.generate_prompt(user_input, documents)
        response = await self.query_openai(prompt)
        return response

async def main():
    database_path = "documents.db"  # Update with your SQLite database path
    jarvis = Jarvis(database_path)

    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        response = await jarvis.handle_input(user_input)
        print("Jarvis:", response)

if __name__ == "__main__":
    asyncio.run(main())
