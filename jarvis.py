import asyncio
import openai
import sqlite3
import difflib
import re
import spacy
import redisai
from googlesearch import search



# Replace 'your_api_key' with your OpenAI API key
openai.api_key = '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'
nlp = spacy.load("en_core_web_sm")
client = redisai.Client(host='localhost', port=6379)

def store_vectors_in_redis(vectors, keys):
    for key, vector in zip(keys, vectors):
        # Convert the vector to a RedisAI tensor
        tensor = redisai.Tensor.from_numpy(vector)
        
        # Store the tensor in RedisAI
        client.tensorset(key, tensor)
        

def get_vectors_from_redis(keys):
    vectors = []
    for key in keys:
        # Retrieve the tensor from RedisAI
        tensor = client.tensorget(key)
        
        # Convert the tensor back to a numpy array
        vector = tensor.to_numpy()
        vectors.append(vector)
    
    return vectors

class Jarvis:
    def __init__(self, database_path):
        self.thoughts = []
        self.expert = "I am Jarvis, a test equipment expert with knowledge in troubleshooting, repairing, and calibrating test equipments. I am here to assist technicians and engineers by providing troubleshooting advice. I will note if i am uncertain about my answer and provide a link to the service manual for further reference. I will attempt to give detailed step to resolve the issue."
        self.conn = sqlite3.connect(database_path)
        self.cursor = self.conn.cursor()
        self.chat_history = []
        
        
        # Connect to RedisAI
        self.client = redisai.Client(host='localhost', port=6379)

    async def rephrase_question(self, question: str, chat_history: list):
        prompt = f"Please rephrase the following question as if it were asked by a technical professional: \"{question}\""
        response = await self.query_openai(prompt, chat_history)
        return response.strip()  # Return the response directly instead of accessing choices[0].text


    def generate_prompt(self, rephrased_question, documents, thoughts):
        documents_prompt = "\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(documents))
        thoughts_prompt = "\n".join(f"Thought {i+1}: {thought}" for i, thought in enumerate(thoughts))

        prompt = f"{self.expert}\n{documents_prompt}\n{thoughts_prompt}\n\nUser: {rephrased_question}\nJarvis:"



        return prompt

    async def query_openai(self, prompt, chat_history):
        messages = chat_history.copy()
        messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1900,
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
        
        # Process the matched_manuals using spaCy for sentence tokenization (chunking)
        chunked_manuals = []
        for manual in matched_manuals:
            doc = nlp(manual)
            sentences = [sent.text for sent in doc.sents]
            chunked_manuals.extend(sentences)

        return chunked_manuals
    
    async def handle_input(self, user_input, current_manufacturer, current_model, chat_history):
        self.chat_history.append({"role": "user", "content": user_input})
        manufacturer, model = self.extract_manufacturer_and_model(user_input)
        rephrased_question = await self.rephrase_question(user_input, chat_history)  # Pass chat_history here
        if "research" in user_input.lower():
            # If user input contains a research-related keyword, perform a Google search
            search_results = await self.google_search(rephrased_question)
            await self.add_research_to_thoughts(search_results)
            prompt = self.generate_prompt(rephrased_question, service_manual, self.thoughts)
            response = await self.query_openai(prompt, self.chat_history)
        if manufacturer and model:
            current_manufacturer = manufacturer
            current_model = model
            service_manual = self.get_service_manual_by_model(current_manufacturer, current_model)
            prompt = self.generate_prompt(rephrased_question, service_manual, self.thoughts)  # Pass self.thoughts here
            response = await self.query_openai(prompt, self.chat_history)
            self.chat_history.append({"role": "assistant", "content": response})
        return response, current_manufacturer, current_model

    async def read_file(self, file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        return content

    async def write_file(self, file_path, content):
        with open(file_path, 'w') as file:
            file.write(content)
    
    async def google_search(self, query, num_results=5):
        search_results = []
        try:
            for j in search(query, num_results=num_results):
                search_results.append(j)
        except Exception as e:
            print(f"Error while searching: {e}")

        return search_results
    
    async def add_research_to_thoughts(self, research_results):
        for result in research_results:
            self.thoughts.append(result)


async def main():
    database_path = "service_manuals.db"  # Update with your SQLite database path
    jarvis = Jarvis(database_path)

    current_make = None
    current_model = None
    chat_history = []  # Define the chat_history variable

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        # Pass chat_history as an argument to the handle_input method
        response, current_make, current_model = await jarvis.handle_input(user_input, current_make, current_model, chat_history)
        print(f"\nJarvis: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())


