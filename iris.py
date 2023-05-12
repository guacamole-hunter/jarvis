import asyncio
import openai
import sqlite3
import difflib
import re
import spacy
import redisai
import pyjokes
import requests
import json
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

class Iris:
    def __init__(self, database_path, user_name):
        self.introduction = "I am acting as Iris, your personal assistant. i am trained in mental health with a personality of a loving grandmother, and expetise at troubleshooting, diagnosing and much more. i will make suggestions as if i am a knowledgable individual"
        self.conn = sqlite3.connect(database_path)
        self.cursor = self.conn.cursor()
        self.short_term_memory = []
        self.long_term_memory = {}
        self.user_name = user_name
        self.save_memory("user_name", user_name)
        self.thoughts = {}
        self.expert = "I am Iris, an technology assistant. I am here to assist you to nagvigate technology! Dont worry if you don't know just ask I will do my best to give detailed instructions, or just have a good conversation if that's what user needs."
        try:
            self.load_long_term_memory()
        except sqlite3.OperationalError:
            self.create_long_term_memory_table()
            self.create_service_manuals_table() 
        
        # Connect to RedisAI
        self.client = redisai.Client(host='localhost', port=6379)
        
    def create_service_manuals_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS service_manuals (
                id INTEGER PRIMARY KEY,
                manufacturer TEXT NOT NULL,
                model TEXT NOT NULL,
                content TEXT NOT NULL
            );
            """
        )
        self.conn.commit()


    def create_long_term_memory_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS long_term_memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def load_long_term_memory(self):
        # Load long-term memory from SQLite database
        self.cursor.execute("SELECT key, value FROM long_term_memory")
        memory_data = self.cursor.fetchall()
        self.long_term_memory = {key: json.loads(value) for key, value in memory_data}

        
    def save_memory(self, key, value):
        serialized_value = json.dumps(value)
        self.cursor.execute(
            "INSERT OR REPLACE INTO long_term_memory (key, value) VALUES (?, ?)",
            (key, serialized_value)
        )
        self.conn.commit()
        self.long_term_memory[key] = value



    async def rephrase_question(self, question: str, chat_history: list):
        prompt = f"Please rephrase the following question as if it were asked by a technical professional: \"{question}\""
        response = await self.query_openai(prompt, chat_history)
        return response.strip()  # Return the response directly instead of accessing choices[0].text


    def generate_prompt(self, rephrased_question, documents, thoughts):
        documents_prompt = "\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(documents))
        thoughts_prompt = "\n".join(f"Thought {i+1}: {thought}" for i, thought in enumerate(thoughts))

        prompt = f"{self.expert}\n{documents_prompt}\n{thoughts_prompt}\n\nUser: {rephrased_question}\niris:"



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
    
    def extract_keywords(self, text):
        doc = nlp(text)
        keywords = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'PROPN', 'VERB')]
        return keywords
    
    async def handle_input(self, user_input, current_manufacturer, current_model, chat_history):
        self.short_term_memory.append({"role": "user", "content": user_input})
        keywords = self.extract_keywords(user_input.lower())

        if "tell" in keywords and "joke" in keywords:
            response = self.tell_joke()
        elif "weather" in keywords:
            location = "New York"  # Default location, can be changed based on user input
            response = self.get_weather(location)
        else:
            manufacturer, model = self.extract_manufacturer_and_model(user_input)
            rephrased_question = await self.rephrase_question(user_input, chat_history)
            if "research" in user_input.lower():
                search_results = await self.google_search(rephrased_question)
                await self.add_research_to_thoughts(search_results)
                prompt = self.generate_prompt(rephrased_question, service_manual, self.thoughts)
                response = await self.query_openai(prompt, self.short_term_memory)
            if manufacturer and model:
                current_manufacturer = manufacturer
                current_model = model
                service_manual = self.get_service_manual_by_model(current_manufacturer, current_model)
                prompt = self.generate_prompt(rephrased_question, service_manual, self.thoughts)
                response = await self.query_openai(prompt, self.short_term_memory)
                self.short_term_memory.append({"role": "assistant", "content": response})
            self.short_term_memory.append({"role": "assistant", "content": response})
            self.save_memory("chat_history", chat_history)

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
            title = result['title']
            self.thoughts[title] = result
    
    def tell_joke(self):
        return pyjokes.get_joke()

    def get_weather(self, location):
        api_key = "your_openweathermap_api_key"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
        response = requests.get(url)
        data = response.json()
        print(data)  # Add this line to print the JSON data

        try:
            weather = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            temp_min = data["main"]["temp_min"]
            temp_max = data["main"]["temp_max"]
            pressure = data["main"]["pressure"]
            humidity = data["main"]["humidity"]
        except KeyError:
            return "I'm sorry, I couldn't fetch the weather data. Please try again later."

        if data["cod"] != "404":
            weather = data["weather"][0]["description"]
            return f"The weather in {location} is {weather}."
        else:
            return f"Sorry, I couldn't find weather information for {location}."



async def main():
    database_path = "long_term_memory.db"  # Update with your SQLite database path
    user_name = input("Please enter your name: ")
    iris = Iris(database_path, user_name)

    current_make = None
    current_model = None
    chat_history = iris.long_term_memory.get("chat_history", [])

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        # Pass chat_history as an argument to the handle_input method
        response, current_make, current_model = await iris.handle_input(user_input, current_make, current_model, chat_history)
        print(f"\niris: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())


