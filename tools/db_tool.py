import faiss
import numpy as np
from pymongo import MongoClient
import google.generativeai as genai
import pandas as pd
from pydantic import BaseModel

client = MongoClient('localhost', 27017)
db = client["bdtest-1303-6"]
collection = db["test_t"]

genai.configure(api_key="AIzaSyCbWtAuWz7QbqdyAkUhC314J0wHarzW1Ms")
MODEL = genai.GenerativeModel('gemini-2.0-flash')


class SearchResultItem(BaseModel):
    subject: str
    credits: int
    description: str


def get_embedding(text):
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text
        )
        return np.array(response["embedding"]).astype('float32')
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def create_faiss_index():
    dimension = 768  # embedding dimensions
    index = faiss.IndexFlatL2(dimension)
    return index


faiss_index = create_faiss_index()
vector_data = []


def insert_data():
    data = [
        {"subject": "Programming I", "credits": 6,
         "description": "Introduction to programming using Python and fundamental programming concepts."},
        {"subject": "Databases", "credits": 6,
         "description": "Fundamentals of databases, database design, and SQL queries."},
        {"subject": "Algorithms and Data Structures", "credits": 6,
         "description": "Study of algorithms and data structures such as lists, stacks, and queues."},
        {"subject": "Operating Systems", "credits": 6,
         "description": "Study of operating systems and their operation, including memory and process management."},
        {"subject": "Computer Networks", "credits": 6,
         "description": "Fundamental concepts of networks, protocols, and computer network architecture."},
        {"subject": "Artificial Intelligence", "credits": 6,
         "description": "Introduction to artificial intelligence, intelligent agents, and search algorithms."},
        {"subject": "Web Development", "credits": 6,
         "description": "Creation of web applications using technologies like HTML, CSS, JavaScript, and frameworks."},
        {"subject": "Software Engineering", "credits": 6,
         "description": "Fundamentals of software development, agile methodologies, and good programming practices."},
        {"subject": "Mobile Devices Programming", "credits": 6,
         "description": "Development of mobile applications for Android and iOS platforms."},
        {"subject": "Computer Graphics", "credits": 6,
         "description": "Fundamentals of computer graphics, rendering, and 3D modeling."}
    ]

    for item in data:
        combined_data = f"Subject: {item['subject']}, Credits: {item['credits']}, Description: {item['description']}"
        embedding = get_embedding(combined_data)
        if embedding is not None:
            doc_id = collection.insert_one(item).inserted_id
            vector_data.append(doc_id)
            faiss_index.add(np.expand_dims(embedding, axis=0))



insert_data()
#faiss_index = faiss.read_index("faiss-index.index")


# top_k -> number of similar vectors to consider
def vector_search(query, top_k=3):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return "There was an error generating the embedding."

    distances, indices = faiss_index.search(np.expand_dims(query_embedding, axis=0), top_k)
    results = [collection.find_one({"_id": vector_data[i]}) for i in indices[0]]
    print(f"\nResults from vector_search: {results}")
    return results


def handle_user_query(query):
    search_results = vector_search(query)
    if not search_results:
        return "No results found."

    search_results_models = [SearchResultItem(**res) for res in search_results if res]
    search_results_df = pd.DataFrame([item for item in search_results_models])

    messages = [
        {"role": "system", "content": "You are a college IT recommendation system."},
        {"role": "user", "content": f"Answer to the following question: {query} with this given context:\n{search_results_df}"}
    ]
    prompt = "\n".join([m["content"] for m in messages])
    system_response = MODEL.generate_content(prompt)

    print(f" - User question: {query}\n - System response: {system_response.text}")
    return system_response


response = handle_user_query("This year I have to choose 2 subjects for my degree, I like algorithms and front-end development. What do you recommend me?")