#Load libraries
from openai import OpenAI
import numpy as np
from typing import List
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os
from bs4 import BeautifulSoup
import pickle

#Load the environment variables from the .env file
load_dotenv()
key = os.environ.get("OPENAI_API_KEY")

#Initialise client
client = OpenAI(api_key=key)

#Specify the embedding model to be used
EMBEDDING_MODEL = "text-embedding-3-small"

#Set file path for embedding cache
embedding_cache_path = "embeddings_cache.pkl"

#Load the cache if it exists, and save a copy to disk
#Code from https://cookbook.openai.com/examples/recommendation_using_embeddings
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

#get_embeddings function
def get_embeddings(texts, model=EMBEDDING_MODEL):
    return [embedding_from_string(text, model=model) for text in texts]

#embedding_from_string function
#Code from https://cookbook.openai.com/examples/recommendation_using_embeddings
def embedding_from_string(
    string: str,
    model: str = EMBEDDING_MODEL,
    embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

#get_embedding function
#Code from: https://platform.openai.com/docs/guides/embeddings/use-cases
def get_embedding(text, model=EMBEDDING_MODEL):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

#Load documents
def load_documents(directory):
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                text = soup.get_text(separator='\n')
                documents[filename] = text
                texts = list(documents.values()) #take the values from the dict and turn into a single list
    return texts

#retrieve_documents function
def retrieve_documents(query, document_embeddings, document_texts, model=EMBEDDING_MODEL):
    query_embedding = get_embedding(query, model=model)
    similarities = cosine_similarity([query_embedding], document_embeddings)
    most_similar_idx = np.argmax(similarities)
    return document_texts[most_similar_idx]


