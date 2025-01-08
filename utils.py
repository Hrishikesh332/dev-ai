import os
from dotenv import load_dotenv
from twelvelabs import TwelveLabs
from pymilvus import connections, Collection
from openai import OpenAI


load_dotenv()

# Load environment variables
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
URL = os.getenv('URL')
TOKEN = os.getenv('TOKEN')
TWELVELABS_API_KEY = os.getenv('TWELVELABS_API_KEY')

# Initialize connections
openai_client = OpenAI()
connections.connect(uri=URL, token=TOKEN)
collection = Collection(COLLECTION_NAME)
collection.load()

def emb_text(text):
    try:
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        embedding = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            text=text
        ).text_embedding
        return embedding.segments[0].embeddings_float
    except Exception as e:
        raise e

def get_rag_response(question):
    try:
        question_embedding = emb_text(question)
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        results = collection.search(
            data=[question_embedding],
            anns_field="vector",
            param=search_params,
            limit=2,
            output_fields=['metadata']
        )

        retrieved_docs = []
        for hit in results[0]:
            try:
                metadata = hit.entity.metadata
                if metadata:
                    similarity = round((hit.score + 1) * 50, 2)  # Convert from [-1,1] to [0,100]
                    similarity = max(0, min(100, similarity))
                    
                    retrieved_docs.append({
                        "title": metadata.get("title", "Untitled"),
                        "description": metadata.get("description", "No description available"),
                        "product_id": metadata.get("product_id", ""),
                        "video_url": metadata.get("video_url", ""),
                        "link": metadata.get("link", ""),
                        "similarity": similarity,
                    })
            except Exception as e:
                continue

        if not retrieved_docs:
            return {
                "response": "I couldn't find any matching products. Try describing what you're looking for differently.",
                "metadata": None
            }

        context = "\n\n".join([f"Title: {doc['title']}\nDescription: {doc['description']}" for doc in retrieved_docs])
        messages = [
            {
                "role": "system",
                "content": """You are a professional fashion advisor and AI shopping assistant.
                Provide stylish, engaging responses about fashion products.
                Focus on style, trends, and helping customers find the perfect items."""
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext: {context}"
            }
        ]

        chat_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        return {
            "response": chat_response.choices[0].message.content,
            "metadata": {
                "sources": retrieved_docs,
                "total_sources": len(retrieved_docs)
            }
        }
    
    except Exception as e:
        return {
            "response": "I encountered an error while processing your request.",
            "metadata": None
        }
