import os
import uuid
from dotenv import load_dotenv
from twelvelabs import TwelveLabs
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import streamlit as st
from openai import OpenAI

load_dotenv()

# Load environment variables
TEXT_COLLECTION_NAME = os.getenv('TEXT_COLLECTION_NAME', 'fashion_text_embeddings')
VIDEO_COLLECTION_NAME = os.getenv('VIDEO_COLLECTION_NAME', 'fashion_video_embeddings')
URL = os.getenv('URL')
TOKEN = os.getenv('TOKEN')
TWELVELABS_API_KEY = os.getenv('TWELVELABS_API_KEY')

# Initialize connections
openai_client = OpenAI()
connections.connect(uri=URL, token=TOKEN)

def create_text_collection():
    """Create collection for text embeddings if it doesn't exist"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="product_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="video_url", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=500)
    ]
    schema = CollectionSchema(fields=fields, description="Text embeddings for fashion products")
    text_collection = Collection(TEXT_COLLECTION_NAME, schema)
    
    # Create index if it doesn't exist
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    text_collection.create_index(field_name="vector", index_params=index_params)
    return text_collection

def create_video_collection():
    """Create collection for video embeddings if it doesn't exist"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="product_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="video_url", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="start_time", dtype=DataType.DOUBLE),
        FieldSchema(name="end_time", dtype=DataType.DOUBLE)
    ]
    schema = CollectionSchema(fields=fields, description="Video embeddings for fashion products")
    video_collection = Collection(VIDEO_COLLECTION_NAME, schema)
    
    # Create index if it doesn't exist
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    video_collection.create_index(field_name="vector", index_params=index_params)
    return video_collection

# Initialize collections
text_collection = create_text_collection()
video_collection = create_video_collection()

def generate_embeddings(product_info):
    """Generate both text and video embeddings for a product"""
    try:
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        
        # Generate text embedding
        text = f"{product_info['title']} {product_info['desc']}"
        text_embedding = twelvelabs_client.embed.create(
            engine_name="Marengo-retrieval-2.6",
            text=text
        ).text_embedding.segments[0].embeddings_float
        
        # Generate video embedding
        video_embedding = twelvelabs_client.embed.create(
            engine_name="Marengo-retrieval-2.6",
            video_url=product_info['video_url']
        ).video_embedding.segments[0].embeddings_float
        
        return {
            'text_embedding': text_embedding,
            'video_embedding': video_embedding
        }, None
    except Exception as e:
        return None, str(e)

def insert_embeddings(embeddings_data, product_info):
    """Insert both text and video embeddings with their respective metadata"""
    try:
        # Common metadata fields
        base_metadata = {
            "product_id": product_info['product_id'],
            "title": product_info['title'],
            "description": product_info['desc'],
            "video_url": product_info['video_url'],
            "link": product_info['link']
        }
        
        # Insert text embedding
        text_data = [{
            "id": int(uuid.uuid4().int & (1<<63)-1),
            "vector": embeddings_data['text_embedding'],
            **base_metadata
        }]
        text_collection.insert(text_data)
        
        # Insert video embedding
        video_data = [{
            "id": int(uuid.uuid4().int & (1<<63)-1),
            "vector": embeddings_data['video_embedding'],
            **base_metadata,
            "start_time": 0.0,  # Add default start time
            "end_time": 0.0     # Add default end time
        }]
        video_collection.insert(video_data)
        
        return True
    except Exception as e:
        st.error(f"Error inserting embeddings: {str(e)}")
        return False

def search_text_embeddings(query_text, limit=2):
    """Search text embeddings collection"""
    try:
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        query_embedding = twelvelabs_client.embed.create(
            engine_name="Marengo-retrieval-2.6",
            text=query_text
        ).text_embedding.segments[0].embeddings_float
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        results = text_collection.search(
            data=[query_embedding],
            anns_field="vector",
            param=search_params,
            limit=limit,
            output_fields=['product_id', 'title', 'description', 'video_url', 'link']
        )
        
        return results
    except Exception as e:
        st.error(f"Error searching text embeddings: {str(e)}")
        return None

def search_video_embeddings(image_file, limit=5):
    """Search video embeddings collection using image query"""
    try:
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        image_embedding = twelvelabs_client.embed.create(
            engine_name="Marengo-retrieval-2.6",
            image_file=image_file
        ).image_embedding.segments[0].embeddings_float
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        results = video_collection.search(
            data=[image_embedding],
            anns_field="vector",
            param=search_params,
            limit=limit,
            output_fields=['product_id', 'title', 'description', 'video_url', 'link', 'start_time', 'end_time']
        )
        
        return results
    except Exception as e:
        st.error(f"Error searching video embeddings: {str(e)}")
        return None

def get_rag_response(question):
    """Get response using text embeddings search"""
    try:
        results = search_text_embeddings(question)
        if not results:
            return {
                "response": "I couldn't find any matching products. Try describing what you're looking for differently.",
                "metadata": None
            }

        retrieved_docs = []
        for hit in results[0]:
            similarity = round((hit.score + 1) * 50, 2)
            similarity = max(0, min(100, similarity))
            
            retrieved_docs.append({
                "title": hit.entity.get('title', 'Untitled'),
                "description": hit.entity.get('description', 'No description available'),
                "product_id": hit.entity.get('product_id', ''),
                "video_url": hit.entity.get('video_url', ''),
                "link": hit.entity.get('link', ''),
                "similarity": similarity,
            })

        context = "\n\n".join([f"Title: {doc['title']}\nDescription: {doc['description']}" 
                              for doc in retrieved_docs])
        
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
            "response": f"I encountered an error while processing your request: {str(e)}",
            "metadata": None
        }

def create_video_embed(video_url, start_time, end_time):
    video_id, platform = get_video_id_from_url(video_url)
    start_seconds = format_time_for_url(start_time)
    
    if platform == 'vimeo':
        return f"""
            <iframe 
                width="100%" 
                height="315" 
                src="https://player.vimeo.com/video/{video_id}#t={start_seconds}s"
                frameborder="0" 
                allow="autoplay; fullscreen; picture-in-picture" 
                allowfullscreen>
            </iframe>
        """
    elif platform == 'direct':
        return f"""
            <video 
                width="100%" 
                height="315" 
                controls 
                autoplay
                id="video-player">
                <source src="{video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <script>
                document.getElementById('video-player').addEventListener('loadedmetadata', function() {{
                    this.currentTime = {start_time};
                }});
            </script>
        """
    else:
        return f"<p>Unable to embed video from URL: {video_url}</p>"
  
