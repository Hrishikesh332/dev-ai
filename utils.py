import os
import uuid
from dotenv import load_dotenv
from twelvelabs import TwelveLabs
from pymilvus import connections, Collection
import streamlit as st
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

def generate_embeddings(product_info):
    """Generate both text and video embeddings for a product"""
    try:
        st.write(f"Processing product: {product_info['title']}")
        
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        
        # Generate text embedding
        text = f"{product_info['title']} {product_info['desc']}"
        text_embedding = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            text=text
        ).text_embedding.segments[0].embeddings_float
        
        # Generate video embedding
        video_embedding = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            video_url=product_info['video_url']
        ).video_embedding.segments[0].embeddings_float
        
        return {
            'text_embedding': text_embedding,
            'video_embedding': video_embedding
        }, None
    except Exception as e:
        return None, str(e)

def insert_embeddings(embeddings_data, product_info):
    """Insert both text and video embeddings into the same collection"""
    try:
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
            "embedding_type": "text",  # Add type identifier
            **base_metadata
        }]
        
        # Insert video embedding
        video_data = [{
            "id": int(uuid.uuid4().int & (1<<63)-1),
            "vector": embeddings_data['video_embedding'],
            "embedding_type": "video",  # Add type identifier
            **base_metadata
        }]
        
        # Insert both embeddings
        insert_result_text = collection.insert(text_data)
        insert_result_video = collection.insert(video_data)
        
        return insert_result_text and insert_result_video
    except Exception as e:
        st.error(f"Error inserting embeddings: {str(e)}")
        return None

def search_similar_videos(image_file, top_k=5):
    """Search for similar videos using image query"""
    try:
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        
        # Generate image embedding
        image_embedding = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            image_file=image_file
        ).image_embedding.segments[0].embeddings_float
        
        # Search in collection for video embeddings
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[image_embedding],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr="embedding_type == 'video'",  # Filter for video embeddings
            output_fields=["metadata"]
        )

        search_results = []
        for hits in results:
            for hit in hits:
                metadata = hit.entity.get('metadata', {})
                if metadata:
                    similarity_score = round((1 - float(hit.distance)) * 100, 2)
                    search_results.append({
                        'Title': metadata.get('title', ''),
                        'Description': metadata.get('description', ''),
                        'Link': metadata.get('link', ''),
                        'Video URL': metadata.get('video_url', ''),
                        'Similarity': f"{similarity_score}%"
                    })
        
        return search_results
    except Exception as e:
        st.error(f"Error searching videos: {str(e)}")
        return None

def get_rag_response(question):
    """Get response using text embeddings search"""
    try:
        # Generate embedding for the question
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        question_embedding = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            text=question
        ).text_embedding.segments[0].embeddings_float
        
        # Search for similar text embeddings
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[question_embedding],
            anns_field="vector",
            param=search_params,
            limit=2,
            expr="embedding_type == 'text'",  # Filter for text embeddings
            output_fields=["metadata"]
        )

        retrieved_docs = []
        for hit in results[0]:
            metadata = hit.entity.get('metadata', {})
            if metadata:
                similarity = round((hit.score + 1) * 50, 2)
                similarity = max(0, min(100, similarity))
                
                retrieved_docs.append({
                    "title": metadata.get('title', 'Untitled'),
                    "description": metadata.get('description', 'No description available'),
                    "product_id": metadata.get('product_id', ''),
                    "video_url": metadata.get('video_url', ''),
                    "link": metadata.get('link', ''),
                    "similarity": similarity
                })

        if not retrieved_docs:
            return {
                "response": "I couldn't find any matching products. Try describing what you're looking for differently.",
                "metadata": None
            }

        context = "\n\n".join([
            f"Title: {doc['title']}\nDescription: {doc['description']}"
            for doc in retrieved_docs
        ])

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

def create_video_embed(video_url, start_time, end_time):
    """Create video embed HTML (implementation remains the same)"""
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
