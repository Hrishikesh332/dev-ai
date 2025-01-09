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

def generate_embedding(product_info):
    """Generate both text and video embeddings for a product"""
    try:
        st.write("Starting embedding generation process...")
        st.write(f"Processing product: {product_info['title']}")
        
        # Log input data structure
        st.write("Input data structure:")
        st.write({
            "title": product_info['title'],
            "description": product_info['desc'][:100] + "...",  # Truncate for logging
            "video_url": product_info['video_url']
        })
        
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        st.write("TwelveLabs client initialized successfully")
        
        # Generate text embedding
        st.write("Attempting to generate text embedding...")
        text = f"{product_info['title']} {product_info['desc']}"
        st.write(f"Combined text length: {len(text)} characters")
        
        try:
            text_embedding_response = twelvelabs_client.embed.create(
                model_name="Marengo-retrieval-2.7",
                text=text
            )
            st.write("Text embedding API call successful")
            text_embedding = text_embedding_response.text_embedding.segments[0].embeddings_float
            st.write(f"Text embedding generated successfully. Vector length: {len(text_embedding)}")
        except Exception as text_error:
            st.error(f"Error in text embedding generation: {str(text_error)}")
            st.write(f"Full text embedding error: {text_error.__class__.__name__}: {str(text_error)}")
            raise
        
        # Generate video embedding
        st.write("Attempting to generate video embedding...")
        st.write(f"Video URL: {product_info['video_url']}")
        
        try:
            video_embedding_response = twelvelabs_client.embed.create(
                model_name="Marengo-retrieval-2.7",
                video_url=product_info['video_url']
            )
            st.write("Video embedding API call successful")
            video_embedding = video_embedding_response.video_embedding.segments[0].embeddings_float
            st.write(f"Video embedding generated successfully. Vector length: {len(video_embedding)}")
        except Exception as video_error:
            st.error(f"Error in video embedding generation: {str(video_error)}")
            st.write(f"Full video embedding error: {video_error.__class__.__name__}: {str(video_error)}")
            raise
        
        st.write("Both embeddings generated successfully")
        return {
            'text_embedding': text_embedding,
            'video_embedding': video_embedding
        }, None
    except Exception as e:
        st.error("Final error in embedding generation")
        st.error(f"Error type: {e.__class__.__name__}")
        st.error(f"Error message: {str(e)}")
        st.error(f"Full error details: {repr(e)}")
        return None, str(e)


def insert_embeddings(embeddings_data, product_info):
    """Insert both text and video embeddings into the same collection"""
    try:
        st.write("Starting embedding insertion process...")
        
        # Log the incoming data structure
        st.write("Embedding data structure check:")
        st.write(f"Text embedding vector length: {len(embeddings_data['text_embedding'])}")
        st.write(f"Video embedding vector length: {len(embeddings_data['video_embedding'])}")
        
        # Create metadata dictionary
        metadata = {
            "product_id": product_info['product_id'],
            "title": product_info['title'],
            "description": product_info['desc'],
            "video_url": product_info['video_url'],
            "link": product_info['link']
        }
        st.write("Metadata prepared successfully")
        
        # Insert text embedding
        text_entry = {
            "id": int(uuid.uuid4().int & (1<<63)-1),
            "vector": embeddings_data['text_embedding'],
            "metadata": metadata,
            "embedding_type": "text"
        }
        st.write("Attempting to insert text embedding...")
        collection.insert([text_entry])
        st.write("Text embedding inserted successfully")
        
        # Insert video embedding
        video_entry = {
            "id": int(uuid.uuid4().int & (1<<63)-1),
            "vector": embeddings_data['video_embedding'],
            "metadata": metadata,
            "embedding_type": "video"
        }
        st.write("Attempting to insert video embedding...")
        collection.insert([video_entry])
        st.write("Video embedding inserted successfully")
        
        st.write("All embeddings inserted successfully")
        return True
    except Exception as e:
        st.error("Error in embedding insertion process")
        st.error(f"Error type: {e.__class__.__name__}")
        st.error(f"Error message: {str(e)}")
        st.error(f"Full error details: {repr(e)}")
        return False

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
