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
    """Generate text and segmented video embeddings for a product"""
    try:
        st.write("Starting embedding generation process...")
        st.write(f"Processing product: {product_info['title']}")
        
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        st.write("TwelveLabs client initialized successfully")
        
        # Generate text embedding
        st.write("Attempting to generate text embedding...")
        text = f"{product_info['title']} {product_info['desc']}"
        text_embedding = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            text=text
        ).text_embedding.segments[0].embeddings_float
        st.write("Text embedding generated successfully")
        
        # Create and wait for video embedding task
        st.write("Creating video embedding task...")
        video_task = twelvelabs_client.embed.task.create(
            model_name="Marengo-retrieval-2.7",
            video_url=product_info['video_url'],
            video_clip_length=6  # Default segment length
        )
        
        def on_task_update(task):
            st.write(f"Video processing status: {task.status}")
        
        st.write("Waiting for video processing to complete...")
        video_task.wait_for_done(sleep_interval=2, callback=on_task_update)
        
        # Retrieve segmented video embeddings
        video_task = video_task.retrieve()
        if not video_task.video_embedding or not video_task.video_embedding.segments:
            raise Exception("Failed to retrieve video embeddings")
        
        video_segments = video_task.video_embedding.segments
        st.write(f"Retrieved {len(video_segments)} video segments")
        
        # Prepare video embeddings with segment information
        video_embeddings = []
        for segment in video_segments:
            video_embeddings.append({
                'embedding': segment.embeddings_float,
                'metadata': {
                    'scope': 'clip',
                    'start_time': segment.start_offset_sec,
                    'end_time': segment.end_offset_sec,
                    'video_url': product_info['video_url']
                }
            })
        
        return {
            'text_embedding': text_embedding,
            'video_embeddings': video_embeddings
        }, None
        
    except Exception as e:
        st.error("Error in embedding generation")
        st.error(f"Error message: {str(e)}")
        return None, str(e)

def insert_embeddings(embeddings_data, product_info):
    """Insert text and all video segment embeddings"""
    try:
        # Create base metadata
        metadata = {
            "product_id": product_info['product_id'],
            "title": product_info['title'],
            "description": product_info['desc'],
            "video_url": product_info['video_url'],
            "link": product_info['link']
        }
        
        # Insert text embedding
        text_entry = {
            "id": int(uuid.uuid4().int & (1<<63)-1),
            "vector": embeddings_data['text_embedding'],
            "metadata": metadata,
            "embedding_type": "text"
        }
        collection.insert([text_entry])
        st.write("Text embedding inserted successfully")
        
        # Insert each video segment embedding
        for video_segment in embeddings_data['video_embeddings']:
            video_entry = {
                "id": int(uuid.uuid4().int & (1<<63)-1),
                "vector": video_segment['embedding'],
                "metadata": {**metadata, **video_segment['metadata']},
                "embedding_type": "video"
            }
            collection.insert([video_entry])
        
        st.write(f"Inserted {len(embeddings_data['video_embeddings'])} video segment embeddings")
        return True
        
    except Exception as e:
        st.error(f"Error inserting embeddings: {str(e)}")
        return False

def search_similar_videos(image_file, top_k=5):
    """Search for similar video segments using image query"""
    try:
        st.write("Generating image embedding...")
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        image_embedding = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            image_file=image_file
        ).image_embedding.segments[0].embeddings_float
        
        st.write("Searching for similar video segments...")
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            data=[image_embedding],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr="embedding_type == 'video'",
            output_fields=["metadata"]
        )

        search_results = []
        for hits in results:
            for hit in hits:
                metadata = hit.metadata
                similarity_score = round((1 - float(hit.distance)) * 100, 2)
                
                search_results.append({
                    'Title': metadata.get('title', ''),
                    'Description': metadata.get('description', ''),
                    'Link': metadata.get('link', ''),
                    'Start Time': f"{metadata.get('start_time', 0):.1f}s",
                    'End Time': f"{metadata.get('end_time', 0):.1f}s",
                    'Video URL': metadata.get('video_url', ''),
                    'Similarity': f"{similarity_score}%"
                })
        
        st.write(f"Found {len(search_results)} matching segments")
        return search_results
        
    except Exception as e:
        st.error(f"Error in visual search: {str(e)}")
        return None
def get_rag_response(question):
    """Get response using text embeddings search"""
    try:
        st.write("Generating question embedding...")
        # Generate embedding for the question
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        question_embedding = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            text=question
        ).text_embedding.segments[0].embeddings_float
        
        st.write("Searching for relevant products...")
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
            expr="embedding_type == 'text'",
            output_fields=["metadata"]
        )

        retrieved_docs = []
        for hits in results:
            for hit in hits:
                metadata = hit.metadata  # Direct access to metadata
                if metadata:
                    similarity = round((1 - float(hit.distance)) * 100, 2)  # Convert distance to similarity score
                    
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

        st.write(f"Found {len(retrieved_docs)} relevant products")
        
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

        st.write("Generating response...")
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
        st.error(f"Error in RAG response: {str(e)}")
        st.error(f"Detailed error: {repr(e)}")
        return {
            "response": "I encountered an error while processing your request.",
            "metadata": None
        }

def get_video_id_from_url(video_url):
    """Extract video ID and platform from URL"""
    try:
        if 'vimeo.com' in video_url:
            video_id = video_url.split('/')[-1].split('?')[0]
            return video_id, 'vimeo'
        else:
            return video_url, 'direct'
    except Exception as e:
        st.error(f"Error processing video URL: {str(e)}")
        return None, None

def format_time_for_url(time_in_seconds):
    """Format time in seconds to URL compatible format"""
    try:
        return str(int(float(time_in_seconds)))
    except:
        return "0"

def create_video_embed(video_url, start_time=0, end_time=0):
    """Create video embed HTML"""
    try:
        video_id, platform = get_video_id_from_url(video_url)
        start_seconds = format_time_for_url(start_time)
        
        if not video_id:
            return f"<p>Unable to process video URL: {video_url}</p>"
        
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
            return f"<p>Unsupported video platform for URL: {video_url}</p>"
            
    except Exception as e:
        st.error(f"Error creating video embed: {str(e)}")
        return f"<p>Error creating video embed for URL: {video_url}</p>"
