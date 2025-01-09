import os
import streamlit as st       
from dotenv import load_dotenv
   
load_dotenv()
from twelvelabs import TwelveLabs
from pymilvus import connections, Collection
from openai import OpenAI

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
    """Generate embeddings for a single product"""
    try:
        st.write(f"Processing product: {product_info['title']}")
        
        twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)
        
        # Combine title and description
        text = f"{product_info['title']} {product_info['desc']}"
        
        # Create embedding for the combined text
        embedding = twelvelabs_client.embed.create(
            engine_name="Marengo-retrieval-2.6",
            text=text
        ).text_embedding  # Get the embeddings_float attribute
        
        embeddings = [{
            'embedding': embedding.segments[0].embeddings_float,
            'video_url': product_info['video_url'],
            'product_id': product_info['product_id'],
            'title': product_info['title'],
            'description': product_info['desc'],
            'link': product_info['link']
        }]
        
        return embeddings, None
    except Exception as e:
        return None, str(e)

def insert_embeddings(collection, embeddings):
    """Insert embeddings into Milvus collection"""
    try:
        data = [{
            "id": int(uuid.uuid4().int & (1<<63)-1),  # Generate unique ID
            "vector": embeddings[0]['embedding'],
            "metadata": {
                "video_url": embeddings[0]['video_url'],
                "product_id": embeddings[0]['product_id'],
                "title": embeddings[0]['title'],
                "description": embeddings[0]['description'],
                "link": embeddings[0]['link']
            }
        }]
        
        insert_result = collection.insert(data)
        return insert_result
    except Exception as e:
        st.error(f"Error inserting embeddings: {str(e)}")
        return None
        

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


def search_similar_videos(image, top_k=5):
    twelvelabs_client = TwelveLabs(api_key=TWELVELABS_API_KEY)

    features = image_embedding(
    twelvelabs_client=twelvelabs_client,
    image_file=image
   )

    results = collection.search(
        data=[features],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["metadata"]
    )
    
    search_results = []
    for hits in results:
        for hit in hits:
            metadata = hit.entity.get('metadata')
            if metadata:
                search_results.append({
                    'Title': metadata['title'],
                    'Description': metadata['description'],
                    'Link': metadata['link'],
                    'Start Time': f"{metadata['start_time']:.1f}s",
                    'End Time': f"{metadata['end_time']:.1f}s",
                    'Video URL': metadata['video_url'],
                    'Similarity': f"{(1 - float(hit.distance)) * 100:.2f}%"
                })
    
    return search_results

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
  
