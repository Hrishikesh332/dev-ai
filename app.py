import streamlit as st
from dotenv import load_dotenv
from utils import get_rag_response, generate_embedding, insert_embeddings, collection, get_multimodal_rag_response

load_dotenv()

st.markdown("""
<style>
    .main {
        padding: 2rem;
        background-color: #fafafa;
    }
    
    .stTitle {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1e1e1e;
        font-weight: 700;
        padding-bottom: 2rem;
    }
    
    .stChatMessage {
        background-color: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .product-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .nav-button {
        background-color: #81E831 !important;  /* Matching pink from app */
        color: white !important;
        border-radius: 4px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        cursor: pointer !important;
    }
    
    .nav-button:hover {
        background-color: #ff3d60 !important;  /* Slightly darker shade of the same pink */
        opacity: 0.9;
    }
    
    
    .nav-container {
        position: fixed;
        top: 80px;
        right: 20px;
        display: flex;
        gap: 10px;
        z-index: 1000;
    }
    .stButton button {
        background-color: #81E831 !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton button:hover {
        background-color: #eeecec !important;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

def create_video_embed(video_url, start_time=0, end_time=0):
    """Create an embedded video player with timestamp support"""
    try:
        if 'vimeo.com' in video_url:
            video_id = video_url.split('/')[-1].split('?')[0]
            start_seconds = str(int(float(start_time)))
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
        else:
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
    except Exception as e:
        st.error(f"Error creating video embed: {str(e)}")
        return f"<p>Error creating video embed for URL: {video_url}</p>"

def render_product_details(source):
    """Helper function to render product details in a consistent format"""
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="product-card">
                <h3 style="color: #81E831;">{source['title']}</h3>
                <div style="margin: 1rem 0;">
                    <div style="background: linear-gradient(90deg, #81E831 {source['similarity']}%, #f1f1f1 {source['similarity']}%); 
                         height: 6px; border-radius: 3px; margin-bottom: 0.5rem;"></div>
                    <p style="color: #666;">Similarity Score: {source['similarity']}%</p>
                </div>
                <p style="color: #333; font-size: 1.1em;">{source['description']}</p>
                <p style="color: #666;">Product ID: {source['product_id']}</p>
                {f'<p style="color: #666;">Segment Time: {source["start_time"]:.1f}s - {source["end_time"]:.1f}s</p>' if source.get("type") == "video" else ""}
                <a href="{source['link']}" target="_blank" style="
                    display: inline-block;
                    background: #81E831;
                    color: white;
                    padding: 0.5rem 1.5rem;
                    border-radius: 25px;
                    text-decoration: none;
                    margin-top: 1rem;
                    transition: all 0.3s ease;
                ">View on Store</a>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if source['video_url']:
                if source.get('type') == 'video':
                    st.markdown(
                        create_video_embed(
                            source['video_url'],
                            source.get('start_time', 0),
                            source.get('end_time', 0)
                        ),
                        unsafe_allow_html=True
                    )
                else:
                    st.video(source['video_url'])

def chat_page():
    """Main chat interface implementation"""
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #81E831; font-size: 3em; font-weight: 800;">🤵‍♂️ Fashion AI Assistant</h1>
            <p style="color: #666; font-size: 1.2em;">Your personal style advisor powered by AI</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="nav-container">
            <a href="?page=add_product" class="nav-button">Add Product Data</a>
            <a href="?page=visual_search" class="nav-button">Visual Search</a>
        </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(
                message["role"],
                avatar="👤" if message["role"] == "user" else "👗"
            ):
                if message["role"] == "assistant":
                    st.markdown(message["content"]["response"])
                    
                    if message["content"].get("metadata") and message["content"]["metadata"].get("sources"):
                        with st.expander("View Product Details 🛍️"):
                            metadata = message["content"]["metadata"]
                            
                            # Ensure metadata has required fields with defaults
                            text_sources = metadata.get("text_sources", 0)
                            video_sources = metadata.get("video_sources", 0)
                            total_sources = metadata.get("total_sources", len(metadata["sources"]))
                            
                            st.markdown(f"""
                                <div style="margin-bottom: 1rem; padding: 1rem; background-color: #f8f9fa; border-radius: 8px;">
                                    <h4 style="color: #333;">Search Results Summary</h4>
                                    <p>Found {total_sources} relevant matches:</p>
                                    <ul>
                                        <li>{text_sources} product descriptions</li>
                                        <li>{video_sources} video segments</li>
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            for source in metadata["sources"]:
                                render_product_details(source)
                                st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])

    prompt = st.chat_input("Hey! Ask me anything about fashion - styles, outfits, trends...")
    
    if prompt:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("assistant", avatar="👗"):
            with st.spinner("Finding perfect matches..."):
                try:
                    response_data = get_multimodal_rag_response(prompt)
                    
                    # Ensure response_data has all required fields
                    if response_data.get("metadata") and response_data["metadata"].get("sources"):
                        metadata = response_data["metadata"]
                        if "text_sources" not in metadata:
                            metadata["text_sources"] = sum(1 for s in metadata["sources"] if s.get("type") == "text")
                        if "video_sources" not in metadata:
                            metadata["video_sources"] = sum(1 for s in metadata["sources"] if s.get("type") == "video")
                        if "total_sources" not in metadata:
                            metadata["total_sources"] = len(metadata["sources"])
                    
                    st.markdown(response_data["response"])
                    
                    if response_data.get("metadata") and response_data["metadata"].get("sources"):
                        with st.expander("View Product Details 🛍️"):
                            metadata = response_data["metadata"]
                            st.markdown(f"""
                                <div style="margin-bottom: 1rem; padding: 1rem; background-color: #f8f9fa; border-radius: 8px;">
                                    <h4 style="color: #333;">Search Results Summary</h4>
                                    <p>Found {metadata["total_sources"]} relevant matches:</p>
                                    <ul>
                                        <li>{metadata["text_sources"]} product descriptions</li>
                                        <li>{metadata["video_sources"]} video segments</li>
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            for source in metadata["sources"]:
                                render_product_details(source)
                                st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    response_data = {
                        "response": "I encountered an error while processing your request. Please try again.",
                        "metadata": None
                    }
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data
        })
    
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1.5rem; background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h2 style="color: #81E831;">Your Fashion Style Guide</h2>
            <p style="color: #666;">I can help you with:</p>
            <ul style="color: #333;">
                <li>Finding perfect outfits based on your preferences</li>
                <li>Style recommendations for different occasions</li>
                <li>Detailed product information and comparisons</li>
                <li>Personal fashion advice and trend insights</li>
                <li>Visual search for similar styles</li>
                <li>Video demonstrations of products</li>
            </ul>
            <p style="color: #666; margin-top: 1rem;">
                Try asking about specific styles, occasions, or product features!
            </p>
        </div>
        """, unsafe_allow_html=True)
def main():
    query_params = st.query_params
    page = query_params.get("page", "chat")[0] if query_params.get("page") else "chat"

    if page == "chat":
        chat_page()
    elif page == "add_product":
        add_product_main()
    elif page == "visual_search":
        visual_search_main()

if __name__ == "__main__":
    main()
