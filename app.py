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
    """Create an embedded video player with timestamp support."""
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
                    allow="fullscreen; picture-in-picture" 
                    allowfullscreen>
                </iframe>
            """
        else:
            return f"""
                <video 
                    width="100%" 
                    height="315" 
                    controls 
                    preload="auto"
                    id="video-player">
                    <source src="{video_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <script>
                    var video = document.getElementById('video-player');
                    video.addEventListener('loadedmetadata', function() {{
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
            is_video = source.get("type") == "video"
            section_title = "📹 Video Segment" if is_video else "📝 Product Details"
            
            # Create store link HTML if link exists
            store_link_html = ""
            if source.get('link') and isinstance(source['link'], str) and len(source['link'].strip()) > 0:
                store_link_html = f"""
                    <div style="margin-top: 1rem;">
                        <a href="{source['link']}" 
                           target="_blank" 
                           style="
                               display: inline-block;
                               background-color: #81E831;
                               color: white;
                               padding: 12px 24px;
                               border-radius: 25px;
                               text-decoration: none;
                               font-weight: 500;
                               margin-top: 15px;
                               box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                               transition: all 0.3s ease;
                           "
                           onmouseover="this.style.backgroundColor='#6bc428'"
                           onmouseout="this.style.backgroundColor='#81E831'">
                            View on Store
                        </a>
                    </div>
                """
            
            # Product card HTML
            card_html = f"""
                <div style="
                    background-color: white; 
                    padding: 1.8rem; 
                    border-radius: 12px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    border: 1px solid #f0f0f0;
                    transition: all 0.3s ease;
                ">
                    <h3 style="
                        color: #333; 
                        margin-bottom: 1rem;
                        font-size: 1.3rem;
                        font-weight: 600;
                    ">{section_title}</h3>
                    
                    <h4 style="
                        color: #81E831;
                        font-size: 1.4rem;
                        margin-bottom: 1rem;
                        font-weight: 700;
                    ">{source.get('title', 'No Title')}</h4>
                    
                    <div style="margin: 1.2rem 0;">
                        <div style="
                            background: linear-gradient(90deg, #81E831 {source.get('similarity', 0)}%, #f1f1f1 {source.get('similarity', 0)}%); 
                            height: 8px; 
                            border-radius: 4px; 
                            margin-bottom: 0.8rem;
                        "></div>
                        <p style="
                            color: #666;
                            font-size: 0.95rem;
                        ">Similarity Score: {source.get('similarity', 0)}%</p>
                    </div>
                    
                    <p style="
                        color: #333; 
                        font-size: 1.1em;
                        line-height: 1.6;
                        margin-bottom: 1rem;
                    ">{source.get('description', 'No description available')}</p>
                    
                    <p style="
                        color: #666;
                        font-size: 0.9rem;
                        margin-bottom: 0.5rem;
                    ">Product ID: {source.get('product_id', 'N/A')}</p>
                    
                    {f'<p style="color: #666; font-size: 0.9rem;">Segment Time: {source.get("start_time", 0):.1f}s - {source.get("end_time", 0):.1f}s</p>' if is_video else ''}
                    {store_link_html}
                </div>
            """
            
            st.markdown(card_html, unsafe_allow_html=True)
        
        with col2:
            if source.get('video_url'):
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
                    st.video(source['video_url'], start_time=0)

def render_results_section(response_data):
    """Helper function to render results in the chat interface"""
    if response_data.get("metadata") and response_data["metadata"].get("sources"):
        with st.expander("View Product Details 🛍️", expanded=True):
            metadata = response_data["metadata"]
            
            # Summary card
            st.markdown(f"""
                <div style="
                    margin-bottom: 2rem; 
                    padding: 1.5rem; 
                    background-color: #f8f9fa; 
                    border-radius: 12px;
                    border: 1px solid #e9ecef;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
                ">
                    <h4 style="
                        color: #333;
                        font-size: 1.2rem;
                        margin-bottom: 1rem;
                        font-weight: 600;
                    ">Search Results Summary</h4>
                    <p style="
                        color: #444;
                        font-size: 1.1rem;
                        margin-bottom: 0.8rem;
                    ">Found {metadata["total_sources"]} relevant matches:</p>
                    <ul style="
                        color: #666;
                        font-size: 1rem;
                        list-style-type: none;
                        padding-left: 0;
                    ">
                        <li style="margin-bottom: 0.5rem;">📝 {metadata["text_sources"]} product descriptions</li>
                        <li>🎥 {metadata["video_sources"]} video segments</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            # Text results
            text_sources = [s for s in metadata["sources"] if s.get("type") == "text"]
            if text_sources:
                st.markdown("""
                    <h3 style="
                        color: #333;
                        font-size: 1.5rem;
                        margin: 2rem 0 1.5rem;
                        font-weight: 600;
                    ">📝 Retrieved Products</h3>
                """, unsafe_allow_html=True)
                for source in text_sources:
                    render_product_details(source)
                    st.markdown('<hr style="margin: 2rem 0; border-color: #f0f0f0;">', unsafe_allow_html=True)
            
            # Video results
            video_sources = [s for s in metadata["sources"] if s.get("type") == "video"]
            if video_sources:
                st.markdown("""
                    <h3 style="
                        color: #333;
                        font-size: 1.5rem;
                        margin: 2rem 0 1.5rem;
                        font-weight: 600;
                    ">📹 Matching Product Videos</h3>
                """, unsafe_allow_html=True)
                for source in video_sources:
                    render_product_details(source)
                    st.markdown('<hr style="margin: 2rem 0; border-color: #f0f0f0;">', unsafe_allow_html=True)
                    
def chat_page():
    """Main chat interface implementation"""
    # Page Header
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #81E831; font-size: 3em; font-weight: 800;">🤵‍♂️ Fashion AI Assistant</h1>
            <p style="color: #666; font-size: 1.2em;">Your personal style advisor powered by AI</p>
        </div>
    """, unsafe_allow_html=True)

    # Navigation buttons
    st.markdown("""
        <div class="nav-container">
            <a href="?page=add_product" class="nav-button">Add Product Data</a>
            <a href="?page=visual_search" class="nav-button">Visual Search</a>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state for messages if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat container for all messages
    chat_container = st.container()
    
    with chat_container:
        # Display all messages in the chat history
        for message in st.session_state.messages:
            with st.chat_message(
                message["role"],
                avatar="👤" if message["role"] == "user" else "👗"
            ):
                if message["role"] == "assistant":
                    # Display assistant's text response
                    st.markdown(message["content"]["response"])
                    
                    # Display product details if metadata exists
                    if message["content"].get("metadata") and message["content"]["metadata"].get("sources"):
                        with st.expander("View Product Details 🛍️", expanded=True):
                            metadata = message["content"]["metadata"]
                            
                            # Display summary statistics
                            st.markdown(f"""
                                <div style="margin-bottom: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 8px;">
                                    <h4 style="color: #333;">Search Results Summary</h4>
                                    <p>Found {metadata["total_sources"]} relevant matches:</p>
                                    <ul>
                                        <li>{metadata["text_sources"]} product descriptions</li>
                                        <li>{metadata["video_sources"]} video segments</li>
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Display text results first
                            text_sources = [s for s in metadata["sources"] if s.get("type") == "text"]
                            if text_sources:
                                st.markdown("### 📝 Retrieved Products")
                                for source in text_sources:
                                    render_product_details(source)
                                    st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)
                            
                            # Display video results second
                            video_sources = [s for s in metadata["sources"] if s.get("type") == "video"]
                            if video_sources:
                                st.markdown("### 📹 Matching Product Videos")
                                for source in video_sources:
                                    render_product_details(source)
                                    st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)
                else:
                    # Display user's message
                    st.markdown(message["content"])

    # Chat input for user
    prompt = st.chat_input("Hey! Ask me anything about fashion - styles, outfits, trends...")
    
    if prompt:
        # Display user's new message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Add user message to session state
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Get and display assistant's response
        with st.chat_message("assistant", avatar="👗"):
            with st.spinner("Finding perfect matches..."):
                try:
                    # Get response from multimodal RAG system
                    response_data = get_multimodal_rag_response(prompt)
                    
                    # Display the text response
                    st.markdown(response_data["response"])
                    
                    # Display product details if available
                    if response_data.get("metadata") and response_data["metadata"].get("sources"):
                        with st.expander("View Product Details 🛍️", expanded=True):
                            metadata = response_data["metadata"]
                            
                            # Display summary statistics
                            st.markdown(f"""
                                <div style="margin-bottom: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 8px;">
                                    <h4 style="color: #333;">Search Results Summary</h4>
                                    <p>Found {metadata["total_sources"]} relevant matches:</p>
                                    <ul>
                                        <li>{metadata["text_sources"]} product descriptions</li>
                                        <li>{metadata["video_sources"]} video segments</li>
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Display text results first
                            text_sources = [s for s in metadata["sources"] if s.get("type") == "text"]
                            if text_sources:
                                st.markdown("### 📝 Retrieved Products")
                                for source in text_sources:
                                    render_product_details(source)
                                    st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)
                            
                            # Display video results second
                            video_sources = [s for s in metadata["sources"] if s.get("type") == "video"]
                            if video_sources:
                                st.markdown("### 📹 Matching Product Videos")
                                for source in video_sources:
                                    render_product_details(source)
                                    st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    response_data = {
                        "response": "I encountered an error while processing your request. Please try again.",
                        "metadata": None
                    }
        
        # Add assistant's response to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data
        })
    
    # Sidebar content
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
