import streamlit as st
from dotenv import load_dotenv
from utils import get_rag_response, generate_embedding, insert_embeddings, collection
from add_product_page import main as add_product_main
load_dotenv()

st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #fafafa;
    }
    
    /* Header styling */
    .stTitle {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1e1e1e;
        font-weight: 700;
        padding-bottom: 2rem;
    }
    
    /* Chat container styling */
    .stChatMessage {
        background-color: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Product card styling */
    .product-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #FF4B6B;
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #FF3358;
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f1f1f1;
    }
    
    /* Video player styling */
    .stVideo {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Custom divider */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #FF4B6B 0%, #FF8E53 100%);
        margin: 1rem 0;
        border-radius: 2px;
    }
    
    /* Add Product Data button */
    .add-product-btn {
        position: fixed;
        top: 20px;
        right: 20px;
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
        z-index: 1000;
    }
    .add-product-btn:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)


def render_product_details(source):
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="product-card">
                <h3 style="color: #FF4B6B;">{source['title']}</h3>
                <div style="margin: 1rem 0;">
                    <div style="background: linear-gradient(90deg, #FF4B6B {source['similarity']}%, #f1f1f1 {source['similarity']}%); 
                         height: 6px; border-radius: 3px; margin-bottom: 0.5rem;"></div>
                    <p style="color: #666;">Similarity Score: {source['similarity']}%</p>
                </div>
                <p style="color: #333; font-size: 1.1em;">{source['description']}</p>
                <p style="color: #666;">Product ID: {source['product_id']}</p>
                <a href="{source['link']}" target="_blank" style="
                    display: inline-block;
                    background: #FF4B6B;
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
                st.video(source['video_url'])

def add_product_data():
    st.markdown('<div class="add-product-container">', unsafe_allow_html=True)
    st.subheader("Add Product Data")
    
    product_id = st.text_input("Product ID")
    title = st.text_input("Title")
    description = st.text_area("Description")
    link = st.text_input("Link")
    video_url = st.text_input("Video URL")
    
    if st.button("Add Product"):
        if product_id and title and description and link and video_url:
            product_data = {
                "product_id": product_id,
                "title": title,
                "desc": description,
                "link": link,
                "video_url": video_url
            }
            
            with st.spinner("Processing product..."):
                embeddings, error = generate_embedding(product_data)
                
                if error:
                    st.error(f"Error processing product: {error}")
                else:
                    insert_result = insert_embeddings(collection, embeddings)
                    
                    if insert_result:
                        st.success("Product data added successfully!")
                    else:
                        st.error("Failed to add product data.")
        else:
            st.warning("Please fill in all fields.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def chat_page():
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #FF4B6B; font-size: 3em; font-weight: 800;">🤵‍♂️ Fashion AI Assistant</h1>
            <p style="color: #666; font-size: 1.2em;">Your personal style advisor powered by AI</p>
        </div>
    """, unsafe_allow_html=True)

    chat_container = st.container()
    
    with chat_container:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "👗"):
                if message["role"] == "assistant":
                    st.markdown(message["content"]["response"])
                    if message["content"]["metadata"]:
                        with st.expander("View Product Details 🛍️"):
                            for source in message["content"]["metadata"]["sources"]:
                                render_product_details(source)
                                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])

    if prompt := st.chat_input("Ask about fashion products..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar="🤵‍♂️"):
            with st.spinner("Finding perfect matches..."):
                response_data = get_rag_response(prompt)
                st.markdown(response_data["response"])
                if response_data["metadata"]:
                    with st.expander("View Product Details 🛍️"):
                        for source in response_data["metadata"]["sources"]:
                            render_product_details(source)
                            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response_data})
    
    st.markdown("""
        <a class="add-product-btn" href="/?page=add_product">
            + Product Data
        </a>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1.5rem; background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h2 style="color: #FF4B6B;">Your Fashion Style Guide</h2>
            <p style="color: #666;">How can I help you with, There are various things I can help - </p>
            <ul style="color: #333;">
                <li>Finding perfect outfits</li>
                <li>Style recommendations</li>
                <li>Product information</li>
                <li>Fashion advice</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def main():
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", ["chat"])[0]

    if page == "chat":
        chat_page()
    elif page == "add_product":
        add_product_main()

if __name__ == "__main__":
    main()
