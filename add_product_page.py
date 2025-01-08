import streamlit as st
from utils import generate_embedding, insert_embeddings, collection

st.markdown("""
<style>
    .custom-btn {
        position: relative;
        display: inline-block;
        padding: 10px 20px;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
        border: none;
        transition: background-color 0.3s;
    }
    .custom-btn:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

def add_product_data():
    st.subheader("Add Product Data")
    
    product_id = st.text_input("Product ID")
    title = st.text_input("Title")
    description = st.text_area("Description")
    link = st.text_input("Link")
    video_url = st.text_input("Video URL")
    
    col1, col2 = st.columns(2)
    
    with col1:
        add_product_clicked = st.button("Add Product")
    
    with col2:
        st.markdown('<a href="/" class="custom-btn">Chat Application</a>', unsafe_allow_html=True)
    
    if add_product_clicked:
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

def main():
    add_product_data()

if __name__ == "__main__":
    main()
