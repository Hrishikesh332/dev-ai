import streamlit as st
from utils import generate_embedding, insert_embeddings, collection

def add_product_data():
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
    
    st.markdown('<a href="/">Return to Chat</a>', unsafe_allow_html=True)

def main():
    add_product_data()

if __name__ == "__main__":
    main()
