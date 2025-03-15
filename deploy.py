import streamlit as st
import pickle
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import io
import os

# Set page config
st.set_page_config(page_title="Fashion Recommender", layout="wide")

# Cloudinary Base URL
CLOUDINARY_BASE_URL = "https://res.cloudinary.com/dg1m6ud6y/image/upload/"

# Load embeddings and filenames
@st.cache_resource
def load_embeddings():
    if not os.path.exists("embeddings.pkl") or not os.path.exists("filenames.pkl"):
        st.error("Error: Required files (embeddings.pkl or filenames.pkl) are missing. Please check deployment.")
        return None, None

    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
    filenames = [f.replace("images\\", "").replace("images/", "") for f in filenames]
    return feature_list, filenames

feature_list, filenames = load_embeddings()


# Load ResNet50 Model
@st.cache_resource
def load_model():
    model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    return tf.keras.Sequential([model, tf.keras.layers.GlobalMaxPooling2D()])

model = load_model()

# Feature extraction
def extract_features(img, model):
    img = img.resize((224, 224))  # Resize image
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array).flatten()
    features /= np.linalg.norm(features) + 1e-10
    return features

# Recommendation function
def recommend(features, feature_list, top_n=5):
    similarities = cosine_similarity([features], feature_list)[0]
    indices = similarities.argsort()[-top_n:][::-1]
    return indices

# Sidebar with instructions
st.sidebar.title("Fashion Recommender System")
st.sidebar.write("Upload or drag & drop an image to find similar fashion items!")

# Main UI
st.markdown("""
    <h1 style='text-align: center; color: #FFFFFF;'>Fashion Recommender System</h1>
    <p style='text-align: center;'>Find similar fashion items by uploading an image</p>
    <hr>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    # Convert uploaded file to image
    img = Image.open(io.BytesIO(uploaded_file.getvalue()))
    st.markdown("<h2 style='text-align: center;'>Uploaded Image</h2>", unsafe_allow_html=True)
    st.image(img, caption="Uploaded Image", use_container_width=False, width=300)

    with st.spinner("Finding similar items..."):
        features = extract_features(img, model)
        indices = recommend(features, feature_list)

    if indices is not None and len(indices) > 0:
        st.markdown("<h2 style='text-align: center;'>Recommended Fashion Items</h2>", unsafe_allow_html=True)
        col_images = st.columns(5)
        for i, col in enumerate(col_images):
            cloudinary_url = filenames[indices[i]]  # Directly use the Cloudinary URL
            col.image(cloudinary_url, use_container_width=True)
    else:
        st.error("No recommendations found.")

st.write("\n\nApplication loaded successfully!")
