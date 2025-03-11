import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load embeddings & filenames
st.write("Loading embeddings...")
try:
    feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
    filenames = pickle.load(open("filenames.pkl", "rb"))
    st.write("Embeddings loaded successfully")
except Exception as e:
    st.error(f"Error loading embeddings: {e}")

# Load model with caching
@st.cache_resource
def load_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

model = load_model()

st.title("Fashion Recommender System")
st.write("Upload an image to find similar fashion items.")

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"File upload error: {e}")
        return None

# Feature extraction function
def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array).flatten()
        return features / norm(features)
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

# Recommendation function
def recommend(features, feature_list):
    try:
        neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.error(f"Recommendation error: {e}")
        return []

# File upload process
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        # Display uploaded image
        display_image = Image.open(file_path)
        st.image(display_image, caption="Uploaded Image", use_container_width=True)

        # Extract features
        features = feature_extraction(file_path, model)
        if features is not None:
            indices = recommend(features, feature_list)

            # Display recommended images
            if indices is not None and len(indices) > 0:
                st.write("Recommended Fashion Items")
                col1, col2, col3, col4, col5 = st.columns(5)

                for col, idx in zip([col1, col2, col3, col4, col5], indices[0][:5]):
                    img_path = filenames[idx]
                    if os.path.exists(img_path):
                        col.image(img_path, use_container_width=True)
                    else:
                        col.error(f"Image not found: {img_path}")
            else:
                st.error("No recommendations found")

    else:
        st.error("Error in file upload")
