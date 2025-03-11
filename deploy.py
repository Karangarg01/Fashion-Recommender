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

# ✅ Load embeddings & filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# ✅ Load ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(324, 324, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# ✅ Streamlit UI
st.title('Fashion Recommender System')

# ✅ Save uploaded file function
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# ✅ Feature extraction function
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(324, 324))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# ✅ Recommendation function
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# ✅ File Upload Process
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image")

        # Extract features
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Get recommendations
        indices = recommend(features, feature_list)

        # Show recommended images
        # Show recommended images with larger size
        # Show recommended images with larger size
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]], use_container_width=True)  # Adjust width automatically
        with col2:
            st.image(filenames[indices[0][1]], use_container_width=True)
        with col3:
            st.image(filenames[indices[0][2]], use_container_width=True)
        with col4:
            st.image(filenames[indices[0][3]], use_container_width=True)
        with col5:
            st.image(filenames[indices[0][4]], use_container_width=True)


    else:
        st.error("Error occurred in file upload!")

print("✅ Deploy script executed successfully!")  # Debugging message
