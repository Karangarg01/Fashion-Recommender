import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from PIL import ImageFile

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to extract features from an image
def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))  # Load image safely
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        print(f"Skipping {img_path} due to error: {e}")  # Log corrupted images
        return None

# Get image file paths
filenames = [os.path.join('images', file) for file in os.listdir('images')]

feature_list = []

# Process images and extract features
for file in tqdm(filenames):
    features = extract_features(file, model)
    if features is not None:  # Only add valid images
        feature_list.append(features)

# Save extracted features and filenames
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
