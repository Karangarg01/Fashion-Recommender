import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import ImageFile

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalAveragePooling2D()  # Keeps features compact
])

# Function to extract features
def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = model.predict(img_array).flatten()
        features /= norm(features) + 1e-10  # Normalize features

        return features
    except Exception as e:
        print(f"Skipping {img_path} due to error: {e}")
        return None

# Get image file paths
image_dir = 'images'
filenames = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('jpg', 'jpeg', 'png'))]

feature_list = []

# Process images and extract features
for file in tqdm(filenames):
    features = extract_features(file, model)
    if features is not None:
        feature_list.append(features)

# Convert to NumPy array
feature_list = np.array(feature_list)

# Save extracted features and filenames
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

# Check feature distribution
plt.figure(figsize=(10, 5))
sns.histplot(feature_list.flatten(), bins=100, kde=True)
plt.title("Feature Value Distribution")
plt.show()

# Print summary statistics
print("Feature Shape:", feature_list.shape)
print("Mean:", np.mean(feature_list))
print("Variance:", np.var(feature_list))
