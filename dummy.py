import pickle

# Load the existing filenames.pkl
with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

# Cloudinary details
CLOUD_NAME = "dg1m6ud6y"  # Replace with your actual Cloudinary cloud name
BASE_URL = f"https://res.cloudinary.com/{CLOUD_NAME}/image/upload/"

# Convert local filenames to Cloudinary URLs
filenames = [BASE_URL + fname for fname in filenames]

# Save the updated filenames.pkl
with open('filenames.pkl', 'wb') as f:
    pickle.dump(filenames, f)

print("Updated filenames.pkl with Cloudinary URLs!")
