import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
MODEL_PATH = "model_rgb.h5"   # updated model name
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
classes = {
    0: 'Speed Limit 20 km/h',
    1: 'Speed Limit 30 km/h',
    2: 'Speed Limit 50 km/h',
    3: 'Speed Limit 60 km/h',
    4: 'Speed Limit 70 km/h',
    5: 'Speed Limit 80 km/h',
    6: 'End of Speed Limit 80 km/h',
    7: 'Speed Limit 100 km/h',
    8: 'Speed Limit 120 km/h',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Streamlit UI
st.set_page_config(page_title="Road Sign Classification", layout="centered")
st.title("🚦 Road Sign Classification App")
st.write("Upload an image of a road sign, and the model will predict the class.")

# File uploader
uploaded_file = st.file_uploader("Upload a road sign image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image in RGB (since model was trained on RGB)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess (resize to 32x32, normalize, keep 3 channels)
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 32, 32, 3)

    # Predict
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    # Show result
    st.markdown(f"###  Predicted Class: {classes[class_index]}")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
