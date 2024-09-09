# app.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the saved model
loaded_model = tf.keras.models.load_model("model.keras")

# Define a dictionary to map numerical classifications to class names
Cancer_label = {
    0: "Glioma",
    1: "Meningioma",
    2: "No_Tumor",
    3: "Pituitary"
}

# Define a function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (180, 180))
    image = image / 255.0
    return image

# Define a function to make predictions
def make_prediction(image):
    image = np.expand_dims(image, axis=0)
    prediction = loaded_model.predict(image)
    return prediction

# Create a Streamlit app
st.title("Brain Tumor Detection App")

# Upload an image file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Preprocess the image
    image = preprocess_image(image)

    # Make a prediction
    prediction = make_prediction(image)
    prediction_class = np.argmax(prediction)

    # Get the class name from the dictionary
    predicted_class_name = Cancer_label[prediction_class]

    # Display the prediction
    st.write("Prediction:", predicted_class_name)

    # Display the image
    st.image(image, caption="Uploaded Image", width=300)