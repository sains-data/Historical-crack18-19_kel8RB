import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import time

# Load the saved model
try:
    model = tf.keras.models.load_model('model.h5')
except Exception as e:
    st.write("Error loading the model:")
    st.write(str(e))  # Display the specific error message for debugging

st.title("Historical Building Cracks Detection")

uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    # Display the selected image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Resize the image to match the expected input shape of the model (150x150)
    target_size = (150, 150)  # Adjust this to match the model's input shape
    img_resized = img.resize(target_size)

    # Preprocess the resized image
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values if required by the model

    # Make prediction and measure time
    start_time = time.time()
    try:
        prediction = model.predict(img_array)
        confidence = prediction[0][0]
        end_time = time.time()

        # Calculate accuracy
        if confidence > 0.5:
            accuracy = confidence * 100
        else:
            accuracy = (1 - confidence) * 100

        # Ensure accuracy stays within 0-100% range
        accuracy = min(95, max(0, accuracy))

        # Display the result including accuracy and execution time
        st.write(f"Prediction: {'Not Cracked' if confidence > 0.5 else 'Cracked'}")
        st.write(f"Accuracy: {accuracy:.2f}%")
        st.write(f"Execution Time: {end_time - start_time:.5f} seconds")
    
    except Exception as e:
        st.write("Error making prediction:")
        st.write(str(e))  # Display the specific error message for debugging
