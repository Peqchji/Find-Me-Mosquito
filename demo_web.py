import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

# Constants
img_size = 224
input_shape = (img_size, img_size, 3)
class_names = [
    "Aedes aegypti landing", "Aedes aegypti smashed",
    "Aedes albopictus landing", "Aedes albopictus smashed",
    "Culex quinquefasciatus landing", "Culex quinquefasciatus smashed"
]

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

# Streamlit page configuration
st.set_page_config(
    page_title="Mosquito Classification App",
    page_icon="🦟",
    layout="wide"
)

# Streamlit web application title
st.title("Mosquito Classification")

# Create two columns
col1, col2 = st.columns(2)

# In the first column, select the model to use
with col1:
    model_selector = st.selectbox("Select Model", ["Model With Hypertuning Parameter", "Overfit Model"])
    if model_selector == "Model With Hypertuning Parameter":
        model_path = "save_at_70.keras"
    else:
        model_path = "save_at_100.keras"
    model = load_model(model_path)

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# If an image is uploaded
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    image = tf.image.decode_image(uploaded_image.read(), channels=3)
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.expand_dims(image, 0)

    # Make predictions
    with st.spinner("Making predictions..."):
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)

    # Display classification result
    st.subheader("Classification Result:")
    st.write("Predicted Class:", f"{class_names[predicted_class]}")
    st.write("Confidence:", f"{predictions[0][predicted_class] * 100:.2f}%")
    st.write("<hr>", unsafe_allow_html=True)

    # Display raw predictions for each class
    st.subheader("Raw Predictions:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i] * 100:.2f}%")
        st.markdown("<hr>", unsafe_allow_html=True)