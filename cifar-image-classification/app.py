
import os

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image



# 1. PAGE CONFIGURATION


st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)



# 2. PROJECT PATH


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_cifar10_model.keras"
)



# 3. CLASS NAMES


classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]






@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


if not os.path.exists(MODEL_PATH):

    st.error(
        "Model not found! Please run train.py first."
    )

    st.stop()


model = load_model()





st.title("🖼️ CIFAR-10 Image Classification")

st.write(
    "Upload an image and let the CNN model "
    "predict its class."
)

st.info(
    "The model recognizes 10 classes: "
    "airplane, automobile, bird, cat, deer, "
    "dog, frog, horse, ship and truck."
)





uploaded_file = st.file_uploader(

    "Drag and drop your image here",

    type=["jpg", "jpeg", "png"]

)


if uploaded_file is not None:

    
    # 7. DISPLAY UPLOADED IMAGE
    

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Original uploaded image",
        width=300
    )


    
    # 8. IMAGE PREPROCESSING
    

    resized_image = image.resize(
        (32, 32)
    )

    image_array = np.array(
        resized_image
    )

    image_array = image_array.astype(
        "float32"
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    
    # 9. MODEL PREDICTION
    

    with st.spinner("Classifying image..."):

        predictions = model.predict(
            image_array,
            verbose=0
        )

    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_class = classes[
        predicted_index
    ]

    confidence = (
        predictions[0][predicted_index] * 100
    )


    
    # 10. DISPLAY PREDICTION
    

    st.subheader("Prediction Result")

    st.success(
        f"Predicted Class: {predicted_class.upper()}"
    )

    st.metric(
        "Confidence Score",
        f"{confidence:.2f}%"
    )


    
    # 11. CONFIDENCE SCORES
    

    confidence_df = pd.DataFrame({

        "Class": classes,

        "Confidence": predictions[0] * 100

    })

    confidence_df = confidence_df.sort_values(
        by="Confidence",
        ascending=False
    )


    
    # 12. BAR CHART
    

    st.subheader(
        "Confidence Scores for All 10 Classes"
    )

    chart_data = confidence_df.set_index(
        "Class"
    )

    st.bar_chart(
        chart_data,
        x_label="Class",
        y_label="Confidence (%)"
    )


    
    # 13. CONFIDENCE TABLE
    

    st.subheader("Detailed Prediction Scores")

    st.dataframe(
        confidence_df.reset_index(drop=True),
        use_container_width=True
    )


    
    # 14. IMAGE INFORMATION
    

    with st.expander("Image preprocessing details"):

        st.write(
            "Original image size:",
            image.size
        )

        st.write(
            "Model input size:",
            resized_image.size
        )

        st.write(
            "Model input shape:",
            image_array.shape
        )
