"""
CIFAR-10 classifier web app (Streamlit) powered by a fine-tuned MobileNetV2.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
import os

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ----------------------------- Config ---------------------------------------
WEIGHTS_PATH = r"C:\Users\SRIHITH REDDY\OneDrive\Desktop\cifar\models\best_cifar10_mobilenetv2.pth"   
IMG_SIZE = 224        # MUST match IMG_SIZE used during training
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️", layout="centered")


# ----------------------------- Model ----------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# The model was trained on 32x32 CIFAR images upscaled to IMG_SIZE, so inference
# uses the same pipeline: resize to 32x32 first, then upscale + normalize.
to_tensor = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def predict(model, img: Image.Image):
    small = img.convert("RGB").resize((32, 32), Image.LANCZOS)   # auto-resize to 32x32
    x = to_tensor(small).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return small, probs.numpy()


# ----------------------------- UI -------------------------------------------
st.title("🖼️ CIFAR-10 Image Classifier")
st.caption("MobileNetV2 fine-tuned with transfer learning. Upload an image of an "
           "airplane, car, bird, cat, deer, dog, frog, horse, ship or truck.")

if not os.path.isfile(WEIGHTS_PATH):
    st.error(f"Weights file `{WEIGHTS_PATH}` not found. Place it next to `app.py`.")
    st.stop()

model = load_model(WEIGHTS_PATH)

uploaded = st.file_uploader("Drag and drop an image here, or click to browse",
                            type=["png", "jpg", "jpeg", "bmp", "webp"])

if uploaded is None:
    st.info("Upload an image to get a prediction.")
    st.stop()

image = Image.open(uploaded)
small, probs = predict(model, image)
top = int(probs.argmax())

col1, col2 = st.columns(2)
with col1:
    st.subheader("Uploaded image")
    st.image(image, use_container_width=True)
with col2:
    st.subheader("Model input (32 × 32)")
    st.image(small.resize((256, 256), Image.NEAREST), caption="Enlarged for viewing")

st.divider()
st.subheader("Prediction")
st.metric(label="Predicted class", value=CLASSES[top].capitalize(),
          delta=f"{probs[top] * 100:.1f}% confidence", delta_color="off")

st.subheader("Confidence scores")
df = pd.DataFrame({"Confidence (%)": probs * 100}, index=CLASSES)
st.bar_chart(df, y="Confidence (%)", horizontal=True)

with st.expander("Show raw scores"):
    st.dataframe(df.sort_values("Confidence (%)", ascending=False).style.format("{:.2f}"))