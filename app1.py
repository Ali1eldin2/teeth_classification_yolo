import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load your trained YOLO classification model
@st.cache_resource  # cache so it doesn’t reload every time
def load_model():
    return YOLO(r"D:\cellula\task1_yolo\runs\classify\yolo_classify_teeth\weights\best.pt")  # path to your best model

model = load_model()

# Streamlit UI
st.title("🦷 Teeth Classification App (YOLOv8)")

st.write("Upload an image to classify using your trained YOLO model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    st.write("Classifying...")
    results = model.predict(image)

    # Show results
    for r in results:
        probs = r.probs  # classification probabilities
        top1 = probs.top1
        conf = probs.top1conf.item()
        label = model.names[top1]

        st.success(f"Prediction: **{label}** with confidence {conf:.2f}")