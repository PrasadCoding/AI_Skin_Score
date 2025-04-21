import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms
import random

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("acne_model.pth", map_location=device)
model.eval()

# Define the class names
class_names = ['Low', 'Moderate', 'Severe']

# Define preprocessing transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Score function based on predicted acne severity
def acne_score(pred_class):
    if pred_class == 0:
        return random.randint(50, 60)
    elif pred_class == 1:
        return random.randint(70, 80)
    elif pred_class == 2:
        return random.randint(90, 100)
    else:
        return 0

# Prediction function
def predict_image(image_pil):
    image = image_pil.convert('RGB')
    img_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    pred_class = predicted.item()
    severity = class_names[pred_class]
    score = acne_score(pred_class)
    return severity, score

# Streamlit UI
st.set_page_config(page_title="AI-Skin-Care", layout="centered")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Instructions", "Analyze Your Face Skin", "Feedback"])

if page == "Home":
    st.title("💆‍♀️ AI-Skin-Care")
    st.markdown("🔬 *This is a demo app. For actual skin analysis, consult a dermatologist.*")

elif page == "Instructions":
    st.title("📖 Instructions")
    st.markdown("""
    1. Make sure you are in a well-lit environment.  
    2. Remove any heavy makeup for an accurate result.  
    3. Position your face clearly in the camera frame.  
    4. Click the 'Take Photo' button.  
    5. Wait for your skin health score!
    """)

elif page == "Analyze Your Face Skin":
    st.title("🔍 Analyze Your Face Skin")
    camera_photo = st.camera_input("📸 Take a clear photo of your face")

    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")
        st.image(image, caption='Your Photo', use_column_width=True)

        # Predict severity and score
        severity, score = predict_image(image)

        st.subheader("🧬 Acne Severity Prediction:")
        st.markdown(f"### 🤖 Detected Severity Level: **{severity}**")
        st.markdown(f"<h1 style='color: teal; font-size: 60px'>{score} / 100</h1>", unsafe_allow_html=True)

        if score >= 90:
            st.success("Severe acne detected. We recommend seeing a dermatologist. 🩺")
        elif score >= 70:
            st.info("Moderate acne. Maintain a consistent skincare routine. 🌿🧴")
        else:
            st.success("Low acne severity! Keep taking care of your skin. 💧✨")

elif page == "Feedback":
    st.title("📝 Feedback")
    feedback_text = st.text_area("Share your thoughts about the app:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! 💙")
