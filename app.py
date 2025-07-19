import streamlit as st
import os
import sys
import torch
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt



# Page configuration
st.set_page_config(page_title="X-ray Lung Classifier", page_icon="🩻", layout="wide")

# Styling
def set_background():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to right, #f9fafe, #dfefff);
        }
        .footer {
            text-align: center;
            font-size: 0.85em;
            margin-top: 3rem;
            color: #777;
        }
        </style>
    """, unsafe_allow_html=True)

set_background()

# Multilingual Labels
LANGUAGES = {
    "English": {"title": "X-ray Lung Classifier", "upload": "📤 Upload X-ray images", "result": "🧠 Prediction Result:"},
    "Hindi": {"title": "एक्स-रे फेफड़े वर्गीकरण", "upload": "📤 एक्स-रे छवियां अपलोड करें", "result": "🧠 पूर्वानुमान परिणाम:"},
    "Spanish": {"title": "Clasificador de Rayos X Pulmonares", "upload": "📤 Subir imágenes de rayos X", "result": "🧠 Resultado de Predicción:"}
}

language = st.sidebar.selectbox("🌐 Choose Language", options=list(LANGUAGES.keys()))
L = LANGUAGES[language]

# Sidebar info
with st.sidebar:
    st.title("About This App")
    st.write("""
        This app uses a deep learning model trained on chest X-ray images to detect signs of **Pneumonia**.
    """)
    st.markdown("**Model:** CNN-based PyTorch")
    st.markdown("**Classes:** Normal, Pneumonia")
    st.markdown("**Input size:** 224x224 RGB")



# Title
st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>🩺 {L['title']}</h1>", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    model = torch.load("model/model.pt", map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# Prediction Function
def predict_image(image):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1).numpy()[0]
    return int(np.argmax(probs)), probs

# Plot Confidence Bar
def plot_confidence(probs):
    labels = ["Normal", "Pneumonia"]
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.barh(labels, probs, color=["green", "red"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)

# File Uploader
uploaded_files = st.file_uploader(
    L["upload"], type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="visible"
)

if uploaded_files:
    st.markdown("---")
    for uploaded_file in uploaded_files:
        col1, col2 = st.columns([1, 2])

        # Save image temporarily
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        # Predict
        prediction, probs = predict_image(image)
        result_label = "Normal" if prediction == 0 else "Pneumonia"
        emoji = "✅" if prediction == 0 else "⚠️"

        with col2:
            st.markdown(f"### {L['result']}")
            if prediction == 0:
                st.success(f"{emoji} **{result_label}** (Confidence: {probs[prediction]:.2%})")
            else:
                st.error(f"{emoji} **{result_label}** (Confidence: {probs[prediction]:.2%})")

            plot_confidence(probs)

# Footer
st.markdown("""
    <div class='footer'>
        Developed with ❤️ by Abhishek Maurya using Streamlit & PyTorch | Multilingual | Batch Image Support
    </div>
""", unsafe_allow_html=True)



























# import streamlit as st
# import os
# import torch
# from torchvision.transforms import transforms
# from PIL import Image
# from pathlib import Path


# # this is for saving images and prediction
# def save_image(uploaded_file):
#     if uploaded_file is not None:
#         save_path = os.path.join("images", "input.jpeg")
#         with open(save_path, "wb") as f:
#             f.write(uploaded_file.read())
#         st.success(f"Image saved to {save_path}")

#         model = torch.load(Path('model/model.pt'))


#         trans = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.Resize(224),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             ])

#         image = Image.open(Path('images/input.jpeg'))

#         input = trans(image)

#         input = input.view(1, 1, 224, 224).repeat(1, 3, 1, 1)

#         output = model(input)

#         prediction = int(torch.max(output.data, 1)[1].numpy())
#         print(prediction)

#         if (prediction == 0):
#             print ('Normal')
#             st.text_area(label="Prediction:", value="Normal", height=100)
#         if (prediction == 1):
#             print ('PNEUMONIA')
#             st.text_area(label="Prediction:", value="PNEUMONIA", height=100)






# if __name__ == "__main__":
#     st.title("Xray lung classifier")
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
#     save_image(uploaded_file)




# import streamlit as st
# import os
# import torch
# from torchvision.transforms import transforms
# from PIL import Image
# from pathlib import Path

# # Set Streamlit page config
# st.set_page_config(
#     page_title="X-ray Lung Classifier",
#     page_icon="🩻",
#     layout="centered",
#     initial_sidebar_state="collapsed",
# )

# # Title and description
# st.markdown("""
#     <h1 style='text-align: center; color: #4CAF50;'>🩺 X-ray Lung Classifier</h1>
#     <p style='text-align: center;'>Upload a chest X-ray image and let AI detect if it's <strong>Normal</strong> or shows signs of <strong>Pneumonia</strong>.</p>
#     <hr style='border-top: 1px solid #bbb;'>
# """, unsafe_allow_html=True)

# # Image preprocessing and prediction function
# def predict_disease(image_path):
#     # Load trained model
#     model = torch.load(Path('model/model.pt'), map_location=torch.device('cpu'))
#     model.eval()

#     # Image transformations
#     trans = transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ])

#     # Load and process image
#     image = Image.open(image_path).convert("RGB")
#     input_tensor = trans(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

#     with torch.no_grad():
#         output = model(input_tensor)
#         prediction = int(torch.argmax(output, 1).item())

#     return prediction

# # File upload
# uploaded_file = st.file_uploader("📤 Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     with st.spinner("Analyzing image..."):
#         # Save image to disk
#         save_dir = "images"
#         os.makedirs(save_dir, exist_ok=True)
#         image_path = os.path.join(save_dir, "input.jpeg")
#         with open(image_path, "wb") as f:
#             f.write(uploaded_file.read())

#         # Show uploaded image
#         st.image(image_path, caption="Uploaded X-ray Image", use_container_width=True)


#         # Predict disease
#         prediction = predict_disease(image_path)

#         # Display result
#         st.markdown("### 🧠 Prediction Result:")
#         if prediction == 0:
#             st.success("✅ The X-ray image appears **Normal**.")
#         elif prediction == 1:
#             st.error("⚠️ The X-ray image shows signs of **Pneumonia**.")
#         else:
#             st.warning("⚠️ Unable to classify the image. Please try again.")

# # Footer
# st.markdown("""
#     <hr>
#     <p style='text-align: center; font-size: 0.9em;'>Developed with ❤️ using Streamlit & PyTorch</p>
# """, unsafe_allow_html=True)



# import streamlit as st
# import os
# import torch
# from torchvision.transforms import transforms
# from PIL import Image
# from pathlib import Path
# import numpy as np

# # Page configuration
# st.set_page_config(
#     page_title="X-ray Lung Classifier",
#     page_icon="🩻",
#     layout="centered"
# )

# # Background styling using HTML/CSS
# def set_background():
#     st.markdown(
#         """
#         <style>
#         .stApp {
#             background: linear-gradient(to right, #f3f9ff, #e6f2ff);
#         }
#         .upload-section {
#             padding: 1rem;
#             background-color: #ffffffcc;
#             border-radius: 1rem;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.1);
#             margin-bottom: 2rem;
#         }
#         .footer {
#             text-align: center;
#             font-size: 0.85em;
#             margin-top: 3rem;
#             color: #555;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# set_background()

# # Sidebar info
# with st.sidebar:
#     st.title("📌 About This App")
#     st.write("""
#         This app uses a deep learning model trained on chest X-ray images to detect signs of **Pneumonia**.
#     """)
#     st.markdown("**Model:** CNN-based PyTorch")
#     st.markdown("**Classes:** Normal, Pneumonia")
#     st.markdown("**Input size:** 224x224 RGB")

# # Title
# st.markdown("""
#     <div style='text-align: center;'>
#         <h1 style='color: #4CAF50;'>🩺 X-ray Lung Classifier</h1>
#         <p>Upload a chest X-ray and let AI detect if it’s <strong>Normal</strong> or shows signs of <strong>Pneumonia</strong>.</p>
#     </div>
# """, unsafe_allow_html=True)

# # Predict function
# def predict_disease(image_path):
#     # Load model
#     model = torch.load(Path('model/model.pt'), map_location=torch.device('cpu'))
#     model.eval()

#     # Preprocess
#     transform = transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ])

#     image = Image.open(image_path).convert("RGB")
#     tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

#     with torch.no_grad():
#         output = model(tensor)
#         probs = torch.nn.functional.softmax(output, dim=1)
#         pred_class = int(torch.argmax(probs, dim=1))
#         confidence = float(torch.max(probs).item())

#     return pred_class, confidence

# # Upload Section
# with st.container():
#     st.markdown("<div class='upload-section'>", unsafe_allow_html=True)

#     uploaded_file = st.file_uploader("📤 Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         with st.spinner("Analyzing image..."):
#             # Save and display
#             save_dir = "images"
#             os.makedirs(save_dir, exist_ok=True)
#             image_path = os.path.join(save_dir, "input.jpeg")
#             with open(image_path, "wb") as f:
#                 f.write(uploaded_file.read())

#             st.image(image_path, caption="📷 Uploaded X-ray Image", use_container_width=True)

#             # Prediction
#             prediction, confidence = predict_disease(image_path)

#             # Result Display
#             st.markdown("### 🧠 Prediction Result:")
#             if prediction == 0:
#                 st.success(f"✅ **Prediction: Normal**\n\n🧪 Confidence: `{confidence:.2%}`")
#             elif prediction == 1:
#                 st.error(f"⚠️ **Prediction: Pneumonia**\n\n🧪 Confidence: `{confidence:.2%}`")
#             else:
#                 st.warning("⚠️ Unable to classify the image. Please try again.")

#     st.markdown("</div>", unsafe_allow_html=True)

# # Footer
# st.markdown("""
#     <div class='footer'>
#         Developed with ❤️ by Abhishek Maurya using <strong>Streamlit</strong> and <strong>PyTorch</strong>
#     </div>
# """, unsafe_allow_html=True)






    


