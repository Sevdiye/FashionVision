import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="FashionVision", layout="centered")

st.markdown("""
    <style>
    /* Full-screen background with clothing/fabric imagery */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.9)), 
                    url('https://images.unsplash.com/photo-1445205170230-053b83016050?auto=format&fit=crop&q=80&w=2071');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Minimalist Branding */
    .brand-title {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 200;
        letter-spacing: 15px;
        color: #ffffff;
        text-align: center;
        text-transform: uppercase;
        padding-top: 2rem;
        margin-bottom: 0px;
    }

    .brand-sub {
        text-align: center;
        color: #888;
        font-size: 0.7rem;
        letter-spacing: 5px;
        margin-bottom: 4rem;
        text-transform: uppercase;
    }

    /* Clean File Uploader styling */
    .stFileUploader {
        border: 1px solid #444 !important;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 0px;
    }

    /* Results section */
    .category-card {
        border-left: 2px solid #ffffff;
        padding-left: 20px;
        margin-top: 20px;
    }

    .category-name {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        color: #ffffff;
        text-transform: uppercase;
        line-height: 1;
        margin-bottom: 10px;
    }

    /* UI Clean-up */
    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_fashion_model():

    return tf.keras.models.load_model("fashionvision_resnet50.keras")

model = load_fashion_model()

CLASS_NAMES = [
    'bodysuits', 'boots', 'dresses', 'jeans', 'long_coats',
    'longsleeves', 'others', 'skirts', 'sportshorts', 'tops', 'tshirts'
]
IMG_SIZE = (224, 224)

st.markdown("<h1 class='brand-title'>FASHION VISION</h1>", unsafe_allow_html=True)
st.markdown("<p class='brand-sub'>AI CLOTHING CLASSIFICATION</p>", unsafe_allow_html=True)

_, col_mid, _ = st.columns([1, 3, 1])
with col_mid:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    label = CLASS_NAMES[pred_index].replace('_', ' ').upper()

    st.markdown("<br><br>", unsafe_allow_html=True)
    res_left, res_right = st.columns([1, 1])
    
    with res_left:
        st.image(image, use_container_width=True)

    with res_right:
        st.markdown(f"""
            <div class="category-card">
                <p style="color: #888; letter-spacing: 2px; font-size: 0.7rem; margin-bottom: 5px;">DETECTED PRODUCT</p>
                <div class="category-name">{label}</div>
            </div>
        """, unsafe_allow_html=True)