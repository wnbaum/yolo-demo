import os 
import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(layout="wide", page_title="YOLO Demo", initial_sidebar_state="expanded")

st.sidebar.markdown(
    f"This [demo](https://github.com/wnbaum/yolo-demo) uses the YOLO v3 network architecture for object detection. Choose an image, edit some parameters, and watch your model update and detect objects! Images sourced from Google Maps."
)
st.sidebar.markdown(
    "Made with 💜 by [Will Baumgartner](https://github.com/wnbaum)"
)

st.title("YOLO Demo")
st.header("This app demonstrates the usage of YOLO for self driving cars and object detection.")

if not os.path.exists("models/yolo_weights.h5"):
    with st.spinner("Loading model..."):
        os.system("wget --no-check-certificate -O models/yolo_weights.h5 \"https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20%20Object%20Detection%20(Autonomous%20Vehicles)/yolo.h5\"")
        from processing import detect_image
else:
    from processing import detect_image

f = st.file_uploader("Upload an Image")

images = { "None":"None", "San Francisco": "images/google1.png", "Paris": "images/google2.png", "London": "images/google3.png", "Tokyo": "images/google4.png"  }
image_options = list(images.keys())
image_choice = st.selectbox("or Select an Already Uploaded Image", image_options)
image_location = images[image_choice]

file_bytes = None

if image_choice == "None":
    if f is not None: 
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
else:
    with open(image_location, 'rb') as image_file:
        if image_file is not None:
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)

if file_bytes is not None:
    st.header("Now edit some parameters to change the object detection algorithm.")
    obj_thresh = st.slider("Object Threshold", min_value=0.01, max_value=1.0, value=0.4, step=0.01)
    nms_thresh = st.slider("Non-Maximum Suppression Threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.01)

    image = cv2.imdecode(file_bytes, 1)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    stimage_location = st.empty()

    stimage_location.image(image, channels="BGR")

    processed_image = detect_image(image_pil, obj_thresh=obj_thresh, nms_thresh=nms_thresh)

    stimage_location.image(processed_image, channels="RGB")