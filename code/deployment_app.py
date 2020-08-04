
import tensorflow as tf
from keras.models import load_model

#Change model accordingly to whichever model you would like to use in the models folder
model = load_model('../models/InceptionV3_1.h5')

import streamlit as st

st.set_option('deprecation.showfileUploaderEncoding', False)

st.write("""
         # Dr. Glava Tikvah's second opinion
         """
         )
st.write("This is a simple image classification web app to help diagnose X-rays. Please do not use me for clinical trials just yet! I'm still a moron. I'm only right 71 % of the time. If I were a human, I would have failed medical school.")
file = st.file_uploader("Please upload an X-ray image in .jpg or .png", type=["jpg", "png"])


import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):

        size = (256,256)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(256, 256),    interpolation=cv2.INTER_CUBIC))/255.

        img_reshape = img_resize[np.newaxis,...]

        prediction = model.predict(img_reshape)

        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if np.argmax(prediction) == 0:
        st.write("Pneumonia detected, please correlate clinically")
    elif np.argmax(prediction) == 1:
        st.write("No pneumonia detected but other lung anomalies present, please investigate ")
    else:
        st.write("Likely to be normal")

    st.text("Probability (0: Pneumonia, 1: No Pneumonia but something else, 2: Normal")
    st.write(prediction)
