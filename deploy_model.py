# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:40:50 2020

@author: DEVIKA
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction


model = tf.keras.models.load_model('my_model.hdf5')

st.write("""
         # Brain Tumor Prediction
         """
         )

st.write("This is a image classification web app to predict the which brain tumor does the person have ")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a Glioma!")
    elif np.argmax(prediction) == 1:
        st.write("It is a Meningioma!")
    elif np.argmax(prediction == 2):
        st.write('it is a No Tumor!')
    else:
        st.write("It is a Pituitary!")
    
    st.text("Probability (0: Glioma, 1: Meningioma, 2: No tumor,3:Pituitary)")
    st.write(prediction)