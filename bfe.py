import torch
import streamlit as st
import numpy as np
from PIL import Image#, ImageFont
#from facenet_pytorch import InceptionResnetV1#, MTCNN, fixed_image_standardization
#import joblib
import os#, sys
#import torch
import util
#import cv2

st.markdown('<h1 style="color:blue;">A Machine Learning Approach to Recognize Masked Facial Expressions of the Bangladeshi People</h1>', unsafe_allow_html = True)
st.markdown('<h2 style="color:gray;">Our model classifies facial expressions into the following categories:</h2>', unsafe_allow_html = True)
st.markdown('<h3 style="color:gray;">Happiness, Sadnesss and Other</h3>', unsafe_allow_html = True)

#model_name = st.selectbox("Choose a Face Detection Model", ("MTCNN", "Caffe Model"))
model_name = st.radio("Choose a Face Detection Model", ("MTCNN", "Caffe Model"))

upload = st.file_uploader('Insert image for classification:', type = ['png', 'jpg'])
#print(upload, end= '\n\n')

#c1, c2 = st.columns(2)

c1 = st.container()
c2 = st.container()
# Setup the model here

#k = 3 # each k image will be processed by networks
#font = ImageFont.truetype(os.path.join(ABS_PATH, 'arial.ttf'), size=22)
#font = ImageFont.truetype('arial.ttf', size=22)

#cmodel = joblib.load('./sgdc_calibrated0.sav')

if upload is not None:
	img = Image.open(upload)
	#container.image(img)
	#print(img, end='\n\n')
	width, height = img.size
	#print(width, height)
	img = np.asarray(img)
	img = np.expand_dims(img, 0)
	c1.header('Input Image')
	c1.image(img)
	#print(img.size)
	# Predict the output here
	c2.header("Output")
	#frame = util.preprocess_image(mtcnn, model, cmodel, IDX_TO_CLASS, upload)
	#frame = util.preprocess_image(caffe_model, model, cmodel, IDX_TO_CLASS, upload)
	frame = util.preprocess_image(model_name, util.model, util.cmodel, util.IDX_TO_CLASS, upload)
	#frame.save("withoutmask.jpg")
	#print(type(frame))
	#print(frame.size)
	c2.image(frame.resize((width, height), Image.BILINEAR))
	# footer
	st.markdown('<footer><p>Thanks for using this app. Please, <b style="color:yellow, outline-color:yellow;">star</b> it in <a href="https://github.com/mamunemon-01/BFE/">GitHub</a></p><p>Developer: <a href="https://www.linkedin.com/in/mamunai/">Mamun Al Imran</a></p><p>Email: <a href="mailto:mamunai0369@gmail.com">mamunai0369@gmail.com</a></p>', unsafe_allow_html=True)
