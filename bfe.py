import torch
import streamlit as st
import numpy as np
from PIL import Image#, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1#, fixed_image_standardization
import joblib
import os#, sys
#import torch
import util

st.markdown('<h1 style="color:blue;">A Machine Learning Approach to Recognize Masked Facial Expressions of the Bangladeshi People</h1>', unsafe_allow_html = True)
st.markdown('<h2 style="color:gray;">Our model classifies facial expressions into the following categories:</h2>', unsafe_allow_html = True)
st.markdown('<h3 style="color:gray;">Happiness, Sadnesss and Other</h3>', unsafe_allow_html = True)

upload = st.file_uploader('Insert image for classification:', type = ['png', 'jpg'])
#print(upload, end= '\n\n')

#c1, c2 = st.columns(2)

c1 = st.container()
c2 = st.container()
# Setup the model here

#k = 3 # each k image will be processed by networks
#font = ImageFont.truetype(os.path.join(ABS_PATH, 'arial.ttf'), size=22)
#font = ImageFont.truetype('arial.ttf', size=22)

mtcnn = MTCNN(keep_all=True, min_face_size=70, device=util.device)
model = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.6, device=util.device).eval()

C_SVM_PATH = os.path.join(util.ABS_PATH, 'sgdc_calibrated0.sav')
#print(C_SVM_PATH, end = '\n\n')
cmodel = joblib.load(C_SVM_PATH)
#cmodel = joblib.load('./sgdc_calibrated0.sav')

IDX2CLS = os.path.join(util.ABS_PATH, 'idx2cls.npy')
IDX_TO_CLASS = np.load(IDX2CLS, allow_pickle=True)

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
	frame = util.preprocess_image(mtcnn, model, cmodel, IDX_TO_CLASS, upload)
	#frame.save("withoutmask.jpg")
	#print(type(frame))
	#print(frame.size)
	c2.image(frame.resize((width, height), Image.BILINEAR))
	# footer
	st.markdown('<footer><p>Thanks for using this app. Please, <b style="color:yellow, outline-color:yellow;">star</b> it in <a href="https://github.com/mamunemon-01/BFE/">GitHub</a></p><p>Developer: <a href="https://www.linkedin.com/in/mamunai/">Mamun Al Imran</a></p><p>Email: <a href="mailto:mamunai0369@gmail.com">mamunai0369@gmail.com</a></p>', unsafe_allow_html=True)
