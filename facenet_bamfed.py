import streamlit as st
import numpy as np
from PIL import Image, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import joblib
import os
import torch
import util

ABS_PATH = ''

st.markdown('<h1 style="color:blue;">Bangladeshi Masked Facial Expression Recognition</h1>', unsafe_allow_html = True)
st.markdown('<h2 style="color:gray;">Our model classifies facial expressions into the following categories:</h2>', unsafe_allow_html = True)
st.markdown('<h3 style="color:gray;">Happiness, Sadnesss and Other</h3>', unsafe_allow_html = True)

upload = st.file_uploader('Insert image for classification:', type = ['png', 'jpg'])

c1, c2 = st.columns(2)

# Setup the model here
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#k = 3 # each k image will be processed by networks
font = ImageFont.truetype(os.path.join(ABS_PATH, 'arial.ttf'), size=22)

mtcnn = MTCNN(keep_all=True, min_face_size=70, device=device)
model = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.6, device=device).eval()

C_SVM_PATH = os.path.join(ABS_PATH, 'sgdc_calibrated0.sav')
cmodel = joblib.load(C_SVM_PATH)

if upload is not None:
	img = Image.open(upload)
	img = np.asarray(img)
	img = np.expand_dims(img, 0)
	c1.header('Input Image')
	c1.image(img)
	# Predict the output here
	c2.header("Output")
	frame = util.preprocess_image(mtcnn, model, cmodel, img)
	c2.image(frame)