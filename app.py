import streamlit as st
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from numpy import asarray
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
from joblib import load
from keras.models import Model, load_model
import gdown
import streamlit_authenticator as stauth
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image as imge
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
model_URLs ={'RBmodel.joblib':'https://drive.google.com/uc?id=1EuqwKe_xAf4BHiovUJgkA6hzF9ThYWsG'}
@st.cache_resource
def load_model(model_name):
  gdown.download(model_URLs[model_name], model_name)
  return load(model_name)

# -- Set page config
apptitle = 'RecycleBot'

st.set_page_config(page_title=apptitle, page_icon=":Recycle:", layout = "wide")
# --- USER AUTHENTICATION ---
names = ["Tushar Mehta", "Administrator"]
usernames = ["tmehta", "admin"]

# load hashed passwords
file_path = "hashed_RB.pkl"
with open(file_path, "rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "RecycleBot", "abcefg", cookie_expiry_days=0)
image = Image.open('Banner_RB.jpg')
st.image(image)

name, authentication_status, username = authenticator.login("RecycleBot Login", "main")
#
if authentication_status == None:
  st.warning("Please enter your username and password.")
  st.write(" :green[If you don't have an account and would like to signup, please send your name and email address to the RecycleBot administrator at mehtat124@gmail.com]")
#
if authentication_status == False:
  st.error("Username/password is incorrect")
  st.write(" :green[If you don't have an account and would like to signup, please send your name and email address to the RecycleBot administrator at mentat124@gmail.com]")
#
if authentication_status == True:
  st.markdown("<h1 style='text-align: center; color: blue;'>Automated Waste Classification System</h1>", unsafe_allow_html=True)
  #st.title(":blue[Automated Waste Classification System]")
  #Text
  st.write(" :green[Purpose: This App provides allows the users to quickly determine if a given waste item can be recycled or not:]")

  Data_tab, Detection_tab, Report_tab = st.tabs([" ## Training Data", "## Recycling Detection", "## Field Trial"])
  #-- sidebar
  authenticator.logout("Logout", "sidebar")
  st.sidebar.title(f"Welcome {name}")
  st.sidebar.markdown("## :blue[User Information] ")
  selected_patient = st.sidebar.text_input(':green[Name/ID]', value = name)
  selected_date = st.sidebar.date_input(':green[Date of Use]', value="today")
  #selected_history = st.sidebar.button(':green[Retrieve Previous Scans and Results]')
  st.sidebar.markdown("## :blue[Model Selector and Parameters] ")
  #-- Choose Diagnosis Type as Detection or Classification
  Selected_Analysis = 'Classification'
  #Selected_Analysis = st.sidebar.selectbox(':green[Analysis Type]', ['Classification', 'Visual Screening'])
  #-- Choose Model
  selected_model = st.sidebar.selectbox('Model', ['VGG16', 'ResNet50'])
  Model_Metrics_Selection = st.sidebar.checkbox(':green[Show Model Performance Metrics]')
  #st.sidebar.markdown("## :blue[Explainability Parameters]")
  #-- Choose Explainability Type
  #selected_explainability = st.sidebar.radio(':green[Explainability Type]', ['Model Level','Object Level'])
  #-- Choose Model
  #if (selected_explainability == 'Model Level'):
  #    selected_ex_display = st.sidebar.selectbox(':green[Display]', ['Feature Importance Pareto and Heat Map','Feature Importance Pareto', 'Heat Map'])
  #else:
  #    selected_ex_display = st.sidebar.selectbox(':green[Display]', ['SHAP Value PLot/Contrast Map'])
  #selected_save = st.sidebar.button(':green[Save Record]')
  with Data_tab:
    image = Image.open("DS1_RB.jpg")
    st.image(image)
  with Detection_tab:
    if (Selected_Analysis == 'Classification'):
      #Load selected model
      if (selected_model == 'VGG16'):
          Selectedmodel = load_model("RBmodel.joblib")
          Model_option = 1
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("metrics_RB.jpg")
            st.image(image)
      else:
          Selectedmodel = load_model("RBmodel.joblib")
          Model_option = 0
          if (Model_Metrics_Selection):
            st.subheader(':green[Model Performance Metrics]')
            image = Image.open("Model_KNCF.jpg")
            st.image(image)
    else:
          st.subheader(':green[Model Performance Metrics]')
    st.divider()

    #SubHeader
    st.subheader(':green[Uplaod the image for Screening Analysis]')
    st.write(':green[Please use the Browse button below to select the image file (jpg, png, gif) from your local drive]')
    #File Input
    uploaded_file = st.file_uploader("Upload Image")
    if uploaded_file is not None:
      file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
      uploaded_file.seek(0)
      if (Model_option == 0):
        st.write("yet to come")
      else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR")
        resize = cv2.resize(image, (224,224))
        img = img_to_array(resize)
        img = img / 255
        img = np.expand_dims(img,axis=0)
#        answer = model.predict_proba(img)
#        nml = (resize-np.min(image))/(np.max(image)-np.min(image))
#        #remove background <-- to do
#        input = np.reshape(nml, (1,224,224,3))
#        input = tf.image.resize(input, (224, 224))
#        input = preprocess_input(input)
        ypred = Selectedmodel.predict(img)
        pred = np.argmax(ypred, axis=1)
        pred_prob = ypred
      st.write("<h4 style='text-align: left; color: blue;'>For the uploaded image shown above, selected model was used to perform the screening analysis.</h4>", unsafe_allow_html = True)
      if (Selected_Analysis == 'Screening'):
        if pred_prob[0][0] > 0.5:
          prob = 100*pred_prob[0][0]
          st.write(f"<h4 style='text-align: left; color: orange;'>This specific object can be **Recycled** with a probability of {prob:.2f}%.</h4>", unsafe_allow_html=True)
        else:
          prob = 100*(1-pred_prob[0][0])
          st.write(f"<h4 style='text-align: left; color: orange;'>This specific object is ** organic **  and can't be recycled with a probability of {prob:.2f}%.</h4>", unsafe_allow_html=True)
      else:
        st.write("yet to come")
    else:
      st.write(':green[If you do not have an image, you can download a sample pictures of waste images from image library,] http://wadaba.pcz.pl/#download')
##  when the records are saved to the database Save
  with Report_tab:
    image = Image.open("Feild_Trial_RB.jpg")
    st.image(image)
