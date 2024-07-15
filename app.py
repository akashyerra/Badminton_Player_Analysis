import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib as plt
import streamlit as st
import numpy as np
import random
import os
from pathlib import Path
import test

st.title("Badminton Player analysis")
st.header("Classification of shot: ")
st.write("Upload image of the beginner player: ")

model = load_model('D:\\CODING\\ML\\A10_MINI_PROJECT\\Image_classification\\Image_classify.h5')
data_cat = ['Drive', 'Net shot', 'Smash']
img_height = 180
img_width = 180

# Get the current working directory
cwd = os.getcwd()

# Create a folder to store the uploaded images (if it doesn't exist)
upload_folder = Path(cwd, "uploaded_images")
upload_folder.mkdir(exist_ok=True)

image = st.file_uploader("Upload the image")

if image is not None:
    file_path = Path(upload_folder, image.name)
    with open(file_path, "wb") as file:
        file.write(image.getbuffer())
    st.success(f"File '{image.name}' uploaded successfully!")

    # Load the saved image
    image_load = tf.keras.utils.load_img(str(file_path), target_size=(img_height, img_width))
    img_arr = tf.keras.utils.array_to_img(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    st.image(image, width=200)
    st.success('The shot is of ' + data_cat[np.argmax(score)] + ' type')
    st.write('With accuracy of ' + str(np.max(score) * 100))

    # Pose estimation
    st.header("Pose estimation: ")
    prof_player = st.selectbox("Select the Professional Player for comparision : ", ['Select Option', 'Lee Chong Wei', 'Lin Dan', 'PV Sindhu', 'Taufik Hidayat', 'Saina Nehwal'])
    if prof_player != 'Select Option':
        type = data_cat[np.argmax(score)]
        player_folder = os.path.join('D:\\CODING\\ML\\A10_MINI_PROJECT\\Professional\\', prof_player, type)
        file_list = os.listdir(player_folder)
        image_files = [f for f in file_list if f.endswith(('.jpg', '.png', '.jpeg'))]

        if image_files:
            # Choose a random image file from the list
            random_image = random.choice(image_files)

            # Construct the full path to the random image file
            prof = os.path.join(player_folder, random_image)
        
        # navigating to the folder of the player 
        # prof = os.path.join('D:\\CODING\\ML\\A10_MINI_PROJECT\\Professional\\', prof_player, type, '1.jpg')


        pose_graph, angles_graph, comparision_statements = test.main(str(file_path), prof)
        st.pyplot(pose_graph)
        st.pyplot(angles_graph)

        for statement in comparision_statements:
            st.success(statement)

    else:
        st.warning("Please select a professional player.")

    

else:
    st.warning("Please upload an image.")

