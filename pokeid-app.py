import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

size = (128, 128)

type_model = tf.keras.models.load_model('Models/type_select_model')
type_names =['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Ghost', 'Grass', 
            'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']

model_dict = {}
for name in type_names:
    model_dict[name] = tf.keras.models.load_model(f'models/{name}_model')

pokemon_dict= pd.read_csv('data/pokemon_class_dict.csv', header=None, index_col=0, squeeze=True).to_dict()

st.title('Welcome to the Pokemon ID Project') 

# Prompt user to upload an image and convert that file into an array/batch the models can use to predict
user_pic = st.file_uploader(label='Please upload your picture to identify what pokemon it is.',
                    type=['png', 'jpg']
                    )

user_array = tf.keras.preprocessing.image.smart_resize(
    user_pic, size, interpolation='bilinear'
)
user_batch = np.expand_dims(test_array2, axis=0)

# Use converted image to predict type of pokemon
user_pred = type_model.predict(user_batch)
user_type = type_names[np.argmax(user_pred)]
st.write(f'The type of Pokemon is {user_type}')

# Based on type, use appropriate model to ID individual pokemon
ind_model = model_dict[user_type]
ind_pred = ind_model.predict(user_batch)
ind_list = pokemon_dict[user_type].split(',')
ind_pokemon = ind_list[np.argmax(ind_pred)]

st.write(f'Your Pokemon is {ind_pokemon}!')