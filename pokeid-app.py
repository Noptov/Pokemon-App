import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

st.title('Welcome to the Pokemon ID Project')

size = (128, 128)
type_model = tf.keras.models.load_model('Models/type-select-model')
type_names =['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Ghost', 'Grass', 
            'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']

user_pic = st.file_uploader(label='Please upload your picture to identify what pokemon it is.',
                    type=['png', 'jpg']
                    )


user_array = tf.keras.preprocessing.image.smart_resize(
    user_pic, size, interpolation='bilinear'
)

user_batch = np.expand_dims(test_array2, axis=0)

user_pred = type_model.predict(user_batch)

user_type = type_names[np.argmax(user_pred)]

st.write(f'The type of Pokemon is {user_type}')