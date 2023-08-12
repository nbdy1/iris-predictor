import pandas as pd
import numpy as np
import joblib
import streamlit as st

# loading in the model to predict on the data
classifier = joblib.load("iris-classifier.joblib")

# defining the function which will make the prediction using
# the data which the user inputs
def prediction(sepal_length, sepal_width, petal_length, petal_width):
	prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
	return prediction

# this is the main function in which we define our webpage
# giving the webpage a title
st.header("Iris Flower Prediction")
st.sidebar.subheader("Input features")
sepal_length = st.sidebar.slider('Sepal length', 0.0, 9.0, (5.0))
sepal_width = st.sidebar.slider('Sepal width', 0.0, 4.5, (2.5))
petal_length = st.sidebar.slider('Petal length', 0.0, 8.0, (4.5))
petal_width = st.sidebar.slider('Petal width', 0.0, 3.0, (1.5))
	
# the below line ensures that when the button called 'Predict' is clicked,
# the prediction function defined above is called to make the prediction
# and store it in the variable result
from PIL import Image

if st.button("Predict"):
	pred = prediction(sepal_length, sepal_width, petal_length, petal_width)
	if pred == 0:
		st.success('The Flower is an Iris-setosa')
		setosa = Image.open('iris_setosa.jpg')
		st.image(setosa, caption = 'Iris-setosa', width = 300)
	elif pred == 1:
		st.success('The Flower is an Iris-versicolor ')
		versicolor = Image.open('iris_versicolor.jpg')
		st.image(versicolor, caption = 'Iris-versicolor', width = 300)
	else:
		st.success('The Flower is an Iris-virginica ')
		virginica = Image.open('iris_virginica.jpg')
		st.image(virginica, caption = 'Iris-virginica', width = 300 )
	
