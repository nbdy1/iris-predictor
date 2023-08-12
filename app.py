import pandas as pd
import numpy as np
import joblib
import streamlit as st

# loading in the model to predict on the data
classifier = joblib.load("iris-classifier.joblib")

# defining the function which will make the prediction using
# the data which the user inputs
def prediction(sepal_length, sepal_width, petal_length, petal_width):
	prediction = classifier.predict(
		[[sepal_length, sepal_width, petal_length, petal_width]])
	print(prediction)
	return prediction

# this is the main function in which we define our webpage
# giving the webpage a title
st.title("Iris Flower Prediction")
st.header('IRIS MODEL DEPLOYMENT')
	
# the following lines create text boxes in which the user can enter
# the data required to make the prediction
sepal_length = st.text_input("Sepal Length", "Type Here")
sepal_width = st.text_input("Sepal Width", "Type Here")
petal_length = st.text_input("Petal Length", "Type Here")
petal_width = st.text_input("Petal Width", "Type Here")
result =""
	
# the below line ensures that when the button called 'Predict' is clicked,
# the prediction function defined above is called to make the prediction
# and store it in the variable result
if st.button("Predict"):
	result = prediction(sepal_length, sepal_width, petal_length, petal_width)
st.success('The output is {}'.format(result))
	
