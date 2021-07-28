import pandas as pd 
import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

def user_input():
  sepal_length= st.sidebar.slider('Sepal Length',4.3,7.9,5.4)
  sepal_width= st.sidebar.slider('Sepal Width',2.0,4.4,3.4)
  petal_length= st.sidebar.slider('Petal Length',1.0,6.9,5.4)
  petal_width= st.sidebar.slider('Petal Width',0.,2.5,1.4)
  data={'sepal_length':sepal_length,
        'sepal_width':sepal_width,
        'petal_length':petal_length,
        'petal_width':petal_width}
  features=pd.DataFrame(data,index=[0])
  return features
st.write("# Simple **Iris Flower** Prediction App")
st.write("# Iris Dataset")
st.write("number of classes:3")
st.write("classifier: KNN")
st.write("Accuracy= 0.95")
st.sidebar.header("User Input Parameters")

df=user_input()
st.write(df)
iris=datasets.load_iris()
x=iris.data
y=iris.target


model=KNeighborsClassifier(n_neighbors=5)
model.fit(x,y)
prediction=model.predict(df)

st.subheader("Class labels and their corresponding index number")
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
