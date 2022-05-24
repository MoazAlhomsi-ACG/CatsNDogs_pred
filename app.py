import streamlit as st 
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np 

st.set_page_config(page_title="Cat or Dog",
					page_icon="🐻")

def load_model():
	model = tf.keras.models.load_model("model/tf_CNN.h5")
	return model 
model = load_model()
st.header("Cat(s) or Dog(s) ?")
st.subheader("A CNN prediction app")
st.write("")

file = st.file_uploader("Upload cat or/and dog photo(s)",accept_multiple_files=True)

for i in file:
	st.image(i)
	ex_image = image.load_img(i,target_size=(64,64))
	ex_image = image.img_to_array(ex_image)
	ex_image = np.expand_dims(ex_image,axis=0)
	output = model.predict(ex_image/255)
	
	st.subheader("Result:")

	if output > 0.55:
		st.write(output)
		st.success("It is a Dog !") 


	else:
	    st.write(output)

	    st.warning("It is a Cat")
