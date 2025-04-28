import streamlit as st
import pickle
import numpy as np
import sklearn

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('🚢 Titanic Survival Predictor')

st.write("Enter passenger details to predict survival:")

# User Inputs
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 100, 25)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', 0, 10, 0)
parch = st.number_input('Number of Parents/Children Aboard', 0, 10, 0)
fare = st.number_input('Passenger Fare', 0.0, 600.0, 50.0)
embarked = st.selectbox('Port of Embarkation', ['S', 'C', 'Q'])

# Preprocess
sex = 0 if sex == 'male' else 1
embarked_dict = {'S': 0, 'C': 1, 'Q': 2}
embarked = embarked_dict[embarked]

features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Prediction
if st.button('Predict'):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success('🎉 Survived!')
    else:
        st.error('💀 Did not survive.')
