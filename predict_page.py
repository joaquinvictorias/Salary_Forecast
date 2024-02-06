import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']
le_age = data['le_age']
le_industry = data['le_industry']
le_education = data['le_education']
le_location = data['le_location']
le_modality = data['le_modality']
le_gender = data['le_gender']

def show_predict_page():
    st.title('Salary Forecast')

    ages = (
        '18-21', 
        '22-25', 
        '26-29', 
        '30-33', 
        '34-37', 
        '38-41', 
        '42-45', 
        '46+'
    )

    industries = (
        'Accounting', 
        'Advertising', 
        'Aerospace', 
        'Automotive', 
        'Banking', 
        'Biotech', 
        'Construction', 
        'Consulting', 
        'Defense', 
        'Education', 
        'Engineering', 
        'Entertainment', 
        'Fashion', 
        'Finance', 
        'Fintech', 
        'Food', 
        'Government', 
        'Healthcare', 
        'Hosptechaltechy', 
        'Human Resources', 
        'Insurance', 
        'Legal', 
        'Logistics', 
        'Manufacturing', 
        'Marketing', 
        'Media', 
        'Medical', 
        'Pharmacy', 
        'Public Relations', 
        'Real Estate', 
        'Retail', 
        'Sales', 
        'Tech', 
        'Telecommunications'
    )

    education = (
        'Doctorate Degree',
        "Master's Degree",
        "Bachelor's Degree",
        'Some College',
        "Associate's Degree",
        'High School/GED',
        'Trade/Vocational',
        'None'
    )

    locations = (
        'Atlanta, GA', 
        'Austin, TX', 
        'Baltimore, MD', 
        'Boston, MA', 
        'Charlotte, NC', 
        'Chicago, IL', 
        'Cincinnati, OH', 
        'Cleveland, OH', 
        'Columbus, OH', 
        'Dallas, TX', 
        'Denver, CO', 
        'Detroit, MI', 
        'Houston, TX', 
        'Indianapolis, IN', 
        'Los Angeles, CA', 
        'Miami, FL', 
        'Minneapolis, MN', 
        'Nashville, TN', 
        'New York, NY', 
        'Orlando, FL', 
        'Philadelphia, PA', 
        'Phoenix, AZ', 
        'Pittsburgh, PA', 
        'Portland, OR', 
        'Raleigh, NC', 
        'Richmond, VA', 
        'Sacramento, CA', 
        'Salt Lake City, UT', 
        'San Antonio, TX', 
        'San Diego, CA', 
        'San Francisco, CA', 
        'San Jose, CA', 
        'Seattle, WA', 
        'St. Louis, MO', 
        'Tampa, FL', 
        'Washington, DC'
    )

    modalities = (
        'Remote',
        'Hybrid', 
        'Onsite'
    )

    genders = (
        'Male', 
        'Female', 
        'LGBTQ+'
    )

    age = st.selectbox('Age Range', ages)
    experience = st.slider('Years of Experience', 0, 50, 3)
    industry = st.selectbox('Industry', industries)
    education = st.selectbox('Education Level', education)
    location = st.selectbox('Location', locations)
    modality = st.selectbox('Modality', modalities)
    gender = st.selectbox('Gender', genders)


    salary = st.button('Calculate Salary')
    if salary:
        x = np.array([[age, experience, industry, education, location, modality, gender]])
        x[:, 0] = le_age.transform(x[:, 0])
        x[:, 2] = le_industry.transform(x[:, 2])
        x[:, 3] = le_education.transform(x[:, 3])
        x[:, 4] = le_location.transform(x[:, 4])
        x[:, 5] = le_modality.transform(x[:, 5])
        x[:, 6] = le_gender.transform(x[:, 6])
        x = x.astype(int)

        salary = regressor.predict(x)
        st.subheader(f'The estimated salary is {salary[0]:.2f}$')

        df = pd.read_csv('salaries_encoded.csv')

        X = df.drop(columns=['Salary'])
        
        explainer = shap.Explainer(regressor, X)
        shap_values = explainer(pd.DataFrame(x, columns=X.columns))
        fig, ax = plt.subplots(1, 1)
        shap.plots.waterfall(shap_values[0])
        st.pyplot(fig)



        