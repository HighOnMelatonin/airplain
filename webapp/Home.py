import streamlit as st

st.set_page_config(
    page_title="Home"
)

st.markdown("""
# Welcome to **Airplain**

We are a team from SUTD developing Airplain as part of our Design Thinking Project module.  
Airplain is a linear regression model written in python to predict the rates of air pollution in a city.  
Parameters:
* Population density (people/km2)
* Land use for main roads (in hectares)
* Land use for parks and public gardens (hectares)
* Normalized proximity to facilities from 0 to 10
* Normalized proximity to facilities from 10 to 20
* Normalized proximity to facilities from 20 to 50   
    * The sum of these 3 normalized parameters must be 1
            
[Our Github Repo](https://github.com/HighOnMelatonin/airplain)
""")

st.page_link("pages/1_Prediction_Model.py", label="Click here for our **Prediction Model**", icon="📈")
st.page_link("pages/2_Sources.py", label="Click here for our **Sources**", icon="🔗")
