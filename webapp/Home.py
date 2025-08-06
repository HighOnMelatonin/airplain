import streamlit as st

st.set_page_config(
    page_title="Home"
)

st.markdown("""
# Welcome to **carGo**

We are a team from SUTD developing carGo as part of our Design Thinking Project module.  
carGo is a linear regression model written in Python to predict urban private car reliance.  
## Parameters:
* Population Density (people/km²)
* Normalized Proximity to Facilities 
    * From 0 to 10
    * From 10 to 20
    * From 20 to 50   
    * The sum of these 3 normalized parameters must be 1
* Average Distance Travelled Per Trip for Public Transport (in kilometres)
    * Includes Trains, Trams, Metros, Buses
    
[Our Github Repo](https://github.com/HighOnMelatonin/airplain)
""")

st.page_link("pages/1_Prediction_Model.py", label="Click here for our **Prediction Model**", icon="📈")
st.page_link("pages/2_Sources.py", label="Click here for our **Sources**", icon="🔗")
