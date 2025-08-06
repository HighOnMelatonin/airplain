import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Prediction Model"
)

st.write("# Prediction Model for Urban Car Reliance")

# Helper function to clear current input in form
def clear_inputs():
    st.session_state.input_pop_density = ""
    st.session_state.input_road_land = ""
    st.session_state.input_park_garden_land = ""
    st.session_state.input_proximity_ten = ""
    st.session_state.input_proximity_twenty = ""
    st.session_state.input_proximity_fifty = ""
    st.session_state.input_public_transport = ""

# Form for all data inputs
with st.form("parameters", clear_on_submit=False):
    pop_density = st.text_input("Estimated Population Density (in people/km²):", key="input_pop_density")
    road_land_use = st.text_input("Main Roads Land Use (in hectares):", key="input_road_land")
    park_garden_land_use = st.text_input("Park and Public Garden Land Use (in hectares):", key="input_park_garden_land")
    zero_to_ten_proximity = st.text_input("Normalized Proximity to Faciltiies, from 0 to 10km (value from 0 to 1):", key="input_proximity_ten")
    ten_to_twenty_proximity = st.text_input("Normalized Proximity to Faciltiies, from 10km to 20km (value from 0 to 1)", key="input_proximity_twenty")
    twenty_to_fifty_proximity = st.text_input("Normalized Proximity to Faciltiies, from 20km to 50km (value from 0 to 1):", key="input_proximity_fifty")
    public_transport = st.text_input("Average Distance Travelled Per Trip for Trains, Buses, Metros and Trams (in kilometres):", key="input_public_transport" )

    submit = st.form_submit_button("Generate Estimated Private Car Use")

if submit:
    if pop_density and road_land_use and park_garden_land_use and zero_to_ten_proximity and ten_to_twenty_proximity and twenty_to_fifty_proximity and public_transport:
        try:
            # Convert all inputs to float
            pop_density = float(pop_density)
            road_land_use = float(road_land_use)
            park_garden_land_use = float(park_garden_land_use)
            zero_to_ten_proximity = float(zero_to_ten_proximity)
            ten_to_twenty_proximity = float(ten_to_twenty_proximity)
            twenty_to_fifty_proximity = float(twenty_to_fifty_proximity)
            public_transport = float(public_transport)

            # Make sure that the proximities add up to 1
            if np.isclose(zero_to_ten_proximity + ten_to_twenty_proximity + twenty_to_fifty_proximity,1.0):
                pass # insert function here
    
        except ValueError: # if float conversion fails
            st.write("Please key in valid values!")

        st.rerun()
    else:
        st.write("Please fill in all the values in the form!")


st.button("Clear Form", on_click=clear_inputs)