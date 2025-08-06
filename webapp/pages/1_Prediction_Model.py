import streamlit as st

st.set_page_config(
    page_title="Prediction Model"
)

st.write("# Prediction Model")

# so rn our predictor variables are:
# pop density (people/km2)
# main road land use (hectares)
# park and public garden land use (hectares)
# normalized proximity to facilities (0<10)
# normalized proximity to facilities (10<20)
# normalized proximity to facilities (20<50)


def clear_inputs():
    st.session_state.input_pop_density = ""
    st.session_state.input_road_land = ""
    st.session_state.input_park_garden_land = ""
    st.session_state.input_proximity_ten = ""
    st.session_state.input_proximity_twenty = ""
    st.session_state.input_proximity_fifty = ""


with st.form("parameters", clear_on_submit=False):
    pop_density = st.text_input("Estimated Population Density (in people/km^2):", key="input_pop_density")
    road_land_use = st.text_input("Main Roads Land Use (in hectares):", key="input_road_land")
    park_garden_land_use = st.text_input("Park and Public Garden Land Use (in hectares):", key="input_park_garden_land")
    zero_to_ten_proximity = st.text_input("Normalized Proximity to Faciltiies, from 0 to 10km (value from 0 to 1):", key="input_proximity_ten")
    ten_to_twenty_proximity = st.text_input("Normalized Proximity to Faciltiies, from 10km to 20km (value from 0 to 1)", key="input_proximity_twenty")
    twenty_to_fifty_proximity = st.text_input("Normalized Proximity to Faciltiies, from 20km to 50km (value from 0 to 1):", key="input_proximity_fifty")

    submit = st.form_submit_button("Generate Estimated PM2.5")

if submit:
    if pop_density and road_land_use and park_garden_land_use and zero_to_ten_proximity and ten_to_twenty_proximity and twenty_to_fifty_proximity:
        # insert function here
        # users.loc[len(users)] = [len(users), new_username, new_name]
        # st.cache_data.clear()
        st.rerun()
    else:
        st.write("Please fill in all the values in the form!")


st.button("Clear Form", on_click=clear_inputs)


