# carGo Webapp, built with Streamlit

## App Overview

This webapp contains 3 pages:
* Home 
    * Basic landing page with information
* Predictive Model 
    * Form to Input Parameters and Obtain Predicted Information
* Sources 
    * Links to all our raw data, as seen in the data files ```README.md```

carGo is a linear regression model written in Python to predict urban private car reliance.

## File Structure

webapp/
│
├── .streamlit/
│   └── config.toml               # Streamlit configuration file, mainly for the theme
├── pages/
│   ├── 1_Prediction_Model.py     # Page for running the prediction model
│   └── 2_Sources.py              # Page listing data sources or references
├── .gitignore                    # Git ignore file for untracked files/folders
├── Home.py                       # Main landing page of the Streamlit app
├── library.py                    # Helper functions for regression model
└── README.md                     # Project documentation

## Expected Output

The predicted total average distance travelled per trip private car reliance from the estimators, calculated using on our model

## Prerequisites

This webapp runs on Streamlit and Python 3.10.

### Installing Streamlit

Run the following in terminal.
```
pip install streamlit
```

### Installing Python

[Download Python here](https://www.python.org/downloads/)

### How to run

```
git clone 
```
Or download the code as a zip file, unzip

If in terminal:
```
python -m streamlit run Home.py
```

If in virtual environment:
```
streamlit run Home.py
```