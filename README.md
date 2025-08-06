# Airplain

## About project
We are a team from SUTD developing carGo as part of our Design Thinking Project module. carGo is a linear regression model written in Python to predict urban car reliance, given certain parameters.

## Goals
To assist urban developers with planning sustainable urban environments, by providing them with an accurate estimate of how much reliance on private cars people living there will have.

## Dependencies
See requirements.txt for the good stuff.

## Repository Structure
### Project Files
.
├── carGo/  
│   ├── datafiles/    
│   │   ├── raw_data_files
│   │   └── processed_data_files
│   ├── webapp/
|   |   ├── pages/
|   |   |   ├── 1_Prediction_Model.py
|   |   |   └── 2_Sources.py
│   │   └── Home.py
|   ├── cleanNether.py
|   ├── linearRegression.py
│   ├── README.md
│   └── requirements.txt

* cleanNether.py : All the code used to clean our data
* linearRegression.py : All the code used to process and train our model 
* datafiles : Contains all our raw and processed data, in json and csv formats. For more information, read the '''README''' inside that folder
* webapp : Coded with Streamlit, for easy use of our model. For more information, read the '''README''' inside that folder


### Training Data
1. [Netherlands Population Density](https://opendata.cbs.nl/statline/portal.html?_la=nl&_catalog=CBS&tableId=70072ned&_theme=246)
2. [Netherlands Region Code](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84929NED/table?dl=343E)
3. [Netherlands Public Transport and Private Transport](https://opendata.cbs.nl/statline/portal.html?_la=en&_catalog=CBS&tableId=84710ENG&_theme=1190)
4. [Netherlands Proximity to Facilites](https://opendata.cbs.nl/statline/#/CBS/en/dataset/85560ENG/table?ts=1754288993424)

#### Limitations of Training Data
- 
- We looked at the data from 2013 to 2023, but not all time periods have datasets.

### Testing Data


## Acknowledgements
We would like to thank the teaching staff of SUTD's 10.020:Data-Driven World module for their care and guidance throughout this project, along with the respective sources for their open data.