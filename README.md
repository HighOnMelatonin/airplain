# Airplain

## About project
We are a team from SUTD developing Airplain as part of our Design Thinking Project module. Airplain is a linear regression model written in python to predict the rates of air pollution in a city, given certain parameters.

## Goals
To assist urban city planners with planning sustainable cities, by providing them with an accurate estimate of how much air pollution their planned city will make.

## Dependencies
See requirements.txt for the good stuff

## Repository Structure
### Project Files


### Training Data
1. [Netherlands Population density data](https://opendata.cbs.nl/statline/portal.html?_la=nl&_catalog=CBS&tableId=70072ned&_theme=246)
2. [Netherlands region code](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84929NED/table?dl=343E)
2. [Netherlands land use data, Main Roads, Park and Public Garden](https://opendata.cbs.nl/statline/portal.html?_la=en&_catalog=CBS&tableId=37105ENG&_theme=1182)
3. [Netherlands PM2.5 data](https://www.luchtmeetnet.nl/rapportages)
4. [Netherlands proximity to facilites](https://opendata.cbs.nl/statline/#/CBS/en/dataset/85560ENG/table?ts=1754288993424)

#### Limitations of Training Data
- The region codes data from CBS did not match with the code values for PM2.5 fron luchtmeet, so more effort was required on our end to relate the two sources directly. This was further made difficult as luchtmeet had sensors placed at strategic locations, with some major cities like Amsterdam and Rotterdam having multiple sensors, so we took the average PM2.5.
- We looked at the data from 2013 to 2023, but not all time periods have datasets.

### Testing Data


## Acknowledgements
We would like to thank the teaching staff of SUTD's 10.020:Data-Driven World module for their care and guidance throughout this project, along with the respective sources for their open data.