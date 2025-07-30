# About datafiles
## Constraints/Constants
- Time period: 2013 - 2023 (Netherlands)
- Time period: 2025 (Seoul)
- Time period: 2011 - 2015 (USA)
- Time period: 2025-07-29 (Mumbai and Jakarta)
- Time period: 2017 - 2022 (Aus)

## Lists
Country Data:
1. [Netherlands](https://www.luchtmeetnet.nl/informatie/download-data/open-data)
2. [Seoul](https://data.seoul.go.kr/dataList/OA-15526/S/1/datasetView.do)
3. [China]()
4. [United States](https://catalog.data.gov/dataset/daily-census-tract-level-pm2-5-concentrations-2011-2015)
5. [Mumbai]()
6. [Jakarta]()
7. [Australia]()
8. [New Delhi]()
9. [Singapore]()

API Data:
1. [Netherlands](https://www.luchtmeetnet.nl/informatie/download-data/api.luchtmeetnet.nl)

## Interpreting the Raw data:
### Netherlands
File: [csv](/netherlands-PM2.5-31-12-2013_1700-31-12-2024_1700-24-07-2025_0641.csv)<br>
Each entry includes the city/location code, and the PM2.5 for the corresponding year

### Seoul
File: [csv](/seoul-air-pollution-measurement.csv) [json](/seoul-air-pollution-measurement.json) <br>
Description in the [information csv](/seoul-air-pollution-meter-information.csv) or [json](/seoul-air-pollution-meter-information.json).<br>
Key indicator: Metric code 9 means PM2.5 <br>
Each entry in the json includes (relevant data) the item code (data being measured), the data value, location code, and the date recorded.

### China
File: 

### United States
File: [csv](/USA.csv)<br>
Includes data at the county level

### Mumbai
File: [csv](/Mumbai%20openaq_location_6948_measurments.csv)<br>
Takes data from the Mumbai airport, includes measurements apart from pm2.5

### Jakarta
File: [csv](/Jakarta%20openaq_location_8320_measurments.csv)
Only pm2.5 from 2025-07-29

### Australia
File: [xlsx](/Australia%20MWM_DataDownload_AirQuality.xlsx)
Data goes below city level (smaller than Sydney)

### New Delhi
File: [csv](/New%20Delhi%20openaq_location_8118_measurments.csv)
Only pm2.5 from 2025-07-09

### Singapore
