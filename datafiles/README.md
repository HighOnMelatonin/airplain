# About datafiles
## Constraints/Constants
- Time period: 2013 - 2023; not all time periods have datasets
- 

## Lists
Data sources:
1. [Netherlands Population density data](https://opendata.cbs.nl/statline/portal.html?_la=nl&_catalog=CBS&tableId=70072ned&_theme=246)
2. [Netherlands region code](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84929NED/table?dl=343E)
3. [Netherlands PM2.5 data](https://www.luchtmeetnet.nl/rapportages)
4. [Netherlands proximity to facilites](https://opendata.cbs.nl/statline/#/CBS/en/dataset/85560ENG/table?ts=1754288993424)

## Interpreting the Raw data:
Refer to [region map](/region_map.json) to translate the region codes to readable names

### File paths (processed data)
1. [PM 2.5](\processed_pm2.5.csv)
2. [Population density](\processed_pop_density.csv)
3. [Processed proximity](\processed_proximity.csv)

All datasets were processed into csv for ease of import into the regression model
