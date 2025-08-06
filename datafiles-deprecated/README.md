# About datafiles
## Constraints/Constants
- Time period: 2013 - 2023; not all time periods have datasets
- Some regions had multiple PM2.5 readings, so the average was taken
- A large number of municipalities don't have measurement stations so no PM2.5 readings were available

## Data sources:
1. [Netherlands Population density data](https://opendata.cbs.nl/statline/portal.html?_la=nl&_catalog=CBS&tableId=70072ned&_theme=246)
2. [Netherlands region code](https://opendata.cbs.nl/statline/#/CBS/nl/dataset/84929NED/table?dl=343E)
3. [Netherlands PM2.5 data](https://www.luchtmeetnet.nl/rapportages)
4. [Netherlands proximity to facilites](https://opendata.cbs.nl/statline/#/CBS/en/dataset/85560ENG/table?ts=1754288993424)
5. [Land use by municipality](https://opendata.cbs.nl/statline/portal.html?_la=en&_catalog=CBS&tableId=70262ENG&_theme=1182)
6. [Transportation modes by province](https://opendata.cbs.nl/statline/portal.html?_la=en&_catalog=CBS&tableId=84710ENG&_theme=1190)

## Interpreting the Raw data:
Refer to [region map](/regionMap.json) to translate the GM region codes to readable names

Refer to [station code](/measurementStations.json) to translate the NL station codes to GM region code (some NL codes overlap with region codes as there can be multiple measurement stations in the same region)

### File paths (processed data)
1. [Population density](\processedPopDensity.csv)
2. [Processed proximity](\processedProximity.csv)
3. [Processed Land Use](\processedLandUse.csv)
4. [Processed private transport](/processedCarTravelPrivate.csv)
5. [Processed public transport](/processedCarTravelPublic.csv)

## Processed Data
All data were processed and converted in csv format for ease of import into `pandas`, the processed data is formatted in the following format:
<table>
    <tr>
        <td></td>
        <td>Year 1</td>
        <td>Year 2</td>
        <td>Year 3</td>
        <td>...</td>
    </tr>
    <tr>
        <td>Region 1</td>
        <td>Datapoint 11</td>
        <td>Datapoint 12</td>
        <td>Datapoint 13</td>
        <td>...</td>
    </tr>
    <tr>
        <td>Region 2</td>
        <td>Datapoint 21</td>
        <td>Datapoint 22</td>
        <td>Datapoint 23</td>
        <td>...</td>
    </tr>
    <tr>
        <td>...</td>
        <td>...</td>
        <td>...</td>
        <td>...</td>
        <td>...</td>
    </tr>
</table>

For datasets that have more than one parameter as a datapoint, the values were stored in a tuple

The independent variables that the team has come up with were:
1. Land use (Roads vs Parks)
2. Population density
3. Proximity to jobs (by km)

<sub>Each independent variable is stored in it's own csv</sub>

The target is:
1. PM2.5 value of the municipality
