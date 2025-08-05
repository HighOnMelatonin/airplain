# Python script to process China PM2.5 data
# Run this script outside of datafiles directory

import os
import json
import numpy as np
import pandas as pd

# China file type is nc, NetCDF

# pyright: ignore[reportUnknownParameterType, reportMissingTypeArgument]
def openJson(filePath: str) -> dict:
    """
    Open a JSON file and return its contents as a dictionary.

    Args:
        filePath (str): The path to the JSON file.

    Returns:
        dict: The data from the JSON file as a Python dictionary.
    """
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"File {filePath} does not exist.")

    with open(filePath, 'r', encoding='utf-8') as f:
        # pyright: ignore[reportAny, reportMissingTypeArgument]
        data: dict = json.load(f)

    return data  # pyright: ignore[reportUnknownVariableType]


def popDensity() -> bool:
    """
    Returns True when population density data has been processed, False otherwise.

    Output file: datafiles/processed_pop_density.csv
    """
    regionCode: dict[str, dict[str, dict[str, str]]] = openJson("datafiles/regionMap.json")
    jsonOutput: dict[str, dict[str, str]] = {}
    output: str = ''
    popDensityFile: str = "datafiles/Netherlands_population_density.csv"

    csvOutPath: str = "datafiles/processed_pop_density.csv"
    # If output csv file does not exist, create it
    if not os.path.exists(csvOutPath):
        f = open(csvOutPath, 'x')
        f.close()

    rawDensity: list[str] = open(popDensityFile, 'r').readlines()
    rawDensity = rawDensity[1:]  # Skip the header row
    with open(csvOutPath, 'w') as f:
        for line in rawDensity:
            entry: list[str] = line.split(sep=",")
            year = entry[2][0:4]

            ## Build a dictionary with region codes as keys; another dictionary as value
            ## Nested dictionary will have years as keys and population density as values
            ## Error handling for region codes that don't have a corresponding region name
            try:
                if regionCode[entry[1].strip()] not in jsonOutput.keys():
                    # pyright: ignore[reportArgumentType]
                    jsonOutput[regionCode[entry[1]].strip()] = {year: entry[3].strip()}
                else:
                    # pyright: ignore[reportArgumentType]
                    jsonOutput[regionCode[entry[1]].strip()][year] = entry[3].strip()

            except:
                if entry[1].strip() not in jsonOutput.keys():
                    jsonOutput[entry[1].strip()] = {year: entry[3].strip()}

                else:
                    jsonOutput[entry[1].strip()][year] = entry[3].strip()

            print(jsonOutput)

    region: str = list(jsonOutput.keys())[0]
    # Make a 3d array to contain all the data to be converted to csv
    # +1 for the header row +1 for newline
    xDim: int = len(jsonOutput[region]) + 2
    yDim: int = len(jsonOutput) + 1  # +1 for the header column
    array: list[list[list[str]]] = [[[]for j in range(xDim)] for i in range(yDim)]

    column = 1
    for year in jsonOutput[region]:
        array[0][column] = [year]
        column += 1
        # Fill the first row with the years

    row = 1
    for region in jsonOutput:
        array[row][0] = [region]
        row += 1
        # Fill the first column with the region codes

    # Fill the rest of the array with population density data
    row = 1
    while row < yDim:
        column = 1
        while column < xDim - 1:
            regionCode = array[row][0][0]
            year = array[0][column][0]
            population_density = jsonOutput[region][year]
            array[row][column] = [population_density]
            column += 1

        array[row][column] = ['\n']  # Add a newline at the end of each row
        row += 1

    print(array)

    # Uncomment the next lines when region code has been fixed
    # output = str(array).replace("], [", "\n").replace("[", "").replace("]", "") # pyright: ignore[reportUnknownVariableType]
    # print(output, file=csvOutPath, mode='w')  # pyright: ignore[reportUnknownVariableType]
    return True


def trimProximity() -> bool:
    """
    Remove unnecessary data from the proximity file and convert it to a CSV format.

    Return True when done correctly

    -1 indicates null data
    """
    proximityPath: str = "datafiles/proximity_to_job.json"
    proximityDict: dict[str, dict[str, str]] = openJson(
        proximityPath)["value"]  # pyright: ignore[reportUnknownVariableType]
    regionMap: dict[str, str] = openJson("datafiles/regionMap.json")
    outputFile: str = "datafiles/processed_proximity.csv"
    if not os.path.exists(outputFile):
        f = open(outputFile, 'x')
        f.close()

    # Make a dataframe to contain all the data to be converted to csv
    output = pd.DataFrame(
        columns=['Region', 'Year', '0 to 10 km', '>10 to 20 km', '>20 to 50km'])

    # Iterate through all the region & periods
    for row in proximityDict.values():
        region: str = row["Regions"]
        region: str = regionMap.get(region, region)
        year: str = row["Periods"][0:4]
        proximityValueDict: dict[str: list[int]] = {
            '10': [], '20': [], '50': []}

        # Place the proximity data into a dict
        for key in row.keys():
            if not 'Within' in key:
                continue
            proximityValue: str = row[key].strip()
            try:
                proximityValue: float = float(proximityValue)
            except:
                proximityValue: float = -1.0
            proximityRange: str = key[6:8]
            proximityValueDict[proximityRange].append(proximityValue)

        # Merge the proximity data together
        for proximityColumn, valueList in proximityValueDict.items():
            finalValue: float = 0
            allBlank: bool = True
            for val in valueList:
                if val == -1.0:
                    continue
                else:
                    allBlank: bool = False
                    finalValue += val
            if allBlank:
                finalValue: float = -1.0
            proximityValueDict[proximityColumn].append(finalValue)

        # Change the cumulative data to ranges
        in10: float = round(proximityValueDict['10'][3], 1)
        in20: float = round(proximityValueDict['20'][3] - max(in10, 0), 1)
        in50: float = round(proximityValueDict['50'][3] - max(in20, 0) - max(in10, 0), 1)

        # Convert the data to percentage of total
        total: float = max(in10 + in20 + in50, 1)
        in10 /= total
        in20 /= total
        in50 /= total

        output.loc[len(output)] = [region, year, in10, in20, in50]
    print(output)

    output.to_csv(outputFile, index=False)
    return True


def trimRegionCode(filename: str) -> None:
    """
    Trim region code Json to only map region codes to their names.

    Naam_2 refers to the region name
    Code_1 refers to the region code

    Args:
        filename (str): The name of the JSON file to process (without .json extension).

    Returns None

    File output: datafiles/regionMap.json
    """
    regionRaw: dict[str, dict[str, str]] = openJson(f"datafiles/{filename}.json")["value"]
    regionMap: dict[str, str] = {}

    for region in regionRaw:  # pyright: ignore[reportUnknownVariableType]
        item: dict[str, str] = regionRaw[region]
        # pyright: ignore[reportUnknownMemberType]
        regionMap[item["Code_1"].strip()] = item["Naam_2"].strip()

    if not os.path.exists("datafiles/regionMap.json"):
        f = open("datafiles/regionMap.json", 'x')
        json.dump({}, f)  # Initialize with an empty JSON object
        f.close()

    json.dump(regionMap, open("datafiles/regionMap.json", "w"), indent=4)

    return None


def trimPM() -> bool:
    """
    Remove unnecessary data from the PM2.5 file

    Returns True when done correctly
    """
    pmFile: str = "datafiles/pm2.5_2013-2023_netherlands.csv"

    outputFile: str = "datafiles/processed_pm2.5.csv"

    # Create the output file if it does not exist
    if not os.path.exists(outputFile):
        f = open(outputFile, 'x')
        f.close()

    # Uncomment the next line once regionMap is available
    # regionMap: dict[str, str] = openJson("datafiles/regionMap2.json")  # pyright: ignore[reportUnknownVariableType]
    with open(pmFile, 'r') as f:
        lines: list[str] = f.readlines()

        # Iterate through the lines and standardise empty values, and translate region codes
        for i in range(1, len(lines)):   # Skip the header row
            line: str = lines[i]
            code = line.split(",")[0]
            lines[i] = line.replace("-", "")
            # lines[i] = line.replace(code, regionMap[code])  # Uncomment once regionMap is available
        print(lines[:61])

    print(lines, file=open(outputFile, 'w'))
    return True


def getFiveNumberSummary(array: np.ndarray | pd.DataFrame) -> dict[str: int]:
    assert array.shape[1] == 1 or len(array.shape) == 1

    min = np.min(array)
    max = np.max(array)
    median = np.median(array)
    q1 = np.quantile(array, 0.25)
    q3 = np.quantile(array, 0.75)

    return min, q1, median, q3, max

# Testing and Example Usage
## ===================================== ##


if __name__ == "__main__":
    # Example usage
    try:
        # trimRegionCode("reregionCode")
        print(popDensity())
        # print(trimPM())
        # print(trimProximity())
        pass
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
