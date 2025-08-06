## Python script to process/clean Netherlands data
## Run this script outside of datafiles directory

import os
import json
import numpy as np
import pandas as pd
import re

# China file type is nc, NetCDF

def openJson(filePath: str) -> dict:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
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
        data: dict = json.load(f)   # pyright: ignore[reportAny, reportMissingTypeArgument]

    return data  # pyright: ignore[reportUnknownVariableType]


def landUse() -> bool:
    """
    Returns True when the land use data has been processed, False otherwise

    Outputs a csv file with the processed land use data

    Ouptutfile: datafiles/processedLandUse.csv
    """
    regionCode: dict[str, str] = openJson("datafiles/regionMap.json")  # pyright: ignore[reportAssignmentType, reportUnknownVariableType, reportRedeclaration]
    provinceCode: dict[str, dict[str, list[str]]] = openJson("datafiles/provinceMap.json")  # pyright: ignore[reportUnknownVariableType]
    ## Dictionary structure: {region: {year: {roads: value, parks: value}}}
    jsonOutput: dict[str, dict[str, dict[str, str]]] = {}
    output: str = ''
    landUseFile: str = "datafiles/provincial_land_use_main_road_public_park_garden.csv"

    csvOutPath: str = "datafiles/processedLandUse.csv"
    ## If output csv file does not exist, create it
    if not os.path.exists(csvOutPath):
        f = open(csvOutPath, 'x')
        f.close()

    rawLand: list[str] = open(landUseFile, 'r').readlines()[1:] ## Skip header row
    for line in rawLand:
        ## Every line in file looks like "ID,Regions,Periods,MainRoad_4,ParkAndPublicGarden_20"
        entry: list[str] = line.split(",")
        year: str = entry[2][0:4]       ## Get value of year  # pyright: ignore[reportRedeclaration]
        region: str = ''  # pyright: ignore[reportRedeclaration]
        ## Match the region code to the region name if it exists in the regionMap
        pvName = ''
        if entry[1] in regionCode.keys():  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            region = regionCode[entry[1].strip()]  # pyright: ignore[reportArgumentType]
            # ## Map region name to province name
            # for pvCode in provinceCode:
            #     if region in provinceCode[pvCode]["municipalities"]:
            #         pvName: str = provinceCode[pvCode]["name"]  # pyright: ignore[reportAssignmentType]
            #         break
        
        ## If regionCode has no name, or pvCode does not have a pvName, set pvName to the regionCode
        if region == '':
            region: str = entry[1].strip()

        if "\u2011" in region:
            region = region.replace("\u2011", '-')

        if region in jsonOutput.keys():
            jsonOutput[region][year] = {"roads": entry[3].strip(), "parks": entry[4].strip()}

        else:
            jsonOutput[region] = {year: {"roads": entry[3].strip(), "parks": entry[4].strip()}}


    ## Make a 3d array to conain all the data to be converted to csv
    ## + 1 for the header row
    pvName: str = list(jsonOutput.keys())[0]
    xDim: int = len(jsonOutput[pvName]) + 1     ## Get the number of years
    yDim: int = len(jsonOutput) + 1             ## Get the number of regions
    array: list[list[list[str | tuple[str, str]]]] = [[[] for j in range(xDim)] for i in range(yDim)]

    ## Populate first row with the years (cell 0,0 will be left empty)
    column = 1
    for year in jsonOutput[pvName]:
        array[0][column] = [year]
        column += 1

    ## Populate the first column with the regions
    row = 1
    for pvName in jsonOutput:
        array[row][0] = [pvName]
        row += 1

    ## Fill the rest of the cells with land use data
    row = 1
    while row < yDim:
        column = 1
        while column < xDim - 1:
            pvCode: str = array[row][0][0]  # pyright: ignore[reportAssignmentType]
            year: str = array[0][column][0]  # pyright: ignore[reportAssignmentType]
            roads: str = jsonOutput[pvCode][year]["roads"]
            parks: str = jsonOutput[pvCode][year]["parks"]
            array[row][column] = [(roads, parks)]
            column += 1

        row += 1

    ## Merges a 2D array into 1 line
    def mergeLine(array2D: list[list[str | tuple[str, str]]]) -> str:
        output = ''
        for item in array2D:  # pyright: ignore[reportAssignmentType]
            if item == []:
                item: list[str] = ['']
            
            output += str(item[0]) + ';'

        return output[:-1]

    for line in array:
        output += mergeLine(line).strip(";") + '\n'

    output = ";" + output
    output = removeProblemCharacters(output)

    pattern = r",+"

    output = re.sub(pattern, ",", output)
    
    with open(csvOutPath, 'r+') as f:
        print(output, file = f)
    
    return True

def popDensity() -> bool:
    """
    Returns True when population density data has been processed, False otherwise.

    Output file: datafiles/processedPopDensity.csv
    """
    regionCode: dict[str, str] = openJson("datafiles/regionMap.json")  # pyright: ignore[reportUnknownVariableType, reportAssignmentType, reportRedeclaration]
    provinceCode: dict[str, dict[str, list[str]]] = openJson("datafiles/provinceMap.json")  # pyright: ignore[reportUnknownVariableType]
    ## Dictionary structure: {region: {year: data}}
    jsonOutput: dict[str, dict[str, str]] = {}
    output: str = ''
    popDensityFile: str = "datafiles/provincial_population_density.csv"

    csvOutPath: str = "datafiles/processedPopDensity.csv"
    ## If output csv file does not exist, create it
    if not os.path.exists(csvOutPath):
        f = open(csvOutPath, 'x')
        f.close()

    rawDensity: list[str] = open(popDensityFile, 'r').readlines()
    rawDensity = rawDensity[1:]  ## Skip the header row
    with open(csvOutPath, 'w') as f:
        for line in rawDensity:
            entry: list[str] = line.split(sep=",")
            year = entry[2][0:4]

            ## Build a dictionary with region codes as keys; another dictionary as value
            ## Nested dictionary will have years as keys and population density as values
            ## Error handling for region codes that don't have a corresponding region name
            region = ''
            if entry[1] in regionCode.keys():
                region: str = regionCode[entry[1].strip()]
                # for pvCode in provinceCode:
                #     if region in provinceCode[pvCode]["municipalities"]:
                #         pvName: str = provinceCode[pvCode]["name"]  # pyright: ignore[reportAssignmentType]
                #         break

            if region == '':
                region: str = entry[1].strip()

            if "\u2011" in region:
                region: str = region.replace("\u2011", '-')
            
            if region in jsonOutput.keys():
                jsonOutput[region][year] = entry[3].strip()

            else:
                jsonOutput[region] = {year: entry[3].strip()}

    pvName: str = list(jsonOutput.keys())[0]
    ## Make a 3d array to contain all the data to be converted to csv
    ## +1 for the header row
    xDim: int = len(jsonOutput[pvName]) + 2
    yDim: int = len(jsonOutput) + 1  # +1 for the header column
    array: list[list[list[str]]] = [[[]for j in range(xDim)] for i in range(yDim)]
    
    column = 1
    for year in jsonOutput[pvName]:
        array[0][column] = [year]
        column += 1
        # Fill the first row with the years
    array[0][column] = ["\n"]

    row = 1
    for pvName in jsonOutput:
        array[row][0] = [pvName]
        row += 1
        # Fill the first column with the region codes



    # Fill the rest of the array with population density data
    row = 1
    while row < yDim:
        column = 1
        while column < xDim - 1:
            pvCode: str = array[row][0][0]
            year = array[0][column][0]
            populationDensity = jsonOutput[pvCode][year]
            array[row][column] = [populationDensity]
            column += 1

        row += 1

        
    def mergeLine(array2D: list[list[str]]) -> str:
        output = ''
        for item in array2D:
            if item == []:
                item = ['']

            output += item[0] + ','
        
        return output[:-2]
    # Uncomment the next lines when region code has been fixed
    for line in array:
        output += mergeLine(line) + '\n'

    output = removeProblemCharacters(output)

    
    with open(csvOutPath, 'r+') as f:
        print(output, file=f)  # pyright: ignore[reportUnknownVariableType]
    
    return True


def trimProximity() -> bool:
    """
    Remove unnecessary data from the proximity file and convert it to a CSV format.

    Return True when done correctly

    -1 indicates null data
    """
    proximityPath: str = "datafiles/provincial_proximity.json"
    proximityDict: dict[str, dict[str, str]] = openJson(
        proximityPath)["value"]  # pyright: ignore[reportUnknownVariableType]
    regionMap: dict[str, str] = openJson("datafiles/regionMap.json")
    provinceCode: dict[str, dict[str, list[str]]] = openJson("datafiles/provinceMap.json")
    outputFile: str = "datafiles/processedProximity.csv"
    if not os.path.exists(outputFile):
        f = open(outputFile, 'x')
        f.close()

    # Make a dataframe to contain all the data to be converted to csv
    output = pd.DataFrame(
        columns=['Region', 'Year', '0 to 10 km', '>10 to 20 km', '>20 to 50km'])

    # Iterate through all the region & periods
    for row in proximityDict.values():
        region: str = row["Regions"]
        # region: str = provinceCode.get(region, region)

        # pvName = ''     ## Province name
        # for pvCode in provinceCode:
        #     if region in provinceCode[pvCode]["municipalities"]:
        #         pvName: str = provinceCode[pvCode]["name"]
        #         break
        
        ## If region does not have an associated name
        # if region == '':
        #     region: str = region

        if "\u2011" in region:
            region: str = region.replace("\u2011", '-')

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

        output.loc[len(output)] = [region.strip(), year, in10, in20, in50]

    output = removeProblemCharacters(output)

    output.to_csv(outputFile, index=False)
    return True

# def translateRegionCode() -> None:
#     """
#     Translates region code from csv format to json format, maps NL codes to region name
#     """
#     NLPath: str = "datafiles/netherlandsRegionCodeTranslation2.csv"
#     outputPath: str = "datafiles/measurementStations.json"
#     file: dict[str, str] = openJson(outputPath)  # noqa: F841  # pyright: ignore[reportUnusedVariable, reportUnknownVariableType]

#     output: dict[str, str] = {}
#     with open(NLPath, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             parted: list[str] = line.split(',')
#             if parted[1] == '\n':
#                 output[parted[0]] = parted[0]
#             else:
#                 try:
#                     output[parted[0]] = parted[2].strip()
#                 except:
#                     output[parted[0]] = parted[0]
    
#     json.dump(output, open(outputPath, 'w'), indent=4)
#     return None


# def trimRegionCode(filename: str) -> None:
#     """
#     Trim region code Json to only map region codes to their names.

#     Naam_2 refers to the region name
#     Code_1 refers to the region code

#     Args:
#         filename (str): The name of the JSON file to process (without .json extension).

#     Returns None

#     File output: datafiles/regionMap.json
#     """
#     regionRaw: dict[str, dict[str, str]] = openJson(f"datafiles/{filename}.json")["value"]  # pyright: ignore[reportUnknownVariableType]
#     regionMap: dict[str, str] = {}

#     for region in regionRaw:  # pyright: ignore[reportUnknownVariableType]
#         item: dict[str, str] = regionRaw[region]  # pyright: ignore[reportUnknownVariableType]
#         regionMap[item["Code_1"].strip()] = item["Naam_2"].strip()  # pyright: ignore[reportUnknownMemberType]

#     if not os.path.exists("datafiles/regionMap.json"):
#         f = open("datafiles/regionMap.json", 'x')
#         json.dump({}, f)  # Initialize with an empty JSON object
#         f.close()

#     json.dump(regionMap, open("datafiles/regionMap.json", "w"), indent=4)

#     return None


# def trimPM() -> bool:
#     """
#     Remove unnecessary data from the PM2.5 file

#     Returns True when done correctly
#     """
#     pmFile: str = "datafiles/pm2.5_2013-2023_netherlands.csv"
#     outputFile: str = "datafiles/processedPm2.5.csv"

#     # Create the output file if it does not exist
#     if not os.path.exists(outputFile):
#         f = open(outputFile, 'x')
#         f.close()
#     regionMap: dict[str, str] = openJson("datafiles/regionMap.json")  # pyright: ignore[reportUnknownVariableType]
#     nlMap: dict[str, str] = openJson("datafiles/measurementStations.json")  # pyright: ignore[reportUnknownVariableType]

#     with open(pmFile, 'r') as f:
#         lines: list[str] = f.readlines()
#         lines = lines[:61]  ## Removes description at the bottom of the csv

#         # Iterate through the lines and standardise empty values, and translate region codes
#         for i in range(1, len(lines)):   # Skip the header row
#             line: str = lines[i]
#             code = line.split(",")[0]
#             lines[i] = line.replace("-", "")
#             gmCode = nlMap.get(code,code)
#             region = regionMap.get(gmCode,gmCode)
#             lines[i] = line.replace(code, region)  # Uncomment once regionMap is available

#     output = ''.join(lines)
#     output = removeProblemCharacters(output)

#     print(output, file=open(outputFile, 'w'))
#     return True

def carTravel() -> bool:

    carTravelFile: str = "datafiles/transportTravelByProvince.json"
    outputPublicFile: str = "datafiles/processedCarTravelPublic.csv"
    outputPrivateFile: str = "datafiles/processedCarTravelPrivate.csv"
    with open(carTravelFile, 'r', encoding='utf-8-sig') as f:
        carTravelDict: dict = json.load(f)['value']
    provinceMap: dict[str, str] = openJson("datafiles/provinceMap.json")

    travelModeDict: dict[str, str] = {"A048583": 'Private', "A048584": "Private", "A018981": "Public", "A018982": "Public", "A018984": None, "A018985": None, "A018986": None }
    
    outputDict: dict[str: dict[str: float]] = {}
    outputPrivate: pd.DataFrame = pd.DataFrame(columns=['Region','Year','Private Transport in km'])
    outputPublic: pd.DataFrame = pd.DataFrame(columns=['Region','Year','Public Transport in km'])
    for _, data in carTravelDict.items():
        travelMode: str = data["TravelModes"]
        travelMode: str | None = travelModeDict[travelMode]
        if travelMode is None:
            continue
        region: str = data["RegionCharacteristics"].strip()
        year: str = data["Periods"][0:4]

        dist1: str = data["Trips_1"]
        dist2: str = data["Trips_4"]
        print(dist1, dist2)
        try:
            dist1 = float(dist1)
            dist2 = float(dist2)
            distTravelled: float = (dist1 + dist2) / 2
        except:
            distTravelled = -1.0
        regionyear: str = region + "@" + year
        if regionyear not in outputDict:
            outputDict[regionyear] = {}
        try:
            outputDict[regionyear][travelMode] += distTravelled
            if distTravelled == -1:
                outputDict[regionyear][travelMode] += 1
                continue
        except:
            outputDict[regionyear][travelMode] = distTravelled

    for regionyear, travelData in outputDict.items():
        region, year = regionyear.split("@")
        try:
            assert region in provinceMap.keys()
        except:
            continue
        travelPublic: float = travelData.get("Public","")
        travelPrivate: float = travelData.get("Private","")
        
        if travelPublic:
            outputPublic.loc[len(outputPublic)] = [region, year, travelPublic] 
        if travelPrivate:
            outputPrivate.loc[len(outputPrivate)] = [region, year, travelPrivate]
        
    outputPublic = outputPublic.map(removeProblemCharacters)
    outputPrivate = outputPrivate.map(removeProblemCharacters)

    print(f'{outputPublic=}')
    print(f'{outputPrivate=}')
            
    outputPublic.to_csv(outputPublicFile, index=False)
    outputPrivate.to_csv(outputPrivateFile, index=False)

    return True



def getFiveNumberSummary(array: np.ndarray | pd.DataFrame) -> dict[str: int]:  # pyright: ignore[reportGeneralTypeIssues, reportInvalidTypeArguments, reportUnknownParameterType]
    assert array.shape[1] == 1 or len(array.shape) == 1

    min = np.min(array)  # pyright: ignore[reportAny]
    max = np.max(array)  # pyright: ignore[reportAny]
    median = np.median(array)
    q1 = np.quantile(array, 0.25)
    q3 = np.quantile(array, 0.75)

    return min, q1, median, q3, max  # pyright: ignore[reportReturnType]

def removeProblemCharacters(string: str) -> str:
    if not isinstance(string, str):
        return string
    problemCharacters: dict[str, str] = {"â" : 'a', "ú": 'u'}
    for problem, fixed in problemCharacters.items():
        string = string.replace(problem, fixed)
    return string


# Testing and Example Usage
## ===================================== ##


if __name__ == "__main__":
    # Example usage
    # print(translateRegionCode())
    # trimRegionCode("reregionCode")
    # print(trimPM())
