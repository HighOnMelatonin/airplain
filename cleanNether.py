## Python script to process China PM2.5 data
## Run this script outside of datafiles directory

import os
import json

## China file type is nc, NetCDF

def openJson(filepath: str) -> dict:  # pyright: ignore[reportUnknownParameterType, reportMissingTypeArgument]
    """
    Open a JSON file and return its contents as a dictionary.
    
    Args:
        filepath (str): The path to the JSON file.

    Returns:
        dict: The data from the JSON file as a Python dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")
    
    f = open(filepath, 'r', encoding='utf-8')  # pyright: ignore[reportUnknownVariableType, reportArgumentType]
    data: dict = json.load(f)  # pyright: ignore[reportAny, reportMissingTypeArgument]

    f.close()
    return data  # pyright: ignore[reportUnknownVariableType]

def popDensity() -> bool:
    """
    Returns True if population density data has been processed, False otherwise.

    Output file: datafiles/processed_pop_density.csv
    """
    region_code: dict[str, dict[str, dict[str, str]]] = openJson("datafiles/region_map.json")  # pyright: ignore[reportUnknownVariableType]
    jsonOutput: dict[str, dict[str, str]] = {}
    output: str = ''
    popDensityFile: str = "datafiles/Netherlands_population_density.csv"

    csvOutPath: str = "datafiles/processed_pop_density.csv"
    ## If output csv file does not exist, create it
    if not os.path.exists(path=csvOutPath):
        f = open(csvOutPath, 'x')
        f.close()
    
    rawDensity: list[str] = open(popDensityFile, 'r').readlines()  # pyright: ignore[reportUnknownVariableType]
    rawDensity = rawDensity[1:]  # Skip the header row
    with open(csvOutPath, 'w') as f:  # pyright: ignore[reportUnknownVariableType]
        for line in rawDensity:
            entry: list[str] = line.split(sep=",")
            year = entry[2][0:4]
            
            ## Build a dictionary with region codes as keys; another dictionary as value
            ## Nested dictionary will have years as keys and population density as values
            if region_code[entry[1]] not in jsonOutput.keys():
                jsonOutput[region_code[entry[1]]] = {year: entry[3].strip()}  # pyright: ignore[reportArgumentType]

            else:
                jsonOutput[region_code[entry[1]]][year] = entry[3].strip()  # pyright: ignore[reportArgumentType]

            print(jsonOutput)
    
    region: str = list(jsonOutput.keys())[0]
    ## Make a 3d array to contain all the data to be converted to csv
    xDim: int = len(jsonOutput[region]) + 2  # +1 for the header row +1 for newline
    yDim: int = len(jsonOutput) + 1  # +1 for the header column
    array: list[list[list[str]]] = [ [ [] for j in range(xDim) ] for i in range(yDim)]

    column = 1
    for year in jsonOutput[region]:
        array[0][column] = [year]
        column += 1
        ## Fill the first row with the years

    row = 1
    for region in jsonOutput:
        array[row][0] = [region]
        row += 1
        ## Fill the first column with the region codes

    ## Fill the rest of the array with population density data
    row = 1
    while row < yDim:
        column = 1
        while column < xDim - 1:
            ## array[row][0][0] is the region code
            ## array[0][column][0] is the year
            ## jsonOutput[region][year] is the population density
            array[row][column] = [jsonOutput[array[row][0][0]][array[0][column][0]]]
            column += 1

        array[row][column] = ['\n'] # Add a newline at the end of each row
        row += 1

    print(array)
    
    return True

def trimRegionCode() -> None:
    """
    Trim region code Json to only map region codes to their names.

    Naam_2 refers to the region name
    Code_1 refers to the region code

    Returns None

    File output: datafiles/region_map.json
    """
    regionRaw: dict[str, dict [str, dict[str, str]]] = openJson("datafiles/reregion_code.json")  # pyright: ignore[reportUnknownVariableType]
    regionMap: dict[str, str] = {}

    for region in regionRaw["value"]:
        item: dict[str, str] = regionRaw["value"][region]
        regionMap[item["Code_1"]] = item["Naam_2"].strip()  # pyright: ignore[reportUnknownVariableType, reportCallIssue]

    if not os.path.exists("datafiles/region_map.json"):
        f = open("datafiles/region_map.json", 'x')  # pyright: ignore[reportUnusedCallResult]
        json.dump({}, f)  # Initialize with an empty JSON object  # pyright: ignore[reportUnusedCallResult]
        f.close()

    json.dump(regionMap, open("datafiles/region_map.json", "w"), indent=4)  # pyright: ignore[reportUnknownVariableType, reportCallIssue]

    return None


## Testing and Example Usage
## ===================================== ##

if __name__ == "__main__":
    # Example usage
    try:
        # trimRegionCode()
        print(popDensity())
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
