## Python script to process China PM2.5 data
## Run this script outside of datafiles directory
import os
import netCDF4 as nc
import json

## China file type is nc, NetCDF

def ncToCsv(year: str) -> None:
    """
    Unpack the China PM2.5 data from a NetCDF file.
    Args:
        year (str): The year for which to unpack the data.

    Returns:
        None

    Functionality:
        - Reads the NetCDF file for the specified year.
        - Extracts PM2.5 data and associated metadata.
        - Processes the data into a usable format for further analysis.
        - Saves the processed data to csv in datafiles.
    
    Error Handling:
        - Raises FileNotFoundError if the specified NetCDF file does not exist.
    """
    filepath: str = f"datafiles/CHAP_PM2.5_Y1K_{year}_V4.nc"
    outputName: str = f"datafiles/processed_CHAP_PM2.5_{year}.csv"
    output: str = ""

    # Check if target file exists, raises FileNotFoundError if not
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist. Please check the year and file path.")

    # Make output file if file does not exist
    if not os.path.exists(outputName):
        file = open(outputName, 'x')
        file.close()
    
    ### Do processing here
    ### NOT DONE YET, still experimenting with netCDF4 library
    with nc.Dataset(filepath, 'r') as dataset:
        # Extract PM2.5 data
        print(dataset.variables['PM2.5'].size)
        print(dataset.variables['PM2.5'][0][0].datatype)  # Check the data type of PM2.5 values  # pyright: ignore[reportAny]
        pm25_data = dataset.variables['PM2.5'][:]  # pyright: ignore[reportAny]
        print(len(pm25_data))  # pyright: ignore[reportAny]

        # for record in pm25_data:
        #     # Assuming the record is a 1D array of PM2.5 values
        #     # Convert to string and join with commas
        #     record_str = ','.join(map(str, record))
        #     output += f"{record_str}\n"

    ### Processing done, output is ready
    with open(outputName, 'w') as f:
        f.write(output)  # pyright: ignore[reportUnusedCallResult]
    print(f"Data for {year} processed and saved to {outputName}")

    return None


def checkProcessed(year:str) -> bool:
    """
    Check if year has been processed, track via json file
    
    Args:
        year (str): The year to check if it has been processed.

    Returns:
        bool: True if the year has been processed, False otherwise.
    """
    filepath: str = "datafiles/processed_years.json"

    # Ensure the file exists, create it if it does not
    if not os.path.exists(filepath):
        file = open(filepath, 'x')
        file.close()

    with open(filepath, 'r') as f:
        years: dict[str, bool] = json.load(f)
        if years.get(year, False):
            return True
        else:
            return False
        
def markProcessed(year: str) -> None:
    """
    Mark a year as processed in the JSON file.
    
    Args:
        year (str): The year to mark as processed.

    Returns:
        None
    """
    filepath: str = "datafiles/processed_years.json"
    
    # Ensure the file exists, create it if it does not
    if not os.path.exists(filepath):
        file = open(filepath, 'x')
        file.write('{}')  # Initialize with an empty JSON object  # pyright: ignore[reportUnusedCallResult]
        file.close()

    with open(filepath, 'r+') as f:
        years: dict[str, bool] = json.load(f)
        years[year] = True
        f.seek(0)  # pyright: ignore[reportUnusedCallResult]
        json.dump(years, f, indent=4)

def readProcessed() -> dict[str, bool]:
    """
    Convert a JSON file to a Python dictionary for easy access to processed years.
    
    Args:
        filepath (str): The path to the JSON file.

    Returns:
        dict: The data from the JSON file as a Python list.
    """
    filepath = "datafiles/processed_years.json"

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist.")
    
    with open(filepath, 'r') as f:
        data: dict[str, bool] = json.load(f)

    return data


if __name__ == "__main__":
    # Example usage
    try:
        ncToCsv("2021")  # Replace "2021" with the desired year
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
