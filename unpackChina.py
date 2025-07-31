## Python script to process China PM2.5 data
import os

## China file type is nc, NetCDF

def unpackChina(year: str) -> None:
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
    """
    filepath: str = f"datafiles/CHAP_PM2.5_Y1K_{year}_v4.nc"
    outputName: str = f"datafiles/processed_CHAP_PM2.5_{year}.csv"
    output: str = ""

    if not os.path.exists(outputName):
        file = open(filepath, 'x')
        file.close()
    
    ### Do processing here

    ### Processing done, output is ready
    with open(outputName, 'w') as f:
        f.write(output)
    print(f"Data for {year} processed and saved to {outputName}")

    return None
