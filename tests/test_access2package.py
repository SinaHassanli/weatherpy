import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import weatherpy as wp
import windpy



# print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print(dir(weatherpy))
lat = 3.12
lon = 101.55

# Find all stations within a given distance from a geographical location
# stations = wp.NOAA.findStation(centre = [lat, lon], radius = 5, unit = 'km')


dataType='ncdc'
# stnNo = '066037'
stnNo = '48647099999' # 10-digit ID for ncdc data

siteName = 'Sultan Abdol Aziz Shah \n International Airport'
firstYear=2000
lastYear=2015

noDirs = 16
isCorrected=False


if dataType=='bom':
    # Labels for BoM data
    dateTime_col = 'Local Time'
    gustSpeed_col = 'Speed of maximum windgust in last 10 minutes in m/s'
    windSpeed_col = 'Wind speed in m/s'
    windDir_col = 'Wind direction in degrees true'
    temp_col = 'Air Temperature in degrees C'
    season_col = 'Season'
    timeOfDay_col = 'TimeOfDay'
    tempLabel_col = "TemperatureLabel"
elif dataType=='ncdc':
    windSpeed_col = 'WindSpeed'
    windDir_col = 'WindDir'
else:
    pass

# df = wp.NOAA.getNOAA(ID=947670, yearStart=firstYear, yearEnd=lastYear)
raw_data = wp.import_data(stnNo, dataType=dataType, firstYear=firstYear, lastYear=lastYear)

if dataType=='ncdc':
    raw_data = windpy.NOAA.fix_NOAA_dimensions(raw_data)

data_cleaned = wp.clean_data(raw_data, dataType=dataType)

data, isCorrected = wp.correct_terrain(data_cleaned, stnNo, dataType=dataType)

wp.windrose(data[windSpeed_col], 
         data[windDir_col],
         firstYear, 
         lastYear, 
         siteName, 
         stnNo, 
         noDirs,
         isCorrected=True)