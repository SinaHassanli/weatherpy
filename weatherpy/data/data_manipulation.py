import pandas as pd
import numpy as np
import os
import windpy

### ADD COLUMN NAMES AS GLOBAL VARIABLES ###
# Labels for BoM data
dateTime_col = 'Local Time'
gustSpd_col = 'Speed of maximum windgust in last 10 minutes in m/s'
meanSpd_col = 'Wind speed in m/s'
windDir_col = 'Wind direction in degrees true'
temp_col = 'Air Temperature in degrees C'
corrMeanSpd_col = 'correctedMeanSpd'
corrFactor_col = 'correctionFactor'
season_col = 'Season'
timeOfDay_col = 'TimeOfDay'
tempLabel_col = "TemperatureLabel"

# TO DO: add NCDC data
# Pipe this to new dev (historic+new data)

def import_data(stnNo, dataType='bom', firstYear=None, lastYear=None):
    """
    Import historic weather data
    
    Parameters
    ----------
    inputParams : Dict
        Input Parameters.

    Returns
    -------
    df : Dataframe
        DESCRIPTION.

    """
    if dataType=='bom':
        weatherFileUrl = "https://s3-ap-southeast-2.amazonaws.com/bomhistoricobs/hourly_data_{}.csv.gz".format(stnNo)
        df = pd.read_csv(weatherFileUrl, compression='gzip', parse_dates=['Local Time'])
    
    elif dataType=='ncdc':
        # if to import ncdc, similar columns with bom should be renamed for consistency
        df = windpy.NOAA.getNOAA(ID=stnNo, yearStart=firstYear, yearEnd=lastYear)
    else:
        print('dataType is bom or ncdc')
        df = pd.DataFrame()
        
    return df


def clean_data(df, dataType='bom', column2clean=[meanSpd_col, windDir_col]):
    """Remove NaN in weather data based on columns to clean
    

    Parameters
    ----------
    df : DataFrame
        DESCRIPTION.
    whichPlot : Dict
        DESCRIPTION.

    Returns
    -------
    clean DataFrame df

    """              
    if dataType=='bom':    
        for columnName in column2clean:
            not_nan = (df[columnName].notnull())
            df_cleaned = df[not_nan]

    elif dataType=='ncdc':

        # removing wind speed over 100, missing data or data with variable wind directions.
        filt = (df['WindSpeed']>100) | (df['WindType']=='9') | (df['WindType']=='V')
        _df_cleaned = df.loc[~filt]

        # changing 
        filt = (_df_cleaned['WindType']=='C') & (_df_cleaned['WindDir']==999) #check if we have to remove windir=999 from data
        _df_cleaned.loc[filt,'WindDir'] = 0

        filt = _df_cleaned['WindDir']<360
        df_cleaned = _df_cleaned[filt]
    else:
        print('dataType should be bom or ncdc')
        df_cleaned = pd.DataFrame()
        
    print('\ndata is cleand')
    return df_cleaned

def correct_terrain(df, stnNo, dataType='bom'):
    
    df_corrected = df.copy()
    
    if dataType=='bom':
        
        dataPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..//data'))
        filePath = os.path.join(dataPath,'correctionFactors.csv')
        print('reading correcton factor from \n{}'.format(filePath))

        _correctionFactor_database=pd.read_csv(filePath,dtype={'stnNo': str})
        
        correctionFactor_database = _correctionFactor_database[_correctionFactor_database['stnNo'].notnull()]

        if stnNo in correctionFactor_database['stnNo'].values:
            [corrFactors] = correctionFactor_database[correctionFactor_database['stnNo'].str.contains(stnNo)].iloc[:,np.r_[2:18]].values.tolist()
            _corrFactors = corrFactors + [corrFactors[0]] # add the north value at the end for interpolation
            isCorrected = True
            print('corection factors applied')
        else:
            _corrFactors = np.ones(16)
            isCorrected = False
            print('no corection factor applied')

        windDirs = np.linspace(0, 360, len(_corrFactors))

        
        df_corrected = df_corrected.assign(correctionFactor=np.interp(df.loc[:, windDir_col], windDirs, _corrFactors)) # interpolate correction factors
        df_corrected.loc[df[windDir_col] == 0, corrFactor_col] = 0  # Set correction factor to zero for calms
        # df_corrected.loc[meanSpd_col] = df_corrected[meanSpd_col]*df_corrected['correctionFactor']-0.4
        df_corrected.loc[meanSpd_col] = df_corrected[meanSpd_col]*df_corrected['correctionFactor']

    else:
        isCorrected = False
        print('no corection factor applied')
        
    return df_corrected, isCorrected