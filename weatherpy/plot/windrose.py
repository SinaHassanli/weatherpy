from collections import defaultdict, namedtuple
from itertools import repeat

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

WIND_DIRECTIONS = ['N','NE','E','SE','S','SW','W','NW']
WIND_ANGLES = list(range(10, 361, 10)) # [10, 20, 30, ..., 340, 350, 360]
DEFAULT_WIND_SPEED_RANGES = [0, 2, 4, 6, 8, 10, 15]
DEFAULT_TEMP_RANGES = list(np.r_[0:51])
NUMBER_OF_WIND_DIRECTIONS = 16

FONTSIZE = 8
TEXT_DESCRIPTION_LOCATION = (0.025, 0.85)
TEXT_TERRAIN_CORRECTION_LOCATION = (0.025, 0.8)

WINDROSE_SECTOR_WIDTH = 0.85 # set to a number between 0 and 1

WindroseSegment = namedtuple('WindroseSegment', ['direction', 'probability', 'offset', 'width', 'colour'])

def _generate_windspeed_ranges(spd_range):
    return list(zip(spd_range, spd_range[1:]+[np.inf]))

def _generate_label(spd_range):
    """Converts wind speed range to a label (str)
    """
    lower_bound, upper_bound = spd_range
    if np.isinf(upper_bound):
        return ">"+str(lower_bound)+" m/s"
    else:
        return ">"+str(lower_bound)+"-"+str(upper_bound)+" m/s"
    
def _generate_label_T(temp_range):
    """Converts wind speed range to a label (str)
    """
    lower_bound, upper_bound = temp_range
    if np.isinf(upper_bound):
        return ">"+str(lower_bound)+" C"
    else:
        return ">"+str(lower_bound)+"-"+str(upper_bound)+" C"

def _wind_speed_colour(spd_range, wind_spd_ranges=DEFAULT_WIND_SPEED_RANGES):
    """Convert wind speed range to colour for plotting
    """
    wind_spd_ranges = _generate_windspeed_ranges(wind_spd_ranges)
    for i, wind_speed_range in enumerate(wind_spd_ranges):
        if _generate_label(wind_speed_range) == spd_range:
            return plt.cm.jet((i+1)/len(wind_spd_ranges))

def _sector_wind_directions(no_sectors):
    return np.linspace(360/no_sectors, 360, no_sectors, endpoint=True)

def windspeed_pdf(wind_dir, wind_speed, wind_spd_ranges=DEFAULT_WIND_SPEED_RANGES):
    """PDF of wind speeds for a given direction and wind speed band.

    Parameters
    ----------------------------------
    wind_dir : numpy.array or pandas.Series
        Vector of wind directions (in degrees, 0 is calm, 360 is north)
    wind_speed : numpy.array or pandas.Series
        Vector of wind speeds (in m/s)
    
    Optional
    ----------------------------------
    wind_spd_ranges : [Int]
        List of wind speeds that bound ranges
        Default: [0, 2, 4, 6, 8, 10, 15]
    
    Returns
    ----------------------------------
    pdf : pandas.Dataframe
    """
    
    def _count(wind_direction, wind_speed_range):
        lower_bound, upper_bound = wind_speed_range
        mask = (wind_dir==direction) & (lower_bound<wind_speed) & (wind_speed<=upper_bound)
        return np.sum(mask)
    
    def _probability(wind_direction, wind_speed_range):
        return _count(wind_direction, wind_speed_range)/wind_speed.size

    wind_speed_ranges = _generate_windspeed_ranges(wind_spd_ranges) # creating ranges (2-4 m/s, etc) for the final pdf dataframe

    wind_speed_pdf = pd.DataFrame()

    for direction in WIND_ANGLES: # if wind angles are not rounded to near 10 degree it does nto work
        wind_speed_pdf[direction] = pd.Series(
            [_probability(direction, spd_range) for spd_range in wind_speed_ranges],
            index = [_generate_label(spd_range) for spd_range in wind_speed_ranges])

    return wind_speed_pdf

def _map_to_nsectors(N):
    """Function to produce a map to reduce number of wind sectors from 36.

    Parameters
    -----------------
    N : int
        Number of sectors to reduce to

    Returns
    ----------------------------------
    defaultdict(<class 'list'>,
        Map providing the fraction of frequencies to be distributed from
        each of the 36 wind directions.

        Example output for 16 wind directions:

            {22.5: [[0.4444444444444444, 10],
                    [0.8888888888888888, 20],
                    [0.6666666666666667, 30],
                    [0.2222222222222222, 40]],
             45.0: [[0.3333333333333333, 30],
                    [0.7777777777777778, 40],
                    [0.7777777777777778, 50],
                    [0.33333333333333337, 60]],
             67.5: [[0.2222222222222222, 50],
                    [0.6666666666666666, 60],
                    .
                    .
                    .
    """
    wind_angles_reduced = _sector_wind_directions(N)

    map_to_sector = defaultdict(list)

    for wind_angle in WIND_ANGLES:
        upper_index = np.searchsorted(wind_angles_reduced, wind_angle)
        upper_wind_angle = wind_angles_reduced[upper_index]
        if upper_index == 0:
            lower_wind_angle = WIND_ANGLES[-1]
            fraction_to_upper = wind_angle/upper_wind_angle
        else:
            lower_wind_angle = wind_angles_reduced[upper_index - 1]
            fraction_to_upper = (wind_angle - lower_wind_angle)/(upper_wind_angle - lower_wind_angle)
        map_to_sector[upper_wind_angle].append([fraction_to_upper, wind_angle])
        map_to_sector[lower_wind_angle].append([1-fraction_to_upper, wind_angle])
    
    return map_to_sector

def sector_windspeed_pdf(wind_dir, wind_speed, n_sectors=NUMBER_OF_WIND_DIRECTIONS, wind_spd_ranges=DEFAULT_WIND_SPEED_RANGES):
    """Produce a wind speed pdf for specified number of wind directions.
    
    Parameters
    ----------------------------------
    wind_dir : numpy.array or pandas.Series
        Vector of wind directions (in degrees, 0 is calm, 360 is north)
    wind_speed : numpy.array or pandas.Series
        Vector of wind speeds (in m/s)
    n_sectors : int, optional (default 16)
        Number of sectors of the wind rose
    wind_spd_ranges : list of int, optional (default [0,2,4,6,8,10,15])
        List of wind speeds that bound wind speed ranges
    
    Returns
    ----------------------------------
    sector_pdf : pandas.Dataframe
    """
    pdfAll = windspeed_pdf(wind_dir, wind_speed, wind_spd_ranges)
    map_to_sector = _map_to_nsectors(n_sectors)
    _sector_windspeed_pdf = pd.DataFrame()
    for sector_wind_direction, probability_share in map_to_sector.items():
        _sector_windspeed_pdf[sector_wind_direction] = sum(fraction*pdfAll[wd] for fraction, wd in probability_share)

    # sort columns before returning
    wind_angles_reduced = list(_sector_wind_directions(n_sectors)) # order of coloums
    pdf=_sector_windspeed_pdf[wind_angles_reduced]
    
    # bring 360.0 Column to the first column and rename to 0.0
    _zeroDir=pdf[360.0]; pdf=pdf.drop(labels=[360.0], axis=1); pdf.insert(0, 0.0, _zeroDir)

    # cumulative pdf
    _cpdf=(pdf.iloc[::-1]).cumsum();
    cpdf=_cpdf.iloc[::-1];
    cpdf_index=[">"+str (i)+" m/s" for i in wind_spd_ranges];
    cpdf.set_index([cpdf_index],inplace=True, drop=True)

    # probability of exceedance
    probExceed=[0.000114,0.00022,0.001,0.01,0.05,0.1,0.2,0.5,0.8,0.95,0.99]
    probExceedLabel=['0.0114%','0.022%','0.1%','1%','5%','10%','20%','50%','80%','95%','99%']
    spdExceed=pd.DataFrame(0, index=probExceedLabel, columns=pdf.columns)
    for i in range(len(probExceed)):
        for j in range(len(pdf.columns)):
            d=probExceed[i]*cpdf.iloc[0,j] # finding the probability exceedance for the speciifc direction
            spdExceed.iloc[i,j]=np.interp(d,cpdf.iloc[:,j],wind_spd_ranges,period=100) #fidning the corresponding speed
    
    return pdf, cpdf, spdExceed


def sector_temp_pdf(wind_dir, temp, siteLabel, n_sectors=NUMBER_OF_WIND_DIRECTIONS, temp_ranges=DEFAULT_TEMP_RANGES):

    pdfAll = windspeed_pdf(wind_dir, temp, temp_ranges)
    map_to_sector = _map_to_nsectors(n_sectors)
    _sector_temp_pdf = pd.DataFrame()
    for sector_wind_direction, probability_share in map_to_sector.items():
        _sector_temp_pdf[sector_wind_direction] = sum(fraction*pdfAll[wd] for fraction, wd in probability_share)

    # sort columns before returning
    wind_angles_reduced = list(_sector_wind_directions(n_sectors)) # order of coloums
    Tpdf=_sector_temp_pdf[wind_angles_reduced]
    # bring 360.0 Column to the first column and rename to 0.0
    _zeroDir=Tpdf[360.0]; Tpdf=Tpdf.drop(labels=[360.0], axis=1); Tpdf.insert(0, 0.0, _zeroDir)

    # cumulative pdf
    _Tcpdf=(Tpdf.iloc[::-1]).cumsum();
    Tcpdf=_Tcpdf.iloc[::-1];
    Tcpdf_index=[">"+str (i)+" C" for i in temp_ranges];
    Tcpdf.set_index([Tcpdf_index],inplace=True, drop=True)
    Tpdf.set_index([Tcpdf_index],inplace=True, drop=True)

    # probability of exceedance
    probExceed=[0.01,0.05,0.1,0.2,0.5,0.8,0.9,0.95,0.99]
    probExceedLabel=['1%','5%','10%','20%','50%','80%','90%','95%','99%']
    Texceed=pd.DataFrame(0, index=probExceedLabel, columns=Tpdf.columns)
    for i in range(len(probExceed)):
        for j in range(len(Tpdf.columns)):
            d=probExceed[i]*Tcpdf.iloc[0,j] # finding the probability exceedance for the speciifc direction
            Texceed.iloc[i,j]=np.interp(d,Tcpdf.iloc[:,j],temp_ranges,period=100) #fidning the corresponding speed
    
    return Tpdf, Tcpdf, Texceed


def _windrose_segments(wind_spd_pdf, n_sectors, wind_spd_ranges):
    """Yields list of windrose segments (or bars).

    Parameters
    ----------------------------------
    wind_spd_pdf : pandas.Dataframe
    n_sectors : Int

    Returns
    ----------------------------------
    [WindroseSegments]
    """
    width = WINDROSE_SECTOR_WIDTH * 2 * np.pi / n_sectors
#    windSpdPdf= wind_spd_pdf.drop(['total','calm'], axis=1) # removing total and calm columns
    segments = []
    for wind_direction, wd_freq in wind_spd_pdf.iteritems():
        offset = 0
        for spd_range, frequency in wd_freq.items():
            precentage = frequency * 100
            segments.append(WindroseSegment(
                np.radians(wind_direction),
                precentage,
                offset,
                width,
                _wind_speed_colour(spd_range, wind_spd_ranges)
            ))
            offset += precentage
    return segments

def windrose(wind_speed, wind_dir , firstYear, lastYear, siteName, stnNo, noDirs=NUMBER_OF_WIND_DIRECTIONS, isCorrected=False, description=" ", wind_spd_ranges=DEFAULT_WIND_SPEED_RANGES, max_radius=False, save=False, save_filename=None):
    """Function to produce a Windrose.

    Parameters
    -----------------
    wind_dir : numpy.array or pandas.Series
        Vector of wind directions (in degrees, 0 is calm, 360 is north)
    wind_speed : numpy.array or pandas.Series
        Vector of wind speeds (in m/s)
    site_name : str
        Name of site e.g. "Sydney Apt 066037"
    years : str
        Years covered e.g. "1992-2017"
    description : str, optional (default None)
        An additional string to add to the plot (e.g. "Summer", "All hours")
    n_sectors : int, optional (default 16)
        Number of sectors of the wind rose
    wind_spd_ranges : list of int, optional (default [0,2,4,6,8,10,15])
        List of wind speeds that bound wind speed ranges
    max_radius : float, optional (default calculated manually)
        The maximum radius of the wind rose.
    corrected : bool, optional (default False)
        Has the wind data been corrected
    save : bool, optional (default False)
        Save figure to the current working directory
        Will show the image if set to false
    save_filename : str, optional (default determine automatically)
        Custom filename to save the windrose.
        Must have a .jpeg or .png file extension
    """
    yearsLabel = "{}-{}".format(firstYear,lastYear)
    siteLabel = "{} ({})".format(siteName,stnNo)
    n_sectors = noDirs
    
    # wind_dir=r
    # Some small input checks
    assert len(wind_dir)>0, "The wind direction vector has zero length, cannot produce windrose."
    assert len(wind_speed)>0, "The wind speed vector has zero length, cannot produce windrose."
    # Get the wind speed histograms
    pdf, cpdf, spdExceed = sector_windspeed_pdf(wind_dir, wind_speed, n_sectors, wind_spd_ranges=wind_spd_ranges)
    

    
    calms = 1-pdf.values.sum()
    segments = _windrose_segments(pdf, n_sectors, wind_spd_ranges)

    ax = plt.subplot(111, projection='polar')
    # format the theta grid
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    ax.set_thetagrids(np.linspace(0, 360, len(WIND_DIRECTIONS), endpoint=False), labels=WIND_DIRECTIONS)
    # format the radial grid
    if not max_radius:
        max_radius = max(s.probability + s.offset for s in segments)
        max_radius = int((1+max_radius//2)*2)
    rticks = np.round(np.linspace(max_radius/4, max_radius, 4, endpoint=True), decimals=1)
    rlabels = list(repeat("", len(rticks)-1))+[str(max_radius)+"%"]
    ax.set_rgrids(rticks, labels=rlabels)
    ax.set_ylim([0,rticks.max()])

    # add data
    bars = ax.bar(
        [s.direction for s in segments],
        [s.probability for s in segments],
        width=[s.width for s in segments],
        bottom=[s.offset for s in segments],
        zorder=10,
        color=[s.colour for s in segments],
        edgecolor=list(repeat(plt.cm.binary(256), len(segments))),
        linewidth=0.1)

    # Legend
    legend_colours = [_wind_speed_colour(_generate_label(spd_rng), wind_spd_ranges) for spd_rng in _generate_windspeed_ranges(wind_spd_ranges)]
    handles = [mpatches.Rectangle((0, 0), 0.15, 0.15, facecolor=colour, edgecolor='black') for colour in legend_colours]
    labels = [_generate_label(spd_range) for spd_range in _generate_windspeed_ranges(wind_spd_ranges)]
    plt.legend(handles=handles, labels=labels, loc=(-0.46,-0.05), fontsize=FONTSIZE)

    # Add text
    
    text = '\n'.join([siteLabel, yearsLabel, description, "Calms: "+str(round(calms*100, 2))+"%"])
    plt.text(*TEXT_DESCRIPTION_LOCATION, text, fontsize=FONTSIZE, transform=plt.gcf().transFigure)
    
    if isCorrected:
        plt.text(*TEXT_TERRAIN_CORRECTION_LOCATION, "Corrected to open terrain", fontsize=FONTSIZE, transform=plt.gcf().transFigure)
    else:
        plt.text(*TEXT_TERRAIN_CORRECTION_LOCATION, "Uncorrected for terrain", fontsize=FONTSIZE, transform=plt.gcf().transFigure)
    
    # save image
    if save:
        if save_filename:
            csv_filename = save_filename.split('.')[0]+".xlsx"
            plt.savefig(save_filename, dpi=600, transparent=False)
        else:
            csv_filename = "windrose - "+str(n_sectors)+" dir - "+siteLabel+" - "+description+".csv"
            plt.savefig("windrose - "+str(n_sectors)+" dir - "+siteLabel+" - "+description+".png", dpi=600, transparent=False)
        
        # adding total and calm columns
        pdf=pdf.assign(total=pdf.sum(axis=1).values)
        pdf.loc[pdf.index[0],'calm']= 1-pdf['total'].sum()
        cpdf=cpdf.assign(total=cpdf.sum(axis=1).values)
        cpdf.loc[cpdf.index[0],'calm']= 1-pdf['total'].sum()
    
        writer = pd.ExcelWriter(csv_filename, engine='xlsxwriter')  
        pdf.to_excel(writer,sheet_name='wind speed pdf ')
        cpdf.to_excel(writer,sheet_name='wind speed cpdf ')
        spdExceed.to_excel(writer,sheet_name='probablity of exceedance')
        writer.save()             # Close the Pandas Excel writer and output the Excel file.            
    plt.show() # after show() is called a new figure is created: so you CANNOT use plt.show() before plt.savefig unless using fig = plt.gca before that