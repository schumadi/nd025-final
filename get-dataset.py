# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:51:17 2022

@author: Dirk Schumacher
"""
import os
import zipfile
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup


# KL_Tageswerte_Beschreibung_Stationen.txt contains information about all available stations
df = pd.read_fwf('./data/KL_Tageswerte_Beschreibung_Stationen.txt', encoding='ISO-8859–1',
                 skiprows=2,
                 names=['Stations_id', 'von_datum', 'bis_datum',
                        'Stationshoehe', 'geoBreite', 'geoLaenge',
                        'Stationsname', 'Bundesland'])


def rad2deg(radians):
    """Converts radians to degree

    Arguments:
        radians -- angle in radians

    Returns:
        angle in degree
    """
    degrees = radians * 180 / np.pi
    return degrees

def deg2rad(degrees):
    """Convertrs degree to radians

    Arguments:
        degrees -- angle in degree

    Returns:
        angle in radians
    """
    radians = degrees * np.pi / 180
    return radians

def get_distance_between_points(latitude1, longitude1, latitude2, longitude2, unit = 'km'):
    """Computed the distance between to locations

    Arguments:
        latitude1 -- latitude of point 1
        longitude1 -- longitude of point 1
        latitude2 -- latitude of point 2
        longitude2 -- longitude of point 2

    Keyword Arguments:
        unit -- the unit 'miles' or 'km' of the result (default: {'miles'})

    Returns:
        the distance between point 1 and point 2 in miles or km
    """
    theta = longitude1 - longitude2

    distance = 60 * 1.1515 * rad2deg(
        np.arccos(
            (np.sin(deg2rad(latitude1)) * np.sin(deg2rad(latitude2))) +
            (np.cos(deg2rad(latitude1)) * np.cos(deg2rad(latitude2)) * np.cos(deg2rad(theta)))
        )
    )

    if unit == 'miles':
        return round(distance, 2)
    if unit == 'kilometers':
        return round(distance * 1.609344, 2)


def get_url_paths(url, ext=''):
    """Gets the url of all files with extension 'ext' at url 'url'

    Arguments:
        url -- url where to look for files

    Keyword Arguments:
        ext -- extension of the files to look for (default: {''})

    Returns:
        List of url
    """
    response = requests.get(url)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    return parent


# Compute the crossproduct of the stations with themself
left = df[['Stations_id', 'Stationsname', 'geoBreite', 'geoLaenge']]
right = df[['Stations_id', 'Stationsname', 'geoBreite', 'geoLaenge']]
cart = left.merge(right, how='cross')

# Compute the distance between all stations
cart['Distance'] = get_distance_between_points(cart['geoBreite_x'], cart['geoLaenge_x'],
                                               cart['geoBreite_y'], cart['geoLaenge_y'],
                                               unit='kilometers')

# Get the ids of the stations within 25 miles around Berlin-Tegel
ids = cart[(cart['Stationsname_x'] == 'Berlin-Tegel') & (cart['Distance'] <= 25)][['Stations_id_y']]
ids['Stations_id_y'] = ids['Stations_id_y'].astype(str).str.zfill(5)

# See what files are available
URL = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/'
EXT = 'zip'
result = get_url_paths(URL, EXT)
found_ids = [x.split('_')[4] for x in result]
files = pd.DataFrame([result, found_ids]).T
files.columns=['filename', 'id']

# The files of stations within the selected range are to be downloaded
files['Download'] = files['id'].isin(ids['Stations_id_y'])
files['localname'] = files.filename.str[97:]
to_be_downloaded = files.loc[files['Download'] == True]

# Download the files
for row in to_be_downloaded.itertuples(index=False):
    URL = row.filename
    res = requests.get(URL)
    open('./data/' + row.localname, "wb").write(res.content)


# Unzip all zipfiles in the data directory
PATH = './data'
EXT = ".zip"

for item in os.listdir(PATH):
    if item.endswith(EXT):
        file_name = PATH + '/' + item
        with zipfile.ZipFile(file_name) as zip_ref:
            zip_ref.extractall(PATH)

# Cleanup: delete all files except the zip files and the files for the dataset
for item in os.listdir(PATH):
    if ( not item.startswith('produkt')) and (not item.startswith('tageswerte')):
        file_name = PATH + '/' + item
        os.remove(file_name)
