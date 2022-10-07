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
        unit -- the unit 'miles' or 'km' of the result (default: {'km'})

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


def get_stations_ids(station, radius, unit):
    """KL_Tageswerte_Beschreibung_Stationen.txt contains information about all available stations.
       This function computes the IDs of all stations within a distance of radius.

    Arguments:
        station -- Name of the station
        radius -- Radius of the area around station to look for other stations
        unit -- 'km' or 'miles'

    Returns:
        IDs of the found stations
    """

    # KL_Tageswerte_Beschreibung_Stationen.txt contains information about all available stations
    available_stations = pd.read_fwf(
                                        './KL_Tageswerte_Beschreibung_Stationen.txt',
                                        encoding='ISO-8859–1',
                                        skiprows=2,
                                        names=['Stations_id', 'von_datum', 'bis_datum',
                                                'Stationshoehe', 'geoBreite', 'geoLaenge',
                                                'Stationsname', 'Bundesland'
                                                ]
                                    )

    # Compute the crossproduct of the stations with themself
    left = available_stations[['Stations_id', 'Stationsname', 'geoBreite', 'geoLaenge']]
    right = available_stations[['Stations_id', 'Stationsname', 'geoBreite', 'geoLaenge']]
    cart = left.merge(right, how='cross')

    # Compute the distance between all stations
    cart['Distance'] = get_distance_between_points(
                                                    cart['geoBreite_x'], cart['geoLaenge_x'],
                                                    cart['geoBreite_y'], cart['geoLaenge_y'],
                                                    unit=unit
                                                  )

    # Get the ids of the stations within 25 miles around Berlin-Tegel
    ids = cart[(cart['Stationsname_x'] == station) & (cart['Distance'] <= radius)][['Stations_id_y']]
    ids['Stations_id_y'] = ids['Stations_id_y'].astype(str).str.zfill(5)

    return ids


def get_dwd_data (station, radius, unit):
    """Get the DWD weather data of all stations within a distance of radius around station

    Arguments:
        station -- Name of the station
        radius -- Radius of the area around station to look for other stations
        unit -- 'km' or 'miles'
    """
    # See what files are available on the DWD website
    URL = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/'
    EXT = 'zip'
    result = get_url_paths(URL, EXT)
    found_ids = [x.split('_')[4] for x in result]
    files = pd.DataFrame([result, found_ids]).T
    files.columns=['filename', 'id']

    # Get the IDs of the stations within a radius distance around station
    ids = get_stations_ids(station, radius, unit)

    # The files of stations within the selected range are to be downloaded
    files['Download'] = files['id'].isin(ids['Stations_id_y'])
    files['localname'] = files.filename.str[97:]
    to_be_downloaded = files.loc[bool(files['Download'])]

    # Download the files
    for row in to_be_downloaded.itertuples(index=False):
        url = row.filename
        res = requests.get(url)
        open('./data/' + row.localname, "wb").write(res.content)

    # Unzip all zipfiles in the data directory
    PATH = './data'
    EXT = ".zip"
    for item in os.listdir(PATH):
        if item.endswith(EXT):
            file_name = PATH + '/' + item
            with zipfile.ZipFile(file_name) as zip_ref:
                zip_ref.extractall(PATH)

    # Cleanup: delete all files except files of the dataset
    for item in os.listdir(PATH):
        if (not item.startswith('produkt')):
            file_name = PATH + '/' + item
            os.remove(file_name)


if __name__ == "__main__":
    # Get the data of station within a distance of 25 km around Berlin-Tegel
    get_dwd_data('Berlin-Tegel', 25, 'km')
