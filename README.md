# nd025 Data Science Capstone Project

## Table of Contents

1. [Libraries used](#libraries-used)
1. [Motivation for the project](#motivation-for-the-project)
1. [Files in the repository](#files-in-the-repository)
1. [Summary of the results of the analysis](#summary-of-the-results-of-the-analysis)
1. [Acknowledgments](#acknowledgments)


## Libraries used
* python                    3.8.13
* pandas                    1.4.3
* numpy                     1.23.3
* seaborn                   0.11.2
* matplotlib                3.5.2
* optuna                    2.10.1
* scikit-learn              1.1.2                   

## Motivation for the project
When talking to older people, they often say that in their youth seasons were different. A summer was a "real" summer, a winter was a "real" winter.  
In this project I'd like to investigate:
* Is a machine learning algorithm capable of predicting the season if it is given the weather conditions of a day?
* Can it be shown that seasons in the past were different?

## Files in the repository
* README.md - this file
* get-dataset.py - Python script to load daily station observations from the DWD website
* data_wrangling.ipynd - Jupyter notebook for analysis and cleaning of data
* machine_learning.ipynb - Jupyter notebook containing the machine learning and statistics part of the project


DWD Climate Data Center (CDC): Historical daily station observations (temperature, pressure, precipitation, sunshine duration, etc.) for Germany, version v21.3, 2021
* KL_Tageswerte_Beschreibung_Stationen.txt - List of available stations
* DESCRIPTION_obsgermany_climate_daily_kl_historical_en.pdf - Dataset description
* /data/*.txt - Historical daily station observations (temperature, pressure, precipitation, sunshine duration, etc.)
* /data/cleaned_dataset.parquet - Created by the data_wrangling jupyter notebook 

## Summary of the results of the analysis
The summary of the results can be seen here https://github.com/schumadi/nd025-final/tree/master/docs#readme

## Acknowledgments
* [Udacity](https://www.udacity.com/) offers this excellent nanodegree program.
* [DWD](https://www.dwd.de/EN/Home/home_node.html) provides the dataset "DWD Climate Data Center (CDC): Historical daily station observations (temperature, pressure, precipitation, sunshine duration, etc.) for Germany, version v21.3, 2021."

