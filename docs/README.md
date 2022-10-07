# Udacity Data Scientist Capstone Project
## Project Definition
### Project Overview
When talking to older people, they often say that in their youth seasons were different. A summer was a "real" summer, a winter was a "real" winter.  
In this project I'd like to investigate:
* Is a machine learning algorithm capable of predicting the season if it is given the weather condition of a day?
* Can it be shown that seasons in the past were different?

[DWD](https://www.dwd.de/EN/Home/home_node.html)'s dataset "DWD Climate Data Center (CDC): Historical daily station observations (temperature, pressure, precipitation, sunshine duration, etc.) for Germany, version v21.3, 2021." will be used to answer those questions. Observations from the Berlin (Germany) area will be used. 

### Problem Statement
#### Machine Learning
In the machine learning part, the task is to find a classifier that is able to predict seasons.
* How good is a classifier at predicting the season (spring, summer, autumn, winter) an observation has been made in?
* Does the quality of the predictions differ when comparing the classifications of observations in the past to current ones?

#### Statistics
In the statistics part, these hypotheses are to be tested. TMK is the daily mean of temperature. TXK is the daily maximum of temperature at 2m height.
* $H_{0-TMK}$: The mean of the winter TMK distribution in the interval 1940 - 1970 is the same as in the interval 2000 - today.  
* $H_{a-TMK}$: The mean of the winter TMK distribution in the interval 1940 - 1970 is different from the on in the interval 2000 - today.


* $H_{0-TXK}$: The mean of the winter TMX distribution in the interval 1940 - 1970 is the same as in the interval 2000 - today.  
* $H_{a-TXK}$: The mean of the winter TMX distribution in the interval 1940 - 1970 is different from the on in the interval 2000 - today.


In addition to comparing the means of the two samples, it will be tested whether there is a trend in the observations of the coldest days in winter. 
* $H_0$: There is no trend in the "It was at least that cold in December" values.  
* $H_a$: There is some trend in the "It was at least that cold in December" values.

### Metrics

https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc


For example, explain why you want to use the accuracy score and/or F-score to measure your model performance in a classification problem,

## Analysis
### Data Exploration
![Basis statistics](./describe-dataset.png)

In the table the SHK_TAG maximum value stand out. That's quite unusual for the Berlin area. 
`clean.query('SHK_TAG == SHK_TAG.max()')` will give us the date this unusual snow depth was measured on. It is 1979-02-18. The winter 1978/1979 was a really [harsh winter in Germany](https://www.vintag.es/2021/01/1978-germany-blizzard.html).

>This blizzard was just the beginning of the winter that crippled everything in Germany, for another round of snow and ice of similar proportions fell later on February 18/19, 1979.


![Correlation matrix](./correlation.png)

We see some quite obvious correlations:
* There is a positive high correlation between min, max and average temparatures.
* There is a negative correlation sunshine duration (SDK), mean cloud cover (NM) and mean relative humidity (UPM).
### Data Visualization
A pairplot of the data might give a how good the chances are to train a classifier to predict seasons.

![Pairplot](./pairplot.png)

In many cases the seasons are separated quite clearly. Therefore a classifier should be able to predict a season. 
## Methodology
### Data Preprocessing
The DWD dataset contains these attributes.
| Column | Meaning | Unit |
| --- | --- | --- |
| STATIONS_ID | station id ||
| MESS_DATUM | date | yyyymmdd |
| QN_3 | quality level of next columns | coding see paragraph "Quality information" |
| FX | daily maximum of wind gust | m/s |
| FM | daily mean of wind speed | m/s |
| QN_4 | quality level of next columns | coding see paragraph "Quality information" |
| RSK | daily precipitation height | mm |
| RSKF | precipitation form ||
||no precipitation (conventional or automatic measurement), relates to WMO code 10 | 0 |
|| only rain (before 1979) | 1 |
||unknown form of recorded precipitation | 4 |
|| only rain; only liquid precipitation at automatic stations, relates to WMO code 11 | 6 |
|| only snow; only solid precipitation at automatic stations, relates to WMO code 12 | 7 |
|| rain and snow (and/or "Schneeregen"); liquid and solid precipitation at automatic stations, relates to WMO code 13 | 8 |
|| error or missing value or no automatic determination of precipitation form, relates to WMO code 15 | 9 |
| SDK | daily sunshine duration | h |
| SHK_TAG | daily snow depth | cm |
| NM | daily mean of cloud cover | 1/8 |
| VPM | daily mean of vapor pressure | hPa |
| PM | daily mean of pressure | hPa |
| TMK | daily mean of temperature | °C |
| UPM | daily mean of relative humidity | % |
| TXK | daily maximum of temperature at 2m height | °C |
| TNK | daily minimum of temperature at 2m height | °C |
| TGK | daily minimum of air temperature at 5cm above ground | °C |
| eor | End of data record |	

*QN_3*, *QN_4* and *eor* are attributes that do not contain information relevant for this project. They can be dropped. *RSKF* is converted to categorical. Unfortunately, more than 70% of the values in *FX* and *FM* are missing. If so many values are missing, it's a good decision to drop these columns. That should not affect the results of the project as there are many attributes remaining that probably are more related to seasons. There are still missing values, but the classifier that has been used for the machine learning part can cope with this. Therefore there is no need to impute values or drop those rows.  
(Please see this jupyter [notebook](./../data_wrangling.ipynb) for details.)


### Implementation
Data split 
	

The process of improving upon the algorithms and techniques used is clearly documented. Both the initial and final solutions are reported, along with intermediate solutions, if necessary.

## Results
### Model Evaluation and Validation

Data split 

![Parameter Tuning](./optimization.png)
https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.lightgbm.train.html

* "lambda_l1": 4.37818456429782
* "lambda_l2": 1.5400901346778327e-05
* "num_leaves": 31
* "feature_fraction": 0.8
* "bagging_fraction": 0.5766192558526406
* "bagging_freq": 7
* "min_child_samples": 20

![ROC Testset](./roc-test.png)

![ROC Current](./roc-current.png)

![Confusion Matrix](./confusion.png)

![Histogram](./means.png)


If a model is used, the following should hold: The final model’s qualities — such as parameters — are evaluated in detail.

Some type of analysis is used to validate the robustness of the model’s solution. For example, you can use cross-validation to find the best parameters.

Show and compare the results using different models, parameters, or techniques in tabular forms or charts.

Alternatively, a student may choose to answer questions with data visualizations or other means that don't involve machine learning if a different approach best helps them address their question(s) of interest.

### Justification
	

The final results are discussed in detail. Explain the exploration as to why some techniques worked better than others, or how improvements were made are documented.

## Conclusion
### Reflection
	
timeseries appropriate statistics

Student adequately summarizes the end-to-end problem solution and discusses one or two particular aspects of the project they found interesting or difficult.

### Improvement

hyperparameter tuning	
other regions
summer change

Discussion is made as to how at least one aspect of the implementation could be improved. Potential solutions resulting from these improvements are considered and compared/contrasted to the current solution.


### Github Repository
	
https://github.com/schumadi/nd025-final is the repository for this project.
