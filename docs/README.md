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
We see some quite obvious correlations  
* There is a positive high correlation between min, max and average temparatures.
* There is a negative correlation sunshine duration (SDK), mean cloud cover (NM) and mean relative humidity (UPM).

Features and calculated statistics relevant to the problem have been reported and discussed related to the dataset, and a thorough description of the input space or input data has been made. Abnormalities or characteristics about the data or input that need to be addressed have been identified.

### Data Visualization
	

Build data visualizations to further convey the information associated with your data exploration journey. Ensure that visualizations are appropriate for the data values you are plotting.

## Methodology
### Data Preprocessing
	

All preprocessing steps have been clearly documented. Abnormalities or characteristics about the data or input that needed to be addressed have been corrected. If no data preprocessing is necessary, it has been clearly justified.

### Implementation
	

The process for which metrics, algorithms, and techniques were implemented with the given datasets or input data has been thoroughly documented. Complications that occurred during the coding process are discussed.

### Refinement
	

The process of improving upon the algorithms and techniques used is clearly documented. Both the initial and final solutions are reported, along with intermediate solutions, if necessary.

## Results
### Model Evaluation and Validation
	

If a model is used, the following should hold: The final model’s qualities — such as parameters — are evaluated in detail.

Some type of analysis is used to validate the robustness of the model’s solution. For example, you can use cross-validation to find the best parameters.

Show and compare the results using different models, parameters, or techniques in tabular forms or charts.

Alternatively, a student may choose to answer questions with data visualizations or other means that don't involve machine learning if a different approach best helps them address their question(s) of interest.

### Justification
	

The final results are discussed in detail. Explain the exploration as to why some techniques worked better than others, or how improvements were made are documented.

## Conclusion
### Reflection
	

Student adequately summarizes the end-to-end problem solution and discusses one or two particular aspects of the project they found interesting or difficult.

### Improvement
	

Discussion is made as to how at least one aspect of the implementation could be improved. Potential solutions resulting from these improvements are considered and compared/contrasted to the current solution.


### Github Repository
	
https://github.com/schumadi/nd025-final is the repository for this project.
