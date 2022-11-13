# Sentiment Analysis of COVID-19 Tweets
Group 11 - ML 7641 (Fall 2022)
Team Members:
1. Harsha Vaddi - hvaddi3
2. Manvith M Reddy - mreddy43
3. Sagar Nandkumar Badlani -  sbadlani6
4. Shanmukh Karra - skarra33
5. Sai Sri Harsha Pinninti - spinninti6

# Project Proposal

## Links
* [Presentation Video](https://gtvault.sharepoint.com/:v:/s/ML7641Group/EbxlMAj5A85InPjVNTXYOxsB57fuOYIhNaEYB0DQmTki7Q?e=aSr94n)
* [Presentation (PPT)](https://gtvault.sharepoint.com/:p:/s/ML7641Group/EUCvL7UlyMdMrA_VuXuei-YBbCf5Tpr_toKxK279LwYtnw?e=h4C4Md)
* [Gantt Chart](https://gtvault.sharepoint.com/:x:/s/ML7641Group/EbdIFInSHgtCnq1xi3FxrikBSQiLFQ_IHpAhqi9xQHOyHw?e=84Ur5Y)
* [Proposal Contribution Table](https://gtvault.sharepoint.com/:x:/s/ML7641Group/EchOkDl_VaZHjKQVxjNw9wYBqydS3j0OZ3XNJyqozGo6sQ?e=aWZGDQ)

## Introduction/Background
Social media is playing an increasingly great role in individuals’ lives and in connecting people to the rest of the world. It is becoming impossible for people to stay on top of the world’s happenings without the help of social media [1].

With the onset of 2020, came the COVID-19 pandemic. Over a series of multiple waves, it prompted governments to frame new policies like social distancing and pushed most forms of human contact to the online world. In this project, we aim to analyze people’s sentiments towards COVID-19 over various waves and build a classifier to predict the sentiment of COVID related tweets. 

Kaur et al. [1] propose a Hybrid Heterogeneous SVM algorithm for COVID-19 tweet sentiment analysis and evaluate its performance against SVM and RNN. Muhammad et al. [2] have compared the performance of different machine learning and deep learning algorithms including SVM, CNN, LSTM, and KNN.

## Problem Definition
Using the wealth of data available to us online, we seek to identify the shift in people’s emotions towards COVID-19 by tweet sentiment analysis over multiple waves. This allows for a better understanding and handling of the pandemic for any future waves and serves as a base on how people would react during the times of epidemic, helping governments to put necessary safeguards in place.

Using Clustering and Topic Modeling [2], we also wish to find the prevalent topics that were being discussed with respect to COVID-19 and analyze the sentiment around those topics. We plan on studying how these sentiments changed during various waves of the pandemic. 

We will use supervised machine learning models to predict the sentiment of a tweet. We will build these models by using sentiment scores given in the dataset as well as sentiment scores that we will compute using various techniques (like lexicon, Vader etc.) and compare the performance of the different approaches.

## Methods

### Dataset
Rabindra Lamsal’s [3] dataset comprises 2 Billion plus tweets along with a sentiment score for each tweet. For this project, we will hydrate a subset of the tweets from each wave and use it for our model ([Dataset Link](https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset)).

### Data Preparation
The dataset in the link consists of unhydrated tweets, which means that they only contain the tweet ID related to COVID-19 tweets. Once the tweet data is fetched using the API, it is returned in the form of a nested dictionary with the sublevels consisting of the retweets and comments to that particular tweet. Most of this data is not useful to our study and we have only considered the actual tweet text itself for the analysis of the sentiment score.

The data was sourced using the twarc module which allows us to retrieve data by referencing the tweet id through twitter's API. Twitter's API times out the python script after a certain number of requests or when the rate of requests to the server exceeds a certain amount. We tried solving this by exploring other modules such as tweepy and twint. tweepy ran into the same issue as twarc as both utilize twitter's API for their data , twint on the other hand is deprecated after twitter's update to API version 2.0. We finally resolved this issue by running the program using twarc on multiple systems using different API keys.
The amount of tweets in the dataset was too large for us to handle on local machines. The unhydrated tweets for one year amounts to approximately 6 Gigabytes of Data. The data was further split into approximately 800 text files for each month in the year.
To process this data,we first identified the relevant months based on the waves of the COVID-19 pandemic and then hydrated a representative sample of the tweets from each month and stored it in "jsonl" format for easier sharing.

The tweet data at this point comprised of various languages, the tweets were filtered by english to maintain homogeneity, all the words were converted to lowercase, punctuation and stop words were removed. To understand the sentiment of our tweets they were then passed to TextBlob which is a lexicon based sentiment analyzer that can be used for sentiment analysis. These values formed the truth labels for our dataset. For modeling we created a subset of our data consisting of equal proportions of positive, negative and neutral tweets.

### Data Analysis
We plan to use a variety of techniques to improve/understand our dataset:
* Time-Series Analysis
* Dimensionality Reduction
* TF-IDF, etc.

### Unsupervised Learning
We plan to use a variety of clustering techniques to identify the salient groups within our dataset. 
* K-Means
* DBSCAN
* Topic Modeling using LDA, etc.

### Supervised Learning
We aim to train various supervised machine learning models that will be able to give the sentiment of a tweet. We can compare these models to see which ones are performing best. 
* SVM
* Random Forest
* Logistic Regression, etc.

## Potential results and Discussion
By fine-tuning our clusters using metrics like Beta-CV, elbow-method, we can see the prevalent clusters/topics and if/how they vary through the various waves of covid. Using various metrics like precision, recall, f1-score, we will be able to compare our machine learning models and see which perform best. Upon further analysis, we will be able to tell what topics invoked generally positive comments on twitter and which topics invoked generally negative sentiments.

## References
[1] Kaur, H., Ahsaan, S.U., Alankar, B. et al. “A Proposed Sentiment Analysis Deep Learning Algorithm for Analyzing COVID-19 Tweets.” Inf Syst Front 23, 1417–1429 (2021). https://doi.org/10.1007/s10796-021-10135-7. 

[2] Mujahid, Muhammad, Ernesto Lee, Furqan Rustam, Patrick Bernard Washington, Saleem Ullah, Aijaz Ahmad Reshi, and Imran Ashraf. "Sentiment analysis and topic modeling on tweets about online education during COVID-19." Applied Sciences 11, no. 18 (2021): 8438.

[3] Rabindra Lamsal, March 13, 2020, "Coronavirus (COVID-19) Tweets Dataset", IEEE Dataport, doi: https://dx.doi.org/10.21227/781w-ef42.
