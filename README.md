# Sentiment Analysis of COVID-19 Tweets

Team Members:
1. Harsha Vaddi - hvaddi3
2. Manvith M Reddy - mreddy43
3. Sagar Nandkumar Badlani -  sbadlani6
4. Shanmukh Karra - skarra33
5. Sai Sri Harsha Pinninti - spinninti6

# ML 7641 (Fall 2022) - Project Group 11
# Project Proposal

## Introduction/Background
Social media is playing an increasingly great role in individuals’ lives and in connecting people to the rest of the world. It is becoming impossible for people to stay on top of the world’s happenings without the help of social media [1].

With the onset of 2020, came the COVID-19 pandemic. Over a series of multiple waves, it prompted governments to frame new policies like social distancing and pushed most forms of human contact to the online world. In this project, we aim to analyze people’s sentiments towards COVID-19 over various waves and build a classifier to predict the sentiment of COVID related tweets. 

Kaur et al. [1] propose a Hybrid Heterogeneous SVM algorithm for COVID-19 tweet sentiment analysis and evaluate its performance against SVM and RNN. Muhammad et al. [3] have compared the performance of different machine learning and deep learning algorithms including SVM, CNN, LSTM, and KNN.

## Problem Definition
Using the wealth of data available to us online, we seek to identify the shift in people’s emotions towards COVID-19 by tweet sentiment analysis over multiple waves. This allows for a better understanding and handling of the pandemic for any future waves and serves as a base on how people would react during the times of epidemic, helping governments to put necessary safeguards in place.

Using Clustering and Topic Modeling [3], we also wish to find the prevalent topics that were being discussed with respect to COVID-19 and analyze the sentiment around those topics. We plan on studying how these sentiments changed during various waves of the pandemic. 

We will use supervised machine learning models to predict the sentiment of a tweet. We will build these models by using sentiment scores given in the dataset as well as sentiment scores that we will compute using various techniques (like lexicon, Vader etc.) and compare the performance of the different approaches.

## Methods

### Dataset
Rabindra Lamsal’s [2] dataset comprises 2 Billion plus tweets along with a sentiment score for each tweet. For this project, we will hydrate a subset of the tweets from each wave and use it for our model ([Dataset Link](https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset)).

### Data Preparation/Analysis
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

[2] Rabindra Lamsal, March 13, 2020, "Coronavirus (COVID-19) Tweets Dataset", IEEE Dataport, doi: https://dx.doi.org/10.21227/781w-ef42.

[3] Mujahid, Muhammad, Ernesto Lee, Furqan Rustam, Patrick Bernard Washington, Saleem Ullah, Aijaz Ahmad Reshi, and Imran Ashraf. "Sentiment analysis and topic modeling on tweets about online education during COVID-19." Applied Sciences 11, no. 18 (2021): 8438.
