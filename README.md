# Sentiment Analysis of COVID-19 Tweets
Group 11 - ML 7641 (Fall 2022)
Team Members:
1. Harsha Vaddi - hvaddi3
2. Manvith M Reddy - mreddy43
3. Sagar Nandkumar Badlani -  sbadlani6
4. Shanmukh Karra - skarra33
5. Sai Sri Harsha Pinninti - spinninti6

# Midterm Report

## Links 
* [Presentation Video](https://gtvault.sharepoint.com/:v:/s/ML7641Group/EbxlMAj5A85InPjVNTXYOxsB57fuOYIhNaEYB0DQmTki7Q?e=aSr94n)(Proposal)
* [Presentation (PPT)](https://gtvault.sharepoint.com/:p:/s/ML7641Group/EUCvL7UlyMdMrA_VuXuei-YBbCf5Tpr_toKxK279LwYtnw?e=h4C4Md)(Proposal)
* [Gantt Chart](https://gtvault.sharepoint.com/:x:/s/ML7641Group/EbdIFInSHgtCnq1xi3FxrikBSQiLFQ_IHpAhqi9xQHOyHw?e=84Ur5Y)(Proposal)
* [Proposal Contribution Table](https://gtvault.sharepoint.com/:x:/s/ML7641Group/EchOkDl_VaZHjKQVxjNw9wYBqydS3j0OZ3XNJyqozGo6sQ?e=aWZGDQ)(updated)

## Introduction/Background
Social media is playing an increasingly great role in individuals’ lives and in connecting people to the rest of the world. It is becoming impossible for people to stay on top of the world’s happenings without the help of social media [1].

With the onset of 2020, came the COVID-19 pandemic. Over a series of multiple waves, it prompted governments to frame new policies like social distancing and pushed most forms of human contact to the online world. In this project, we aim to analyze people’s sentiments towards COVID-19 over various waves and build a classifier to predict the sentiment of COVID related tweets. 

Kaur et al. [1] propose a Hybrid Heterogeneous SVM algorithm for COVID-19 tweet sentiment analysis and evaluate its performance against SVM and RNN. Muhammad et al. [2] have compared the performance of different machine learning and deep learning algorithms including SVM, CNN, LSTM, and KNN.

## Problem Definition
Using the wealth of data available to us online, we seek to identify the shift in people’s emotions towards COVID-19 by tweet sentiment analysis over multiple waves. This allows for a better understanding and handling of the pandemic for any future waves and serves as a base on how people would react during the times of epidemic, helping governments to put necessary safeguards in place.

Using Clustering and Topic Modeling [2], we also wish to find the prevalent topics that were being discussed with respect to COVID-19 and analyze the sentiment around those topics. We plan on studying how these sentiments changed during various waves of the pandemic. 

We will use supervised machine learning models to predict the sentiment of a tweet. We will build these models by using sentiment scores that we will compute using various models (like TextBlob, Vader etc.) and compare the performance of the different supervised models.

## Methods

### Dataset
Rabindra Lamsal’s [3] dataset comprises 2 Billion plus tweets along with a sentiment score for each tweet. For this project, we will hydrate a subset of the tweets from each wave and use it for our model ([Dataset Link](https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset)). The sentiment scores in the dataset were calculated using uncleaned tweets. However, we felt that we could do better by removing the stopwords and utilizing cleaned data for our training to build more accurate models.

### Data Preparation
The dataset in the link consists of unhydrated tweets, which means that they only contain the tweet ID related to COVID-19 tweets. Once the tweet data is fetched using the API, it is returned in the form of a nested dictionary with the sublevels consisting of the retweets and comments to that particular tweet. Most of this data is not useful to our study and we have only considered the actual tweet text itself for the analysis of the sentiment score.

The data was sourced using the "twarc" module which allows us to retrieve data by referencing the tweet id through twitter's API. Twitter's API times out the python script after a certain number of requests or when the rate of requests to the server exceeds a certain amount. We tried solving this by exploring other modules such as "tweepy" and "twint". tweepy ran into the same issue as twarc as both utilize twitter's API for their data , twint on the other hand is deprecated after twitter's update to API version 2.0. We finally resolved this issue by running the program using twarc on multiple systems using different API keys.

The amount of tweets in the dataset was too large for us to handle on local machines. The unhydrated tweets for one year amounts to approximately 6 Gigabytes of Data. The data was further split into approximately 800 text files for each month in the year.

To process this data,we first identified the relevant months based on the waves of the COVID-19 pandemic and then hydrated a representative sample of the tweets from each month and stored it in "jsonl" format for easier sharing. The tweet data at this point comprised of various languages, the tweets were filtered by english to maintain homogeneity, all the words were converted to lowercase, punctuation and stop words were removed. Some of the tweets returned contained only either blank spaces or empty strings, we dropped these rows as they constituted a very small percentage of our dataset (0.1%).

To understand the sentiment of our tweets they were then passed to TextBlob which is a lexicon based sentiment analyzer that can be used for sentiment analysis. These values formed the truth labels for our dataset. For modeling, we created a subset of our data consisting of equal proportions of positive, negative and neutral tweets.
<p align="center">
  <img src="https://user-images.githubusercontent.com/112896256/201557655-4bacd6ef-4a40-4445-8838-40d2f81f8029.png"/>
                      <figcaption>Fig.1 - Distribution of Labels in the Sample Dataset</figcaption></p>
<!-- ![alt text](https://user-images.githubusercontent.com/112896256/201557655-4bacd6ef-4a40-4445-8838-40d2f81f8029.png)[alt] -->

### Data Processing
After the data was preprocessed, we converted the tweets into vectors to feed into our models. We applied the Bag of Words representation using the CountVectorizer from Scikit Learn. We limited our features to 500 as we didn't find a sharp increase in accuracy with greater number of features and we ran into RAM limitations when training models with more than 500 features. We then standardized features by subtracting the mean and scaling to unit variance using StandardScaler from Scikit Learn. Standardization of a dataset is a common preprocessing technique for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data. The data was divided into an 80:20 split for training and testing respectively.

#### Unsupervised Learning
To reduce the number of features in our model further, we used the unsupervised learning technique Principal Component Analysis (PCA). PCA uses the Singular Value Decomposition of the data to project it to a lower dimensional space. Instead of manually setting the number of components, we set the variance of the input that is supposed to be explained by the generated components to 95%. PCA returned 419 features which we then used to train our models.

#### Supervised Learning
We trained our data on several machine learning algorithms including :
* Multinomial Naive Bayes
* Decision Tree
* Random Forest
* Neural Network (MLP)
* SVM

## Results and Discussion

We also used several different metrics to compare the performance of the models like :
* Accuracy
* Precision
* Recall
* F1-score

#### Multinomial Naive Bayes

The Multinomial Naive Bayes Model achieved an Accuracy of 0.693.


|              | Precision | Recall | f1-score | Support |
|--------------|-----------|--------|----------|---------|
| negative     | 0.73      | 0.62   | 0.67     | 8088    |
| neutral      | 0.65      | 0.78   | 0.70     | 8042    |
| Positive     | 0.72      | 0.69   | 0.70     | 7870    |


#### Decision Tree

The Decision tree classifier achieved an Accuracy of 0.743.

|              | Precision | Recall | f1-score | Support |
|--------------|-----------|--------|----------|---------|
| negative     | 0.73      | 0.71   | 0.72     | 8088    |
| neutral      | 0.74      | 0.79   | 0.76     | 8042    |
| Positive     | 0.77      | 0.73   | 0.75     | 7870    |


#### Random Forest

The Random Forest classifier achieved an Accuracy of 0.790.

|              | Precision | Recall | f1-score | Support |
|--------------|-----------|--------|----------|---------|
| negative     | 0.84      | 0.70   | 0.77     | 8088    |
| neutral      | 0.71      | 0.91   | 0.80     | 8042    |
| Positive     | 0.86      | 0.76   | 0.80     | 7870    |


#### Neural Network

The Neural Network (MLP) achieved an Accuracy of 0.748.

|              | Precision | Recall | f1-score | Support |
|--------------|-----------|--------|----------|---------|
| negative     | 0.85      | 0.60   | 0.70     | 8088    |
| neutral      | 0.65      | 0.95   | 0.77     | 8042    |
| Positive     | 0.84      | 0.70   | 0.76     | 7870    |

#### SVM

The SVM classifier was not able to train to completion due to the time complexity being (O(n_samples^2 * n_features)). Since this is a language based dataset it is important to retain features to capture the sentiment of the tweet effectively. The training time for SVM cannot be reduced further without sacrificing information.

## Conclusion
In the study so far, we have processed about 6 months of data representing the tweets during multiple waves of COVID-19. We have utilized PCA for feature selection and have trained several supervised models to compare their performance in classifying the sentiment of these tweets. The models were compared using four different metrics. We found that Random Forest performs better than the rest of the models with only the Neural Network (MLP) showing a comparable performance based on the metrics that we have tested. 

## References
[1] Kaur, H., Ahsaan, S.U., Alankar, B. et al. “A Proposed Sentiment Analysis Deep Learning Algorithm for Analyzing COVID-19 Tweets.” Inf Syst Front 23, 1417–1429 (2021). https://doi.org/10.1007/s10796-021-10135-7. 

[2] Mujahid, Muhammad, Ernesto Lee, Furqan Rustam, Patrick Bernard Washington, Saleem Ullah, Aijaz Ahmad Reshi, and Imran Ashraf. "Sentiment analysis and topic modeling on tweets about online education during COVID-19." Applied Sciences 11, no. 18 (2021): 8438.

[3] Rabindra Lamsal, March 13, 2020, "Coronavirus (COVID-19) Tweets Dataset", IEEE Dataport, doi: https://dx.doi.org/10.21227/781w-ef42.
