# Sentiment Analysis of COVID-19 Tweets
Group 11 - ML 7641 (Fall 2022)
Team Members:
1. Harsha Vaddi - hvaddi3
2. Manvith M Reddy - mreddy43
3. Sagar Nandkumar Badlani -  sbadlani6
4. Shanmukh Karra - skarra33
5. Sai Sri Harsha Pinninti - spinninti6

# Final Report

## Links 
* [Presentation Video](https://gtvault.sharepoint.com/:v:/s/ML7641Group/EbxlMAj5A85InPjVNTXYOxsB57fuOYIhNaEYB0DQmTki7Q?e=aSr94n) (Proposal)
* [Presentation (PPT)](https://gtvault.sharepoint.com/:p:/s/ML7641Group/EUCvL7UlyMdMrA_VuXuei-YBbCf5Tpr_toKxK279LwYtnw?e=h4C4Md) (Proposal)
* [Gantt Chart](https://gtvault.sharepoint.com/:x:/s/ML7641Group/EbdIFInSHgtCnq1xi3FxrikBSQiLFQ_IHpAhqi9xQHOyHw?e=84Ur5Y) (Proposal)
* [Final Contribution Table](https://gtvault.sharepoint.com/:x:/s/ML7641Group/EchOkDl_VaZHjKQVxjNw9wYBqydS3j0OZ3XNJyqozGo6sQ?e=aWZGDQ) (Updated)

## Introduction/Background
Social media is playing an increasingly great role in individuals’ lives and in connecting people to the rest of the world. It is becoming impossible for people to stay on top of the world’s happenings without the help of social media [1].

With the onset of 2020, came the COVID-19 pandemic. Over a series of multiple waves, it prompted governments to frame new policies like social distancing and pushed most forms of human contact to the online world. In this project, we aim to analyze people’s sentiments towards COVID-19 over various waves and build a classifier to predict the sentiment of COVID related tweets. 

Kaur et al. [1] propose a Hybrid Heterogeneous SVM algorithm for COVID-19 tweet sentiment analysis and evaluate its performance against SVM and RNN. Muhammad et al. [2] have compared the performance of different machine learning and deep learning algorithms including SVM, CNN, LSTM, and KNN.

## Problem Definition
Using the wealth of data available to us online, we seek to identify the shift in people’s emotions towards COVID-19 by tweet sentiment analysis over multiple waves. This allows for a better understanding and handling of the pandemic for any future waves and serves as a base on how people would react during the times of epidemic, helping governments to put necessary safeguards in place.

Using Clustering and Topic Modeling [2], we also wish to find the prevalent topics that were being discussed with respect to COVID-19 and analyze the sentiment around those topics. We plan on studying how these sentiments changed during various waves of the pandemic. 

We will use supervised machine learning models to predict the sentiment of a tweet. We will build these models by using sentiment scores that we will compute using various models (like TextBlob, Vader, roBERTa etc.) and compare the performance of the different supervised models.

## Data Collection

### Dataset
Rabindra Lamsal’s [3] dataset comprises 2 Billion plus tweets along with a sentiment score for each tweet. For this project, we will hydrate a subset of the tweets from each wave and use it for our model ([Dataset Link](https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset)). The sentiment scores in the dataset were calculated using uncleaned tweets. However, we felt that we could do better by removing the stopwords and utilizing cleaned data for our training to build more accurate models.

## Methods

### Data Preparation
The dataset in the link consists of unhydrated tweets, which means that they only contain the tweet ID related to COVID-19 tweets. Once the tweet data is fetched using the API, it is returned in the form of a nested dictionary with the sublevels consisting of the retweets and comments to that particular tweet. Most of this data is not useful to our study and we have only considered the actual tweet text itself for the analysis of the sentiment score.

The data was sourced using the "twarc" module which allows us to retrieve data by referencing the tweet id through twitter's API. Twitter's API times out the python script after a certain number of requests or when the rate of requests to the server exceeds a certain amount. We tried solving this by exploring other modules such as "tweepy" and "twint". tweepy ran into the same issue as twarc as both utilize twitter's API for their data, twint on the other hand is deprecated after twitter's update to API version 2.0. We finally resolved this issue by running the program using twarc on multiple systems using different API keys.

The amount of tweets in the dataset was too large for us to handle on local machines. The unhydrated tweets for one year amounts to approximately 6 Gigabytes of Data. The data was further split into approximately 800 text files for each month in the year.

To process this data, we first identified the relevant months based on the waves of the COVID-19 pandemic and then hydrated a representative sample of the tweets from each month and stored it in "jsonl" format for easier sharing. The tweet data at this point comprised of various languages, so tweets were filtered by English to maintain homogeneity. Some of the tweets returned contained only either blank spaces or empty strings, we dropped these rows as they constituted a very small percentage of our dataset (0.1%).

To understand the sentiment of our tweets, we used three methods. Each of the methods assigned sentiment scores, which formed the truth labels for our dataset. For modeling, we created a subset of our data consisting of equal proportions of positive, negative and neutral tweets.

#### TextBlob
The words in the tweets were converted to lowercase, the punctuation and stop words were removed, and the text was then lemmatized using spaCy. The tweets with cleaned text were then then passed to [TextBlob](https://textblob.readthedocs.io/en/dev/index.html) which is a lexicon based sentiment analyzer that can be used for sentiment analysis. 

#### VADER (Valence Aware Dictionary for Sentiment Reasoning)
[VADER](https://github.com/cjhutto/vaderSentiment) is a model used for text sentiment analysis that is sensitive to both polarity (positive/negative) and intensity (strength) of emotion. VADER takes text as input and returns a dictionary with scores in four categories (negative, neutral, positive, and compound). VADER is known for being specifically attuned to sentiments expressed in social media. The tweets were passed to VADER without cleaning/lemmatization, as the VADER documents recommend using un-cleaned data.

#### Twitter-roBERTa-base for Sentiment Analysis
We ran [cardiffnlp/twitter-roberta-base-sentiment-latest](cardiffnlp/twitter-roberta-base-sentiment-latest), which is a roBERTa-base model trained on ~124M tweets from January 2018 to December 2021 finetuned for sentiment analysis. It returns scores in three categories (negative, neutral, positive), and also recommends using un-cleaned data. Therefore, we passed tweets to the model without cleaning/lemmatization.

<!-- <p align="center"><figcaption>Fig.1 - Distribution of Labels in the Sample Dataset</figcaption></p> -->
<!-- ![alt text](https://user-images.githubusercontent.com/112896256/201557655-4bacd6ef-4a40-4445-8838-40d2f81f8029.png)[alt] -->

### Data Processing
After the data was preprocessed, we converted the tweets into vectors to feed into our models. We applied the Bag of Words representation using the CountVectorizer from Scikit Learn. We limited our features to 500 as we didn't find a sharp increase in accuracy with greater number of features and we ran into RAM limitations when training models with more than 500 features. We then standardized features by subtracting the mean and scaling to unit variance using StandardScaler from Scikit Learn. Standardization of a dataset is a common preprocessing technique for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data. The data was divided into an 80:20 split for training and testing respectively.

### Unsupervised Learning
We used the following unsupervised learning algorithms:
* Principal Component Analysis (PCA)
* Latent Semantic Analysis (LSA)
* Latent Dirichlet Allocation (LDA)

#### Principal Component Analysis (PCA)
To reduce the number of features in our model further, we used the unsupervised learning technique Principal Component Analysis (PCA). PCA uses the Singular Value Decomposition of the data to project it to a lower dimensional space. Instead of manually setting the number of components, we set the variance of the input that is supposed to be explained by the generated components to 95%. PCA returned 419 features which we then used to train our models.

#### Topic Modeling
Topic Modeling helps in automatically organizing, understanding, searching, and summarizing large sets of documents. Topic modeling is an unsupervised learning technique that can be used to find word patterns in a set of documents. It clusters the word groupings and related expressions that best represent a cluster. Hence, Topic Modeling can be used to discover abstract topics in a collection of documents.

##### Latent Semantic Analysis (LSA)
Latent Semantic Analysis is one of the foundational techniques used in Topic Modeling. The core idea is to decompose a matrix of documents and terms into two separate matrices - 
* A document-topic matrix
* A topic-term matrix

Therefore, the learning of LSA for latent topics includes matrix decomposition on the document-term matrix using Singular Value Decomposition (SVD).

Steps for Latent Semantic Analysis:
* Convert raw text into a document-term matrix: Before deriving topics from documents, the text has to be converted into a document-term matrix. We do this using the Bag of Words approach from the gensim python library. 
* Implement Truncated Singular Value Decomposition: We use the gensim lsimodel to implement fast truncated SVD (Singular Value Decomposition). This operation decomposes the document-term matrix A. Mathematically, this can be stated as:
```math
A_{n X m} = U_{n X r}S_{r X r}V^T_{m X r}
```
where U represents the document-topic matrix. Essentially, its values show the strength of association between each document and its derived topics. The matrix has n x r dimensions, with n representing the number of documents and r representing the number of topics. S represents a diagonal matrix that evaluates the strength of each topic in the collection of documents. The variable V represents the word-topic matrix. Its values show the strength of association between each word and the derived topics.
* Encode the words/documents with the derived topics: We use the information obtained from SVD to decide what each derived topic represents and determine which documents belong to which topic.

##### Latent Dirichlet Allocation (LDA)
Latent Dirichlet Allocation (LDA) is used as a Topic Modeling technique. It uses Dirichlet distribution to find topics for each document model and words for each topic model. The LDA makes two key assumptions:
* Documents are a mixture of topics
* Topics are a mixture of tokens (or words)
The aim behind LDA is to find topics that the document belongs to, on the basis of words contained in it. It assumes that documents with similar topics will use a similar group of words. This enables the documents to map the probability distribution over latent topics where the topics are probability distributions.

### Supervised Learning
We trained our data on several machine learning algorithms including :
* Multinomial Naive Bayes
* Decision Tree
* Random Forest
* Neural Network (MLP)
* SVM

## Results and Discussion

### Unsupervised Learning
To evaluate the topic coherence of the various topics obtained from the different topic modeling algorithms, we make use of the:
* UMass Coherence Score: It calculates how often two words, w<sub>i</sub> and w<sub>j</sub> appear together in the corpus and it is defined as 
```math 
C_{UMass}(w_{i}, w_{j}) = \log \frac{D(w_{i}, w_{j}) + 1}{D(w_{i})},
``` 
where D(w<sub>i</sub>,w<sub>j</sub>) indicates how many times words w<sub>i</sub> and w<sub>j</sub> appear together in documents, and D(w<sub>i</sub>) is how many times word w<sub>i</sub> appeared alone. We calculate the global coherence of the topic as the average pairwise coherence scores on the top N words which describe the topic. For the gensim implementation of the UMass Coherence Score, a more negative value indicates a better topic coherence.
* CV Coherence Score: It creates content vectors of words using their co-occurrences and, after that, calculates the score using normalized pointwise mutual information (NPMI) and the cosine similarity. A larger value indicates better topic coherence.
* UCI Coherence Score: This coherence score is based on sliding windows and the pointwise mutual information of all word pairs using top N words by occurrence. Instead of calculating how often two words appear in the document, we calculate the word co-occurrence using a sliding window. It means that if our sliding window has a size of 10, for one particular word w<sub>i</sub>, we observe only 10 words before and after the word w<sub>i</sub>. The UCI coherence between words w<sub>i</sub> and w<sub>j</sub> is defined as 
```math
C_{UCI}(w_{i}, w_{j}) = \log \frac{P(w_{i}, w_{j}) + 1}{P(w_{i}) \cdot P(w_{j})} 
```
where P(w) is probability of seeing word w in the sliding window and P(w<sub>i</sub>, w<sub>j</sub>) is probability of seeing words w<sub>i</sub> and w<sub>j</sub> together in the sliding window.

In order to analyze the shift in people's emotions towards COVID-19 over multiple waves, we implement the topic models independently for the first and the second COVID-19 waves. The topic modeling results for the first wave are summarized in the table below. LDA_BERT corresponds to the LDA model that has been run on data cleaned and labled using roBERTa while LDA_VADER corresponds to the LDA model that has been implemented on data processed using the VADER model. Similarly, LSA_BERT and LSA_VADER correspond to LSA models implemented on data obtained after being processed by roBERTa and VADER models respectively. The number of topics or clusters for the LDA models were fixed at 3 based on domain heuristics. It is assumed that these topics will capture tweets representing the positive, negative, and neutral sentiments of the people. Thus, the tweets data has been modeled into 3 topics/clusters. However, LSA being a simpler model may not capture the non-linear dependencies in the data and model the 3 topics appropriately. Thus, for LSA, we run the topic model for the number of topics/clusters ranging from 2 to 

|Model	  |No. of Topics	 | UMass Coherence Score |	CV Coherence Score |	UCI Coherence Score |
|LDA_BERT	| 3	             | -3.520704581	         |  0.2928482936	     |    0.04519712077     |
|LDA_VADER|	3	             |  -3.610224525	       |  0.2706889428	     |    -0.07458269286    |
|LSA_BERT	| 7              |	-4.128453703	       |  0.346167002	       |    -0.9805364877     |
|LSA_VADER|	10             |	-4.454721729	       |  0.3036853252	     |    -0.9353867443     |

####

### Supervised Learning
We also used several different metrics to compare the performance of the models like :
* Accuracy
* Precision
* Recall
* F1-score

### Preliminary Models
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

### Final Models

We ran an updated model with data from roBERTa as Textblob didnt give accurate sentiment scores and labels to our data. The results below include those from an initial run with default parameters for each of the models and a final model with optimized hyperparameters.

#### MultinomialNB
Accuracy =  0.641

|         |precision |   recall  |f1-score  |   support|
|----------|-----------|----------|----------|---------|
|Negative   |    0.63   |  0.68    |   0.66   |  52823  |
|Positive   |    0.68   |  0.66    |   0.67   |  63446  |
|Neutral    |  0.50     |  0.44    |   0.47   |  15871  |


#### Decision Tree
For the optimization of parameters we first trained a tree with max depth and then pruned the tree with cost complexity pruning to find an effective alpha to produce significant results while minimizing training time.

<p align="center">
  <img src="https://user-images.githubusercontent.com/112896256/205879040-b9602327-1f7e-474b-a5a0-cad6d93d3150.png"/></p>
<h4 align="center">Fig.1 - Distribution of Labels in the Sample Dataset</h4>


##### Intial

Accuracy =  0.545

|         |precision |   recall  |f1-score  |   support|
|----------|-----------|----------|----------|---------|
|Negative   |    0.61   |  0.32    |    0.42   |  52823  |
|Positive   |    0.53   |  0.85     |   0.65   |  63446  |
|Neutral    |  0.53    |  0.05    |   0.09   |  15871  |
 
           
##### Final

Accuracy =  0.601
Classification Report
|         |precision |   recall  |f1-score  |   support|
|----------|-----------|----------|----------|---------|
|Negative   |    0.60   |  0.56    |    0.58   |  52823  |
|Positive   |    0.60   |  0.73     |  0.66   |  63446  |
|Neutral    |  0.62   |  0.22    |   0.33   |  15871  |

           
           
#### Random Forest
The inital random forest was built with a max depth of 10 to check performance. It did not appear to capture the complex patterns in the data so we removed the limit on the depth to allow the decision trees to capture more patterns.

|Hyperparameters| Initial | Final|
|---------------|---------------|---------------|
|max_depth      | 'None' | 10|
|max_features   | "all" | "sqrt"|

##### Initial

Accuracy =  0.589

|         |precision |   recall  |f1-score  |   support|
|----------|-----------|----------|----------|---------|
|Negative   |     0.69    | 0.41     |    0.51   |  52823  |
|Positive   |   0.56    |  0.89     |  0.68   |  63446  |
|Neutral    |  0.95   |  0.02    |  0.03    |  15871  |

##### Final Tuned

Accuracy =  0.750

|         |precision |   recall  |f1-score  |   support|
|----------|-----------|----------|----------|---------|
|Negative   |     0.74    | 0.77     |    0.76   |  52823  |
|Positive   |   0.75    |  0.81     |  0.78   |  63446  |
|Neutral    |  0.82   |  0.43    |  0.56    |  15871  |

           
#### MLP Classifer
For the MLP the inital model was built with 100 layers and second model was built with 10 layers. We did not find significant changes in performance between the models.

##### Initial
Accuracy =  0.666

|         |precision |   recall  |f1-score  |   support|
|----------|-----------|----------|----------|---------|
|Negative   |     0.68    | 0.66    |    0.67   |  52823  |
|Positive   |   0.66    |  0.77     |  0.71   |  63446  |
|Neutral    |  0.68   |  0.29    |  0.41   |  15871  |

                
##### Final

Accuracy =  0.667

|         |precision |   recall  |f1-score  |   support|
|----------|-----------|----------|----------|---------|
|Negative   |     0.68    | 0.65    |    0.67   |  52823  |
|Positive   |   0.66    |  0.78     |  0.71   |  63446  |
|Neutral    |  0.68   |  0.29    |  0.41   |  15871  |


## Conclusion
In the study so far, we have processed about 6 months of data representing the tweets during multiple waves of COVID-19. We have experimented with different Language Models to obtain sentiment scores for the twitter data including TextBlob, Vader and roBERTa. On deeper inspection of the results we found that roBERTa provided the best results for sentiment classification . We have utilized PCA for feature selection and have trained several supervised models to compare their performance in classifying the sentiment of these tweets. The models were compared using four different metrics. We found that Random Forest performs better than the rest of the models with only the Neural Network (MLP) showing a comparable performance based on the metrics that we have tested. We believe it performed best because of its nature as an ensemble model which combines multiple decision trees to reduce the generalization error of the prediction. Naive bayes performed the worst relatively and the neural network will probably perform better when more data is fed into the model.

## References
[1] Kaur, H., Ahsaan, S.U., Alankar, B. et al. “A Proposed Sentiment Analysis Deep Learning Algorithm for Analyzing COVID-19 Tweets.” Inf Syst Front 23, 1417–1429 (2021). https://doi.org/10.1007/s10796-021-10135-7. 

[2] Mujahid, Muhammad, Ernesto Lee, Furqan Rustam, Patrick Bernard Washington, Saleem Ullah, Aijaz Ahmad Reshi, and Imran Ashraf. "Sentiment analysis and topic modeling on tweets about online education during COVID-19." Applied Sciences 11, no. 18 (2021): 8438.

[3] Rabindra Lamsal, March 13, 2020, "Coronavirus (COVID-19) Tweets Dataset", IEEE Dataport, doi: https://dx.doi.org/10.21227/781w-ef42.
