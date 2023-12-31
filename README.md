# Sentiment Analysis of COVID-19 Tweets

## Introduction

This project analyzed tweet sentiments over multiple ‘waves’ of COVID-19 using Twitter data from 2020 to 2022. Tweet sentiments were computed using three different pre-trained NLP models (TextBlob, VADER, and roBERTa). 6 different supervised and unsupervised learning techniques were implemented and compared using coherence scores and performance metrics. 

[Presentation Video](https://youtu.be/TV82gs31doo)
  
## Data Collection and Preparation

### Dataset
Rabindra Lamsal’s [3] dataset comprises 2 Billion plus tweets along with a sentiment score for each tweet. For this project, we will hydrate a subset of the tweets from each wave and use it for our model ([Dataset Link](https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset)). The sentiment scores in the dataset were calculated using uncleaned tweets. However, we felt that we could do better by removing the stopwords and utilizing cleaned data for our training to build more accurate models.

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

## Methods

### Exploratory Data Analysis

Before feeding our dataset into machine learning models, it was imperative that we carefully studied and understood the dataset that we collected to maximize the information gained from the features of our dataset. Using exploratory data analysis we were able to investigate the dataset in terms of the 3 sentiment scoring models (TextBlob, Vader, RoBERTa) as well as across the 2 waves of Covid-19.

Our dataset consisted of 660977 tweets related to Covid-19 spanning more than 6 months. We used only English language tweets to aid our models and the tweets had an average length of 11.94 words and 80.4 characters. Across the three sentiment scoring models, we noticed that TextBlob and Vader labeled the tweets more or less evenly across the 3 labels whereas RoBERTa’s labeling was skewed more towards the Neutral/Negative spectrum.

|          | Positive | Neutral | Negative |
|----------|----------|---------|----------|
| TextBlob | 263146   | 251824  | 146007   |
| Vader    | 271747   | 166268  | 222691   |
| roBERTa  | 79961    | 315880  | 264865   |

Upon further investigation of RoBERTa’s labeling, we also noticed that the model labeled a tweet as positive only if the tweet was extremely positive whereas it labeled a tweet as negative with a more even distribution of scoring as evidenced below. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/44746109/206121971-908cbd89-fc67-4588-8d0f-92c7efb50dc8.png"/></p>
<h4 align="center">RoBERTa Positive Score Distribution</h4>

<p align="center">
  <img src="https://user-images.githubusercontent.com/44746109/206121852-3b4536ac-05c7-43fd-aee9-f13bb4964b2e.png"/></p>
<h4 align="center">RoBERTa Negative Score Distribution</h4>

Another aspect of the data that we investigated was the prevalent keywords used in Positive and Negative tweets and how they were similar/different. The word clouds we plotted for both labels helped us answer this question. A lot of the keywords used in the tweets were overlapping for both Positive and Negative word clouds with common words like Masks, Lockdown, Vaccinations occurring in both sets of tweets. However, the two word clouds also showed subtle differences amongst them. The negatively labeled tweets had negatively inclined keywords like Hate, Death, sad, hospital, etc and the positively labeled tweets had keywords like Family, Care, Love, etc. The word clouds can be found below :

<p align="center">
  <img src="https://user-images.githubusercontent.com/44746109/206122196-d9f7f62c-aad1-4f2b-aa1d-b4efb1bac8a1.png"/></p>
<h4 align="center">Positive Word Cloud</h4>

<p align="center">
  <img src="https://user-images.githubusercontent.com/44746109/206122263-e90a7bc4-510c-474c-ab48-1d8f6f49a325.png"/></p>
<h4 align="center">Negative Word Cloud</h4>

Finally, we also wanted to analyze the difference in themes between the 2 waves. As an example, we look at word clouds of negatively labeled tweets from wave 1 and 2 below. We noticed that the conversation surrounding Covid-19 had also changed in between waves. The first wave used keywords like Trump, lockdown, government whereas the second wave saw a decrease in interest in those keywords and saw new keywords like Omicron, vaccinations and variants.

<p align="center">
  <img src="https://user-images.githubusercontent.com/44746109/206122522-7bd56a51-40e7-462c-a2dc-f271c4061191.png"/></p>
<h4 align="center">Wave 1 Word Cloud</h4>

<p align="center">
  <img src="https://user-images.githubusercontent.com/44746109/206122719-3bfc2b04-231b-4991-8e18-fe562562431b.png"/></p>
<h4 align="center">Wave 2 Word Cloud</h4>


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

#### Multinomial Naive Bayes
MultinomialNB implements the naive Bayes algorithm for multinomially distributed data, and is one of the two classic naive Bayes variants used in text classification.
The distribution is parametrized by vectors $\theta_y=(\theta_{y1},....\theta_{yn})$ for each class y, where n is the number of features (in text classification, the size of the vocabulary) and $\theta_{yi}$ is the probability $P(x_i|y)$ of feature i appearing in a sample belonging to class y.

#### Decision Tree
Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.

#### Random Forest
In random forests, each tree in the ensemble is built from a sample drawn with replacement (bootstrap sample) from the training set. Furthermore, when splitting each node during the construction of a tree, the best split is found either from all input features or a random subset of size max_features.
The purpose of these two sources of randomness is to decrease the variance of the forest estimator. Individual decision trees typically exhibit high variance and tend to overfit.

#### Neural Network (MLP)
The MLP or multi layer perceptron is a form of a fully connected artificial neural network. It uses backpropagation for updating its parameters and It is capable of separating points that are not linearly separable by utilizing non-linear activation functions.

#### SVM
It is a supervised learning algorithm that is effective in high dimensional spaces. It aims at defining a hyperplane/hyperplanes separating each class of data points while maximizing the margin between the support vectors. New points are then classified based on which side of the hyperplane they fall on. It is relatively memory efficient but has an expensive time complexity of $O(n^3)$.

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

In order to analyze the shift in people's emotions towards COVID-19 over multiple waves, we implement the topic models independently for the first and the second COVID-19 waves. The tweet data for the time period November 2020 to February 2021 is considered to be constituting the first wave, while the tweet data corresponding to December 2021 to January 2022 is considered as the second wave. The topic modeling results for the first wave are summarized in Table 1. LDA_BERT corresponds to the LDA model that has been run on data cleaned and labled using roBERTa while LDA_VADER corresponds to the LDA model that has been implemented on data processed using the VADER model. Similarly, LSA_BERT and LSA_VADER correspond to LSA models implemented on data obtained after being processed by roBERTa and VADER models respectively. The number of topics or clusters for the LDA models were fixed at 3 based on domain heuristics. It is assumed that these topics will capture tweets representing the positive, negative, and neutral sentiments of the people. Thus, the tweets data has been modeled into 3 topics/clusters. However, LSA being a simpler model may not capture the non-linear dependencies in the data and model the 3 topics appropriately. Thus, for LSA, we run the topic model for the number of topics/clusters ranging from 2 to 10 and observe the coherence score for each of these topic models. We select the appropriate number of topics based on the best value for the observed topic coherence scores. For LSA_BERT, we obtain the best topic coherence for 7 clusters in the first wave, while the best coherence is obtained for 10 clusters for the LSA_VADER model.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47854124/205942796-93ccb92e-34a5-4aa1-8a40-8c43808f2161.png"/></p>
<h4 align="center">Table 1 - Topic Modeling Results for First Wave</h4>

Table 1 elaborates the different coherence metrics for each of the models. The UMass Coherence Score is an intrinsic measure while the CV Coherence Score and the UCI Coherence Score are extrinsic measures that consider external benchmark datasets while evaluating topic coherence. Based on a combination of the three coherence scores, we can conclude that the LSA_BERT model results in the most coherent topics for the first wave. Table 2 summarizes the topic modeling results obtained for the second wave. For the second wave, the best topic coherence is obtained for 6 clusters for the LSA_BERT model, while LSA_VADER produces the best topic coherence for 8 clusters. Based on the different coherence measures for the different topic models, we can observe that the LSA_BERT model results in the most coherent topics for the second wave.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47854124/205947809-fefec6f1-a858-4d77-b14b-2604dc253f8f.png"/></p>
<h4 align="center">Table 2 - Topic Modeling Results for Second Wave</h4>

However, these metrics do not provide us with human-interpretable topic models. As a result, we cannot compare the shift in people's emotions from one COVID-19 wave to another. For this purpose, we make use of wordclouds. The topic models provide us with terms/tokens that model each topic. We can use these to create wordclouds that can help us to easily capture the essence of the different topics in each wave. Figure 2 shows the wordcloud obtained from the topics modeled by LSA_BERT for the first wave, while Figure 3 shows the wordcloud corresponding to the second wave as modeled by LSA_BERT. Figure 4 shows the wordcloud corresponding to the first wave as modeled by LDA_VADER, while Figure 5 corresponds to the wordcloud obtained from the topics modeled by LDA_VADER for the second wave.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47854124/205974062-de69cc8a-ad98-4fed-822e-777f82dcf6f5.png"/></p>
<h4 align="center">Figure 2 - LSA_BERT wordcloud for First Wave</h4>

<p align="center">
  <img src="https://user-images.githubusercontent.com/47854124/205974446-f64142f5-5680-4ec9-a588-cbca60394ece.png"/></p>
<h4 align="center">Figure 3 - LSA_BERT wordcloud for Second Wave</h4>

<p align="center">
  <img src="https://user-images.githubusercontent.com/47854124/205974611-7994755d-ef75-4949-98d0-1ddc51e1cb6f.png"/></p>
<h4 align="center">Figure 4 - LDA_VADER wordcloud for First Wave</h4>

<p align="center">
  <img src="https://user-images.githubusercontent.com/47854124/205974823-30900796-1466-48af-a29c-8296d8800ad7.png"/></p>
<h4 align="center">Figure 5 - LDA_VADER wordcloud for Second Wave</h4>

The wordclouds clearly indicate a shift in people's tweets as COVID-19 progresses from one wave to the next. The topics in the first wave mostly revolve around lockdowns, work from home, wearing masks, the COVID-19 pandemic, vaccines, and new cases. However, the emphasis on lockdowns and masks reduces in the second wave. The second wave is characterized by topics like the omicron variant, vaccine mandates and the booster dose.

The LDA topic models were explored in further depth by using the pyLDAvis python package. This helped us produce interactive visualizations that map the intertopic distance and show the estimated term frequency within each topic along with the overall freuency of the terms for the top sailent terms for that topic. This supports human interpretation of the topics modeled by the LDA model.

![image](https://user-images.githubusercontent.com/47854124/205978833-3ef43fe9-99cf-4a01-99af-95824d308d2c.png)


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


##### Initial

Accuracy =  0.545

|         |precision |   recall  |f1-score  |   support|
|----------|-----------|----------|----------|---------|
|Negative   |    0.61   |  0.32    |    0.42   |  52823  |
|Positive   |    0.53   |  0.85     |   0.65   |  63446  |
|Neutral    |  0.53    |  0.05    |   0.09   |  15871  |
 
           
##### Final

Accuracy =  0.601

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
In the study, we processed about 6 months of data representing the tweets during multiple waves of COVID-19. We experimented with different Language Models, including TextBlob, Vader and roBERTa, to obtain sentiment scores for the Twitter data. On deeper inspection of the results, we found that roBERTa provided the best results for sentiment classification. However, we ran our models on all three sets of data. We ran three unsupervised learning algorithms (PCA, LSA, and LDA). We utilized PCA for feature selection, and LSA and LDA for topic modeling. The topic modeling results were evaluated using three different metrics (UMass coherence score, CV coherence score, and UCI coherence score), and demonstrated using tables and visualizations such as word clouds and bar graphs (using pyLDAvis). For supervised learning, we trained several supervised models (multinomial Naive Bayes, decision tree, random forest, SVM, etc.) to compare their performance in classifying the sentiment of these tweets. The models were compared using four different metrics (accuracy, precision, recall, and F1-score). The detailed results were demonstrated above, but the summary is that Random Forest Model performed best because of its ability to handle big data with numerous variables running into thousands. It can also to a degree automatically balance data sets when a class is more infrequent than other classes in the data. Overall, applying all these methods helped highlight the general shift in public sentiment towards COVID-19 as the pandemic evolved.
