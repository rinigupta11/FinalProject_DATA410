# MBTI Personality Classification Based on Social Media Posts 
### By Rini Gupta and Kimya Shirazi 
![image](https://user-images.githubusercontent.com/76021844/163291849-e1b45ca5-5ebf-4a69-ac7a-520a5ebe52e8.png)

Link to the dataset: https://www.kaggle.com/datasets/datasnaek/mbti-type 

## Introduction
kimya 
## Description of Data
kimya
## Description of the Methods
rini
### Preprocessing Steps
The dataset that we are working with contains primarily textual data. As a result, there is a decent amount of preprocessing work to be done in order to draw meaningful information from the textual data. There are several libraries in Python that deal with textual preprocessing -- namely, NLTK, Gensim, and spaCy. Regardless of which library we ultimately choose to use, there are several standard steps that need to be taken to transform the text data into something that can go into a machine learning model. At this point, some experimentation can be done with lemmatization and stemming. Next, we will remove url links, punctuation, and excess white space in the social media posts. Then, we will make all of the letters lowercase to standardize the data. Finally, we will gather a list of common English stopwords (commonly used words that do not contribute to overall meaning) and remove them from the text data to eliminate them from influencing classification as well as lower the number of words per observation. After our string data is much cleaner, we can at this stage of the analysis tokenize the text (split it into individual words in a list). This step is essential so we can transform our text data into matrix or vector form. The two options for doing this are using the CountVectorizer function or by creating a TF-IDF (Term Frequency Inverse Document Frequency) matrix. These approaches utilize the frequency count of the words in the post as the metric for future modeling input. A functional programming approach is ideal for setting up these processing steps in order to best streamline cleaning training data and future data that might be plugged into the model. 

### Machine Learning Methodology 
The core goal with this dataset is classification. The classification for this dataset logically can be separated out into four separate classifications for the four components of the Myers Brigg label. These four classifications would be E versus I (extraverted versus introverted), S versus N (sensing versus intuitive), T versus F (thinking versus feeling), and J versus P (judging versus perceiving). 

There are several options that will be evaluated as options for the classification task. These models include random forest classifier, XGBClassifier (extreme gradient boosting for classification), LightGBMClassifier (light gradient boosting classifier), linear support vector machines (SVM Classifiers), and the BERT (Bidirectional Encoder Representation from Transformers) deep learning model. A GPU will be utilized to accelerate runtime since this dataset is quite comprehensive. 

## Pipeline
rini

## Validation Procedure 
rini

## Results/Inferences
rini

## Concluding Thoughts and Future Work 
rini/kimya

## References 
kimya 
