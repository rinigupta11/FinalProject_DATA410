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
The dataset that we are working with contains primarily textual data. As a result, there is a decent amount of preprocessing work to be done in order to draw meaningful information from the textual data. There are several libraries in Python that deal with textual preprocessing -- namely, NLTK, Gensim, and spaCy. Regardless of which library we ultimately choose to use, there are several standard steps that need to be taken to transform the text data into something that can go into a machine learning model. At this point, some experimentation can be done with lemmatization and stemming. Next, we will remove url links, punctuation, and excess white space in the social media posts. Then, we will make all of the letters lowercase to standardize the data. Finally, we will gather a list of common English stopwords (commonly used words that do not contribute to overall meaning) and remove them from the text data to eliminate them from influencing classification as well as lower the number of words per observation. 

After our string data is much cleaner, we can at this stage of the analysis tokenize the text (split it into individual words in a list). This step is essential so we can transform our text data into matrix or vector form. The two options for doing this are using the CountVectorizer function or by creating a TF-IDF (Term Frequency Inverse Document Frequency) matrix. These approaches utilize the frequency count of the words in the post as the metric for future modeling input. A functional programming approach is ideal for setting up these processing steps in order to best streamline cleaning training data and future data that might be plugged into the model. 

### Machine Learning Methodology 
The core goal with this dataset is classification. The classification for this dataset logically can be separated out into four separate classifications for the four components of the Myers Brigg label. These four classifications would be E versus I (extraverted versus introverted), S versus N (sensing versus intuitive), T versus F (thinking versus feeling), and J versus P (judging versus perceiving). 

There are several options that will be evaluated as options for the classification task. These models include random forest classifier, XGBClassifier (extreme gradient boosting for classification), LightGBMClassifier (light gradient boosting classifier), linear support vector machines (SVM Classifiers), and the BERT (Bidirectional Encoder Representation from Transformers) deep learning model. A GPU will be utilized to accelerate runtime since this dataset is quite comprehensive. 

In simple terms, boosting takes weak learners and makes them into strong learners (Singh 2018). The trees that are fit are on a modified version of the original data. Each tree tries to improve upon the weights placed on the previous tree. Gradient boosting is a greedy algorithm that gradually trains many models. Friedman's extreme gradient boosting was developed in 2001 with regularization mechanisms to avoid overfitting (Maklin 2020). Like gradient boosting, extreme gradient boosting is a tree-based algorithm. One of the main strengths of extreme gradient boosting is the speed at which it runs, particularly in comparison to a deep neural network.

LightGBM is a gradient boosting (tree-based) framework developed by Microsoft to improve upon accuracy, efficiency, and memory-usage of other boosting algorithms. XGBoost is the current star among boosting algorithms in terms of the accuracy that it produces; however, XGBoost can take more time to compute results. As a result, LightGBM aims to compete with its "lighter", speedier framework. LightGBM splits the decision tree by the leaf with the best fit. The way that LightGBM chooses the leaf is by finding the split that will create the greatest loss decrease. In contrast, other boosting algorithms split the tree based on depth. Splitting by the leaf has proven to be a very effective loss reduction technique that boosts accuracy. Furthermore, LightGBM uses a histogram-like approach and puts continuous features into bins to speed training time. This approach has been demonstrated to dramatically increase time and space complexity. The two specific techniques that are part of the LightGBM algorithm are Exclusive Feature Bundling (a feature reduction technique) and Gradient-Based One Side Sampling (higher gradients contribute more information). 

Linear Support Vector Machines (SVMs) are used for regression and classification. For this task, we will be using the SVM for classification. SVMs translate the observations to a higher dimensional feature space. From there, the data points can be classified using the concept of hyperplanes and separators. The transformation process utilizes a kernel function which can take a variety of forms. It is useful to experiment with different kernel functions to see which performs the best with the SVM. 

TowardsDataScience calls BERT the "state of the art" model for natural language processing. BERT utilizes the bidirectional training of a widely-used attention model, Transformer. Given the complexity of comprehending language, bidirectional training has led to meaningful improvements in the textual analysis. As mentioned previously, BERT is a deep learning model. BERT utilizes the encoding aspect of the Transformer model and that encoder reads text all at once (perhaps non-directional is a better term than bidirectional). BERT can be used for a wide variety of NLP tasks such as classification, question answering, and named-entity recognition. 

These methods are all supervised learning methods. We are able to use a supervised approach since we have a labeled dataset to train the models on. 

## Pipeline
![Untitled drawing](https://user-images.githubusercontent.com/76021844/163297864-ca3c80a7-e2f2-45a4-ae8e-458b9ea62016.png)

## Validation Procedure 
More information will be written here for the final submission, but our plan is to use the score function of these classifiers and compare scores across the different models. The model with the highest score will be considered the best model for personality prediction. It will likely be difficult to run many cross validations given the size of the dataset, so we do not plan on performing cross validation at the present moment. 

## Results/Inferences
rini

## Concluding Thoughts and Future Work 
rini/kimya

## References 
kimya 

Bachman, E. (2020, March 27). Light GBM vs XGBOOST: Which algorithm takes the Crown. Analytics Vidhya. Retrieved March 6, 2022, from https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/

Ke, Guolin; Meng, Qi; Finley, Thomas; Wang, Taifeng; Chen, Wei; Ma, Weidong; Ye, Qiwei; Liu, Tie-Yan (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". Advances in Neural Information Processing Systems.
