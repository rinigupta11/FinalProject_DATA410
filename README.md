# MBTI Personality Classification Based on Social Media Posts 
### By Rini Gupta 
![image](https://user-images.githubusercontent.com/76021844/163291849-e1b45ca5-5ebf-4a69-ac7a-520a5ebe52e8.png)

Link to the dataset: https://www.kaggle.com/datasets/datasnaek/mbti-type 

## Introduction
The Myers Briggs Type Indicator (MBTI) is a personality type system that categorizes people into 16 personality types across 4 categories: 
- Introversion (I) – Extroversion (E)
- Intuition (N) – Sensing (S)
- Thinking (T) – Feeling (F)
- Judging (J) – Perceiving (P)

The MBTI test seeks to categorize random variations in personality in 16 distinct personality types and remains one of the most popular personality tests in the world. Personality labels are given based off of a series of questions on a personality examination and can generally categorize people's personalities into one of the 16 categories. Recently, the utility of the MBTI has been questioned after further experimentation. Nevertheless, many still use the Myers Briggs assessment as a useful categorization of personality.

For the purposes of our analysis, we will be examining if social media posts can serve as a predictor for MBTI personality type using an extensive dataset comprised of MBTI labels and previous social media posts. We will be using various classification techniques explained in depth in later sections to accomplish this task, as well as harnessing the power of NLP to capture meaning from the social media posts. If social media posts can predict MBTI type to a certain degree of accuracy then there will be an alternative way of determining personality through a machine learning algorithm as well as a validation mechanism that the Myers Brigg test provides meaningful classifications of personality that can be observed in daily life. Because the Myers Briggs assessment is relatively time consuming to complete, having a machine learning mechanism to label personality could be an efficient way to test personality. 

## Description of Data
The Myers Briggs Type Indicator (MBTI) personality type dataset includes information on people's MTBI type and content written by them. This dataset contains 8600+ observations. The data was collected from the PersonalityCafe forum. That particular source includes personality labels as well as textual social media posts people have made. 

- Personality Type (MBTI personality type) (labels) (target) 
- A section of each of the last 50 things they have posted on social media (textual data)(features)

Although there are only two columns of data, the text column with the social media posts actually consists of 50 previous social media posts and will be broken down into a matrix that summarizes the frequency of words within the social media post. The extensiveness of this dataset should also be helpful for further inferences. There is a substantial amount of data for classification within each observation as well as many (almost 9000) observations. 

Here is a visual display of the distribution of personality types in the dataset.
![image](https://user-images.githubusercontent.com/76021844/163433518-6e1cd1a0-c460-440b-9743-af4d0a45df71.png)

This distribution is acceptable given that there are some personality types that are more common among people than others, so we do not expect a uniform distribution. 

## Description of the Methods
We will be utilizing algorithms that center around classification. Given a set of observations and labels to train on, classification entails being able to categorize the data points. Since classification algorithms involve a labeled dataset, classification is a type of supervised learning. The two main types of classification algorithms are lazy learners and eager learners. Lazy learners store the training data in memory and classify the testing data point based on related stored train data. Training does not take as long with a lazy learner, but prediction can be slower. In contrast, an eager learner trains on the train data to cover all possible cases and then makes predictions after. Training takes a long time with this method, but prediction is much faster. There is no clear-cut best classification algorithm and performance changes depending on the dataset. As a result, we will experiment with a variety of classification algorithms. 

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
This section will be filled in for the final project using the score results and comparing the different methodologies to understand why some may outperform others. 

## Concluding Thoughts and Future Work 
This section will also be filled in for the final project with concluding thoughts and ideas for how this research should be expanded in the future to better understand classification tasks using text data. 


## References 
Asiri, S. (2018, June 11). Machine learning classifiers. Medium. Retrieved April 14, 2022, from https://towardsdatascience.com/machine-learning-classifiers-a5cc4e1b0623 
Bachman, E. (2020, March 27). Light GBM vs XGBOOST: Which algorithm takes the Crown. Analytics Vidhya. Retrieved April 14, 2022, from https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/ 
Horev, R. (2018, November 17). Bert explained: State of the art language model for NLP. Medium. Retrieved April 14, 2022, from https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270 
How SVM Works. How SVM works. (2021, August 17). Retrieved April 14, 2022, from https://www.ibm.com/docs/it/spss-modeler/SaaS?topic=models-how-svm-works 
Ke, Guolin; Meng, Qi; Finley, Thomas; Wang, Taifeng; Chen, Wei; Ma, Weidong; Ye, Qiwei; Liu, Tie-Yan (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". Advances in Neural Information Processing Systems.
Take the MBTI Instrument. The Myers & Briggs Foundation - take the MBTI® Instrument. (n.d.). Retrieved April 14, 2022, from https://www.myersbriggs.org/my-mbti-personality-type/take-the-mbti-instrument/ 
Waseem, M. (2022, March 28). Classification in machine learning: Classification algorithms. Edureka. Retrieved April 14, 2022, from https://www.edureka.co/blog/classification-in-machine-learning/ 
