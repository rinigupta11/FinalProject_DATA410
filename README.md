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
The dataset that I am working with contains primarily textual data. As a result, there is a decent amount of preprocessing work to be done in order to draw meaningful information from the textual data. There are several libraries in Python that deal with textual preprocessing -- namely, NLTK, Gensim, and spaCy. For the purposes of this analysis, the preprocessing involved was relatively straightforward so I decided to go with NLTK. There are several standard steps that need to be taken to transform the text data into something that can go into a machine learning model to produce effective predictions. First, I removed url links, punctuation, and excess white space in the social media posts. Another important step is to remove all labels of personality type if they exist within the social media posts so there is no cheating involved in the training process (Vijay 2019). Then, we will make all of the letters lowercase to standardize the data. We stem the words in order to create more unified meanings for words with different endings but of the same word family. Finally, we will gather a list of common English stopwords (commonly used words that do not contribute to overall meaning) and remove them from the text data to eliminate them from influencing classification as well as lower the number of words per observation. The purpose of all of these preprocessing steps is to eliminate as much non-useful data from the observations as possible so that the models can train on only significant information. 

```
personality_labels=list(mbti_df['type'].unique())
personality_labels = [label.lower() for label in personality_labels]
for i in range(len(mbti_df)) :  
    mbti_df['posts'][i] = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', mbti_df['posts'][i])
    mbti_df['posts'][i] = re.sub(' +', ' ', mbti_df['posts'][i])
    for k in range(16):
        mbti_df['posts'][i]=re.sub(personality_labels[k], ' ', mbti_df['posts'][i])
        
mbti_df['posts'] = mbti_df['posts'].str.strip()
mbti_df['posts'] = mbti_df['posts'].str.lower()
def process(text):
    text = re.sub('[^a-zA-Z0-9 ]','',text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    txt = [ps.stem(word) for word in text]
    text = ' '.join(text)
    return text
corpus = mbti_df["posts"].apply(process)
```
The core goal with this dataset is classification. The classification for this dataset logically can be separated out into four separate classifications for the four components of the Myers Brigg label. These four classifications would be E versus I (extraverted versus introverted), S versus N (sensing versus intuitive), T versus F (thinking versus feeling), and J versus P (judging versus perceiving). For the purposes of our classification, I took inspiration from the approach taken by Vijay and split the classification into these portions by creating encodings for each of the four tasks using 0s and 1s by using the concept of a map in Python so the model can train on the encodings. 
```
# code adapted from approach taken by Vijay/Sharma 2019
map1 = {"I": 0, "E": 1}
map2 = {"N": 0, "S": 1}
map3 = {"T": 0, "F": 1}
map4 = {"J": 0, "P": 1}
mbti_df['I-E'] = mbti_df['type'].astype(str).str[0]
mbti_df['I-E'] = mbti_df['I-E'].map(map1)
mbti_df['N-S'] = mbti_df['type'].astype(str).str[1]
mbti_df['N-S'] = mbti_df['N-S'].map(map2)
mbti_df['T-F'] = mbti_df['type'].astype(str).str[2]
mbti_df['T-F'] = mbti_df['T-F'].map(map3)
mbti_df['J-P'] = mbti_df['type'].astype(str).str[3]
mbti_df['J-P'] = mbti_df['J-P'].map(map4)
```

After our string data is much cleaner, we can at this stage of the analysis tokenize the text (split it into individual words in a list). This step is essential so we can then transform our text data into matrix or vector form. The two options for doing this are using the CountVectorizer function or by creating a TF-IDF (Term Frequency Inverse Document Frequency) matrix. These approaches utilize the frequency count of the words in the post as the metric for future modeling input. I decided to use the count vectorizer function to prepare the data for the machine learning models. This function transforms textual data into a vector of token counts where tokens are the individual terms. Using the encodings above, I create training sets for each of the four classification tasks by assigning the approach column labels to the personality classification task (ex. introverted versus extroverted). A functional programming approach is ideal for setting up these processing steps in order to best streamline cleaning training data and future data that might be plugged into the model. 

```
# Code also adapted from Vijay/Sharma approach
cv = CountVectorizer(max_features = 2000)
features = cv.fit_transform(mbti_df['posts']).toarray()
IE= mbti_df.iloc[:, 2].values
NS= mbti_df.iloc[:, 3].values
TF=mbti_df.iloc[:, 4].values
JP=mbti_df.iloc[:, 5].values
```

Finally, we split the data into training and testing sets for the four classification tasks. 

```
from sklearn.model_selection import train_test_split
posts_train, posts_test, IE_train, IE_test, NS_train, NS_test, TF_train, TF_test, JP_train, JP_test = train_test_split(features, IE,NS,TF,JP, test_size = 0.20)
```
## Pipeline
<img width="919" alt="Screen Shot 2022-05-02 at 10 41 46 AM" src="https://user-images.githubusercontent.com/76021844/166253756-a2b8a401-d8f9-42be-bb1b-01b6f600616b.png">

### Machine Learning Methodology 

There are several options that will be evaluated as options for the classification task. These models include random forest classifier, XGBClassifier (extreme gradient boosting for classification), LightGBMClassifier (light gradient boosting classifier), linear support vector machines (SVM Classifiers), CatBoost,and a neural network (deep learning) modeling approach. A GPU will be utilized to accelerate runtime where possible since this dataset is quite comprehensive.

The first classifier tested was the random forest classifier. In order to understand how the random forest classifier works, we first introduce the concept of a decision tree. A decision tree is actually quite a simple tree-like structure which trains on some labeleled data to then make predictions about new data. This process occurs by forming a hierarchy of decisions to make in the training process. The process of separating the different levels of the tree happens recursively, separating into homogenous groups (nodes) down to terminal nodes (Gromping 2009). 
![image](https://user-images.githubusercontent.com/76021844/153754695-8e7d0a5c-fbec-4b84-b90c-fb756e6696fd.png)

The random forest regressor model is an ensemble model that incorporates many decision trees into its structure to make a final prediction on data. Unlike an ordinary linear classifier, random forests can fit to accomodate non-linearities in the dataset. As a result, random forests are non-parametric (Gromping 2009). Random forests are advantageous over decision trees because they are better at preventing overfitting due to the ensemble nature of the model (incorporating several predictions). The individual decision trees within the forest are, as the name suggests, quite random and yield differing predictions. The random forest algorithm takes the average of each individual decision tree to make final predictions (Gromping 2019). Additionally, random forests group weak learners together to form stronger learners (boosting), another theoretical strength of the model. Random forests are regarded by data scientists as one of the "best performing learning algorithms" (Schonlau 2020).

<img width="867" alt="image" src="https://user-images.githubusercontent.com/76021844/153780621-864777ac-93fc-48ee-83c1-d180ad89623f.png">

```
from sklearn.ensemble import RandomForestClassifier

IE_model = RandomForestClassifier()
IE_model.fit(posts_train, IE_train)
IE_model_train=IE_model.score(posts_train,IE_train)
IE_model_test=IE_model.score(posts_test,IE_test)

NS_model = RandomForestClassifier()
NS_model.fit(posts_train, NS_train)
NS_model_train=NS_model.score(posts_train,NS_train)
NS_model_test=NS_model.score(posts_test,NS_test)


TF_model = RandomForestClassifier()
TF_model.fit(posts_train, TF_train)
TF_model_train=TF_model.score(posts_train,TF_train)
TF_model_test=TF_model.score(posts_test,TF_test)

JP_model = RandomForestClassifier()
JP_model.fit(posts_train, JP_train)
JP_model_train=JP_model.score(posts_train,JP_train)
JP_model_test=JP_model.score(posts_test,JP_test)
```

In simple terms, boosting takes weak learners and makes them into strong learners (Singh 2018). The trees that are fit are on a modified version of the original data. Each tree tries to improve upon the weights placed on the previous tree. Gradient boosting is a greedy algorithm that gradually trains many models. Friedman's extreme gradient boosting was developed in 2001 with regularization mechanisms to avoid overfitting (Maklin 2020). Like gradient boosting, extreme gradient boosting is a tree-based algorithm. One of the main strengths of extreme gradient boosting is the speed at which it runs, particularly in comparison to a deep neural network. 
```
from xgboost import XGBClassifier

IE_model = XGBClassifier()
IE_model.fit(posts_train, IE_train)
IE_model_train=IE_model.score(posts_train,IE_train)
IE_model_test=IE_model.score(posts_test,IE_test)

NS_model = XGBClassifier()
NS_model.fit(posts_train, NS_train)
NS_model_train=NS_model.score(posts_train,NS_train)
NS_model_test=NS_model.score(posts_test,NS_test)


TF_model = XGBClassifier()
TF_model.fit(posts_train, TF_train)
TF_model_train=TF_model.score(posts_train,TF_train)
TF_model_test=TF_model.score(posts_test,TF_test)

JP_model = XGBClassifier()
JP_model.fit(posts_train, JP_train)
JP_model_train=JP_model.score(posts_train,JP_train)
JP_model_test=JP_model.score(posts_test,JP_test)
```

LightGBM is a gradient boosting (tree-based) framework developed by Microsoft to improve upon accuracy, efficiency, and memory-usage of other boosting algorithms. XGBoost is the current star among boosting algorithms in terms of the accuracy that it produces; however, XGBoost can take more time to compute results. As a result, LightGBM aims to compete with its "lighter", speedier framework. LightGBM splits the decision tree by the leaf with the best fit. The way that LightGBM chooses the leaf is by finding the split that will create the greatest loss decrease. In contrast, other boosting algorithms split the tree based on depth. Splitting by the leaf has proven to be a very effective loss reduction technique that boosts accuracy. Furthermore, LightGBM uses a histogram-like approach and puts continuous features into bins to speed training time. This approach has been demonstrated to dramatically increase time and space complexity. The two specific techniques that are part of the LightGBM algorithm are Exclusive Feature Bundling (a feature reduction technique) and Gradient-Based One Side Sampling (higher gradients contribute more information). 
```
import lightgbm as lgb

IE_model = lgb.LGBMClassifier()
IE_model.fit(posts_train, IE_train)
IE_model_train=IE_model.score(posts_train,IE_train)
IE_model_test=IE_model.score(posts_test,IE_test)

NS_model = lgb.LGBMClassifier()
NS_model.fit(posts_train, NS_train)
NS_model_train=NS_model.score(posts_train,NS_train)
NS_model_test=NS_model.score(posts_test,NS_test)


TF_model = lgb.LGBMClassifier()
TF_model.fit(posts_train, TF_train)
TF_model_train=TF_model.score(posts_train,TF_train)
TF_model_test=TF_model.score(posts_test,TF_test)

JP_model = lgb.LGBMClassifier()
JP_model.fit(posts_train, JP_train)
JP_model_train=JP_model.score(posts_train,JP_train)
JP_model_test=JP_model.score(posts_test,JP_test)

```

Linear Support Vector Machines (SVMs) are used for regression and classification. For this task, we will be using the SVM for classification. SVMs translate the observations to a higher dimensional feature space. From there, the data points can be classified using the concept of hyperplanes and separators. The transformation process utilizes a kernel function which can take a variety of forms. It is useful to experiment with different kernel functions to see which performs the best with the SVM. 
```
from sklearn.svm import SVC

IE_model = SVC()
IE_model.fit(posts_train, IE_train)
IE_model_train=IE_model.score(posts_train,IE_train)
IE_model_test=IE_model.score(posts_test,IE_test)

NS_model = SVC()
NS_model.fit(posts_train, NS_train)
NS_model_train=NS_model.score(posts_train,NS_train)
NS_model_test=NS_model.score(posts_test,NS_test)


TF_model = SVC()
TF_model.fit(posts_train, TF_train)
TF_model_train=TF_model.score(posts_train,TF_train)
TF_model_test=TF_model.score(posts_test,TF_test)

JP_model = SVC()
JP_model.fit(posts_train, JP_train)
JP_model_train=JP_model.score(posts_train,JP_train)
JP_model_test=JP_model.score(posts_test,JP_test)
```

CatBoost is another gradient boosting algorithm. One major advantage of CatBoost is it can work well with categorical variables directly without any hot-encodings. The name CatBoost comes from the words category and boosting (gradient boosting). CatBoost does not require tedious hyperparameter fine-tuning and as a result reduces the risk of creating an overfit model. CatBoost is relatively efficient and is hypothesized to produce results on par with the other state-of-the-art gradient boosting algorithms discussed above. CatBoost has become one of the most widely used machine learning frameworks in recent past according to Kaggle. 

```
from catboost import CatBoostClassifier as CBC

IE_model = CBC()
IE_model.fit(posts_train, IE_train)
IE_model_train=IE_model.score(posts_train,IE_train)
IE_model_test=IE_model.score(posts_test,IE_test)

NS_model = CBC()
NS_model.fit(posts_train, NS_train)
NS_model_train=NS_model.score(posts_train,NS_train)
NS_model_test=NS_model.score(posts_test,NS_test)


TF_model = CBC()
TF_model.fit(posts_train, TF_train)
TF_model_train=TF_model.score(posts_train,TF_train)
TF_model_test=TF_model.score(posts_test,TF_test)

JP_model = CBC()
JP_model.fit(posts_train, JP_train)
JP_model_train=JP_model.score(posts_train,JP_train)
JP_model_test=JP_model.score(posts_test,JP_test)
```

Finally, I tested out a neural network architecture in TensorFlow inspired by Uzsoy. Neural networks are a type of deep learning that can be used for classification tasks. 
![image](https://user-images.githubusercontent.com/76021844/166251801-d3256155-c839-472b-9af4-cf715dc3dd0e.png)
Neural networks are inspired by the structure of the human brain and have various neurons that interact with neurons in the next layer of the model by taking numerical input and sending numerical output as well as adding a bias term. There are several layers within a typical neural network so there are multiple interacting features. A neural network adjusts weights as it trains. One difficult area involves understanding what bias term there should be and what the optimal weights are. A cost function is used at the end to understand how the neural network performed. 
```
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

def create_model(): 
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = (Bidirectional(LSTM(200, return_sequences=True)))(x)
    x = (Dropout(0.3))(x)
    x = (Bidirectional(LSTM(20)))(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(16, activation="softmax")(x)
    
    op = tf.keras.optimizers.Adam(learning_rate=0.00001)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(op, 'categorical_crossentropy', metrics = ['accuracy'])
    return model
    
use_tpu = True
if use_tpu:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    with strategy.scope():
        model = create_model()
else:
    model = create_model()    
model.fit(train_padded, one_hot_labels, epochs =30, verbose = 1, 
          validation_data = (val_padded, val_labels), callbacks = [tf.keras.callbacks.EarlyStopping(patience = 3)])
```

The code above is taken from Uzosoy and essentially initializes various layers within the network. There is a bidirectional layer, a typical dropout layer, another bidirectional layer, and then a dense layer followed by the final dropout layer. The Keras library and TensorFlow library was used to create this model and the Adam optimizer was used as well as categorical cross entropy for the loss function. 

These methods are all supervised learning methods. We are able to use a supervised approach since we have a labeled dataset to train the models on. 


## Validation Procedure 
The model with the highest score will be considered the best model for personality prediction. It is computationally difficult to run many cross validations given the size of the dataset, so we do not plan on performing cross validation at the present moment. This would be an area of future work to expand upon, however. 

## Results/Inferences
These are the summarized test scores (using the respective libraries' score function) from each of the model runs where the test scores are in the order IE, NS, TF, and then JP classification:

XGB Scores: [0.7827089337175792, 0.8645533141210374, 0.7412103746397695, 0.6720461095100865]

LightGBM Scores: [0.7838616714697406, 0.8651296829971181, 0.7463976945244957, 0.6818443804034582]

RF Classifier Scores: [0.7752161383285303, 0.8639769452449567, 0.7100864553314121, 0.6380403458213256]

SVC Scores: [0.7786743515850144, 0.8639769452449567, 0.7631123919308357, 0.6743515850144092]

Cat Scores: [0.7861671469740634, 0.8662824207492795, 0.7561959654178675, 0.6795389048991355]

As you can see, the various models were comparable in performance with no clear winner. The gradient boosting models did appear to consistently outperform the random forest scores, especially for the last two classifications. There is, however, one clear loser. That is the neural network. I created encodings for the various 16 personality types in order to work with the neural network more efficiently and the final score after 30 epochs was 0.2636 accuracy. This is significantly worse than the other models are likely indicates that the architecture of the model needs to be altered. Furthermore, running more epochs might be worthwhile to see if the score can increase further. 

Here is the full output of the neural network training:

Epoch 1/30
153/153 [==============================] - 186s 1s/step - loss: 2.7450 - accuracy: 0.0836 - val_loss: 2.7062 - val_accuracy: 0.1346
Epoch 2/30
153/153 [==============================] - 160s 1s/step - loss: 2.6956 - accuracy: 0.0703 - val_loss: 2.6735 - val_accuracy: 0.0270
Epoch 3/30
153/153 [==============================] - 160s 1s/step - loss: 2.6727 - accuracy: 0.0508 - val_loss: 2.6537 - val_accuracy: 0.0098
Epoch 4/30
153/153 [==============================] - 160s 1s/step - loss: 2.6524 - accuracy: 0.0844 - val_loss: 2.6411 - val_accuracy: 0.0953
Epoch 5/30
153/153 [==============================] - 160s 1s/step - loss: 2.6429 - accuracy: 0.1322 - val_loss: 2.6309 - val_accuracy: 0.1567
Epoch 6/30
153/153 [==============================] - 160s 1s/step - loss: 2.6312 - accuracy: 0.1728 - val_loss: 2.6205 - val_accuracy: 0.1653
Epoch 7/30
153/153 [==============================] - 160s 1s/step - loss: 2.6206 - accuracy: 0.1898 - val_loss: 2.6112 - val_accuracy: 0.1961
Epoch 8/30
153/153 [==============================] - 160s 1s/step - loss: 2.6142 - accuracy: 0.1869 - val_loss: 2.6028 - val_accuracy: 0.2022
Epoch 9/30
153/153 [==============================] - 160s 1s/step - loss: 2.6050 - accuracy: 0.2007 - val_loss: 2.5945 - val_accuracy: 0.2028
Epoch 10/30
153/153 [==============================] - 160s 1s/step - loss: 2.5996 - accuracy: 0.1959 - val_loss: 2.5877 - val_accuracy: 0.2053
Epoch 11/30
153/153 [==============================] - 160s 1s/step - loss: 2.5961 - accuracy: 0.1996 - val_loss: 2.5806 - val_accuracy: 0.2096
Epoch 12/30
153/153 [==============================] - 160s 1s/step - loss: 2.5843 - accuracy: 0.2005 - val_loss: 2.5739 - val_accuracy: 0.2090
Epoch 13/30
153/153 [==============================] - 160s 1s/step - loss: 2.5817 - accuracy: 0.1984 - val_loss: 2.5663 - val_accuracy: 0.2120
Epoch 14/30
153/153 [==============================] - 160s 1s/step - loss: 2.5742 - accuracy: 0.2039 - val_loss: 2.5574 - val_accuracy: 0.2120
Epoch 15/30
153/153 [==============================] - 160s 1s/step - loss: 2.5725 - accuracy: 0.2052 - val_loss: 2.5498 - val_accuracy: 0.2114
Epoch 16/30
153/153 [==============================] - 160s 1s/step - loss: 2.5592 - accuracy: 0.2134 - val_loss: 2.5403 - val_accuracy: 0.2120
Epoch 17/30
153/153 [==============================] - 160s 1s/step - loss: 2.5508 - accuracy: 0.2173 - val_loss: 2.5311 - val_accuracy: 0.2120
Epoch 18/30
153/153 [==============================] - 160s 1s/step - loss: 2.5389 - accuracy: 0.2150 - val_loss: 2.5233 - val_accuracy: 0.2200
Epoch 19/30
153/153 [==============================] - 160s 1s/step - loss: 2.5320 - accuracy: 0.2203 - val_loss: 2.5153 - val_accuracy: 0.2065
Epoch 20/30
153/153 [==============================] - 160s 1s/step - loss: 2.5245 - accuracy: 0.2199 - val_loss: 2.5072 - val_accuracy: 0.1998
Epoch 21/30
153/153 [==============================] - 160s 1s/step - loss: 2.5127 - accuracy: 0.2232 - val_loss: 2.4957 - val_accuracy: 0.2237
Epoch 22/30
153/153 [==============================] - 160s 1s/step - loss: 2.5028 - accuracy: 0.2253 - val_loss: 2.4873 - val_accuracy: 0.2268
Epoch 23/30
153/153 [==============================] - 161s 1s/step - loss: 2.4912 - accuracy: 0.2367 - val_loss: 2.4765 - val_accuracy: 0.2237
Epoch 24/30
153/153 [==============================] - 160s 1s/step - loss: 2.4840 - accuracy: 0.2386 - val_loss: 2.4745 - val_accuracy: 0.2176
Epoch 25/30
153/153 [==============================] - 160s 1s/step - loss: 2.4660 - accuracy: 0.2412 - val_loss: 2.4692 - val_accuracy: 0.2028
Epoch 26/30
153/153 [==============================] - 160s 1s/step - loss: 2.4664 - accuracy: 0.2404 - val_loss: 2.4513 - val_accuracy: 0.2274
Epoch 27/30
153/153 [==============================] - 160s 1s/step - loss: 2.4476 - accuracy: 0.2527 - val_loss: 2.4408 - val_accuracy: 0.2342
Epoch 28/30
153/153 [==============================] - 160s 1s/step - loss: 2.4309 - accuracy: 0.2642 - val_loss: 2.4330 - val_accuracy: 0.2379
Epoch 29/30
153/153 [==============================] - 160s 1s/step - loss: 2.4220 - accuracy: 0.2660 - val_loss: 2.4289 - val_accuracy: 0.2329
Epoch 30/30
153/153 [==============================] - 160s 1s/step - loss: 2.4127 - accuracy: 0.2636 - val_loss: 2.4195 - val_accuracy: 0.2440
<keras.callbacks.History at 0x7f95ac75f8d0>


## Concluding Thoughts and Future Work 

TowardsDataScience calls BERT the "state of the art" model for natural language processing. BERT utilizes the bidirectional training of a widely-used attention model, Transformer. Given the complexity of comprehending language, bidirectional training has led to meaningful improvements in the textual analysis. As mentioned previously, BERT is a deep learning model. BERT utilizes the encoding aspect of the Transformer model and that encoder reads text all at once (perhaps non-directional is a better term than bidirectional). BERT can be used for a wide variety of NLP tasks such as classification, question answering, and named-entity recognition. My original plan was to test out BERT on this NLP classification task; however, I did not have the computational ability to do so as I repeatedly received resources exhausted errors. I hypothesize that BERT would outperform the neural network that I did use and think that incorporating BERT into this analysis must be a key portion of future work. 

Furthermore, cross-validation would be a useful exercise to ensure there is no one specific model that outperforms the others. This area was another computational efficiency problem when I was conducting my analysis, but cross-validation is a very important way to understand the results of a particular model. 


## References 
Asiri, S. (2018, June 11). Machine learning classifiers. Medium. Retrieved April 14, 2022, from https://towardsdatascience.com/machine-learning-classifiers-a5cc4e1b0623 

Bachman, E. (2020, March 27). Light GBM vs XGBOOST: Which algorithm takes the Crown. Analytics Vidhya. Retrieved April 14, 2022, from https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/ 

Horev, R. (2018, November 17). Bert explained: State of the art language model for NLP. Medium. Retrieved April 14, 2022, from https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270 
How SVM Works. How SVM works. (2021, August 17). Retrieved April 14, 2022, from https://www.ibm.com/docs/it/spss-modeler/SaaS?topic=models-how-svm-works 

Grömping, U. (2009). Variable importance assessment in regression: Linear regression versus Random Forest. The American Statistician, 63(4), 308–319. https://doi.org/10.1198/tast.2009.08199 

Ke, Guolin; Meng, Qi; Finley, Thomas; Wang, Taifeng; Chen, Wei; Ma, Weidong; Ye, Qiwei; Liu, Tie-Yan (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". Advances in Neural Information Processing Systems.
Take the MBTI Instrument. The Myers & Briggs Foundation - take the MBTI® Instrument. (n.d.). Retrieved April 14, 2022, from https://www.myersbriggs.org/my-mbti-personality-type/take-the-mbti-instrument/ 

Knocklein, O. (2019, June 15). Classification using neural networks. Medium. Retrieved May 2, 2022, from https://towardsdatascience.com/classification-using-neural-networks-b8e98f3a904f 

Vijay, T. (2019, June 27). Personality classifier MBTI. Kaggle. Retrieved May 2, 2022, from https://www.kaggle.com/code/tapanvijay/personality-classifier-mbti 

Uzoy, A. S. (2020, June 17). Myers-Briggs types with Tensorflow/Bert. Kaggle. Retrieved May 2, 2022, from https://www.kaggle.com/code/anasofiauzsoy/myers-briggs-types-with-tensorflow-bert 

Waseem, M. (2022, March 28). Classification in machine learning: Classification algorithms. Edureka. Retrieved April 14, 2022, from https://www.edureka.co/blog/classification-in-machine-learning/ 
