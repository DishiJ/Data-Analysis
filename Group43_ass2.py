#!/usr/bin/env python
# coding: utf-8

# In[3]:


# importing the required libraries
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# object for the Porter stemmer library for stemming
stem = PorterStemmer()

# reading the training data file
df_final = pd.read_csv('train_data_final.csv')

# initializing the tokenizer to extract words from the text
tokenizer = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

# reading the stopword file into a list
stopwords_list = stopwords.words('english')


# function to perform tokenization,stemming and lower case all the words
def clean_line(t):
    # tokenizing the abstract using tokenizer object
    token_list = tokenizer.tokenize(t)
    # removing the stopwords,stemming the words and changing it to lower case of the abstract
    final_list_tokens = [stem.stem(x).lower() for x in token_list if x not in stopwords_list]
    # concating the list of tokens to a string
    string_final = ' '.join(final_list_tokens)

    # return the final string of the abstract
    return string_final


# initializing the object of the TfidfVectorizer
tfidf = TfidfVectorizer()


# function to call the clean function and transforming using TfidfVectorizer object
def load_and_process_data(df_final):
    all_text = [clean_line(t) for t in df_final.abstract]
    # making dataframe
    all_data_df = pd.DataFrame({'text': all_text, 'topics': df_final.label})
    # initializing the X train data
    X_raw = all_data_df['text'].values
    # initializing the Y Train data
    y_raw = all_data_df['topics'].values

    # fitting and transforming the X Train data using the object of the Tfidf vectorizer object
    X_train_tfidf = tfidf.fit_transform(X_raw)
    # returning the X train and Y train data
    return X_train_tfidf, y_raw


# calling the function load and process
X_train, y_raw = load_and_process_data(df_final)

# intializing the logistic regression model
lr_clf = LogisticRegression(multi_class='multinomial', solver='saga')
# fitting the model using Train X and Train Y data
lr_clf.fit(X_train, y_raw)

# reading the test data for predictions
df_test = pd.read_csv("test_data.csv")
# cleaning the abstract column using clean function
all_abstract = [clean_line(t) for t in df_test.abstract]
# Transforming the abstract using tfidf vectorizer object
test_data = tfidf.transform(all_abstract)
# predicting the data using the model
predicted_check = lr_clf.predict(test_data)
# including the prediction into the dataframe
df_test['label'] = pd.Series(predicted_check)
# droping the abstract column
df_test.drop(['abstract'], inplace=True, axis=1)
# writing the dataframe with prediction into a csv file
df_test.to_csv('pred_labels.csv', index=False)


