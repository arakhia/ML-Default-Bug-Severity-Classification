from nltk.util import pr
import pandas as pd
import numpy as np
import re
import string

from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer

import statistics

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

from imblearn.over_sampling import RandomOverSampler

porter = PorterStemmer()
tokenizer = WordPunctTokenizer()

def vectorizeSeverityValue(value):
    if value == 'Critical' or value == 'Blocker':
        return 0
    elif value == 'Minor' or value == 'Trivial':
        return 1
    elif value == 'Major':
        return 2
    else:
        return 3

def vectorizeSemanticValue(value):
    if value == 'negative':
        return 0
    elif value == 'neutral':
        return 1
    elif value == 'positive':
        return 2

def getTokenizedText(text):
    text = str(text)
    tokens = tokenizer.tokenize(text)
    stemmed = []
    for token in tokens:
        stemmed.append(porter.stem(token))
        stemmed.append(" ")
    stemmed = "".join(stemmed)
    
    #text cleaning
    text_without_punctuation = [char for char in stemmed if char not in string.punctuation]
    text_without_punctuation = ''.join(text_without_punctuation)

    tokenized_text = [word for word in text_without_punctuation.split() if word.lower() not in stopwords.words('english')]
    tokenized_text = ' '.join(tokenized_text)
    return tokenized_text


logDataset = pd.read_csv('data/five_projects_log.csv')
logDataset.dropna(subset=['title', 'description', 'new_priority'] , inplace=True)
logDataset.new_priority = logDataset.new_priority.apply(vectorizeSeverityValue)
#logDataset.PREDICTED = logDataset.PREDICTED.apply(vectorizeSemanticValue)
logDataset.drop(logDataset[(logDataset['new_priority'] == 2) | (logDataset['new_priority'] == 3)].index, inplace=True)


logDataset = logDataset.reset_index()

logDataset.title = logDataset.title.apply(getTokenizedText)
logDataset.description = logDataset.description.apply(getTokenizedText)

X = logDataset[['title', 'description', 'PREDICTED']]
y = logDataset['new_priority']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('Results with Semantics')

preprocessor = ColumnTransformer(
    transformers=[
        ('text_title', TfidfVectorizer(), 'title'),
        ('text_description', TfidfVectorizer(), 'description'),
        ('category', OneHotEncoder(), ['PREDICTED']),
    ],
)

classifier = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression()),
],)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print('\n \n')
print('Results without Semantics')

preprocessor2 = ColumnTransformer(
    transformers=[
        ('text_title', TfidfVectorizer(), 'title'),
        ('text_description', TfidfVectorizer(), 'description'),
    ],
)

classifierWithoutSemantics = Pipeline(steps=[
    ('preprocessor', preprocessor2),
    ('classifier', LogisticRegression()),
],)

classifierWithoutSemantics.fit(X_train[['title', 'description']], y_train)
_y_pred = classifierWithoutSemantics.predict(X_test[['title', 'description']])
print(classification_report(y_test, _y_pred))
print(confusion_matrix(y_test, _y_pred))

print('######################')

print(cross_val_score(classifier, X, y, cv=10, scoring=make_scorer(f1_score, average='weighted')))
print(statistics.mean(cross_val_score(classifier, X, y, cv=10, scoring=make_scorer(f1_score, average='weighted'))))

print(cross_val_score(classifierWithoutSemantics, X[['title', 'description',]], y, cv=10, scoring=make_scorer(f1_score, average='weighted')))
print(statistics.mean(cross_val_score(classifierWithoutSemantics, X[['title', 'description']], y, cv=10, scoring=make_scorer(f1_score, average='weighted'))))
