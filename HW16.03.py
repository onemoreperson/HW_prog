# -*- coding: utf-8 -*-
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import  wordpunct_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.dummy import DummyClassifier

clf = tree.DecisionTreeClassifier()
rfc = RandomForestClassifier()
wml = WordNetLemmatizer()

df = pd.read_csv('C:/Users/Оля/Desktop/All-seasons.csv')

Stan = df[df['Character'] == 'Stan']
Cartman = df[df['Character'] == 'Kyle']
Kyle = df[df['Character'] == 'Cartman']
Kenny = df[df['Character'] == 'Kenny']

#для лемматизации текста и удаения стопслов, обсценную лексику удаляить не стоит, есть некоторая
#специфическая для персонажей. ф-я используется в векторах  в качестве лемматизатора и токенизатора.
def lemm(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = wordpunct_tokenize(text)    
    mass = [wml.lemmatize(word) for word in word_tokens if word not in stop_words]
    return mass    

train = pd.concat([Stan, Kyle, Cartman, Kenny], ignore_index = True)

#test = pd.concat([Stan, Kyle, Cartman, Kenny], ignore_index = True)

#
x_train, x_test, y_train, y_test = train_test_split(list(train['Line']), np.array(train['Character'])) 
#
xtrain, xtest, ytrain, ytest = train_test_split(x_train, y_train)
#--------------------------------------------------------------------------------------------------
# .
cv = CountVectorizer(token_pattern = '[^a-zA-Z0-9]', tokenizer = lemm, max_df = 0.7)
cvtrain = cv.fit_transform(xtrain)
cvtest = cv.transform(xtest)

y_pred = clf.fit(cvtrain, ytrain).predict(cvtest)

mat = confusion_matrix(ytest, y_pred)
# смотрим результаты по матрицам (кто из персонажей лучше определяется и при каких условиях вектора)
#выше в векторе указаны наилучшие параметры из тех , что я перебрала
#-----------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#-----------------------------------------------------------------------------
#комментирую, чтобы лишний раз не запускать
#np.set_printoptions(precision=2)
#
#plt.figure()
#plot_confusion_matrix(mat, classes=['Stan', 'Kyle', 'Cartman', 'Kenny'],
#                      title='Confusion matrix for South Park')
#plt.figure()
#plot_confusion_matrix(mat1, classes=['Stan', 'Kyle', 'Cartman', 'Kenny'],
#                      title='Confusion matrix for South Park2')
#plt.show()
#------------------------------------------------------------------------------

print('Tree')
#print(classification_report(ytest, y_pred, target_names=['Stan', 'Kyle', 'Cartman', 'Kenny']))
print(accuracy_score(ytest, y_pred))

y_pred1 = rfc.fit(cvtrain, ytrain).predict(cvtest)
print('Forest')
#print(classification_report(ytest, y_pred1, target_names=['Stan', 'Kyle', 'Cartman', 'Kenny']))
print(accuracy_score(ytest, y_pred1))

lr = linear_model.LogisticRegression(C=1000, class_weight='balanced') 
y_pred2 = lr.fit(cvtrain, ytrain).predict(cvtest)
print('Regression')
#print(classification_report(ytest, y_pred2, target_names=['Stan', 'Kyle', 'Cartman', 'Kenny']))
print(accuracy_score(ytest, y_pred2))

mnb = MultinomialNB()
y_pred3 = mnb.fit(cvtrain, ytrain).predict(cvtest)

print('Bayes')
#print(classification_report(ytest, y_pred3, target_names=['Stan', 'Kyle', 'Cartman', 'Kenny']))
print(accuracy_score(ytest, y_pred3))
#------------------------------------------------------------------------------
# лучший результат дает Байес. В данном случае самый низкийрезультат у tree.
# полная тестовая выборка ниже
#------------------------------------------------------------------------------
#cv1train = cv.fit_transform(x_train)
cv1test = cv.transform(x_test)

y_pred_b = mnb.predict(cv1test)

print('Bayes best')
#print(classification_report(ytest, y_pred3, target_names=['Stan', 'Kyle', 'Cartman', 'Kenny']))
print(accuracy_score(y_test, y_pred_b))

#Baseline
print('Baseline')
dc = DummyClassifier()
cv_test = cv.fit_transform(x_train)
y_pred_b1 = dc.fit(cv1train, y_train).predict(cv1test)
print(accuracy_score(y_test, y_pred_b1))

#Байес значительно превосходит базовый классификатор по тестовой выборке


