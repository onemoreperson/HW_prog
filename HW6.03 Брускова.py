# -*- coding: utf-8 -*-
from sklearn.dummy import DummyClassifier
#from sklearn import cross_validation
import pandas as pd
import numpy as np
import re
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree

tknzr = TweetTokenizer()
st = EnglishStemmer()
wnl = WordNetLemmatizer() 
gnb = GaussianNB()

db0 = pd.read_csv('SMSSpamCollection', sep='\t', names = ['A','B'])#вообще не сбалансирован
db0['A'] = db0['A'].map({'spam': 0, 'ham':1}).astype(int)
x1 = db0['A']
y1 = db0['B']
db0['C'] = range(len(x1))

#------------------------------------------------------------------------
#xtrain, xtest, ytrain, ytest = train_test_split(x1, y1)
#
#clf = DummyClassifier(constant=None, strategy='most_frequent',random_state=0,)
#clf.fit(xtrain, ytrain)
#a = clf.score(xtest, ytest) 
#
#print(a)

#------------------------------------------------------------------------

idx = 0   #деление выборки на равные части
for item in x1:
    idx += 1
num = [idx for idx, item in enumerate(x1) if item == 0]
num1 = [idx for idx, item in enumerate(x1) if item == 1]
summ = [i for i in num[:740]] + [i1 for i1 in num1[:740]]

db = pd.DataFrame(db0, index = sorted(summ))# по 50% спама и хама    
x = db['A']
y = db['B']


####для стемов и лемматизации
#def lem(y): 
#    y = re.sub(r'\s+', ' ', str(y))
##    y = y.lower()
#    tok = tknzr.tokenize(y)
#    lem = [wnl.lemmatize(i) for i in tok]
#    return( " ".join(lem)) 
#n = y.size
#wer = [lem(y[i]) for i in sorted(summ)]

#def ste(y):  
#    y = re.sub(r'\s+', ' ', str(y))
##    y = y.lower()
#    tok = tknzr.tokenize(y)
#    ste = [st.stem(i) for i in tok]
#    return( " ".join(ste))   
#n = y.size
#wer = [ste(y[i]) for i in sorted(summ)]
#
#db['C'] = wer #  снять при выборе одной из 2х предыдущих функций, чтобы у У учитывались изменения
#y = db['C']

xtrain, xtest, ytrain, ytest = train_test_split(x, y)
vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None)    
#                             stop_words = 'english', 
#                             token_pattern = '[^a-zA-Z0-9]')
#                             min_df = 0.3) 
vectorizer.fit_transform(y)
vocab = vectorizer.get_feature_names()
data = vectorizer.transform(y)

naive_model = MultinomialNB()
naive_model.fit(data, db['A'])
cv_results = cross_val_score(naive_model, data, db['A'], cv=10, scoring='accuracy')
print(cv_results.mean(), cv_results.std())
#
transformer = TfidfVectorizer(analyzer = "word",                             
                                tokenizer = None)    
#                                stop_words = 'english', 
#                                token_pattern = '[^a-zA-Z0-9]')
#                                min_df = 0.3)
vectorizer.fit_transform(y)
vocab = vectorizer.get_feature_names()
data1 = vectorizer.transform(y)
naive_model = MultinomialNB()
naive_model.fit(data1, db['A'])
cv_results = cross_val_score(naive_model, data1, db['A'], cv=10, scoring='accuracy')
print(cv_results.mean(), cv_results.std())

# результаты по векторам: count: 0.958108108108(первый) 0.0141085223093(второй)
#tfidf 0.958108108108 0.0141085223093
# stop_words - влияния не обнаружено.
#  token_pattern - влияет на числа. В первом уменьшает на 0.15, во втором увеличивает на 0.005.
# min_df на 0.4 хуже в первом, на 0.2 лучше во втором.
# max_df - влияния не обнаружено.
# отсутствие лемматизации и стемминга - высокий результат в 1 случае, маленький во втором
#lemmatize, stem серьезных различий нет, ниже исходного уровня. Лемматицация дает незначительно больший результат.

#------------------------------2-----------------------------------------

#for t in range(1,100):
#    clf= tree.DecisionTreeClassifier(random_state= None, max_features= None, 
#                                     class_weight= None, 
#                                     max_leaf_nodes= None, 
#                                     min_samples_leaf= 1, criterion= 'gini', 
#                                     splitter= 'best', min_weight_fraction_leaf= 0.0, 
#                                     min_samples_split= 2, max_depth= None, 
#                                     presort= False, min_impurity_split= 1e-07)
#
#    clf = clf.fit(xtrain, ytrain)
##    clf = clf.get_params(deep=True)
#
#    ypred = clf.predict(xtest)
#cv_results = cross_val_score(clf, clf, db["A"], cv=10, scoring='accuracy')
#print(cv_results.mean(), cv_results.std())
