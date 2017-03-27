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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk import word_tokenize, wordpunct_tokenize
from sklearn.ensemble import RandomForestClassifier

tknzr = TweetTokenizer()
st = EnglishStemmer()
wnl = WordNetLemmatizer() 
gnb = GaussianNB()

db0 = pd.read_csv('SMSSpamCollection', sep='\t', names = ['A','B'])#вообще не сбалансирован
db0['A'] = db0['A'].map({'spam': 0, 'ham':1}).astype(int)
x1 = db0['A']
y1 = db0['B']


a = db0[db0['A']==0]
b = db0[db0['A']==1]

summ = [i for i in a['C'][:300]] + [i1 for i1 in b['C'][:300]]# делаем сбалансированную выборку
#в нее войдет равное кол-во примеров
db = pd.DataFrame(db0, index = sorted(summ))# по 50% спама и хама благодаря массиву 

x = db['B']
y = db['A']


db = pd.DataFrame(db0, index = sorted(summ))# по 50% спама и хама    
x = db['A']
y = db['B']


####для стемов и лемматизации в векторах, здесь все шло методом перебора, поэтому результаты внизу
def lem(y): 
    y = re.sub(r'\s+', ' ', str(y))
##    y = y.lower()
    tok = tknzr.tokenize(y)
    lem = [wnl.lemmatize(i) for i in tok]
    return( " ".join(lem)) 


def ste(y):  
    y = re.sub(r'\s+', ' ', str(y))
##    y = y.lower()
    tok = tknzr.tokenize(y)
    ste = [st.stem(i) for i in tok]
    return( " ".join(ste))   


xtrain, xtest, ytrain, ytest = train_test_split(list(x), np.array(y))
clf = CountVectorizer(analyzer = "word",   
                             tokenizer = None,   
#                             stop_words = 'english', 
#                             token_pattern = '[^a-zA-Z0-9]',
                             max_df = 1.0) 

transformer = TfidfVectorizer(analyzer = "word",                             
                                tokenizer = None,    
#                                stop_words = 'english', 
#                                token_pattern = '[^a-zA-Z0-9]',
                                max_df = 1.0)

vectorizer = clf.fit_transform(xtrain)
data = clf.transform(xtest)

naive_model = MultinomialNB()
cv_results = naive_model.fit(vectorizer, ytrain).predict(data)

#посмотрим результаты по метрикам разных векторов
print('Bayes CV')
print(classification_report(ytest, cv_results, target_names=['S', 'H']))

vec = transformer.fit_transform(xtrain)
data1 = transformer.transform(xtest)
cv_results1 = naive_model.fit(vec, ytrain).predict(data1)
print('Bayes TF')
print(classification_report(ytest, cv_results1, target_names=['S', 'H']))
# 
#в большинстве случаев тфидф работает хуже, только лучший результат вышел одинаково на обоих векторах
#самые высокие результаты дает параметр глубины max_df при значении 0.5 (высшие из всех испробованных)
#лемматизация и стемизация мало отличаются друг от друга (токенизация уже включена в ф-ции)
#если считать знаки препинания отдельными токенами (wordpunkt и token_pattern были использованы) - результат выше
#стоп слова мало что меняют, по крайней мере встроенные
#
#
#------------------------------2-----------------------------------------


tr= tree.DecisionTreeClassifier(random_state= None, max_features= None, 
                                     class_weight= None, 
                                     max_leaf_nodes= None, 
                                     min_samples_leaf= 1, criterion= 'gini', 
                                     splitter= 'best', min_weight_fraction_leaf= 0.0, 
                                     min_samples_split= 2, max_depth= None, 
                                     presort= False, min_impurity_split= 1e-07)

cv_results3 = tr.fit(vectorizer, ytrain).predict(data)
print('tree CV')
print(classification_report(ytest, cv_results3, target_names=['S', 'H']))

print('tree TF')
cv_results4 = tr.fit(vec, ytrain).predict(data1)
print(classification_report(ytest, cv_results4, target_names=['S', 'H']))


rfc = RandomForestClassifier(bootstrap=True, min_impurity_split=1e-07, 
                                 n_estimators=10, verbose=0,
                                 max_leaf_nodes= None, oob_score=False, 
                                 min_samples_leaf=1,
                                 class_weight=None, max_features='auto', 
                                 max_depth=None, min_samples_split=2,
                                 random_state=None, 
                                 min_weight_fraction_leaf=0.0, warm_start=False,
                                 criterion='gini', n_jobs=1)

cv_results5 = rfc.fit(vectorizer, ytrain).predict(data)
print('tree CV')
print(classification_report(ytest, cv_results5, target_names=['S', 'H']))

print('tree TF')
cv_results6 = rfc.fit(vec, ytrain).predict(data1)
print(classification_report(ytest, cv_results6, target_names=['S', 'H']))

# лучший вариант дает байесовский классификатор при CountVectorize
