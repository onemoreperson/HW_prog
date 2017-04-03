# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import tree
from sklearn_pandas import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus


df  = pd.read_csv('titanic.csv', index_col = 'PassengerId')

#-----------------------------------3-------------------------------------
df = df.dropna() # удаляю пункты с Nan
columns = ["Pclass", "Fare", "Sex", "Age", "Survived"]
columns2 = ["Pclass", "Fare", "male", 'female', "Age"]#уже без класса выживших
x0 = df[columns]
x0.head()
df_sex = pd.get_dummies(x0['Sex']) # для удобства создаются две новые колонки
df_new = pd.concat([df, df_sex], axis=1)

x = df[columns2]
x['Sex'] = x['Sex'].map({'female': 0, 'male':1}).astype(int) #  для заданий 4 и 5

col = ["Survived"]
y = df_new[col]
#---------------------------------1----------------------------------
#здесь закомментированно, чтобы не запускать
new = df_new.groupby('Survived').sum()
f = new['female']
m = new['male']
FS, FNs = f[1], f[0]
MS, MNs = m[1], m[0]

#fig, axes = plt.subplots(nrows=2, ncols=2)
#ax0, ax1, ax2, ax3 = axes.flatten()
#labels1 = 'FS', 'FNs'# F - female, M - male, S - survived
#sizes1 = [FS, FNs] 
#labels2 = 'MS', 'MNs'# F - female, M - male, S - survived
#sizes2 = [MS, MNs]
#labels3 = 'F', 'M'
#sizes3 = [f[1], m[1]]
#labels4 = 'F', 'M'
#sizes4 = [f[0], m[0]]
#ax0.pie(sizes1, labels=labels1, autopct='%1.1f%%',
#        shadow=True, startangle=90)
#ax0.axis('equal')
#ax0.set_title('Выживание среди женщин')
#ax1.pie(sizes2, labels=labels2, autopct='%1.1f%%',
#        shadow=True, startangle=90)
#ax1.axis('equal')
#ax1.set_title('Выживание среди мужчин')
#ax2.pie(sizes3, labels=labels3, autopct='%1.1f%%',
#        shadow=True, startangle=90)
#ax2.axis('equal')
#ax2.set_title('Выживание среди женщин и мужчин')
#ax3.pie(sizes4, labels=labels4, autopct='%1.1f%%',
#        shadow=True, startangle=90)
#ax3.axis('equal')
#ax3.set_title('Невыживание среди женщин и мужчин')
#plt.axis('equal')
#plt.show()

#Вывод: из выживших большинство женщины, их же и погибло меньше всего;
#мужчинам пришлось труднее, выжило меньше половины.

new = df_new.groupby(['Pclass']).sum()
s = new['Survived']
summ = new['male'] + new['female']
FS, SS, TS = s[1], s[2], s[3]
FNs = summ[1] - FS
SNs = summ[2] - SS
TNs = summ[3] - TS

#fig, axes = plt.subplots(nrows=2, ncols=2)
#ax0, ax1, ax2, ax3 = axes.flatten()
#labels1 = 'FS', 'FNs'# F - female, M - male, S - survived
#sizes1 = [FS, FNs] 
#labels2 = 'SS', 'SNs'# F - female, M - male, S - survived
#sizes2 = [SS, SNs]
#labels3 = 'TS', 'TNs'
#sizes3 = [TS, TNs]
#labels4 = '1', '2', '3'
#sizes4 = [FS, SS, TS]
#ax0.pie(sizes1, labels=labels1, autopct='%1.1f%%',
#        shadow=True, startangle=90)
#ax0.axis('equal')
#ax0.set_title('Выживание среди 1 класса')
#ax1.pie(sizes2, labels=labels2, autopct='%1.1f%%',
#        shadow=True, startangle=90)
#ax1.axis('equal')
#ax1.set_title('Выживание среди 2 класса')
#ax2.pie(sizes3, labels=labels3, autopct='%1.1f%%',
#        shadow=True, startangle=90)
#ax2.axis('equal')
#ax2.set_title('Выживание среди 3 класса')
#ax3.pie(sizes4, labels=labels4, autopct='%1.1f%%',
#        shadow=True, startangle=90)
#ax3.axis('equal')
#ax3.set_title('Выжившие по классам')
#plt.axis('equal')
#plt.show()

#Вывод: Среди выживших превалирует первый класс. Выжила большая часть второго
#и половина первого. 


mass = []
new = x.groupby(['Pclass']).mean()
s = new['Fare']
cl1, cl2, cl3 = s[1], s[2], s[3]
mass.append(cl1)
mass.append(cl2)
mass.append(cl3)

#labels = ['1', '2', '3']
#x = [1, 2, 3]
#plt.plot(x, mass, 'ro')
#plt.xticks(x, labels, rotation='horizontal')
#plt.xlabel('Class')
#plt.ylabel('$')
#plt.title('Средняя стоимость билета')
#plt.show()
#Вывод: В данном случае я решила, что будет наиболее показательна средняя стоимость 
#(и без графика ясно, что между самым дешевым и дорогим билетами будет огромная разница).
#Второй класс хоть и дороже первого, но не намного. А вот первому классу явно жилось лучше
#с учетом разницы примерно в 70 долларов со вторым классом.
#-----------------------------------2-----------------------------------------
new = df_new.groupby(["male", 'Pclass']).mean()#среднее по выжившим
#new = df_new.groupby(["male", 'Pclass']).sum()#по кол-ву выживших
s = new["Survived"]
f = s[0]
m = s[1]

men = (m[1], m[2], m[3])
women = (f[1], f[2], f[3])
bar_width = 0.35
error_config = {'ecolor': '0.3'}

#rects1 = plt.bar(np.arange(3), men, bar_width,
#                 alpha=0.4,
#                 color='b',
#                 error_kw=error_config,
#                 label='Male')
#
#rects2 = plt.bar(np.arange(3) + bar_width, women, bar_width,
#                 alpha=0.4,
#                 color='r',
#                 error_kw=error_config,
#                 label='Female')
#
#plt.xlabel('Class')
#plt.ylabel('probability')
#plt.title('Num. 2')
#plt.xticks(np.arange(3) + bar_width / 2, ('1', '2', '3'))
#plt.legend()
#
#plt.tight_layout()
#plt.show()
#Вывод: наибольшая верятность выжить у женщины перого класса. Наименьшая у мужчины 
#третьего. Что интересно, у мужчины 2 класса вероятность выжить неожиданно больше,
#чем у женщины третьего класса.
#А - женщина, В1 - первый класс, С - мужчина, 
#Р(А|В1) = 0.9 (вероятность выжить женщине 1 класса), Р(С|В2) = 0.7
#(вероятность выжить мужчине 2 класса)
#----------------------------------4----------------------------------------
xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(x, y)

#presort to True - предсортировка, мне показалось важным найти лучшие сплиты
clf= tree.DecisionTreeClassifier(random_state= None, max_features= None, 
                                 class_weight= None, 
                                 max_leaf_nodes= None, 
                                 min_samples_leaf= 1, criterion= 'gini', 
                                 splitter= 'best', min_weight_fraction_leaf= 0.0, 
                                 min_samples_split= 2, max_depth= None, 
                                 presort= False, min_impurity_split= 1e-07)
        
clf1= tree.DecisionTreeClassifier(random_state= None, max_features= None, 
                                  class_weight= None, 
                                  max_leaf_nodes= None, 
                                  min_samples_leaf= 1, criterion= 'entropy', 
                                  splitter= 'best', min_weight_fraction_leaf= 0.0, 
                                  min_samples_split= 2, max_depth= None, 
                                  presort= True, min_impurity_split= 1e-07)

clf = clf.fit(xtrain, ytrain)

clf1 = clf1.fit(xtrain, ytrain)
ypred = clf.predict(xtest)
ypred1 = clf1.predict(xtest)
#
#print(classification_report(ytest, ypred))
#print(classification_report(ytest, ypred1))

#
#fir = (f1_score(ytest, ypred),recall_score(ytest, ypred),precision_score(ytest, ypred))
#sec = (f1_score(ytest, ypred1), recall_score(ytest, ypred1),precision_score(ytest, ypred1))

#bar_width = 0.35
#error_config = {'ecolor': '0.3'}
#
#rects1 = plt.bar(np.arange(3), fir, bar_width,
#                 alpha=0.4,
#                 color='b',
#                 error_kw=error_config,
#                 label='with presort = False')
#
#rects2 = plt.bar(np.arange(3) + bar_width, sec, bar_width,
#                 alpha=0.4,
#                 color='r',
#                 error_kw=error_config,
#                 label='with presort = True')
#
#
#plt.xlabel('metrics')
#plt.ylabel('total')
#plt.title('Результаты по деревьям')
#plt.xticks(np.arange(3) + bar_width / 2, ('f1', 'recall', 'precision'))
#plt.legend()

# результаты улучшились, но не намного. Возможно, стоит поменять несколько пунктов.
dot_data = tree.export_graphviz(clf, out_file=tree.dot) 
#graph = pydotplus.graph_from_dot_data(dot_data)  
#Image(graph.create_png())  #чтобы не запускалось
#---------------------------------5------------------------------------------

# criterion to entropy. для расширения split area.   

#rfc = RandomForestClassifier(bootstrap=True, min_impurity_split=1e-07, 
#                                 n_estimators=10, verbose=0,
#                                 max_leaf_nodes= None, oob_score=False, 
#                                 min_samples_leaf=1,
#                                 class_weight=None, max_features='auto', 
#                                 max_depth=None, min_samples_split=2,
#                                 random_state=None, 
#                                 min_weight_fraction_leaf=0.0, warm_start=False,
#                                 criterion='gini', n_jobs=1)
#rfc1 = RandomForestClassifier(bootstrap=True, min_impurity_split=1e-07, 
#                                 n_estimators=10, verbose=0,
#                                 max_leaf_nodes= None, oob_score=False, 
#                                 min_samples_leaf=1,
#                                 class_weight=None, max_features='auto', 
#                                 max_depth=None, min_samples_split=2,
#                                 random_state=None, 
#                                 min_weight_fraction_leaf=0.0, warm_start=False,
#                                 criterion='entropy', n_jobs=1)
#
#rfc = rfc.fit(xtrain, ytrain)
#ypred = rfc.predict(xtest)
#rfc1 = rfc1.fit(xtrain, ytrain)
#ypred1 = rfc1.predict(xtest)
#print(classification_report(ytest, ypred))
#print(classification_report(ytest, ypred1))
#gini = (f1_score(ytest, ypred),recall_score(ytest, ypred),precision_score(ytest, ypred))
#entropy = (f1_score(ytest, ypred1), recall_score(ytest, ypred1),precision_score(ytest, ypred1))

#bar_width = 0.35
#error_config = {'ecolor': '0.3'}
#
#rects1 = plt.bar(np.arange(3), gini, bar_width,
#                 alpha=0.4,
#                 color='b',
#                 error_kw=error_config,
#                 label='with presort = False')
#
#rects2 = plt.bar(np.arange(3) + bar_width, entropy, bar_width,
#                 alpha=0.4,
#                 color='r',
#                 error_kw=error_config,
#                 label='with presort = True')
#
#
#plt.xlabel('metrics')
#plt.ylabel('total')
#plt.title('Результаты по лесам')
#plt.xticks(np.arange(3) + bar_width / 2, ('f1', 'recall', 'precision'))
#plt.legend()

# в обоих случаях сильного повышения результатов во всех метриках не обнаружилось
