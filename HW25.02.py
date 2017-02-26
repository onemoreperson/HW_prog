# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from sklearn import cross_validation
from sklearn import tree
from sklearn_pandas import cross_val_score
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

df  = pd.read_csv('titanic.csv', index_col = 'PassengerId')

#-----------------------------------3-------------------------------------
df = df.dropna() # удаляю пункты с Nan
columns = ["Pclass", "Fare", "Sex", "Age", "Survived"]
x = df[columns]
x.head()
df_sex = pd.get_dummies(x['Sex']) # для удобства создаются две новые колонки
df_new = pd.concat([df, df_sex], axis=1)
columns1 = ["Survived"]
y = df_new[columns1]
#---------------------------------1----------------------------------
new = df_new.groupby('Survived').sum()
f = new['female']
m = new['male']
FS, FNs = f[1], f[0]
MS, MNs = m[1], m[0]

#labels = 'FS', 'FNs', 'MS', 'MNs'# F - female, M - male, S - survived
#sizes = [FS, FNs, MS, MNs] 
#plt.pie(sizes, labels=labels, autopct='%1.1f%%',
#        shadow=True, startangle=90)
#plt.axis('equal') 
#plt.title('Num. 1.1') 
#plt.show()
#Вывод: из выживших большинство женщины(44,8%), их же и погибло меньше всего(3.3%);
#мужчинам пришлось труднее: разница между выжившими и нет всего 7.1% в пользу
#(если так можно сказать) вторых.

new = df_new.groupby(['Pclass']).sum()
s = new['Survived']
summ = new['male'] + new['female']
FS, SS, TS = s[1], s[2], s[3]
FNs = summ[1] - FS
SNs = summ[2] - SS
TNs = summ[3] - TS

#labels = 'FS', 'FNs', 'SS', 'SNs', 'TS', 'TNs' # F -first, S - second, T - third
#sizes = [FS, FNs, SS, SNs, TS, TNs]  
#plt.pie(sizes, labels=labels, autopct='%1.1f%%',
#        shadow=True, startangle=90)
#plt.axis('equal')
#plt.title('Num. 1.2')   
#plt.show()
#Вывод: Среди выживших превалирует первый класс. 


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
#Р(А|В) = 0.9, Р(С|В2) = 0.7
#----------------------------------4----------------------------------------
xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(x, y)
mass = []
#presort to True - предсортировка, мне показалось важным найти лучшие сплиты
for t in range(1,100):
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
                                     min_samples_leaf= 1, criterion= 'gini', 
                                     splitter= 'best', min_weight_fraction_leaf= 0.0, 
                                     min_samples_split= 2, max_depth= None, 
                                     presort= True, min_impurity_split= 1e-07)

    clf = clf.fit(xtrain, ytrain)
#    clf = clf.get_params(deep=True)
    clf1 = clf1.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)
    ypred1 = clf1.predict(xtest)
    mass.append(cross_val_score(ytest, ypred1), cv = 100)

print(classification_report(ytest, ypred))
print(classification_report(ytest, ypred1))


#false = (f1_score(ytest, ypred),recall_score(ytest, ypred),precision_score(ytest, ypred))
#true = (f1_score(ytest, ypred1), recall_score(ytest, ypred1),precision_score(ytest, ypred1))
#
#bar_width = 0.35
#error_config = {'ecolor': '0.3'}
#
#rects1 = plt.bar(np.arange(3), false, bar_width,
#                 alpha=0.4,
#                 color='b',
#                 error_kw=error_config,
#                 label='with presort = False')
#
#rects2 = plt.bar(np.arange(3) + bar_width, true, bar_width,
#                 alpha=0.4,
#                 color='r',
#                 error_kw=error_config,
#                 label='with presort = True')
#
#
#plt.xlabel('metrics')
#plt.ylabel('total')
#plt.title('Num. 4')
#plt.xticks(np.arange(3) + bar_width / 2, ('f1', 'recall', 'precision'))
#plt.legend()
#plt.plot(mass)
#plt.show()
# результаты улучшились, но не намного. Возможно, стоит поменять несколько пунктов.
