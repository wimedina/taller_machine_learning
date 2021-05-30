# -*- coding: utf-8 -*-
"""
Created on Sat May 29 21:14:59 2021

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

url = 'bank-full.csv'
data = pd.read_csv(url)
# Observamos cual es la persona de menor edad
data.age.min()
# Observamos cual es la persona d mayor edad
data.age.max()
# Preprocesamiento de los datos
rangos = [18, 20, 23, 28, 35, 45, 55, 65, 75, 85, 95]
nombres = ['1','2','3','4','5','6','7','8', '9', '10']
data['age'] = pd.cut(data.age, rangos, labels = nombres)
data['job'].replace(['admin.','blue-collar','entrepreneur','housemaid','management','retired',
                     'self-employed','services','student','technician','unemployed','unknown'], 
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace = True)
data['marital'].replace(['married','divorced','single'], [0, 1, 2], inplace = True)
data['education'].replace(['primary','secondary','tertiary','unknown'], [0, 1, 2, 3], inplace = True)
data['contact'].replace(['cellular','telephone','unknown'], [0, 1, 2], inplace = True)
data['loan'].replace(['yes','no'], [0, 1], inplace = True)
data['pdays'].replace(-1, 0, inplace = True)
data['housing'].replace(['yes','no'], [0, 1], inplace = True)
data['default'].replace(['yes','no'], [0, 1], inplace = True)
data['month'].replace(['jan','feb','mar','apr','may','jun',
                     'jul','aug','sep','oct','nov','dec'], 
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace = True)
data['poutcome'].replace(['success','failure','other','unknown'], [0, 1, 2, 3], inplace = True)
data['y'].replace(['yes','no'], [0, 1], inplace = True)
#data.drop(['day','month','duration','pdays','previous'], axis = 1, inplace=True)
data.drop('balance', axis = 1, inplace=True)
data.dropna(axis=0, how='any', inplace=True)

# Aplicacion de los modelos 

# Parto Dataset en dos, uno para todo lo referente a entrenamiento y otro para datos que
# nunca vea el modelo solo hasta validar 
data_train = data[:30000]
data_test = data[30000:]

# Separar la data menos la columna categorica Survived
# X y Y de la primera parte del Dataset
x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y) # 0 Murió #1 Vivió

# x_train, x_test, y_train, y_test de la primera parte del dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# Saco un X y un Y de la otra parte del dataset 
x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)

def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix,);
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print (classification_report(y_test, pred_y))
    
    
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
#Entrenamiento de los modelos

# Regresion Logistica con validacion cruzada
logreg = LogisticRegression(solver='lbfgs', max_iter=7600)
kfold = KFold(n_splits=10)
cvscores = []
for train,test in kfold.split(x_train, y_train):
    logreg.fit(x_train[train], y_train[train])
    scores = logreg.score(x_train[test], y_train[test]) 
    cvscores.append(scores)
#print(np.mean(cvscores))

print('Regresión logística validacion cruzada')
# Accuracy de entrenamiento
acTrain = logreg.score(x_train, y_train)
#print(f'Accuracy de entrenamiento: {logreg.score(x_train, y_train)}') 
# Accuracy de test
acTest = logreg.score(x_test, y_test)
#print(f'Accuracy de Test: {logreg.score(x_test, y_test)}') 
y_pred = logreg.predict(x_test_out)
# Accuracy de Validacion
acValida = accuracy_score(y_pred, y_test_out)
series = pd.Series([acTrain, acValida, acTest],
index = ['Accuracy de entrenamiento', 'Accuracy de validacion', 'Accuracy de test'])
print(series)
#print(f'Accuracy de Validacion: {accuracy_score(y_pred, y_test_out)}') 
print('Matriz de Confusión') 
print(confusion_matrix(y_test_out, y_pred))
mostrar_resultados(y_test_out, y_pred)
print('*' * 50)
probs = logreg.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test_out, probs)
plot_roc_curve(fpr, tpr)


#Maquinas de soporte vectorial
print('Máquina de soporte vectorial')
# svc = SVC(gamma = 'auto')
# svc.fit(x_train,y_train)

# print(f'Accuracy de entrenamiento: {svc.score(x_train, y_train)}') 
# print(f'Accuracy de Test: {svc.score(x_test, y_test)}') 
# print('*' * 50)


print('Maquina de soporte vectorial con validacion cruzada')

svc = SVC(gamma = 'auto')
kfold = KFold(n_splits=10)
cvscores = []
for train,test in kfold.split(x_train, y_train):
    svc.fit(x_train[train], y_train[train])
    scores = svc.score(x_train[test], y_train[test]) 
    cvscores.append(scores)
#print(np.mean(cvscores))


# Accuracy de entrenamiento
acTrain = svc.score(x_train, y_train)
#print(f'Accuracy de entrenamiento: {svc.score(x_train, y_train)}') 
# Accuracy de test
acTest = svc.score(x_test, y_test)
#print(f'Accuracy de Test: {svc.score(x_test, y_test)}') 
y_pred = svc.predict(x_test_out)
acValida = accuracy_score(y_pred, y_test_out)
series = pd.Series([acTrain, acValida, acTest],
index = ['Accuracy de entrenamiento', 'Accuracy de validacion', 'Accuracy de test'])
print(series)
#print(f'Accuracy de Validacion: {accuracy_score(y_pred, y_test_out)}') 
print('Matriz de Confusión') 
print(confusion_matrix(y_test_out, y_pred))
mostrar_resultados(y_test_out, y_pred)
print('*' * 50)
probs = svc.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test_out, probs)
plot_roc_curve(fpr, tpr)



#Vecinos mas cercanos clasificado

print('Clasificador Vecinos mas cercanos')
knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(x_train, y_train)
# svc.fit(x_train,y_train)
kfold = KFold(n_splits=10)
cvscores = []
for train,test in kfold.split(x_train, y_train):
    knn.fit(x_train[train], y_train[train])
    scores = knn.score(x_train[test], y_train[test]) 
    cvscores.append(scores)
#print(np.mean(cvscores))

acTrain = knn.score(x_train, y_train)
acTest = knn.score(x_test, y_test)
y_pred = knn.predict(x_test_out)
acValida = accuracy_score(y_pred, y_test_out)
series = pd.Series([acTrain, acValida, acTest],
index = ['Accuracy de entrenamiento', 'Accuracy de validacion', 'Accuracy de test'])
print(series)
print('Matriz de Confusión') 
print(confusion_matrix(y_test_out, y_pred))
mostrar_resultados(y_test_out, y_pred)
print('*' * 50)
probs = knn.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test_out, probs)
plot_roc_curve(fpr, tpr)
# print(f'Accuracy de entrenamiento: {knn.score(x_train, y_train)}') 
# print(f'Accuracy de Test: {knn.score(x_test, y_test)}') 
print('*' * 50)

# Gaussian Naive Bayes
print('Clasificador Naive Bayes Gaussiano')
clf = GaussianNB()
kfold = KFold(n_splits=10)
cvscores = []
for train,test in kfold.split(x_train, y_train):
    clf.fit(x_train[train], y_train[train])
    scores = clf.score(x_train[test], y_train[test]) 
    cvscores.append(scores)
    
acTrain = clf.score(x_train, y_train)
acTest = clf.score(x_test, y_test)
y_pred = clf.predict(x_test_out)
acValida = accuracy_score(y_pred, y_test_out)
series = pd.Series([acTrain, acValida, acTest],
index = ['Accuracy de entrenamiento', 'Accuracy de validacion', 'Accuracy de test'])
print(series)

print('Matriz de Confusión') 
print(confusion_matrix(y_test_out, y_pred))
mostrar_resultados(y_test_out, y_pred)
print('*' * 50)
probs = clf.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test_out, probs)
plot_roc_curve(fpr, tpr)
print('*' * 50)


# Multinomial Naive Bayes
multiNB = MultinomialNB()
kfold = KFold(n_splits=10)
cvscores = []
for train,test in kfold.split(x_train, y_train):
    multiNB.fit(x_train[train], y_train[train])
    scores = multiNB.score(x_train[test], y_train[test]) 
    cvscores.append(scores)
    
acTrain = multiNB.score(x_train, y_train)
acTest = multiNB.score(x_test, y_test)
y_pred = multiNB.predict(x_test_out)
acValida = accuracy_score(y_pred, y_test_out)
series = pd.Series([acTrain, acValida, acTest],
index = ['Accuracy de entrenamiento', 'Accuracy de validacion', 'Accuracy de test'])
print(series)

print('Matriz de Confusión') 
print(confusion_matrix(y_test_out, y_pred))
mostrar_resultados(y_test_out, y_pred)
print('*' * 50)
probs = multiNB.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test_out, probs)
plot_roc_curve(fpr, tpr)
print('*' * 50)





