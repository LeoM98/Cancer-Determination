from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from random import randrange
import sklearn.metrics as m
import pandas as pd
import numpy as np

dataSet = pd.read_csv("wdbc.txt",delimiter=',')#Valores delimitados por comas

X, y = dataSet.values[:,2:], dataSet.values[:,1] #Valores columna 2 y valores colunma 1
np.place(y, y=='M',1) #Remplazar lo que esta en Y como en M y se representa como 1
np.place(y, y=='B',0)
y=y.astype('int') #Conversion a enteros

#Train_split divide el dataset en subgrupo y hace pruebas al azar con sus parametros
#Toma el 30% de datos son para pruebas y el otro 70 de entrenamiento... Divide el dataset, randonizado
#Si no hay valor de prueba el default es .25
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0, stratify=y)
c = [randrange(1, 10000, 100)/float(1000) for i in range(100) ]#Se toman valores aleatorios y se dividen entre 1000, 100 veces
c = list(set(c))#Se crea una lista con los 100 valores que se tienen en c

#Regresión Logistica regularizada por liblinear
#Converjencia de solver (maxiter, defecto 100)
#fit(es necesario los datos de prueba y entrenamiento para el ajuste de estos)
Logis1 = [LogisticRegression(C=i, solver="lbfgs", max_iter=1000).fit(X_train,y_train) for i in c]
#Evaluacion de puntacion por validación cruzada
#X_train (datos de encaje. Y_train(Datos para aprendizaje supervisado))
#.mean (media aritmetica de la matriz)
#cv determina el numero de pliegues
cros = [cross_val_score(i,X_train,y_train,cv=10).mean() for i in Logis1]
#Inidice maximo de la validacion cruzada
LR = LogisticRegression(C=c[cros.index(max(cros))], solver="lbfgs", max_iter=1000).fit(X_train,y_train)
Model = LR.predict(X_test)#Se inicia a predecir con el 30% de los datos 

#Hallan cada una de las estadisticas con formulas implicitas
recall = m.recall_score(y_test,Model)
accuracy = m.accuracy_score(y_test,Model)
precision = m.precision_score(y_test,Model)
tn, fp, fn, tp = m.confusion_matrix(y_test,Model).ravel()#Calcula la matriz de confusion  para evaluar la precision


print("\n\t   MATRIZ DE CONFUSION")
print("         Negativos     Positivos")
print("Negativos   {0}           {1}".format(tn,fp))
print("Positivos  {0}            {1}".format(fn,tp))
print("Recall: {0}\nAccuracy: {1}\nPrecision: {2}".format(recall,accuracy,precision))
