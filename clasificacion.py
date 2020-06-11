# -*- coding: utf-8 -*-

# CLASIFICACIÓN: OPTICAL RECOGNITION OF HANDWRITTEN DIGITS

import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import zero_one_loss, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Fijamos la semilla
np.random.seed(1)

# -------------------- FUNCIONES --------------------

# Leer un fichero con todos los datos disponibles
def readRawFile(file):
    # Leemos las filas del archivo
    with open(file) as f:
        contenido = f.readlines()

    contenido = [c.strip() for c in contenido]
    contenido = [c.split(',') for c in contenido]

    for i in range(5):
        del contenido[0]

    return contenido

# Separar etiquetas de los atributos
def extractLabels(data, index):
    labels = [item.pop(index) for item in data]
    
    return data, labels

def dataToFloat(data):
    # Pasamos los atributos a decimal
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j]) 

    return np.array(data)

def removeAtribute(data, index):
    for i in range(len(data)):
        del data[i][index]

def encodeLabels(labels, categories = []):
    # Buscamos todos los tipos de categorias que hay
    for i in range(len(labels)):
        if labels[i] not in categories:
            categories.append(labels[i])
        labels[i] = categories.index(labels[i])
            
    return np.array(labels), categories
    
# Función para normalizar los datos
def normalizar(data_tra, data_tes):
    scaler = preprocessing.MinMaxScaler()
    normalized_tra = scaler.fit_transform(data_tra)
    normalized_tes = scaler.transform(data_tes)
    
    return normalized_tra, normalized_tes

# Función para reducir dimensionalidad
def reducir(data_tra,data_tes):
    reduc = PCA(0.95)
    reduced_tra = reduc.fit_transform(data_tra)
    reduced_tes = reduc.transform(data_tes)
    return reduced_tra, reduced_tes

# Función para tratar outliers
def tratar_outliers(datax,datay):
    clf = LocalOutlierFactor(contamination='auto')
    outliers = clf.fit_predict(datax,datay)
    return datax[outliers==1], datay[outliers==1]

# Función para aplicar transformaciones polinomiales
def transformar_polinomial(data,exponente):
    poly = preprocessing.PolynomialFeatures(exponente)
    return poly.fit_transform(data)

# ------------------ CLASIFICACIÓN ------------------
    
print("--------- CLASIFICACIÓN ---------")

# Lectura de datos
print("\nLeyendo datos...")

x_train = readRawFile("datos/segmentation.tes")
x_test = readRawFile("datos/segmentation.tra")

x_train, y_train = extractLabels(x_train,0)
x_test, y_test = extractLabels(x_test,0)

dataToFloat(x_train)
dataToFloat(x_test)

removeAtribute(x_train, 2)
removeAtribute(x_test, 2)

y_train, cat = encodeLabels(y_train)
y_test, cat = encodeLabels(y_test, categories = cat)

# Preprocesado
print("Preprocesando datos...")

x_train, x_test = normalizar(x_train, x_test)

x_train, y_train = tratar_outliers(x_train,y_train)
x_test, y_test = tratar_outliers(x_test,y_test)

#x_train, x_test = reducir(x_train, x_test)

# Complementación clase de funciones
print("Calculando clase de funciones...")

x_train_sq = transformar_polinomial(x_train,2)
x_test_sq = transformar_polinomial(x_test,2)

#----------------------------------------------------------
#                 Modelo Lineal
#----------------------------------------------------------

# Validación cruzada
print("\n Modelos lineales: ")
#print("Ejecutando validación cruzada...\n")

a = (0.0001, 0.001, 0.01, 0.1)

parameters = {'loss' : ('log', 'perceptron'), 'alpha' : a}

linear_model = SGDClassifier()

sg = GridSearchCV(linear_model, parameters, n_jobs = 2, scoring = 'accuracy', cv = 5, iid = False)

print("\nEntrenando el modelo...")

model = sg.fit(x_train,y_train)
Eval = 1-model.best_score_.mean()

print("\n Mejores parametros: ",model.best_params_.items())
print("\nEval (según CV): ",Eval)

print("\nEin: ",zero_one_loss(y_train,model.predict(x_train)))
print("\n",confusion_matrix(y_train,model.predict(x_train)))
print("\nEtest: ",zero_one_loss(y_test,model.predict(x_test)))
print("\n",confusion_matrix(y_test,model.predict(x_test)))

input("\nPulse una tecla para continuar\n")

#----------------------------------------------------------
#           Perceptron Multicapa
#----------------------------------------------------------
print("\n Perceptron Multicapa: ")

nd = [(i,) for i in range(50,101,10)]
a = (0.001, 0.01, 0.1)
s = ('lbfgs', 'adam')

parameters = {'hidden_layer_sizes' : nd, 'alpha' : a, 'solver' : s}

mlp = MLPClassifier(max_iter = 500, activation = 'identity')

sg = GridSearchCV(mlp, parameters, n_jobs = 4, scoring = 'accuracy', cv = 5, iid = False)

print("\nEntrenando el modelo...")

model = sg.fit(x_train,y_train)
Eval = 1-model.best_score_.mean()

print("\n Mejores parametros: ",model.best_params_.items())
print("\nEval(según CV): ",Eval)
print("\nEin: ",zero_one_loss(y_train,model.predict(x_train)))
print("\n",confusion_matrix(y_train,model.predict(x_train)))
print("\nEtest: ",zero_one_loss(y_test,model.predict(x_test)))
print("\n",confusion_matrix(y_test,model.predict(x_test)))

input("\nPulse una tecla para continuar\n")

# #----------------------------------------------------------
# #                   Boosting
# #----------------------------------------------------------
print("\n Boosting: ")

ne = [150]

parameters = {'n_estimators' : ne}

gbc = GradientBoostingClassifier()

sg = GridSearchCV(gbc, parameters, n_jobs = 4, scoring = 'accuracy', cv = 5, iid = False)

print("\nEntrenando el modelo...")

model = sg.fit(x_train,y_train)
Eval = 1-model.best_score_.mean()

print("\n Mejores parametros: ",model.best_params_.items())
print("\nEval(según CV): ",Eval)
print("\nEin: ",zero_one_loss(y_train,model.predict(x_train)))
print("\n",confusion_matrix(y_train,model.predict(x_train)))
print("\nEtest: ",zero_one_loss(y_test,model.predict(x_test)))
print("\n",confusion_matrix(y_test,model.predict(x_test)))

input("\nPulse una tecla para continuar\n")

# #----------------------------------------------------------
# #                 Random Forest
# #----------------------------------------------------------
print("\n Random Forest: ")

ne = [100]
c = ['gini', 'entropy']
mf = ['auto', 'log2', 0.6, 0.7, 0.8, 0.9]
b = [True, False]

parameters = {'n_estimators' : ne, 'criterion' : c, 'max_features' : mf, 'bootstrap' : b}

rforest = RandomForestClassifier()

sg = GridSearchCV(rforest, parameters, n_jobs = 4, scoring = 'accuracy', cv = 5, iid = False)

print("\nEntrenando el modelo...")

model = sg.fit(x_train,y_train)
Eval = 1-model.best_score_.mean()

print("\n Mejores parametros: ",model.best_params_.items())
print("\nEval(según CV): ",Eval)
print("\nEin: ",zero_one_loss(y_train,model.predict(x_train)))
print("\n",confusion_matrix(y_train,model.predict(x_train)))
print("\nEtest: ",zero_one_loss(y_test,model.predict(x_test)))
print("\n",confusion_matrix(y_test,model.predict(x_test)))

input("\nPulse una tecla para terminar\n")

