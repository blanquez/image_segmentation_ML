# -*- coding: utf-8 -*-

# CLASIFICACIÓN: OPTICAL RECOGNITION OF HANDWRITTEN DIGITS

import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import zero_one_loss, confusion_matrix

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
    reduc = PCA(0.99)
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

x_train = readRawFile("datos/segmentation.tra")
x_test = readRawFile("datos/segmentation.tes")

x_train, y_train = extractLabels(x_train,0)
x_test, y_test = extractLabels(x_test,0)

dataToFloat(x_train)
dataToFloat(x_test)

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
print("\nModelos lineales: ")
print("Ejecutando validación cruzada...\n")

log_reg = SGDClassifier(loss="log", penalty = 'l1')
percep = SGDClassifier(loss="perceptron", penalty = 'l1')

score_log = cross_val_score(log_reg, x_train, y_train, cv=5)
print("R. Logaritmica: ",1-score_log.mean())
score_log_sq = cross_val_score(log_reg,x_train_sq,y_train,cv=5)
print("R. Logaritmica sq:",1-score_log_sq.mean())
score_per = cross_val_score(percep,x_train,y_train,cv=5)
print("Perceptron: ",1-score_per.mean())
score_per_sq = cross_val_score(percep,x_train_sq,y_train,cv=5)
print("Perceptron sq: ",1-score_per_sq.mean())

# Entrenamiento
print("\nModelo elegido: Perceptrón con combinaciones cuadráticas")
print("\nEntrenando el modelo...")

percep.fit(x_train_sq,y_train)

print("\nEin: ",zero_one_loss(y_train,percep.predict(x_train_sq)))
print("\n",confusion_matrix(y_train,percep.predict(x_train_sq)))
print("\nEtest: ",zero_one_loss(y_test,percep.predict(x_test_sq)))
print("\n",confusion_matrix(y_test,percep.predict(x_test_sq)))
print("\nEout(según CV): ",1-score_per_sq.mean())

#----------------------------------------------------------
#                 Modelo 1
#----------------------------------------------------------

#----------------------------------------------------------
#                 Modelo 2
#----------------------------------------------------------
