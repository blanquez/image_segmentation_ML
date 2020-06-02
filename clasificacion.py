# -*- coding: utf-8 -*-

# CLASIFICACIÓN: OPTICAL RECOGNITION OF HANDWRITTEN DIGITS

import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import zero_one_loss, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

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
print("\nModelos lineales: ")
print("Ejecutando validación cruzada...\n")

log_reg = SGDClassifier(loss="log", penalty = 'l1')
percep = SGDClassifier(loss="perceptron", penalty = 'l1')

score_log = cross_val_score(log_reg, x_train, y_train, cv=5)
Emin = 1-score_log.mean()
model = log_reg
training = x_train
test = x_test
print("R. Logaritmica: ",1-score_log.mean())

score_log_sq = cross_val_score(log_reg,x_train_sq,y_train,cv=5)
if 1-score_log_sq.mean() < Emin:
    Emin = 1-score_log_sq.mean()
    training = x_train_sq
    test = x_test_sq
print("R. Logaritmica sq:",1-score_log_sq.mean())

score_per = cross_val_score(percep,x_train,y_train,cv=5)
if 1-score_per.mean() < Emin:
    Emin = 1-score_per.mean()
    model = percep
    training = x_train
    test = x_test
print("Perceptron: ",1-score_per.mean())

score_per_sq = cross_val_score(percep,x_train_sq,y_train,cv=5)
if 1-score_per_sq.mean() < Emin:
    Emin = 1-score_per_sq.mean()
    model = percep
    training = x_train_sq
    test = x_test_sq
print("Perceptron sq: ",1-score_per_sq.mean())

# Entrenamiento
#print("\nModelo elegido: Perceptrón con combinaciones cuadráticas")
print("\nEntrenando el modelo...")


model.fit(training,y_train)

print("\nEin: ",zero_one_loss(y_train,model.predict(training)))
print("\n",confusion_matrix(y_train,model.predict(training)))
print("\nEtest: ",zero_one_loss(y_test,model.predict(test)))
print("\n",confusion_matrix(y_test,model.predict(test)))
print("\nEval(según CV): ",Emin)

input("\nPulse una tecla para continuar\n")

#----------------------------------------------------------
#           Perceptron Multicapa
#----------------------------------------------------------
print("\n Perceptron Multicapa: ")

m_it = 1000
a = 0.01

Emin = 1

for nd in range(50,101,10):
    for lr in range(1,10,1):
        print("ND: ",nd," LR: ",lr/100)
        boost = MLPClassifier(hidden_layer_sizes=(nd, ), max_iter = m_it, learning_rate_init = lr/100, alpha = a )

        score = cross_val_score(boost, x_train, y_train, cv=5)
        print("MLP: ",1-score.mean())

        if 1-score.mean() < Emin:
            Emin = 1-score.mean()
            best_nd = nd
            best_lr = lr/100

model = MLPClassifier(hidden_layer_sizes=(nd, ), max_iter = m_it, learning_rate_init = best_lr, alpha = a)
model.fit(x_train,y_train)

print("\nND Final: ",best_nd," LR Final: ",best_lr)
print("\nEin: ",zero_one_loss(y_train,model.predict(x_train)))
print("\n",confusion_matrix(y_train,model.predict(x_train)))
print("\nEtest: ",zero_one_loss(y_test,model.predict(x_test)))
print("\n",confusion_matrix(y_test,model.predict(x_test)))
print("\nEval(según CV): ",Emin)

#----------------------------------------------------------
#                   Boosting
#----------------------------------------------------------

print("\n Boosting: ")

boost = GradientBoostingClassifier(learning_rate = 0.1, n_estimators=50, max_features = 'auto')

score = cross_val_score(boost, x_train, y_train, cv=5)
Emin = 1-score.mean()
ne = 50
print("Nº Estimators = ",ne)
print("Boosting: ",Emin)

for i in range(60,101,10):
    print("\nNº Estimators = ",i)
    boost = GradientBoostingClassifier(learning_rate = 0.1, n_estimators=i)

    score = cross_val_score(boost, x_train, y_train, cv=5)
    print("Boosting: ",1-score.mean())

    if 1-score.mean() < Emin:
        Emin = 1-score.mean()
        ne = i

model = GradientBoostingClassifier(learning_rate = 0.1, n_estimators=ne, max_features = 'auto')
model.fit(x_train,y_train)

print("\n Nº Estimadores Finales = ",ne)
print("\nEin: ",zero_one_loss(y_train,model.predict(x_train)))
print("\n",confusion_matrix(y_train,model.predict(x_train)))
print("\nEtest: ",zero_one_loss(y_test,model.predict(x_test)))
print("\n",confusion_matrix(y_test,model.predict(x_test)))
print("\nEval(según CV): ",Emin)
