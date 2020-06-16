# -*- coding: utf-8 -*-

# CLASIFICACIÓN: OPTICAL RECOGNITION OF HANDWRITTEN DIGITS

import numpy as np

from sklearn import preprocessing
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

# Función para aplicar transformaciones polinomiales
def transformar_polinomial(data,exponente):
    poly = preprocessing.PolynomialFeatures(exponente)
    return poly.fit_transform(data)

# ------------------ CLASIFICACIÓN ------------------
    
print("--------- CLASIFICACIÓN ---------")

# Lectura de datos
print("\nLeyendo datos...")

x_train = readRawFile("datos/segmentation.data")
x_test = readRawFile("datos/segmentation.test")

x_train, y_train = extractLabels(x_train,0)
x_test, y_test = extractLabels(x_test,0)

# Pasamos los valores leidos de string a float
dataToFloat(x_train)
dataToFloat(x_test)

# Eliminamos el atributo 2 por ser igual en todas las instancias
removeAtribute(x_train, 2)
removeAtribute(x_test, 2)

# Preprocesado
print("Preprocesando datos...")

# Codificamos las etiquetas como números
y_train, cat = encodeLabels(y_train)
y_test, cat = encodeLabels(y_test, categories = cat)
# Imprimimos por pantalla la codificación realizada
print("Codificación de las etiquetas:")
for i in range(len(cat)):
    print(i, ": ",cat[i])

x_train, x_test = normalizar(x_train, x_test)

#----------------------------------------------------------
#                 Modelo Lineal
#----------------------------------------------------------

# Validación cruzada
print("\n Modelos lineales: ")
#print("Ejecutando validación cruzada...\n")

# Hiperparámetros que va a comprobar la búsqueda en grid
a = (0.0001,0.01)

parameters = {'loss' : ('log', 'perceptron'), 'alpha' : a}

linear_model = SGDClassifier()

# Ajuste de parámetros
sg = GridSearchCV(linear_model, parameters,  scoring = 'accuracy', cv = 5, iid = False)

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

# #----------------------------------------------------------
# #                   Boosting
# #----------------------------------------------------------
print("\n Boosting: ")

# Hiperparámetros 
sub = (0.9, 0.5, 0.3)
max_f = ('sqrt', 0.5, 1.0)
lr = (0.1,0.5)

parameters = {'subsample' : sub, 'max_features' : max_f, 'learning_rate' : lr}

gbc = GradientBoostingClassifier(n_estimators = 100)

# Ajuste de Hiperparámetros
sg = GridSearchCV(gbc, parameters, scoring = 'accuracy', cv = 5, iid = False)

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
md = [8]

parameters = {'n_estimators' : ne, 'criterion' : c, 'max_features' : mf, 'max_depth' : md}

rforest = RandomForestClassifier()

sg = GridSearchCV(rforest, parameters, scoring = 'accuracy', cv = 5, iid = False)

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
# #               Multi-layer Perceptron
# #----------------------------------------------------------
print("\n Multi-layer Perceptron: ")

hidden = [(8,)]
mi = [100000]
a = [0.1]
act = ['tanh']
s = ['lbfgs']

parameters = {'hidden_layer_sizes' : hidden, 'alpha' : a,'max_iter' : mi, 'activation' : act, 'solver' : s}

mlp = MLPClassifier()

sg = GridSearchCV(mlp, parameters, scoring = 'accuracy', cv = 5, iid = False)

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