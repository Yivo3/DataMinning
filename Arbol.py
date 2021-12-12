# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 17:44:49 2021

@author: Jorge
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# Configuración warnings
# =================================================================================
import warnings
warnings.filterwarnings('ignore')

#Cargar datos
#==================================================================================
data = pd.read_csv("data.csv")

data=data[data.Position == 'Adc']

#Banderas para la columna result
#=================================================================================
data.loc[data['Result'] == 'W', "Result"] = '1'
data.loc[data['Result'] == 'L', "Result"] = '0'

#Separar las columnas en dependientes e independientes
#==================================================================================
feature_cols = ['Kills', 'Deaths', 'Gold Earned']
X = data[feature_cols] # Features
y = data.Result # Target variable

#Llenar subconjuntos de prueba y de entrenamiento
#=================================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Crear el objeto del arbol de clasificación
#=================================================================================
clf = DecisionTreeClassifier()

# Entrenar el arbol
#=================================================================================
clf = clf.fit(X_train,y_train)

#Predecir la respuesta del modelo ante el dataset
#=================================================================================
y_pred = clf.predict(X_test)

# Evaluar la precisión del modelo
#=================================================================================
print("Precisión:",metrics.accuracy_score(y_test, y_pred))


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['Derrota','Victoria'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Arbol.png')
Image(graph.create_png())










