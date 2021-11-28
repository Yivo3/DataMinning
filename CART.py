# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 14:57:32 2021

@author: Jorge
"""

# Tratamiento de datos
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd

# Gráficos
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Preprocesado y modelado
# ------------------------------------------------------------------------------
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Configuración warnings
# ------------------------------------------------------------------------------
import warnings
warnings.filterwarnings('once')

#Manejo del conjunto de datos para realizar la seleccion de campos
datos = pd.read_csv("data.csv")

dmgData=datos[['Position','Champion','Kills','Deaths','Champion Damage Share','Result']]
dmgProv=dmgData[dmgData.Position=='Adc']
dmgAdc=dmgProv[['Kills','Deaths','Champion Damage Share']]
# División de los datos en train y test
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
                                        dmgAdc.drop(columns = "Champion Damage Share"),
                                        dmgAdc['Champion Damage Share'],
                                        random_state = 123
                                    )
# Creación del modelo
# ------------------------------------------------------------------------------
modelo = DecisionTreeRegressor(
            max_depth         = 3,
            random_state      = 123
          )

# Entrenamiento del modelo
# ------------------------------------------------------------------------------
modelo.fit(X_train, y_train)

# Estructura del árbol creado
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))

print(f"Profundidad del árbol: {modelo.get_depth()}")
print(f"Número de nodos terminales: {modelo.get_n_leaves()}")

plot = plot_tree(
            decision_tree = modelo,
            feature_names = dmgAdc.drop(columns = "Champion Damage Share").columns,
            class_names   = 'Champion Damage Share',
            filled        = True,
            impurity      = False,
            fontsize      = 10,
            precision     = 2,
            ax            = ax
       )

# Imprimir arbol en consola
#-------------------------------------------------------------------------------
texto_modelo = export_text(
                    decision_tree = modelo,
                    feature_names = list(dmgAdc.drop(columns = "Champion Damage Share").columns)
               )
print(texto_modelo)
