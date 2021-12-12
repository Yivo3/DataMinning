# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:49:49 2021

@author: Jorge
"""
#Bibliotecas Tratamiento de datos 
import numpy as np
import pandas as pd
#Bibliotecas gráficas
import matplotlib.pyplot as plt
#Bibliotecas para establecer reglas de asociación
from mlxtend.frequent_patterns import apriori, association_rules

#Tratamiento de datos
#En este caso se pone el atributo "header=none" para que no se cuente la primera fila del conjunto como la cabecera del conjunto, esto debido a la naturaleza de este
data=pd.read_csv('store_data.csv',header=None)
print(data)
#print(data.head());
#Para usar el conjunto de datos con el algoritmo apriori lo que se tiene que hacer es reacomodar los datos en una tabla cuyas columnas sean los productos comprados y en las filas se especifique la cantidad de productos comprada
newData = []
for i in range(0, 7501):
    newData.append([str(data.values[i,j]) for j in range(0, 20)])


#print(newData)

#Aplicación algoritmo Apriori
#Primer parametro=Lista a usar
#Segundo parametro = Parametro usado para seleccionar los elementos con valores de soporte superiores al valor especificado por el parámetro
#Tercer parametro = Parametro usado para filtrar las reglas que tienen una confianza mayor que el umbral de confianza especificado por el parámetro.
#Cuarto parametro = Parametro usado para especificar el valor de elevación mínimo para las reglas de la lista corta.
#Quinto parametro= Parametro que especifica el número mínimo de elementos que desea en las reglas generadas.
#association_rules = apriori(newData, min_support=0.0045)

#Las reglas van a una lista llamada "Association_rules"
#association_results = list(association_rules)

#Mostrar la cantidad de reglas generadas
#print(len(association_rules))

