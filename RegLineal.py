# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:26:26 2021

@author: Jorge
"""
# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import sympy as sym
from sympy import symbols
from sympy.plotting import plot
# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')
# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')
# Datos
# ==============================================================================
data = pd.read_csv("data.csv")

generalData=data[['Position','Champion','Kills','Creep Score','Result','Time']]
generalData=generalData[generalData.Position=='Adc']
#generalData=generalData[generalData.Champion=='Miss Fortune']
generalData=generalData[generalData.Result=='W']
generalData=generalData[['Creep Score','Time']]

generalData.head()

# Gráfico
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 3.84))

generalData.plot(
    x    = 'Time',
    y    = 'Creep Score',
    c    = 'firebrick',
    kind = "scatter",
    ax   = ax
)
ax.set_title('Farm por segundo respecto al tiempo');

# Correlación lineal entre las dos variables
# ==============================================================================
corr_test = pearsonr(x = generalData['Time'], y =  generalData['Creep Score'])
print("Coeficiente de correlación de Pearson: ", corr_test[0])
print("P-value: ", corr_test[1])

# División de los datos en train y test
# ==============================================================================
X = generalData[['Time']]
y = generalData['Creep Score']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo
# ==============================================================================
modelo = LinearRegression()
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

# Información del modelo
# ==============================================================================
print("Intercept:", modelo.intercept_)
print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
print("Coeficiente de determinación R^2:", modelo.score(X, y))

# Error de test del modelo 
# ==============================================================================
predicciones = modelo.predict(X = X_test)
print(predicciones[0:3,])

rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = predicciones,
        squared = False
       )
print("")
print(f"El error (rmse) de test es: {rmse}")
print(f"Coeficiente: {modelo.coef_}")
print(f"Independiente: {modelo.intercept_}")
