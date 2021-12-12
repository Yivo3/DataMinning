# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 12:24:07 2021

@author: Jorge
"""
#Importar librerías
#==================================================================================
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# Configuración warnings
# =================================================================================
import warnings
warnings.filterwarnings('ignore')

#Cargar datos
#==================================================================================
data = pd.read_csv("champs.csv")

#data=data[['Team','Champion','Result']]

#print(data)
#print(pd.unique(data.Champion))

data=list(data['Champion'].apply(lambda x:x.split(',')))
te= TransactionEncoder()
te_data=te.fit(data).transform(data)

df=pd.DataFrame(te_data,columns=te.columns_).astype(int)

#print(df.sum())
#first = pd.DataFrame(df.sum() / df.shape[0], columns = ["Support"]).sort_values("Support",ascending = False)
#print(first)

frequent_itemsets = apriori(df, min_support=0.06, use_colnames=True)
print(frequent_itemsets.head())

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
print(rules.head())


