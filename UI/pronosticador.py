import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


#En nuestro dataset los accidentes por exceso de velocidad son representados con el valor 3.
def categorizar_accidentes(valor):
  if(valor==3):
    return 1
  else:
    return 0

def cargar_datos(fileName):
    datos = pd.read_csv(fileName, delimiter=";")
    df = pd.DataFrame(datos)
    df['CAUSA'] = df['CAUSA'].apply(categorizar_accidentes)
    dataframe = df.drop(df[df['CAUSA']==0].index[0:16000])
    dataframe = dataframe.drop(columns=["PROVINCIA", "NUM_FALLECIDO", "NUM_LESIONADO", "ZONA", "TOTAL_VICTIMAS","CLASE"])
    dataframe["INTERCEPT"] = 1 #Se le agrega esta columna de 1 para el intercepto
    X = dataframe.drop(columns = [ "CAUSA"]) #Aqui agregar columnas que se quieran quitar de los features
    Y = dataframe["CAUSA"]
    X_train, X_test, y_train, y_test = train_test_split( X, Y, train_size= 0.8 )
    return X_train, X_test, y_train, y_test

def entrenar_modelo(X_train, y_train, epocas):
    #Entrenamiento del modelo ================================================================================
    modelo = LogisticRegression()
    for i in range(epocas):
        modelo.fit(X_train, y_train)
    return modelo



