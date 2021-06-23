
#------------------------------------------------------------------------------------------------
#                                    INTRODUCCIÓN AL PROBLEMA
#------------------------------------------------------------------------------------------------

# El conjunto de datos con el que vamos a tratar, almacena características de los clientes de una
# empresa fictísia de telecomunicaciones y e indica si estos abandonaron sus servicios o no.

# Se identifica que la problemática a tratar es el abandono de clientes a los servicios de una
# empresa de telecomunicaciones, es bien sabido que esta industria posee una tasa de abandono
# relativamente alta, variando entre el 15 y 20%, debido a que existe una gran competitividad en
# el mercado, por lo tanto al momento de replantearnos cuál o cuáles pueden ser los factores
# definitivos que propician el abandono de los clientes nos ponemos a pensar que puede ser
# causado por múltiples factores, desde un mal servicio, costes muy altos para su presupuesto,
# o porque simplemente desean probar otras opciones (lo que como ya mencionamos, es muy normal
# debido a la fuerte competencia en el sector). 

# Debido a todo lo anterior mencionado, las empresas buscan a toda costa retener a sus clientes,
# y aqui es donde surge la interrogante, ¿A que cliente es al que debo retener?. Puesto que sin
# un analísis de los datos es imposible saber que clientes tienen intenciones de abandonar su
# servicio o no. Intentar ejecutar una retención individualizada a cada uno de sus clientes no es
# factible debido a que estas empresas cuentan con demasiadas personas suscritas a sus servicios,
# por lo cual seria una gran pérdida de tiempo intentar retener a esta gran cantidad de personas,
# además que se superaría rápidamente el presupuesto asignado para esta labor y sin aun lograr
# resultados convincentes.

# Llegado a este punto, es necesario que la empresa utilice los datos a su favor, los vea como
# un activo estratégico para la resolución de problemas y toma de decisiones acertivas, que en
# este caso es identificar los clientes propensos a abandonar su servicio y centrar todos sus
# esfuerzos en intentar retenerlos, a la vez que tambien los vea como una inversión, puesto que
# al solventar el problema de la deserción, lograrán tener mas clientes, y por ende, aumentar
# sus ingresos, los cuales pueden ser usados para mantener o incrementar su posición en el
# mercado o en otras tareas de mayor o igual relevancia. Para lograr este objetivo, utilizaremos
# el análisis exploratorio de los datos para lograr responder algunas preguntas acerca del comportamiento
# de los clientes y su relación con el abandono de la empresa, a la vez que aprovecharemos la potencia
# y eficacia que nos ofrecen los algoritmos de machine learning, para que dado una serie de
# caracteristicas del cliente, nos de una estimación concreta acerca si en el futuro abandonará los
# servicios de la empresa o no.

# Es por ello que se proponen los siguientes objetivos:

# Analizar los datos y encontrar patrones y comportamientos que expliquen la deserción de los clientes.
# Construir un modelo de aprendizaje automático para la predicción de clientes desertores en la empresa.


#------------------------------------------------------------------------------------------------
#                             IMPORTACIÓN DE LIBRERIAS Y CARGA DE DATOS
#------------------------------------------------------------------------------------------------

# Librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTENC

# Carga de datos
data = pd.read_csv("Telco_Churn.csv")


#------------------------------------------------------------------------------------------------
#                                     EXPLORACIÓN DE DATOS
#------------------------------------------------------------------------------------------------

#-----------------------------------------------
#  ELIMINACIÓN Y CODIFICACIÓN DE CARACTERÍSTICAS
#-----------------------------------------------

data.head()

data.info()

# Obsevamos que en nuestro conjunto de datos tenemos una columna llamada "customerID", el cuál es
# un conjunto de números y letras que hacen referencia al ID del cliente, debido a que no es una
# variable relevante para nuestro estudio y construcción del modelo predictivo, se procederá a 
# eliminarlo.
data = data.drop(['customerID'], axis=1)

# También se observa que algunas variables estan etiquetadas incorrectamente con un tipo de dato
# que no les corresponde, como en el caso de "SeniorCitizen" : float y "TotalCharges" : object,
# es por ello que se procederá a convertirlas al tipo de dato correcto.

# Conversión de la columna "SeniorCitizen" a object
data = data.astype({"SeniorCitizen":object})

# Conversión de la columna "TotalCharges" a float
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

data.info()


#------------------------------------------------------------------------------------------------
#                                  PREPROCESAMIENTO DE DATOS
#------------------------------------------------------------------------------------------------

#----------------------------
# IDENTIFICACIÓN DE OUTLIERS
#----------------------------

# Mediante un diagrama de caja o bigote identificaremos visualmente si existen outliers en las
# columnas numéricas de nuestro conjunto de datos a través del rango intercuartílico.

# Se separarán en dos gráficos, debido a que la variable "TotalCharges" posee valores muy altos
# en comparación con las demas variables, lo cual ocasiona que no se visabilizen bien los gráficos

fig, ax = plt.subplots(1, 2)
sns.boxplot(ax=ax[0], data= data[["tenure", "MonthlyCharges"]])
sns.boxplot(ax=ax[1], data= data[["TotalCharges"]])
plt.show()

# Podemos observar la inexistencia de outliers, por lo que no será necesario tomar medidas al respecto.

#-------------------------------------------------
# IDENTIFICACIÓN E IMPUTACIÓN DE VALORES FALTANTES
#-------------------------------------------------

# Observamos cuantos valores faltantes hay en nuestro conjunto de datos
data.isnull().sum().sum()

# Observamos cuantos valores faltantes hay en cada columna
data.isnull().sum()

# Porcentaje de valores nulos respecto del total
data.isnull().sum().sum() / (data.shape[0] * (data.shape[1]-1)) * 100

# Los resultados nos arrojan un total de 4824 valores nulos de los 133817 que cuenta el conjunto
# de datos, estos valores nulos suponen un 3,6% del total de datos. Obtenida esta informacion,
# procederemos a imputarlos mediante el uso de algoritmos de regresión, técnica conocida con el
# nombre de imputación simple e imputación iterativa.

# Para ello empezaremos a dividir nuestro conjunto de datos en tres grupos, el primero de
# variables numéricas, el segundo de variables categóricas y el último de la variable de salida,
# ya que las técnicas de imputacion para los dos primeros conjuntos seran distintas, y el tercer
# conjunto lo excluimos de la imputación puesto que nuestra variable de salida no puede influir
# en este proceso.
numericas = data.iloc[:, [4,17,18]]
categoricas = data.iloc[:, [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16]]
salida = data.iloc[:, -1]

# Extraeremos los nombres tanto de nuestras variables categóricas como numéricas, ya que en el
# proceso de imputación estos nombres seran eliminados.
col_numericas = numericas.columns.values
col_categoricas = categoricas.columns.values

# Empezaremos imputando nuestras variables categóricas mediante un imputador simple utilizando
# la técnica de reemplazar por el mas frecuente, es decir, la moda.
imputer_categorico = SimpleImputer(strategy='most_frequent')
imputer_categorico.fit(categoricas)
categoricas = imputer_categorico.transform(categoricas)
# Y ahora le regresaremos el formato original en el que se encontraba en nuestro conjunto de datos
categoricas = pd.DataFrame(categoricas)
categoricas.columns = col_categoricas

# Proseguiremos imputando nuestras variables numéricas mediante un imputador iterativo, utilizando
# el algoritmo de los bosques aleatorios de regresión para estimar los valores faltantes en base a los
# valores no faltantes de las demás variables. Cabe mencionar que no es necesario escalar nuestros
# datos numéricos cuando utilizamos un algoritmo Random Forest.
imputer_numerico = IterativeImputer(estimator=RandomForestRegressor())
imputer_numerico.fit(numericas)
numericas = imputer_numerico.transform(numericas)
# Como hicimos con el conjunto anterior, le regresamos el formato original
numericas = pd.DataFrame(numericas)
numericas.columns = col_numericas
# Y redondeamos los decimales para tener el mismo formato númerico de los datos originiales
numericas["tenure"] = numericas["tenure"].round()

# Transformamos también la variable de salida a su formato original
salida = pd.DataFrame(salida)

# Por último, unimos los tres conjuntos de datos para tener un solo DataFrame como al inicio de
# la sección
data = pd.concat([categoricas, numericas, salida], axis=1)

# Comprobamos nuevamente si existen valores faltantes
data.isnull().sum().sum()
data.isnull().sum()

# Y efectivamente, los métodos utilizados imputaron de forma satisfactoria los valores faltantes.
