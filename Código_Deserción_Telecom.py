
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