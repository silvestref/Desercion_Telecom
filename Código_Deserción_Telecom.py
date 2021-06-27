
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
import xgboost as xgb
from xgboost import XGBClassifier
import optuna  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve


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


#------------------------------------------------------------------------------------------------
#                                ANÁLISIS Y VISUALIZACIÓN DE DATOS
#------------------------------------------------------------------------------------------------

# Empezaremos la sección formulando algunas hipótesis que seran respondidas mediante el proceso
# de análisis de los datos

# H1: ¿El género del cliente propicia la deserción de los servicios de la empresa?
# H2: ¿Son los clientes de la tercera edad mas propensos a desertar?
# H3: ¿Los clientes sin pareja son mas propensos a permanecer en la empresa?
# H4: ¿Si el cliente no vive con personas dependientes tiene mas probabilidades de abandonar la empresa?
# H5: ¿Es más probable que los clientes que no tienen servicio telefónico abandonen la empresa?
# H6: ¿Los clientes con múltiples lineas telefónicas son más propensos a permanecer en la empresa?
# H7: ¿Es más probable que los clientes abandonen la empresa si no tienen conexión a internet?
# H8: ¿Los clientes sin servicio de seguridad en línea tienden a abandonar la empresa?
# H9: ¿Los clientes sin servicio de copia de seguridad en línea tienden a abandonar la empresa?
# H10: ¿Los clientes sin servicio de protección de dispositivos tienden a abandonar la empresa?
# H11: ¿Los clientes sin servicio de soporte técnico tienden a abandonar la empresa?
# H12: ¿Los clientes sin servicio de transmisión televisiva tienden a abandonar la empresa?
# H13: ¿Los clientes sin servicio de transmisión de películas tienden a abandonar la empresa?
# H14: ¿Los clientes con mayor cantidad de meses en la empresa son más propensos a seguir permaneciendo en ella?
# H15: ¿Los clientes con poca cantidad de dinero mensual a pagar son más propensos a permanecer en la empresa?
# H16: ¿Los clientes con poca cantidad de dinero total a pagar son más propensos a permanecer en la empresa?
# H17: ¿El el tipo de contrato elegido por el cliente un factor que indique su deserción de la empresa?
# H18: ¿Los clientes que eligen facturación electrónica son más propensos a abandonar la empresa?
# H19: ¿Existe algún metodo de pago en particular preferido por los clientes desertores?


# Visualizaremos la distribución de los datos respecto a cada uno de los tres conjuntos de
# variables que se han identificado: Variables de información del cliente - Variables de servicio
# - Variables de contrato. Esta segmentación nos permitirá realizar un análisis mas ordenado e
# identificar patrones e información util para entender nuestros datos.

# Variables de información del cliente

fig, ax = plt.subplots(2, 2, figsize=(16, 8))
sns.countplot(data=data, x="gender", ax=ax[0,0])
sns.countplot(data=data, x="SeniorCitizen", ax=ax[0,1])
sns.countplot(data=data, x="Partner", ax=ax[1,0])
sns.countplot(data=data, x="Dependents", ax=ax[1,1])
fig.suptitle('Distribución de las variables de información del cliente', fontsize=16)
plt.show()

# Observamos que tenemos una distribución equitativa en nuestras variables "gender" y 
# "Partner", lo cual nos indica que ningún género predomina sobre el otro en la empresa, a la 
# vez que hay tantas personas con pareja como sin pareja.

# Por otro lado observamos que predominan más las personas menores de 65 años, y que la
# mayoría de ellos no viven con personas dependientes.


# Variables de servicio

fig, ax = plt.subplots(3, 3, figsize=(16, 12))
sns.countplot(data=data, x="PhoneService", ax=ax[0,0])
sns.countplot(data=data, x="MultipleLines", ax=ax[0,1])
sns.countplot(data=data, x="InternetService", ax=ax[0,2])
sns.countplot(data=data, x="OnlineSecurity", ax=ax[1,0])
sns.countplot(data=data, x="OnlineBackup", ax=ax[1,1])
sns.countplot(data=data, x="DeviceProtection", ax=ax[1,2])
sns.countplot(data=data, x="TechSupport", ax=ax[2,0])
sns.countplot(data=data, x="StreamingTV", ax=ax[2,1])
sns.countplot(data=data, x="StreamingMovies", ax=ax[2,2])
fig.suptitle('Distribución de las variables de servicio', fontsize=16)
plt.show()

# De los resultados obtenimos podemos indicar lo siguiente:

# Existe una inmensa mayoría de clientes que han adquirido los servicios de telefonía de la
# empresa, lo cual nos indica que es el servicio más demandado y básico que ofrece la
# compañía.

# Hay una distribución relativamente equitativa entre el número de clientes que tiene múltiples
# líneas y no, lo cual es común puesto que es un servicio opcional.

# La mayoria de los usuarios prefiere una conexión de fibra óptica como servicio de internet,
# ya que esta es mucho más rápida y de calidad que otros tipos de conexión convencionales.

# Observamos una tendencia de los usuarios a no contar con los servicios de seguridad
# que ofrece la empresa, podemos suponer múltiples razones, desde una mala calidad en estos
# servicios hasta costos elevados en la mensualidad por adquirirlos.

# Por último, observamos que existe una ligera diferencia entre la cantidad de clientes que eligen
# adquirir servicios de transmisión de TV a través de su servicio de internet y los que no lo
# hacen.


# Variables de contrato

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(data=data, x="tenure", kde=True, ax=ax[0])
sns.histplot(data=data, x="MonthlyCharges", kde=True, ax=ax[1])
sns.histplot(data=data, x="TotalCharges", kde=True, ax=ax[2])
fig.suptitle('Distribución de las variables de contrato', fontsize=16)

# De estos resultados se extrae la siguiente información:
    
# Existen dos grandes picos en la distribución que muestra la cantidad de meses que el cliente
# a permanecido en la empresa al finalizar el trimestre, siendo estos los que han permanecido en
# un rango menor a 5 meses, y los que han permanecido en un rango mayor a 65 meses, lo que significa
# que la empresa cuenta con tantos clientes fieles como nuevos en la adquisición de sus servicios.

# Por otra parte, observamos que la variable de los cargos mensuales "MonthlyCharges" presenta
# 3 picos notables, siendo el mayor de estos los que tienen cargos mensuales alrededor de
# 20 dólares, seguido de otro con alrededor de 80 dólares, y por ultimo, uno con alrededor de
# 50 dólares. Esto quiere decir que tenemos una cantidad considerable de clientes que
# prefieren los contratos con poca mensualidad a pagar (los cuales probablemente
# incluyan menos servicios). 

# La variable de los cargos totales "TotalCharges" presenta una distribución de cola en donde
# el único pico que presenta es en los cargos con poca cantidad de dólares a pagar por los
# clientes, lo cual guarda relación con los anteriores gráficos donde vimos que existe una
# gran cantidad de clientes con pocos meses en la empresa y con cargos mensuales bajos.

fig, ax = plt.subplots(1, 2, figsize=(16, 4))
sns.countplot(data=data, x="Contract", ax=ax[0])
sns.countplot(data=data, x="PaperlessBilling", ax=ax[1])
fig, ax = plt.subplots(1, 1, figsize=(16, 4))
sns.countplot(data=data, y="PaymentMethod")
plt.show()

# Observamos que el contrato preferido por los clientes es el de "Month to month", el cual es el
# más corto de todos, lo cual guarda cierta relación con el gráfico anterior donde vimos que había
# una gran cantidad de clientes con pocos meses de permanencia en la empresa.

# Por otra parte, observamos que los usuarios de la empresa mayormente prefieren facturación
# electronica.

# Por ultimo, observamos que la mayoría de clientes prefiere el método de pago con cheque
# electrónico, la distribución de los demas metodos se mantiene de forma equitativa entre ellos.

# En base a todo lo anterior visto, procederemos a responder las hipótesis que inicialmente
# habíamos planteado, esto lo lograremos mediante un análisis bivariado de nuestras variables de
# entrada con nuestra variable de salida.


# Variables de información del cliente vs "Churn"

fig, ax = plt.subplots(2, 2, figsize=(16, 8))

sns.countplot(data=data, x="gender", ax=ax[0,0], hue=data.Churn)
sns.countplot(data=data, x="SeniorCitizen", ax=ax[0,1], hue=data.Churn)
sns.countplot(data=data, x="Partner", ax=ax[1,0], hue=data.Churn)
sns.countplot(data=data, x="Dependents", ax=ax[1,1], hue=data.Churn)
fig.suptitle('Variables de información del cliente vs Churn', fontsize=16)
plt.show()

# Observamos que tanto el número de desertores en el género masculino es el mismo que en el
# género femenino, por lo tanto se puede decir que esta variable no influye en la deserción
# de clientes de la empresa

# Sin embargo, en el gráfico de la variable "SeniorCitizen", observamos que los clientes que no
# son de la tercera edad (mayor a 65 años) son menos propensas a abandonar los servicios de la
# empresa, en comparación con las personas que si cumplen con esta franja de edad, las cuales
# tienen una distribución mas equilibrada. Por lo tanto, se puede decir que esta variable influye
# en cierta medida a la deserción de clientes.

# Del gráfico de la variable "Partner" podemos deducir que los clientes que no tienen pareja son
# ligeramente más propensos a abandonar los servicios de la empresa.

# Y por último, de la gráfica de la variable "Dependents" podemos observar que los clientes que no
# viven con personas dependientes tienen más probabilidades de abandonar los servicios de la
# empresa, por ende, es una variable influyente en la deserción de usuarios.

# Resumiendo toda la información obtenida tenemos que: Tanto hombres como mujeres tienen la misma
# probabilidad de deserción, si estas personas son mayores de 65 años, esta probabilidad aumenta.
# Y el hecho que no tengan pareja y que no vivan con personas dependientes aumenta en cierta forma
# sus probabilidades de abandonar los servicios de la empresa.

# Respondiendo a las hipótesis tenemos que:
# H1: El género del cliente no afecta de ninguna forma en la deserción de los servicios de la empresa.
# H2: Los clientes de la tercera edad son más propensos a ser desertores comparados con los que no pasan esta franja de edad
# H3: Los clientes sin pareja tienen ligeramente mas probabilidades de desertar que aquellos que sí tienen
# H4: Los clientes que no viven con personas dependientes tienen más probabilidades de abandonar la empresa


# Variables de servicio vs "Churn"

fig, ax = plt.subplots(3, 3, figsize=(16, 12))
sns.countplot(data=data, x="PhoneService", ax=ax[0,0], hue=data["Churn"])
sns.countplot(data=data, x="MultipleLines", ax=ax[0,1], hue=data["Churn"])
sns.countplot(data=data, x="InternetService", ax=ax[0,2], hue=data["Churn"])
sns.countplot(data=data, x="OnlineSecurity", ax=ax[1,0], hue=data["Churn"])
sns.countplot(data=data, x="OnlineBackup", ax=ax[1,1], hue=data["Churn"])
sns.countplot(data=data, x="DeviceProtection", ax=ax[1,2], hue=data["Churn"])
sns.countplot(data=data, x="TechSupport", ax=ax[2,0], hue=data["Churn"])
sns.countplot(data=data, x="StreamingTV", ax=ax[2,1], hue=data["Churn"])
sns.countplot(data=data, x="StreamingMovies", ax=ax[2,2], hue=data["Churn"])
fig.suptitle('Variables de servicio vs Churn', fontsize=16)
plt.show()

# Para nuestra variable "PhoneService" no observamos relación alguna con la deserción de
# clientes, ya que ambas proporciones de abandonos en el caso de tener o no servicio 
# telefónico se reparte de forma equitativa respecto al total de muestras.

# El mismo patrón se observa en la variable "MultipleLines", aunque es ligeramente más probable
# abandonar si el cliente cuenta con múltiples lineas de telefonía.

# En el caso de la variable "InternetService" observamos claramente que existe una alta
# probabilidad de desertar los servicios de la empresa si el usuario tiene un servicio de
# internet de fibra óptica.

# Lo mismo sucede en las variables "OnlineSecurity", "OnlineBackup", "DeviceProtection" y
# "TechSupport", donde es mucho más probable encontrar abandono de usuarios si estos no cuentan
# con los servicios mencionados, el cual es un comportamiento interesante ya que todos estos
# servicios están asociados a la seguridad y protección de red y dispositivos, y dependen únicamente
# si el cliente cuenta con servicio de internet o no.

# Por último, de las gráficas respecto a las variables "StreamingTV" y "StreamingMovies" tenemos
# una probabilidad similar de desertar si el usuario cuenta o no con estos servicios, y si no tiene
# servicio de internet, esta probabilidad desciende en gran medida

# Resumiendo toda la informacion obtenida tenemos que: Contar con servicio telefónico o no,
# no afecta en la deserción de clientes, sin embargo el contar con múltiples líneas telefónicas
# puede llegar a afectar ligeramente esta probabilidad. Si el usuario tiene servicio de
# internet de fibra óptica, las probabilidades de desertar aumentan exponencialmente, y si a
# esto lo sumamos no adquirir ninguno de los servicios de protección y seguridad como
# ("OnlineSecurity", "OnlineBackup", "DeviceProtection", y "TechSupport") esta probabilidad
# aumenta aun más, por ende podemos deducir que existe un problema grave en los servicios de
# fibra óptica y los servicios de seguridad que brinda la empresa. Por último, podemos
# decir que el cliente tiene igual probabilidad de desertar en el caso que adquiera o no adquiera
# servicios de transmisión televisiva o de películas, y si no tiene servicios de internet, esta
# probabilidad disminuye en gran medida.

# Respondiendo a las hipótesis tenemos que:
# H5: El contar o no con servicio telefónico no influye en la deserción de clientes en la empresa
# H6: Los clientes con múltiples líneas telefónicas son ligeramente mas probables a desertar
# H7: Es muy probable que los clientes abandonen la empresa si estos cuentan con internet de fibra óptica
# H8: Los clientes sin servicio de seguridad en línea tienden a abandonar la empresa
# H9: Los clientes sin servicio de copia de seguridad en linea tienden a abandonar la empresa
# H10: Los clientes sin servicio de protección de dispositivos tienden a abandonar la empresa
# H11: Los clientes sin servicio de soporte técnico tienden a abandonar la empresa
# H12: Los clientes sin servicio de transmisión televisiva tienen similar probabilidad de desertar en comparación
# con los que si cuentan con este servicio
# H13: Los clientes sin servicio de transmisión de películas tienen similar probabilidad de desertar en comparacion
# con los que si cuentan con este servicio


# Variables de contrato vs "Churn"

fig, axs = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(data=data, x="tenure", kde=True, ax=axs[0], hue=data.Churn)
sns.histplot(data=data, x="MonthlyCharges", kde=True, ax=axs[1], hue=data.Churn)
sns.histplot(data=data, x="TotalCharges", kde=True, ax=axs[2], hue=data.Churn)
fig.suptitle('Variables de servicio vs Churn', fontsize=16)
plt.show()

# En primer lugar observamos que existe una relación entre nuestra variable "tenure" (número de
# meses que el cliente permaneció en la empresa) con la deserción del cliente, ya que el histograma
# deja en clara evidencia que los usuarios que menos meses permanecieron se dividen de forma
# equitativa en usuarios desertores y no desertores, y que mientras más meses permanezcan en
# la empresa, menos probabilidades tendran de desertar y más probabilidades tendran de quedarse.

# En el caso de la variable "MonthlyCharges", podemos observar que un aumento del costo
# mensual a pagar por el cliente provocará un leve aumento en las probabilidades de desertar,
# a la vez que también observamos que los clientes con poca cantidad mensual a pagar son los
# que indudablemente permanecen en la empresa.

# Por último, para el caso de "TotalCharges", tenemos un comportamiento similar entre los
# clientes que desertaron y no desertaron, ya que podemos observar que si el monto total a pagar
# es pequeño, las probabilidades de desertar o quedarse son similares, por otra parte, mientras
# mayor sea este monto, menor sera la probabilidad de abandonar (para los clientes
# que desertaron) y no abandonar (para los clientes que no desertaron) los servicios de la
# empresa

# Resumiendo toda la información obtenedia tenemos que: Existe una distribución equitativa
# entre los clientes que desertaron y no desertaron cuando estos tienen pocos meses afiliados
# a la empresa, y que mientras mayor sea la cantidad de meses que permanecen en esta, mayor será
# su probabilidad de quedarse, a la vez que tambien observamos una leve relación entre el aumento
# de la mensualidad a pagar de los clientes y el abandono de estos, ya que mientras mayor sea
# el monto, mayores probabilidades hay de desertar. Por último tenemos que en los montos totales
# con menor valor monetario hay una probabilidad casi equitativa de desertar o no de los servicios
# de la empresa, y que esto se relaciona con los meses de permanencia en la empresa("tenure"), ya
# que si un cliente pasa menos tiempo afiliado a la empresa, es de esperarse que su monto total
# a pagar sea igual de bajo como el tiempo que paso afiliado.

# Respondiendo a las hipótesis tenemos que:
# H14: Los clientes con mayor número de meses en la empresa tienden a permancer más tiempo en ella
# H15: Los clientes con poca cantidad de dinero mensual a pagar son más propensos a permanecer en la empresa
# H16: Los clientes con poca cantidad de dinero total a pagar son igualmente propensos a abandonar como permanecer en la empresa


fig, axs = plt.subplots(1, 2, figsize=(16, 4))
sns.countplot(data=data, x="Contract", ax=axs[0], hue=data.Churn)
sns.countplot(data=data, x="PaperlessBilling", ax=axs[1], hue=data.Churn)
fig.suptitle('Variables de servicio vs Churn', fontsize=16)
fig, axs = plt.subplots(1, 1, figsize=(16, 4))
sns.countplot(data=data, x="PaymentMethod", hue=data.Churn)
plt.show()

# Se puede observar que las probabilidades de deserción de un cliente aumentan en gran medida
# si este tiene un contrato corto de mes a mes, y que los clientes desertores rara vez escogen
# contratos largos como los de un año o dos años.

# La facturacion electrónica es una variable que influye ligeramente en la deserción de los clientes
# ya que podemos ver que los usuarios que escogen este tipo de documento tienen un mayor número
# de desertores comparado con los que no escogen esto.

# Por último, tenemos que los clientes que escogen el cheque electrónico como método de pago 
# son mas propensos a abandonar los servicios de la empresa, mientras que los clientes que
# escogen otros métodos como los automaticos o los enviados por correo electroncio tienden a
# no desertar

# Resumiendo toda la información obtenedia tenemos que: Los contratos cortos son los preferidos
# de los usuarios que no estan seguros si los servicios que brinda la empresa cumpliran sus
# expectativas, y por ende, son los que mas probabilidad tienen de desertar, al igual que
# el documento y método de pago preferido por este tipo de clientes es el cheque electrónico y
# la facturación electrónica.

# Respondiendo a las hipótesis tenemos que:
# H17: El tipo de contrato elegido ayuda a determinar si un cliente es propenso a desertar o no, ya
# que en la mayoría de ocaciones estos eligen contratos mes a mes.
# H18: Los clientes que eligen facturación electrónica son ligeramente mas propensos a abandonar la empresa
# H19: Los clientes desertores en la mayoría de ocaciones eligen el cheque electrónico como metodo de pago.


# A lo largo de este proceso de análisis nos hemos encontrado con un patrón repetitivo en los
# clientes desertores, el cual consiste en no tener los servicios basados en la seguridad de red
# y dispositivos, y que estos en su gran mayoría provenian de usuarios con servicio de internet
# y especificamente con conexión de fibra óptica, es por ello que nos proponemos a identificar
# la combinación de servicios que mayor abandono y mayor permanencia de clientes tienen para poder
# observar que cantidad de servicios adquiridos y que tipo de servicios en particular son los
# que propician la deserción y la permanencia de cliente en la empresa.


#-----------------------------
# ¿Que combinación de servicios propicia el abandono de clientes?
#-----------------------------

# Creamos un nuevo conjunto de datos en la que codificaremos numéricamente la variable "Churn"
# para poder construir una tabla de pivotaje que cuente los valores positivos de esta variable
churn_dummy = pd.get_dummies(data, columns=["Churn"])

# Identificamos que combinación de servicios tiene más deserción en base a la variable "PhoneService"
mayor_aban = pd.pivot_table(churn_dummy,index=["PhoneService"], columns=["InternetService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"],
                            values=["Churn_Yes"],aggfunc=lambda x: x.sum() if x.sum() > 70 else np.nan)

mayor_aban.plot(kind="bar" )
plt.title("Combinaciones de servicios con mayor cantidad de abandonos de clientes", y=1.09)
plt.suptitle("InternetService | MultipleLines | OnlineSecurity | OnlineBackup | DeviceProtection | TechSupport | StreamingTV | StreamingMovies", y=0.93)
plt.yticks([0,40,80,120,160,190])
plt.xticks(rotation=0)
plt.xlabel('')
plt.legend('',frameon=False)
plt.text(0.245, 0.87, 'DSL|No|No|No|No|No|No|No', verticalalignment='center', transform=ax.transAxes)
plt.text(0.42, 0.59, 'FO|Yes|No|No|No|No|No|No', verticalalignment='center', transform=ax.transAxes)
plt.text(0.585, 0.41, 'FO|Yes|No|No|No|No|No|No', verticalalignment='center', transform=ax.transAxes)
ax = plt.gca()
ax.grid(alpha=0.5, axis="y")
ax.set_axisbelow(True)

# Del gráfico mostrado identificamos que las combinaciones que mas desersión tienen son las que
# incluyen menos servicios del catálogo que ofrece la empresa, estos clientes solo cuentan con
# servicio de telefonía e internet, sin embargo, no adquieren los servicios complementarios al de
# internet, como vendrian a ser "TechSupport", "DeviceProtection", "OnlineBackup" y "OnlineSecurity"
# como ya habiamos visto previamente en análisis anteriores, lo cual nos da a entender que si el
# cliente cuenta con servicios de conexión a internet pero no con sus complementarios, entonces
# hay una mayor probabilidad de que estos en un futuro deserten, puesto que los consideran importantes
# y que hay algún motivo que esta impidiendo que los adquieran, el cual podría ser el factor económico.


#---------------------------------
# ¿Que combinación de servicios propician la permanencia de clientes?
#---------------------------------

mayor_perm = pd.pivot_table(churn_dummy,index=["PhoneService"], columns=["InternetService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"],
                            values=["Churn_No"],aggfunc=lambda x: x.sum() if x.sum() > 112 else np.nan)

mayor_perm.plot(kind="bar" )
plt.title("Combinaciones de servicios con mayor cantidad de permanencia de clientes", y=1.09)
plt.suptitle("InternetService | MultipleLines | OnlineSecurity | OnlineBackup | DeviceProtection | TechSupport | StreamingTV | StreamingMovies", y=0.93)
plt.yticks([0,100,300,500,700,900])
plt.xticks(rotation=0)
plt.xlabel('Internet Service')
plt.legend('',frameon=False)
plt.text(0.25, 0.9, 'No|No|NoIS|NoIS|NoIS|NoIS|NoIS|NoIS', verticalalignment='center', transform=ax.transAxes)
plt.text(0.50, 0.27, 'No|Yes|NoIS|NoIS|NoIS|NoIS|NoIS|NoIS', verticalalignment='center', transform=ax.transAxes)
ax = plt.gca()
ax.grid(alpha=0.5, axis="y")
ax.set_axisbelow(True)

# De este gráfico observamos un patrón interesante, ya que los clientes que tienen mayor permanencia
# en la empresa son en su gran mayoría los que solo cuentan con servicio telefónico, estos clientes
# en comparación con la combinación que mayor propicia la deserción no cuentan con servicio de
# internet, por lo tanto no se ven afectados al no tener los servicios complementarios
# que derivan de este, lo que causa que sus probabilidades de desertar disminuyan y sean mas
# propensos a permanecer en la empresa.

# Entonces podemos concluir que nuestra variable "InternetService" es un factor muy importante al
# momento de determinar si un cliente abandona o permanece con los servicios de la empresa, ya que
# condiciona el comportamiento de las demás variables de servicio y su adquisición sin los servicios
# complementarios que derivan de él propician la desercion de los usuarios. 


# Para terminar con esta sección, graficaremos una matriz de correlación para identificar el
# comportamiento conjunto de nuestras variables sobre otras, como estamos tratando tanto con
# variables categóricas como numéricas, sera necesario primero codificar las variables categóricas para
# poder graficar de forma correcta la matriz de correlación.

data_corr = pd.get_dummies(data, columns = ["gender","SeniorCitizen","Partner","Dependents",
                                            "PhoneService","MultipleLines","InternetService",
                                            "OnlineSecurity","OnlineBackup","DeviceProtection",
                                            "TechSupport","StreamingTV","StreamingMovies",
                                            "Contract","PaperlessBilling","PaymentMethod","Churn"],
                                            drop_first=True)

# Debido a que contamos con muchas variables sera necesario dividir nuestro conjunto de datos y
# graficar la matriz de correlación en base a cada una de las divisiones para poder apreciar mejor
# la gráfica.

data_corr_1 = data_corr[["tenure","MonthlyCharges","TotalCharges","gender_Male","SeniorCitizen_1.0",
                        "Partner_Yes","Dependents_Yes","PhoneService_Yes","MultipleLines_No phone service",
                        "MultipleLines_Yes","Churn_Yes"]]
data_corr_2 = data_corr.iloc[:,10:31]

# Matriz de correlación para el primer conjunto
plt.figure(figsize=(30, 20))
corr = data_corr_1.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
ax = sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)

# Matriz de correlación para el segundo conjunto
plt.figure(figsize=(30, 20))
corr = data_corr_2.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
ax = sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)

# Identificamos la existencia de una gran correlación entre las variables que estan asociadas
# a los servicios que ofrece la empresa, siendo las mas influyentes y recurrentes aquellas 
# relacionadas con "InternetService".

# A continuacion, visualizaremos algunas de las correlaciones mas altas y bajas mediante un gráfico de
# barras, puesto que estamos tratando en su mayoría con variables categóricas y no numéricas.

#-------
# "StreamingTV" vs "StreamingMovies"
STV_SMOV=pd.crosstab(index=data['StreamingTV'],columns=data['StreamingMovies'])
STV_SMOV.plot.bar(figsize=(7,4), rot=0)

# Observamos que las variables "StreamingTV" y "StreamingMovies" estan correlacionadas
# positivamente, especialmente en la clase "No internet service" como nos muestra nuestra tabla
# de correlaciones, puesto que para cada clase de la variable "StreamingTV", la variable "StreamingMovies"
# se comportará en gran medida de la misma manera, es decir, si el cliente no cuenta con servicio
# de transmisión de TV, con mucha frecuencia tampoco contara con servicio de transmisión de películas,
# y este mismo patrón se repite en las demas clases.
# Este comportamiento se puede explicar de la siguiente forma: Si un usuario no esta interesado
# en usar su servicio de internet para adquirir servicios de transmisión televisiva, es muy
# probable que tampoco este interesado en adquirir servicios de transmisión de películas, puesto
# que sus gustos no se centran en este tipo de entretenimiento, el mismo comportamiento se aplica
# si el cliente si adquiere servicios televisivos, sin embargo, en el caso de no contar con servicio
# de internet, no existen alternativas que el cliente pueda elegir.

#-------
# "DeviceProtection" vs "TechSupport"
DP_TS=pd.crosstab(index=data['DeviceProtection'],columns=data['TechSupport'])
DP_TS.plot.bar(figsize=(7,4), rot=0)

# Un patron similar observamos en estas variables, con la diferencia que la última clase de la
# variable "DeviceProtection" tiene una distribución mas balanceada, sin embargo, aun posee
# correlación con "TechSupport", ya que sigue influenciando en su comportamiento. 
# La razon de este comportamiento sigue siendo el mismo que el del gráfico anterior.

#-------
# "OnlineSecurity" vs "TechSupport"
OS_TS=pd.crosstab(index=data['OnlineSecurity'],columns=data['TechSupport'])
OS_TS.plot.bar(figsize=(7,4), rot=0)

# Y lo mismo observamos al comparar "DeviceProtection" vs "TechSupport", en donde se aprecia
# correlación positiva e igual interpretacion de comportamiento.

#-------
# "MultipleLines" vs "PhoneService"
ML_PS=pd.crosstab(index=data['MultipleLines'],columns=data['PhoneService'])
ML_PS.plot.bar(figsize=(7,4), rot=0)

# Por último, observamos una correlación altamente negativa entre ambas variables, puesto que
# "MultipleLines" tiende a adquirir un valor de "No" cuando "PhoneService" adquiere un valor de
# "Yes", condicionando en forma inversa su valor.


#--------------------------------------------------------------------------------------------
#                                   TRANSFORMACIÓN DE DATOS
#--------------------------------------------------------------------------------------------

# Antes de empezar con la verificación e implementación de técnicas para la transformación de
# datos, empezaremos codificando nuestras variables categóricas a numéricas, puesto que es un
# paso necesario para que los algoritmos de aprendizaje automático (XGBoost en nuestro caso)
# puedan aprender correctamente de los datos.

data_cod = data.copy()
encoder = LabelEncoder()
data_cod["Churn"] = encoder.fit_transform(data_cod["Churn"])

data_cod = pd.get_dummies(data_cod, columns=["gender","SeniorCitizen","Partner","Dependents",
                                             "PhoneService","MultipleLines","InternetService",
                                             "OnlineSecurity","OnlineBackup","DeviceProtection",
                                             "TechSupport","StreamingTV","StreamingMovies",
                                             "Contract","PaperlessBilling","PaymentMethod"],
                                             drop_first=True)

# Posterior a ello segmentaremos la totalidad de nuestros datos en dos conjuntos: variables
# de entrada (X) y variable de salida (y). Para después volver a dividir estos conjuntos
# en: datos de entrenamiento (X_train, y_train) y datos de validación (X_test, y_test).
# Esta división nos ayudara a evitar un problema conocido como "fuga de datos", el 
# cual es causado al realizar transformaciones en la totalidad de los datos o incluir
# información en la fase de entrenamiento del modelo que no se esperaría que estuviese
# disponible al momento de realizar una predicción con datos no antes visto, lo cual provoca
# que no tengamos recursos al momento de querer validar nuestro modelo o que las métricas
# de evaluación arrojen falsos resultados.

# Conjunto de variables de entrada y salida
X = data_cod.drop(["Churn"], axis=1)
X = X.iloc[: , :].values
y = data_cod.iloc[: , 3].values

# Conjunto de entrenamiento y evaluación
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=21, stratify=y)

# Una vez realizados todos estos pasos, estamos listos para empezar con la implementación de
# técnicas para la transformación de datos.


# REBALANCEO DE DATOS
#--------------------

# Empezaremos comprobando el número de muestras para cada una de las clases que tiene nuestra
# variable de salida para identificar si tenemos un conjunto de datos desbalanceado.

sns.countplot(data=data_cod, x="Churn")

from collections import Counter
counter_total = Counter(data_cod["Churn"])
print(counter_total)

# Efectivamente notamos que existe una diferencia notable en el número de datos clasificados a
# cada clase, en este caso, nuestra clase minoritaria vendria a ser "1" (clientes desertores),
# el cual es aproximandamente 5 veces menor a nuestra clase mayoritaria "0" (clientes no desertores).

# Las consecuencias de tener datos desbalanceados se dan a relucir cuando el modelo de predicción
# que utilizemos tenga un rendimiento deficiente al momento de predecir datos catalogados con la
# clase minoritaria, y buenas predicciones hacia datos de la clase mayoritaria, puesto que estará
# sesgado hacia la clase que mayor presencia tiene en el conjunto de datos, lo cual no es el
# resultado que esperamos.

# Existen diversas técnicas para solucionar este problema, como el sobremuestreo (creación de
# nuevas muestras sintéticas en la clase minoritaria para igualar la cantidad de muestras de la
# clase mayoritaria), submuestreo (reducción de la cantidad de muestras de la clase mayoritaria
# para igualar la cantidad de muestras de la clase minoritaria), modelos híbridos(aplica ambas
# técnicas mencionadas), entre otros.

# En este caso, debido a que en nuestro conjunto de datos tenemos variables categóricas y
# numéricas, haremos uso del sobremuestreo, siendo mas especificos, utilizaremos la técnica
# SMOTE-NC, la cual está basada en el algoritmo de aprendizaje automatico KNN, el cual
# utilizará la distancia euclidiana para generar nuevos datos que mayor se ajusten a la
# realidad a partir de los que ya tenemos.

# Como anteriormente habíamos explicado, para evitar sufrir de fuga de datos implementaremos
# la técnica SMOTE-NC solo en nuestros conjuntos de entrenamiento, dejando intactos los de
# evaluación, ya que es recomendable que estos esten íntegros para obtener resultados confiables
# en las métricas que evalúen nuestro modelo.

#--------
# Antes del rebalanceo

# Número de muestras para cada clase en el conjunto de entrenamiento antes del rebalanceo
counter_before = Counter(y_train)
print(counter_before)
print(y_train.shape)
plt.bar(counter_before.keys(), counter_before.values())

#--------
# Después del rebalanceo

# Lista que almacenará la posición de nuestras variables categóricas en el conjunto de datos
categoricas = []  
for i in range(3,30):
    categoricas.append(i)
    
# Inicializamos y ejecutamos la técnica SMOTE-NC
smnc = SMOTENC(categorical_features= categoricas, random_state=21)
X_train_bal, y_train_bal = smnc.fit_resample(X_train, y_train)

# Número de muestras para cada clase en el conjunto de entrenamiento después del rebalanceo
counter_after = Counter(y_train_bal)
print(counter_after)
print(y_train_bal.shape)
plt.bar(counter_after.keys(), counter_after.values())

# Podemos observar que ahora el número de muestras para cada clase en nuestros datos de entrenamiento
# están perfectamente balanceados los unos de los otros, por ende, habremos alivianado en cierta
# medida el problema de los datos desbalanceados y el sesgo hacia la clase mayoritaria.


# REDUCCIÓN DE LA DIMENSIONALIDAD
#--------------------------------

# Debido a que tenemos muchas variables de entrada nos vemos en la necesidad de reducir la dimensión
# de nuestro conjunto de datos para evitar los problemas asociados a la alta dimensionalidad. El
# tener una alta dimensionalidad provoca que nuestro modelo predictivo caiga con mucha frecuencia
# en el sobreajuste y sea incapáz de generalizar al momento de realizar una predicción en base a
# datos nunca antes vistos por el modelo, otro problema de tener muchas variables de entrada es
# que el coste computacional aumenta exponencialmente en el proceso de entrenamiento y predicción,
# haciendo tedioso el proceso que conyeva la construcción y validación de nuestro algoritmo
# predictivo.

# Es por ello que en esta ocasión utilizaremos una técnica estadística muy popular llamada "Análisis
# de los componentes principales", conocido por sus siglas como PCA, el cual consta en tomar
# todas las variables de entrada de nuestro conjunto de datos y realizar tantas combinaciones
# lineales como variables de entrada tengamos, estas combinaciones lineales se les denomina
# componentes, y es a traves de estos que según sea el número de componentes que elijamos nos
# encontraremos con un nuevo conjunto de datos mas pequeño que el original que explica una parte
# de su información total y varianza de sus datos. 

# Antes de implementar PCA, será necesario estandarizar nuestros datos para que no existan variables
# con más peso que otras y el algoritmo pueda trabajar de forma correcta al momento de calcular
# cada componente.

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_bal)
X_test_sc = sc.transform(X_test)

# Una vez estandarizados nuestros datos, procederemos a implementar PCA en ellos.

# Como en un principio desconocemos el número de componentes óptimos que explican la mayor
# información y varianza de nuestro conjunto de datos, estableceremos "None" en el parámetro
# "n_components" con el objetivo de que la función nos muestre todos los componentes que pueda calcular
# y raiz de ello poder visualizar y elegir a nuestro criterio el numero de componentes óptimos
# que resumen la mayor parte de la información de nuestros datos.

pca = PCA(n_components=None)
X_train_none = pca.fit_transform(X_train_sc)
X_test_none = pca.transform(X_test_sc)

# Posterior a esto procederemos a crear una variable que almacene el array que contiene los
# porcentajes de la varianza explicada en forma ascendente para cada componente, la cual
# utilizaremos para construir una gráfica y visualizar mejor que número de componentes que nos 
# conviene elegir.

varianza_explicada = pca.explained_variance_ratio_

# Para realizar el gráfico necesitaremos un array donde los procentajes obtenidos anteriormente
# se sumen de forma secuencial cada vez que aumentemos de componente, es por ello que aplicaremos
# la función "cumsum" a nuestro array anterior.

varianza_acumulada = varianza_explicada.cumsum()

# Con este nuevo array ahora si procederemos a realizar la gráfica

plt.plot(range(1,31), varianza_acumulada, marker = 'o')
plt.grid()
plt.show()

# Podemos observar que varianza explicada deja de crecer en el componente 27, y que en el componente
# 20 tenemos aproximadamente un 98% de la varianza explicada, el cual es un valor excelente ya que
# casi no estamos perdiendo información y habremos reducido en 10 el numero de variables
# de entrada de nuestro conjunto de datos, por lo cual, consideraremos este número de componentes
# como el más optimo al ejecutar esta vez de forma definitiva la técnica del PCA.

pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)

# Con este último paso realizado, podemos observar que nuestros conjuntos de datos pasaron de tener
# 30 variables a tener 20, lo cual indica que la técnica se ejecuto correctamente y que estamos
# listos para la contrucción y evaluación de nuestro modelo predictivo.


#-------------------------------------------------------------------------------------------------
#                          CONSTRUCCIÓN Y EVALUACIÓN DEL MODELO PREDICTIVO
#-------------------------------------------------------------------------------------------------

# Para este proyecto utilizaremos el algoritmo del aumento del gradiente (Gradient Boosting),
# especificamente en una de sus versiones potenciadas y optimizadas, XGBoost.

# El principal motivo por el que usaremos este tipo de algoritmo es debido a que en la mayoria
# de ocaciones, si se le suministra una correcta combinacion de hiperparametros, obtiene mejores
# resultados al momento de predecir a comparacion con sus predecesores, los arboles de decision y
# los bosques aleatorios, y porque esta familia de algoritmos generalmente se ajustan muy bien a
# este tipo de problemas en donde la clasificacion de clases solo depende de la interaccion entre
# variables.

# Resumiendo el funcionamiento de los alogirtmos de Gradient Boosting tenemos:
# Entrena un primer árbol de decisión en base a nuestro conjunto de entrenamiento
# Predice el valor de nuestra variable de salida, compara la predicción con el resultado real y calcula el error cometido
# Entrena un segundo árbol de decisión para tratar de corregir y reducir el error cometido del primer árbol
# Predice nuevamente el valor de nuestra variable de salida y calcula el error cometido
# Entrena un tercer árbol para tratar de corregir y reducir el error cometido de manera conjunta por el primer y segundo árbol
# Predice otra vez el valor de nuestra variable de salida y calcula el error cometido
# Este proceso se realiza iterativamente hasta llegar a un punto en donde no se pueda reducir más
# el error cometido y se da por válido el modelo.
# Este modelo predecirá nuevos datos en base al promedio de todas las predicciones de los árboles
# de decisión con el que ha sido entrenado, dando más peso a aquellos árboles con poco margen de
# error cometido.

# Una vez hecha un pequeña introducción sobre los algoritmos de Gradient Boosting procederemos
# a realizar la construcción y evalución de nuestro modelo


# ELECCIÓN DE LA MEJOR COMBINACIÓN DE PARÁMETROS
#-----------------------------------------------

# XGBoost depende mucho de la combinación de hiperparámetros que se le suministren para tener una
# precisión y eficacia superior a la otros modelos, es por ello que utilizaremos un framework
# muy popular llamado Optuna para entrenar distintos modelos con distintas combinaciones de
# hiperparámetros, con el fin de elegir la combinación que una mayor precisión nos arroje

# Procederemos a crear en un diccionario con los valores de los hiperparámetros más relevantes con
# los que queremos evaluar a nuestro modelo 


#------------------------------
# DATOS BALANCEADOS POR XGBOOST

def objective(trial):   
    
    # Estos parámetros con los que se realizaran las combinaciones previamente han sido elegidos
    # a partir del rango que mejor resultados mostraron
    
    # Inicializaremos el modelo con una métrica de evaluación basada en las curvas AUC-ROC, ya que
    # es una buena métrica para determinar si el modelo distingue bien las clases de nuestra variable
    # de salida, y con un objetivo binario logístico para que AUC-ROC pueda funcionar correctamente

    params = {"n_estimators": trial.suggest_int("n_estimators",200,1200,50),
              "max_depth": trial.suggest_int("max_depth", 12, 25, 1),
              "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.5),
              "subsample": trial.suggest_discrete_uniform("subsample", 0.1, 1, 0.1),
              "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.3, 1,0.1),
              "scale_pos_weight": 2.76,
              "tree_method": "gpu_hist", 
              "eval_metric": "auc",
              "objective": "binary:logistic",
              "use_label_encoder": "False"}
    
    model = XGBClassifier(**params)   
    
    model.fit(X_train,y_train,eval_set=[(X_test,y_test)],early_stopping_rounds=100,verbose=False)
    
    preds = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, preds)
    
    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=70)

print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

best_1 = study.trials_dataframe()

optuna.visualization.plot_optimization_history(study)  #Necesita plotly

# Se ejecutó la función tres veces de forma independiente, y posterior a ello, se registro
# la mejor combinación de parametros que arrojo cada ejecución.

# 79.22% | n_estimators=300, max_depth=18, learning_rate=0.0116, subsample=0.2, colsample_bytree=0.8 
# 79.17% | n_estimators=300, max_depth=16, learning_rate=0.0137, subsample=0.1, colsample_bytree=0.8 
# 79.03% | n_estimators=400, max_depth=18, learning_rate=0.0013, subsample=0.2, colsample_bytree=0.9 


# Procederemos a entrenar un nuevo modelo XGBoost en base a las tres combinaciones de hiperparámetros
# obtenidas para determinar cual de ellas presenta mejores resultados al clasificar nuestros datos

# Para la primera combinación
xgb_1a = XGBClassifier(tree_method='gpu_hist', objective="binary:logistic", use_label_encoder=False, seed=21,
                       n_estimators=300, max_depth=18, learning_rate=0.0116, subsample=0.2, 
                       colsample_bytree=0.8, scale_pos_weight=2.76)

xgb_1a.fit(X_train, y_train)
y_pred_1a = xgb_1a.predict(X_test)

# Para la segunda combinación
xgb_1b = XGBClassifier(tree_method='gpu_hist', objective="binary:logistic", use_label_encoder=False, seed=21,
                       n_estimators=300, max_depth=16, learning_rate=0.0137, subsample=0.1, 
                       colsample_bytree=0.8, scale_pos_weight=2.76)

xgb_1b.fit(X_train, y_train)
y_pred_1b = xgb_1b.predict(X_test)

# Para la tercera combinación
xgb_1c = XGBClassifier(tree_method='gpu_hist', objective="binary:logistic", use_label_encoder=False, seed=21,
                       n_estimators=400, max_depth=18, learning_rate=0.0013, subsample=0.2, 
                       colsample_bytree=0.9, scale_pos_weight=2.76)

xgb_1c.fit(X_train, y_train)
y_pred_1c = xgb_1c.predict(X_test)


# COMPARACIÓN DE RENDIMIENTO ENTRE COMBINACIONES

# Para la primera combinación
f1_1a = f1_score(y_test, y_pred_1a)
auc_1a = accuracy_score(y_test, y_pred_1a)
report_1a = classification_report(y_test,y_pred_1a)

# Para la segunda combinación
f1_1b = f1_score(y_test, y_pred_1b)
auc_1b = accuracy_score(y_test, y_pred_1b)
report_1b = classification_report(y_test,y_pred_1b)

# Para la tercera combinación
f1_1c = f1_score(y_test, y_pred_1c)
auc_1c = accuracy_score(y_test, y_pred_1c)
report_1c = classification_report(y_test,y_pred_1c)


print("F1 primera comb.: %.2f%%" % (f1_1a * 100.0))
print("Accuracy primera comb.: %.2f%%" % (auc_1a * 100.0))
print("-------------------------------")
print("F1 segunda comb.: %.2f%%" % (f1_1b * 100.0))
print("Accuracy segunda comb.: %.2f%%" % (auc_1b * 100.0))
print("-------------------------------")
print("F1 tercera comb.: %.2f%%" % (f1_1c * 100.0))
print("Accuracy tercera comb.: %.2f%%" % (auc_1c * 100.0))

print(report_1a)
print("-------------------------------------------------")
print(report_1b)
print("-------------------------------------------------")
print(report_1c)

# Observamos que la tercera combinación tiene mejores valores de métrica que las demas combinaciones

# Procederemos a graficar la matriz de confusión y la curva ROC-AUC, por ultimo obtendremos su valor
# como métrica para tomar la decisión final sobre que combinación elegir 

plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, y_pred_1a), annot=True, fmt = "d", linecolor="k", linewidths=3)
plt.title("CONFUSION MATRIX 1A",fontsize=14)
plt.show()

plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, y_pred_1b), annot=True, fmt = "d", linecolor="k", linewidths=3)
plt.title("CONFUSION MATRIX 1B",fontsize=14)
plt.show()

plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, y_pred_1c), annot=True, fmt = "d", linecolor="k", linewidths=3)
plt.title("CONFUSION MATRIX 1C",fontsize=14)
plt.show()

# A simple vista observamos que la combinación 3 tiene un mejor balance entre verdaderos positivos
# y falsos positivos respecto a las demás combinaciones

y_pred_prob1a = xgb_1a.predict_proba(X_test)[:,1]
fpr_1a, tpr_1a, thresholds_1a = roc_curve(y_test, y_pred_prob1a)
y_pred_prob1b = xgb_1b.predict_proba(X_test)[:,1]
fpr_1b, tpr_1b, thresholds_1b = roc_curve(y_test, y_pred_prob1b)
y_pred_prob1c = xgb_1c.predict_proba(X_test)[:,1]
fpr_1c, tpr_1c, thresholds_1c = roc_curve(y_test, y_pred_prob1c)

plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_1a, tpr_1a, label='Combinación 1',color = "r")
plt.plot(fpr_1b, tpr_1b, label='Combinación 2',color = "g")
plt.plot(fpr_1c, tpr_1c, label='Combinación 3',color = "b")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve',fontsize=16)
plt.legend()
plt.show()

# En el grafico de la curva no es posible distinguir con claridad cual combinacion es la que mejor
# AUC tiene, asi que procederemos a calcular su valor en forma de porcentaje

auc_1a = roc_auc_score(y_test, y_pred_1a)
auc_1b = roc_auc_score(y_test, y_pred_1b)
auc_1c = roc_auc_score(y_test, y_pred_1c)

print("AUC primera comb.: %.2f%%" % (auc_1a * 100.0))
print("AUC segunda comb.: %.2f%%" % (auc_1b * 100.0))
print("AUC tercera comb.: %.2f%%" % (auc_1c * 100.0))

# Con este ultimo paso observamos que la combinación 3 tiene un mayor valor tanto de AUC como
# puntaje F1 en comparacion con las demas combinaciones, se ha elegido la mejor combiancion en
# base a estas metricas debido a que son mucho mas utiles que la precision al momento de evaluar
# un modelo de clasificacion binaria desbalanceado, por lo tanto, utilizaremos esta
# combinacion como referente del modelo de "Datos rebalanceados con XGBoot".
 

#-------------------------------
# DATOS REBALANCEADOS CON SMOTE-NC

def objective(trial):   
    
    params = {"n_estimators": trial.suggest_int("n_estimators",200,1200,50),
              "max_depth": trial.suggest_int("max_depth", 10, 25, 1),
              "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.5),
              "subsample": trial.suggest_discrete_uniform("subsample", 0.3, 1, 0.1),
              "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.3, 1,0.1),
              "tree_method": "gpu_hist", 
              "eval_metric": "auc",
              "objective": "binary:logistic",
              "use_label_encoder": "False"}
    
    model = XGBClassifier(**params)   
    
    model.fit(X_train_bal,y_train_bal,eval_set=[(X_test,y_test)],early_stopping_rounds=100,verbose=False)
    
    preds = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, preds)
    
    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=70)

print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

best_2 = study.trials_dataframe()

# 78.89% | n_estimators=700, max_depth=17, learning_rate=0.0113, subsample=0.8, colsample_bytree=0.8  
# 78.65% | n_estimators=450, max_depth=15, learning_rate=0.0021, subsample=0.6, colsample_bytree=0.8   
# 78.56% | n_estimators=450, max_depth=14, learning_rate=0.0026, subsample=0.8, colsample_bytree=0.6  

# Procederemos a entrenar un nuevo modelo XGBoost en base a las tres combinaciones de hiperparámetros
# obtenidas para determinar cual de ellas presenta mejores resultados al clasificar nuestros datos

# Para la primera combinación
xgb_2a = XGBClassifier(tree_method='gpu_hist', objective="binary:logistic", use_label_encoder=False, seed=21,
                       n_estimators=700, max_depth=17, learning_rate=0.0113, subsample=0.8, 
                       colsample_bytree=0.8)

xgb_2a.fit(X_train_bal, y_train_bal)
y_pred_2a = xgb_2a.predict(X_test)

# Para la segunda combinación
xgb_2b = XGBClassifier(tree_method='gpu_hist', objective="binary:logistic", use_label_encoder=False, seed=21,
                       n_estimators=450, max_depth=15, learning_rate=0.0021, subsample=0.6, 
                       colsample_bytree=0.8)

xgb_2b.fit(X_train_bal, y_train_bal)
y_pred_2b = xgb_2b.predict(X_test)

# Para la tercera combinación
xgb_2c = XGBClassifier(tree_method='gpu_hist', objective="binary:logistic", use_label_encoder=False, seed=21,
                       n_estimators=450, max_depth=14, learning_rate=0.0026, subsample=0.8, 
                       colsample_bytree=0.6)

xgb_2c.fit(X_train_bal, y_train_bal)
y_pred_2c = xgb_2c.predict(X_test)


# COMPARACIÓN DE RENDIMIENTO ENTRE COMBINACIONES

# Para la primera combinación
f1_2a = f1_score(y_test, y_pred_2a)
auc_2a = accuracy_score(y_test, y_pred_2a)
report_2a = classification_report(y_test,y_pred_2a)

# Para la segunda combinación
f1_2b = f1_score(y_test, y_pred_2b)
auc_2b = accuracy_score(y_test, y_pred_2b)
report_2b = classification_report(y_test,y_pred_2b)

# Para la tercera combinación
f1_2c = f1_score(y_test, y_pred_2c)
auc_2c = accuracy_score(y_test, y_pred_2c)
report_2c = classification_report(y_test,y_pred_2c)


print("F1 primera comb.: %.2f%%" % (f1_2a * 100.0))
print("Accuracy primera comb.: %.2f%%" % (auc_2a * 100.0))
print("-------------------------------")
print("F1 segunda comb.: %.2f%%" % (f1_2b * 100.0))
print("Accuracy segunda comb.: %.2f%%" % (auc_2b * 100.0))
print("-------------------------------")
print("F1 tercera comb.: %.2f%%" % (f1_2c * 100.0))
print("Accuracy tercera comb.: %.2f%%" % (auc_2c * 100.0))

print(report_2a)
print("-------------------------------------------------")
print(report_2b)
print("-------------------------------------------------")
print(report_2c)

# Observamos que la tercera combinación tiene mejores valores de métrica que las demas combinaciones

# Procederemos a graficar la matriz de confusión y la curva ROC-AUC, por ultimo obtendremos su valor
# como métrica para tomar la decisión final sobre que combinación elegir 

plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, y_pred_2a), annot=True, fmt = "d", linecolor="k", linewidths=3)
plt.title("CONFUSION MATRIX 2A",fontsize=14)
plt.show()

plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, y_pred_2b), annot=True, fmt = "d", linecolor="k", linewidths=3)
plt.title("CONFUSION MATRIX 2B",fontsize=14)
plt.show()

plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, y_pred_2c), annot=True, fmt = "d", linecolor="k", linewidths=3)
plt.title("CONFUSION MATRIX 2C",fontsize=14)
plt.show()

# A simple vista podemos descartar la combinacion 1 ya que su balance entre verdaderos positivos
# y falsos positivos es menor en comparacion con las demás combinaciones, y que tanto la combinacion
# 2 como 3 tienen distribuciones similares

y_pred_prob2a = xgb_2a.predict_proba(X_test)[:,1]
fpr_2a, tpr_2a, thresholds_2a = roc_curve(y_test, y_pred_prob2a)
y_pred_prob2b = xgb_2b.predict_proba(X_test)[:,1]
fpr_2b, tpr_2b, thresholds_2b = roc_curve(y_test, y_pred_prob2b)
y_pred_prob2c = xgb_2c.predict_proba(X_test)[:,1]
fpr_2c, tpr_2c, thresholds_2c = roc_curve(y_test, y_pred_prob2c)

plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_2a, tpr_2a, label='Combinación 1',color = "r")
plt.plot(fpr_2b, tpr_2b, label='Combinación 2',color = "g")
plt.plot(fpr_2c, tpr_2c, label='Combinación 3',color = "b")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve',fontsize=16)
plt.legend()
plt.show()

# En el grafico distinguimos que la combinacion 1 tiene la peor curva AUC, y que tanto las
# combinaciones 2 como 3, tienen un ajuste similar, por lo que calcularan sus valores en forma
# de porcentaje

auc_2a = roc_auc_score(y_test, y_pred_2a)
auc_2b = roc_auc_score(y_test, y_pred_2b)
auc_2c = roc_auc_score(y_test, y_pred_2c)

print("AUC primera comb.: %.2f%%" % (auc_2a * 100.0))
print("AUC segunda comb.: %.2f%%" % (auc_2b * 100.0))
print("AUC tercera comb.: %.2f%%" % (auc_2c * 100.0))


# Con todo lo anterior mostrado, siendo meticulosos con los resultados, observamos que la tercera
# combinacion tiene ligeramente mejores valores de métrica que las demás combinaciones, por lo
# tanto, se eligira a esta combinacion como referente del modelo de "Datos rebalanceados con SMOTE-NC".


#--------------------
# DATOS ESCALADOS, REBALANCEADOS Y PCA

def objective(trial):   
    
    params = {"n_estimators": trial.suggest_int("n_estimators",300,1500,50),
              "max_depth": trial.suggest_int("max_depth", 10, 25, 1),
              "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.5),
              "subsample": trial.suggest_discrete_uniform("subsample", 0.3, 1, 0.1),
              "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.3, 1,0.1),
              "tree_method": "gpu_hist", 
              "eval_metric": "auc",
              "objective": "binary:logistic",
              "use_label_encoder": "False"}
    
    model = XGBClassifier(**params)   
    
    model.fit(X_train_pca,y_train_bal,eval_set=[(X_test_pca,y_test)],early_stopping_rounds=100,verbose=False)
    
    preds = model.predict(X_test_pca)
    
    accuracy = accuracy_score(y_test, preds)
    
    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=70)

print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

a = study.trials_dataframe()

# 77.71% colsample=0.4, learning_rate=0.0033, max_depth=15, n_estimators=1000, subsample=0.4
# 77.71% colsample=0.7, learning_rate=0.091, max_depth=25, n_estimators=800, subsample=0.4
# 77.78% colsample=0.3, learning_rate=0.001, max_depth=22, n_estimators=1200, subsample=0.5

xgb_gs = XGBClassifier(tree_method='gpu_hist', objective="binary:logistic", use_label_encoder=False, seed=21,
                       colsample_bytree=0.8, learning_rate=0.0021, max_depth=15, n_estimators=450,
                       subsample=0.6)

xgb_gs.fit(X_train_bal, y_train_bal)
y_pred_gs = xgb_gs.predict(X_test)


accuracy = accuracy_score(y_test, y_pred_gs)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


report = classification_report(y_test,y_pred_gs)
print(report)


scores = cross_val_score(xgb_gs, X_train, y_train, cv=10)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, y_pred_gs),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.title("FINAL CONFUSION MATRIX",fontsize=14)
plt.show()



# Mejor modelo hasta ahora
# colsample=0.7, learning_rate=0.001, max_depht=19, n_estimator=900, subsample=0.5

# Mejor modelo precision hasta ahora 
# colsample=0.7, learning_rate=0.001, max_depht=16, n_estimator=1150, subsample=0.4





