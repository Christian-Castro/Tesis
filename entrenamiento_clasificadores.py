import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

#función cargar archivo y asigno X y y, y hago 80-20

def mifun(n_File):

  ruta='/content/output_' + str(n_File) + '.csv'
  df = pd.read_csv(ruta, encoding='latin1')
  

  # Seleccionamos columnas de características
  feature_cols = ['cosine', 'euclidean', 'manhattan', 'chi2', 'cityblock', 'l1', 'l2', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'minkowski', 'sqeuclidean']

  X = df[feature_cols]
  #X

  # seleccionamos columna objetivo
  y = df['truth_binary']


  # Usaremos validación cruzada para evaluar
  from sklearn.model_selection import cross_validate
  from sklearn.model_selection import train_test_split

  # Dejamos 20% para validación final
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

  return X_train, y_train

#función eval_classifiers

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_validate

def eval_classifiers(X_train, y_train):
  clfs = [
          ('Decision tree', DecisionTreeClassifier(random_state=45)),
          ('RandomForest', RandomForestClassifier(n_estimators=20, random_state=45)),
          ('MLP', MLPClassifier(max_iter=1000, random_state=45)),
          ('GradientBoost',GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)),
          ]


  # Vamos devolver los resultados como una tabla
  # Cada fila un algoritmo, cada columna un resultado
  metrics = ['accuracy', 'precision', 'recall', 'f1']
  results = pd.DataFrame(columns=metrics)
  for alg, clf in clfs:
    scores = cross_validate(clf, X_train, y_train, cv=10, scoring=metrics) # por defecto, es estratificado
    results.loc[alg,:] = [scores['test_'+m].mean() for m in metrics]
  return results

#Principal que llama funciones para la cantidad de palabras

from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore') # filtramos warnings

#aqui empieza todo


erows = np.arange(50, 1001, 50)  # etiquetas numero de palabras

nwords = np.arange(0,len(erows),1)   # len(nwords) sería el número de ejecuciones



met = ['Decision tree','RandomForest','MLP','GradientBoost']
nmet = np.arange(0,len(met),1)

#creo el df de cada métrica
accu = pd.DataFrame(columns=met)
prec = pd.DataFrame(columns=met)
rec = pd.DataFrame(columns=met)
funo = pd.DataFrame(columns=met)


# 4 decimales para cada valor en Pandas
pd.options.display.float_format = '{:,.4f}'.format


# este es el for principal para nwords
for e in erows:

  # llamo función cargar archivos, asigna X y y hace 80-20
  #v = erows[n]
  X_train, y_train = mifun(e)

  #Pre Procesamiento - Escalado (antes de pasar al modelo?)
  scaler = preprocessing.StandardScaler()

  # llama función eval_classifiers que devuelve un results para cada n palabras
  resul = eval_classifiers(scaler.fit_transform(X_train), y_train)

  ac = resul['accuracy']
  pr = resul['precision']
  re = resul['recall']
  f1 = resul['f1']

  # aquí separo para cada métrica y grafico

  #ESTO ESTA OK
  accu.loc[e] = [ac[nm] for nm in nmet]
  prec.loc[e] = [pr[nm] for nm in nmet]
  rec.loc[e] = [re[nm] for nm in nmet]
  funo.loc[e] = [f1[nm] for nm in nmet]

print('F1 TODAS LAS PALABRAS')
print(funo)

#funcion para graficar
fig, ax = plt.subplots(4)

fig.set_figheight(20)
fig.set_figwidth(20)

ax[0].plot(accu)
ax[0].set_title('Accuracy', loc = "left", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
ax[0].legend(met)
ax[0].set_xlim([0,1])
ax[0].set_xticks(range(0,1001,50))
ax[1].plot(prec)
ax[1].set_title('Precision', loc = "left", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
ax[1].legend(met)
ax[1].set_xticks(range(0,1001,50))
ax[2].plot(rec)
ax[2].set_title('Recall', loc = "left", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
ax[2].legend(met)
ax[2].set_xticks(range(0,1001,50))
ax[3].plot(funo)
ax[3].set_title('f1', loc = "left", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
ax[3].legend(met)
ax[3].set_xticks(range(0,1001,50))
plt.show()

#termina el for principal






print(resul)

resul1 = resul.transpose()
resul1.plot()

