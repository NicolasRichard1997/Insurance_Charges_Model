Title: Formation et Déploiement d'un Modèle d'Apprentissage Automatique
Date: 16 novembre 2023
Catégorie : Blog
Slug : modèle-de-regression, 
Auteurs : Nicolas Richard, Brian Schmidt


Répertoire GitHub public: https://github.com/NicolasRichard1997/Insurance_Charges_Model

Résumé : Ce blog est une compilation, une adaptation et une traduction de deux projets par Brian Schmidt (https://www.tekhnoal.com/regression-model.html,https://www.tekhnoal.com/logging-for-ml-models) qui vise à intégrer plusieurs techniques de science de données et de génie logiciel à u  modèle d'apprentissage automatique. Dans ce blog, j'utiliserai des packages Python open source pour effectuer une exploration de données automatisée, une ingénierie de fonction automatisée, un apprentissage automatique automatisé et une validation de modèle. Dans un second, j'incopererai les "logs" du second blog afin de pouvoir plus efficacement utiliser et dépanner le modèle. J'utiliserai également Docker et Kubernetes pour déployer le modèle localement. Le blog couvrira l'ensemble de la base de code du modèle, de l'exploration initiale des données au déploiement du modèle derrière une API RESTful dans Kubernetes.

# Introduction

L'ingénierie de fonction automatisée est une technique utilisée pour automatiser la création de fonctionnalités à partir d'un ensemble de données sans avoir à les concevoir manuellement et à écrire le code pour les créer. L'ingénierie de fonction est très importante pour pouvoir créer des modèles ML qui fonctionnent bien sur un ensemble de données, mais cela prend beaucoup de temps et d'efforts. L'ingénierie de fonction automatisée peut générer de nombreuses fonctionnalités candidates à partir d'un ensemble de données donné, parmi lesquelles nous pouvons ensuite sélectionner les plus utiles. Dans ce billet de blog, j'utiliserai la bibliothèque feature_tools, qui aide à effectuer le prétraitement des fonctionnalités, la sélection des fonctionnalités, la sélection du modèle et la recherche d'hyperparamètres.

L'apprentissage automatique automatisé est un processus grâce auquel nous pouvons créer des modèles d'apprentissage automatique sans avoir à explorer de nombreux types de modèles et hyperparamètres différents. AutoML peut automatiser le processus de choix de la meilleure solution pour un ensemble de données, passant d'un ensemble de données brut à un modèle entraîné. Les outils AutoML permettent aux non-experts de créer des modèles ML sans avoir à comprendre tout ce qui se passe sous le capot. Tout ce dont on a besoin est un ensemble de données correctement traité et n'importe qui peut générer un modèle à partir des données. Dans ce billet de blog, j'utiliserai la bibliothèque TPOT, qui aide à effectuer le prétraitement des fonctionnalités, la sélection des fonctionnalités, la sélection du modèle et la recherche d'hyperparamètres.

Dans ce billet de blog, je montrerai également comment créer un service RESTful pour le modèle qui nous permettra de déployer rapidement et simplement le modèle. Nous montrerons également comment déployer le service modèle à l'aide de Docker et de Kubernetes. Ce billet de blog contient de nombreux outils et techniques différents pour la construction et le déploiement de modèles ML, et il n'est pas destiné à être une plongée approfondie dans l'une des techniques individuelles. Je voulais simplement montrer comment amener un modèle depuis l'exploration des données, jusqu'à la formation et enfin jusqu'au déploiement.

# Structure du Package

La structure du package que nous allons développer dans ce billet de blog est la suivante :
```
- blog_post
- configuration
    - kubernetes_rest_config.yaml
    - rest_configuration.yaml
-data
    - insurance.csv
    - testing_set.csv
    - trainning_set.csv
- insurance_charges_model
    - model_files (fichiers de sortie de l'entraînement du modèle)
    - prediction, package pour le code de prédiction
        - __init__.py
        - model.py (code de prédiction)
        - schemas.py (schémas d'entrée et de sortie du modèle)
        - transformers.py (transformateurs de données)
    - training (package pour le code d'entraînement)
        - data_exploration.ipynb (code d'exploration des données)
        - data_preparation.ipynb (code de préparation des données)
        - model_training.ipynb (code d'entraînement du modèle)
        - model_validation.ipynb (code de validation du modèle)
    - __init__.py
- kubernetes 
    - model_service.yaml
- ml_model_logging
    - filters.py
    - __init__.py
    - logging_decorator.py
- tests
    - __init__.py
    - model_test.py
    - transformers_test.py
- venv (environment virtuel)
- LICENCE
- Dockerfile (instructions pour générer une image Docker)
- Makefile
- requirements.txt (liste des dépendances)
- rest_config.yaml (configuration pour le service modèle REST)
- service_contract.yaml (contrat de service OpenAPI)
- setup.py
- test_requirements.txt (dépendances de test)
```
Le code original est disponible dans le répertoire GitHub de Brian Schmidt: https://github.com/schmidtbri/regression-model

# Obtention des Données

Afin de former un modèle de régression, nous devons d’abord disposer d’un ensemble de données.
Nous sommes allés dans Kaggle et avons trouvé [un ensemble de données](https://www.kaggle.com/mirichoi0218/insurance) qui
contenait des informations sur les frais d’assurance. Pour faciliter le téléchargement du
données, nous avons installé le [package kaggle python](https://pypi.org/project/kaggle/). Ensuite, nous avons exécuté ces
commandes pour télécharger les données et les décompresser dans le dossier de données dans le
projet:

```
mkdir -p data
kaggle datasets download -d mirichoi0218/insurance -p ./data \--unzip
```

Pour faciliter encore plus le téléchargement des données, nous avons ajouté une cible Makefile
pour les commandes :

```
download-dataset: ## télécharger l'ensemble de données depuis Kaggle
    mkdir -p data
    kaggle datasets download -d mirichoi0218/insura
```

Il ne nous reste plus qu'à exécuter cette commande :

```
make download-data
```

Au lieu de devoir se rappeler comment obtenir les données nécessaires à la modélisation,
J'essaie toujours de créer un processus reproductible et documenté pour créer
l’ensemble de données. Nous veillons également à ne jamais stocker l'ensemble de données dans la source
contrôle, nous allons donc ajouter cette ligne au fichier .gitignore :

```
data/
```

# Entraînement d'un Modèle de Régression

Maintenant que nous avons l'ensemble de données, nous allons commencer à travailler sur l'entraînement d'un modèle de régression. Nous effectuerons une exploration des données, une préparation des données, une ingénierie des fonctionnalités, un entraînement automatique et une sélection de modèle, ainsi qu'une validation du modèle.

## Exploration des Données

L'exploration des données est une étape clé qui peut nous fournir beaucoup d'informations sur l'ensemble de données que nous devons modéliser. L'exploration des données peut être hautement personnalisée pour un ensemble de données spécifique, mais il existe également des outils qui nous permettent de calculer automatiquement les choses les plus courantes que nous voulons savoir sur un ensemble de données. ydata_profiling est un package qui prend un dataframe pandas et crée un rapport HTML avec un profil de l'ensemble de données dans le dataframe. Selon la documentation de ydata_profiling, il a ces capacités :

    Inférence de type : détecter les types de colonnes dans un dataframe.
    Essentiels : type, valeurs uniques, valeurs manquantes
    Statistiques de quantiles comme la valeur minimale, Q1, médiane, Q3, maximum, plage, plage interquartile
    Statistiques descriptives comme la moyenne, le mode, l'écart type, la somme, la déviation médiane absolue, le coefficient de variation, le kurtosis, l'asymétrie
    Valeurs les plus fréquentes
    Histogrammes
    Mise en évidence des corrélations entre variables fortement corrélées, matrices Spearman, Pearson et Kendall
    Matrice des valeurs manquantes, comptage, heatmap et dendrogramme des valeurs manquantes
    Lignes dupliquées : liste des lignes dupliquées les plus fréquentes
    Analyse de texte : apprenez-en davantage sur les catégories (majuscules, espaces), les scripts (latin, cyrillique) et les blocs (ASCII) des données textuelles

Ce sont les aspects que nous examinerions pour en savoir plus sur l'ensemble de données. Pour utiliser le package ydata_profiling, nous allons d'abord charger l'ensemble de données dans un dataframe pandas :

```python
import pandas as pd
import os
current_directory = os.getcwd()
repertory_path = os.path.abspath(os.path.join(current_directory, "..", ".."))
from ydata_profiling import ProfileReport
```

Télécharger les données:

```python
data = pd.read_csv(repertory_path + "/data/insurance.csv")
data.head()
```

Maintenant, nous pouvons interroger le dataframe pour connaître les types de colonnes :

```python
data.dtypes
```

```
age int64
sex object
bmi float64
children int64
smoker object
region object
charges float64
dtype: object
```

Pour créer le profil, nous exécuterons ce code :

```python
profile = ProfileReport(data, 
                        title='Insurance Dataset Profile Report',
                        pool_size=4,
                        html={'style': {'full_width': True}})
profile.to_notebook_iframe()
```

Une fois créé, on peut sauvegarder le rapport en HTML :

```python
profile.to_file("data_exploration_report.html")
```
Il est maintenant possible de consulter le rapport en version HTML

## Préparation des Données

Afin de modéliser l'ensemble de données, nous devrons d'abord préparer et prétraiter les données. Pour commencer,
chargeons à nouveau l'ensemble de données dans un dataframe :

```python
import sys
from copy import deepcopy
import warnings
import numpy as np
from numpy import inf, nan
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
import featuretools as ft
import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
current_directory = os.getcwd()
repertory_path = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append("../../")

from insurance_charges_model.prediction.transformers import DFSTransformer
from insurance_charges_model.prediction.transformers import InfinityToNaNTransformer
from insurance_charges_model.prediction.transformers import IntToFloatTransformer
from insurance_charges_model.prediction.transformers import BooleanTransformer

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
```

```python
df = pd.read_csv("../../data/insurance.csv")
```

Pour effectuer la préparation des données, nous utiliserons le package feature_tools pour créer des fonctionnalités à partir des données déjà présentes dans l'ensemble de données. Pour créer des fonctionnalités, nous devrons informer le package feature_tools sur nos données en identifiant les entités dans les données :

```python
entityset = ft.EntitySet(id="Transactions")
entityset = entityset.add_dataframe(dataframe_name="Transactions",
                                    dataframe=df,
                                    make_index=True,
                                    index="index")
entityset
```

Dans le code ci-dessus, nous avons créé un ensemble d'entités avec l'identifiant "Transactions", qui est l'entité présente dans le dataframe. Le package feature_tools a identifié les variables associées à l'entité Transactions :

```python
entityset["Transactions"].variables
```

```
[<Variable: index (dtype = index)>,
<Variable: age (dtype = numeric)>,
<Variable: sex (dtype = categorical)>,
<Variable: bmi (dtype = numeric)>,
<Variable: children (dtype = numeric)>,
<Variable: smoker (dtype = categorical)>,
<Variable: region (dtype = categorical)>,
<Variable: charges (dtype = numeric)>]
```

We can now generate some new features on the entity:

```python
feature_dataframe, features = ft.dfs(entityset=entityset,
                                     target_entity="Transactions",
                                     trans_primitives=["add_numeric", "subtract_numeric",
                                                       "multiply_numeric", "divide_numeric",
                                                       "greater_than", "less_than"],
                                     ignore_variables={"Transactions": ["sex", "smoker", "region", 
                                                                        "charges"]})
```

Le package feature_tools utilise un ensemble d'opérations primitives pour générer de nouvelles fonctionnalités à partir des données. Dans ce cas, nous utilisons la primitive "add_numeric" pour générer une nouvelle fonctionnalité en additionnant les valeurs de toutes les paires de variables numériques. En combinant les variables numériques de cette manière, nous allons générer trois nouvelles colonnes :

    age + bmi
    age + enfants
    bmi + enfants

Les primitives subtract_numeric, multiply_numeric et divide_numeric créent également de nouvelles colonnes de manière similaire, en appliquant respectivement la soustraction, la multiplication et la division. Les primitives greater_than et less_than créent de nouvelles colonnes booléennes en comparant les valeurs de toutes les paires de variables numériques. La primitive greater_than a généré ces nouvelles fonctionnalités :

    age > bmi
    age > enfants
    bmi > age
    bmi > enfants
    enfants > age
    enfants > bmi

À la fin de la génération de fonctionnalités, nous avons 30 nouvelles fonctionnalités dans l'ensemble de données qui ont été générées à partir des données déjà présentes. Avant de pouvoir utiliser ces nouvelles fonctionnalités, nous devons comprendre comment intégrer le transformateur avec les pipelines scikit-learn, que nous utiliserons pour construire notre modèle. Pour ce faire, nous avons créé un transformateur qui est instancié de la manière suivante :

```python
from insurance_charges_model.prediction.transformers import DFSTransformer
```

```python
target_entity = "Transactions"
trans_primitives = ["add_numeric", "subtract_numeric", "multiply_numeric", "divide_numeric", "greater_than", "less_than"]
ignore_variables = {"Transactions": ["sex", "smoker", "region"]}

dfs_transformer = DFSTransformer(target_entity=target_entity, trans_primitives=trans_primitives, ignore_variables=ignore_variables)
```

Étant donné que la génération de fonctionnalités crée parfois des valeurs infinies, nous aurons également besoin d'un transformateur pour les convertir en valeurs NaN. Ce transformateur est instancié de la manière suivante :

```python
infinity_transformer = InfinityToNaNTransformer()
```

Pour gérer les valeurs NaN générées par le transformateur InfinityToNaN, nous utiliserons un SimpleImputer de la bibliothèque scikit-learn. Il est instancié comme ceci :

```python
simple_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
```

Le transformateur SimpleImputer a des problèmes pour imputer des valeurs qui ne sont pas des nombres flottants lorsqu'on utilise la stratégie 'mean'. Pour résoudre cela, nous allons créer un transformateur qui convertira toutes les colonnes entières en colonnes de nombres flottants :

```python
int_to_float_transformer = IntToFloatTransformer()
```

Enfin, nous placerons les transformateurs DFSTransformer, IntToFloatTransformer, InfinityToNaNTransformer et SimpleImputer dans un pipeline pour qu'ils fonctionnent tous ensemble comme une unité :

```python
dfs_pipeline = Pipeline([
    ("dfs_transformer", dfs_transformer),
    ("int_to_float_transformer", int_to_float_transformer),
    ("infinity_transformer", infinity_transformer),
    ("simple_imputer", simple_imputer),
])
```

Ensuite, nous nous occuperons des fonctionnalités booléennes dans l'ensemble de données. Pour ce faire, nous avons créé un transformateur qui convertit les valeurs de chaîne en valeurs booléennes correspondantes. Il est instancié comme suit :

```python
boolean_transformer = BooleanTransformer(true_value="yes", false_value="no")
```

Ce transformateur sera utilisé pour convertir la variable "smoker" en une valeur booléenne. Les valeurs trouvées dans l'ensemble de données sont "yes" et "no". L'encodeur est configuré pour convertir "yes" en True et "no" en False.

Ensuite, nous allons créer un encodeur qui codera les fonctionnalités catégorielles. Les fonctionnalités catégorielles que nous encoderons seront 'sex' et 'region'. Nous utiliserons OrdinalEncoder de la bibliothèque scikit-learn :

```python
ordinal_encoder = OrdinalEncoder()
```

Maintenant, nous pouvons créer un ColumnTransformer qui combine tous les pipelines et transformateurs que nous avons créés ci-dessus en un pipeline plus grand :

```python
column_transformer = ColumnTransformer(remainder="passthrough",
                                       transformers=[
                                           ("dfs_pipeline", dfs_pipeline, ["age", "sex", "bmi",
                                                                           "children", "smoker", "region"]),
                                           ("boolean_transformer", boolean_transformer, ["smoker"]),
                                           ("ordinal_encoder", ordinal_encoder, ["sex", "region"])
                                       ])
```

Le ColumnTransformer applique le pipeline de synthèse de fonctionnalités profondes à toutes les variables d'entrée, puis applique le transformateur booléen à la variable "smoker" et l'encodeur ordinal aux variables "sex" et "region".

Maintenant, nous faisons un petit test pour nous assurer que les transformations se déroulent comme prévu :

```python
test_df = pd.DataFrame([[65, "male", 12.5, 0, "yes", "southwest"],
                        [75, "female", 78.770, 1, "no", "southeast"]],
                       columns=["age", "sex", "bmi", "children", "smoker", "region"])

column_transformer.fit(test_df)

result = column_transformer.transform(test_df)

if len(result[0]) != 33: # expecting 33 features to come out of the ColumnTransformer
    raise ValueError("Unexpected number of columns found in the dataframe.")
```

Pour tester le pipeline, nous avons créé un dataframe avec deux lignes, puis nous avons ajusté le pipeline et transformé le dataframe. Nous nous attendons à obtenir 33 colonnes dans le dataframe de sortie en raison de la synthèse de fonctionnalités profondes, donc nous testons cela et élevons une exception si ce n'est pas le cas.

Le ColumnTransformer peut maintenant être sauvegardé afin que nous puissions l'utiliser ultérieurement dans le processus d'entraînement du modèle :

```python
joblib.dump(column_transformer, "transformer.joblib")
```

Dans cette section, nous avons utilisé des pipelines scikit-learn pour composer une série complexe de transformations de données qui seront exécutées lors de l'entraînement du modèle et également lors de son utilisation pour les prédictions. En utilisant des pipelines, nous nous assurons que les étapes se déroulent toujours dans le même ordre et avec les mêmes paramètres. Si nous n'utilisions pas de pipelines, nous aurions à réécrire les transformations deux fois, une fois pour l'entraînement du modèle et une fois pour la prédiction. Tout le code de préparation des données se trouve dans le cahier data_preparation.ipynb.



## Entrainement du Modèle

La prochaine étape après la préparation des données consiste à entraîner un modèle. Pour ce faire, nous utiliserons le package TPOT, qui est un outil d'apprentissage automatique automatisé capable de rechercher parmi de nombreux types de modèles possibles et hyperparamètres et de trouver le meilleur pipeline pour l'ensemble de données. Le package utilise la programmation génétique pour explorer l'espace des pipelines ML possibles.

Pour entraîner le modèle, nous allons d'abord charger l'ensemble de données :

```python
df = pd.read_csv("../../data/insurance.csv")
```

Ensuite, nous créerons un ensemble d'entraînement et un ensemble de test en sélectionnant des échantillons de manière aléatoire. La répartition entre l'ensemble d'entraînement et l'ensemble de test sera de 80:20.

```python
mask = np.random.rand(len(df)) < 0.8
training_set = df[mask]
testing_set = df[~mask]
```

Ensuite, nous sauvegarderons les ensembles de données dans le dossier "data" car nous en aurons besoin lorsque nous effectuerons la validation du modèle. Étant donné que nous avons choisi de le faire dans un autre cahier Jupyter, nous devons conserver les ensembles de données sur le disque dur jusqu'à ce moment-là.

```python
training_set.to_csv("../../data/training_set.csv")
testing_set.to_csv("../../data/testing_set.csv")
```

Maintenant que nous avons un ensemble d'entraînement, nous devrons séparer les colonnes des caractéristiques de la colonne cible :

```python
feature_columns = ["age", "sex", "bmi", "children", "smoker", "region"]
target_column = "charges"
X_train = training_set[feature_columns]
y_train = training_set[target_column]
X_test = testing_set[feature_columns]
y_test = testing_set[target_column]
```

Ensuite, nous appliquerons le pipeline de prétraitement que nous avons construit dans le code de prétraitement des données. Tout d'abord, nous chargerons le transformateur que nous avons sauvegardé sur le disque :

```python
transformer = joblib.load("transformer.joblib")
```

Maintenant, nous pouvons l'appliquer au dataframe des caractéristiques afin de calculer les fonctionnalités que nous avons créées à l'aide de l'ingénierie des fonctionnalités automatisée :

```python
features = transformer.fit_transform(X_train)
```

Maintenant que nous avons un dataframe de caractéristiques avec lequel nous pouvons entraîner un modèle, nous lancerons l'entraînement en instanciant un objet TPOTRegressor et en appelant la méthode fit :

```python
tpot_regressor = TPOTRegressor(generations=50,
                               population_size=50,
                               random_state=42,
                               cv=5,
                               n_jobs=8,
                               verbosity=2,
                               early_stop=10)

tpot_regressor = tpot_regressor.fit(features, y_train)
```

Le TPOTRegressor utilise la programmation génétique, nous devons donc fournir certains paramètres qui définiront la taille de la population et le nombre de générations. Le paramètre random_state facilite la réplication de l'exécution de l'entraînement, le paramètre cv est le nombre de splits de validation croisée que nous voulons utiliser, le paramètre n_jobs indique à TPOT combien de processus lancer pour entraîner le modèle.

Le processus peut être plus ou moins long, dépendamment de votre machine. Dans mon cas, le processus a pris plus ou moins 45 minutes avec 
le processeur suivant	: Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz


Voici un échantillon de la sortie du tpot_regressor pendant l'entraînement :

```
Optimization Progress: 100%
2550/2550 [35:22<00:00, 1.15pipeline/s]
Generation 1 - Current best internal CV score: -19328040.90181576
Generation 2 - Current best internal CV score: -19328040.90181576
Generation 3 - Current best internal CV score: -19291161.694311526
Generation 4 - Current best internal CV score: -19216662.844604537
Generation 5 - Current best internal CV score: -19194856.36477192

...

Generation 48 - Current best internal CV score: -18848299.473418456
Generation 49 - Current best internal CV score: -18848299.473418456
Generation 50 - Current best internal CV score: -18848299.473418456

Best pipeline:
RandomForestRegressor(MaxAbsScaler(SGDRegressor(Normalizer(input_matrix,
norm=l2), alpha=0.01, eta0=1.0, fit_intercept=True, l1_ratio=0.0,
learning_rate=invscaling, loss=squared_loss, penalty=elasticnet,
power_t=0.1)), bootstrap=True, max_features=0.7500000000000001,
min_samples_leaf=16, min_samples_split=14, n_estimators=100)
```

Il semble que le meilleur pipeline trouvé par TPOT inclut un RandomForestRegressor combiné avec plusieurs étapes de prétraitement. Maintenant que nous avons un pipeline optimal créé par TPOT, nous allons y ajouter nos propres préprocesseurs. Pour ce faire, nous aurons besoin d'avoir un objet pipeline non ajusté, ce que nous n'avons pas actuellement car le pipeline TPOTRegressor a été ajusté.

Pour obtenir un pipeline non ajusté, nous demanderons à TPOT le pipeline ajusté et le clonerons :

```python
unfitted_tpot_regressor = clone(tpot_regressor.fitted_pipeline_)
```

Maintenant que nous avons un pipeline non ajusté qui est le même pipeline trouvé par le package TPOT, nous ajouterons nos propres préprocesseurs au pipeline. Cela garantira que le pipeline final acceptera les fonctionnalités de l'ensemble de données d'origine et traitera correctement les fonctionnalités. Nous composerons le pipeline TPOT non ajusté et le pipeline du transformateur en un seul pipeline :

```python
model = Pipeline([("transformer", transformer),
                  ("tpot_pipeline", unfitted_tpot_regressor)
                  ])
```

On peut maintenant entrainer le modèle:

```python
model.fit(X_train, y_train)
```

The final fitted pipeline contains all of the transformations that we
used to do deep feature synthesis and data preprocessing, and all of the
transformations that were added by TPOT. This is the final pipeline:

```
Pipeline(steps=[('transformer',
                 ColumnTransformer(remainder='passthrough',
                                   transformers=[('dfs_pipeline',
                                                  Pipeline(steps=[('dfs_transformer',
                                                                   DFSTransformer(ignore_variables={'Transactions': ['sex',
                                                                                                                     'smoker',
                                                                                                                     'region']},
                                                                                  target_entity='Transactions',
                                                                                  trans_primitives=['add_numeric',
                                                                                                    'subtract_numeric',
                                                                                                    'multiply_numeric',
                                                                                                    'divide_numeric',
                                                                                                    'greater_than',
                                                                                                    'less_...
                                                  BooleanTransformer(),
                                                  ['smoker']),
                                                 ('ordinal_encoder',
                                                  OrdinalEncoder(),
                                                  ['sex', 'region'])])),
                ('tpot_pipeline',
                 Pipeline(steps=[('variancethreshold',
                                  VarianceThreshold(threshold=0.1)),
                                 ('standardscaler', StandardScaler()),
                                 ('randomforestregressor',
                                  RandomForestRegressor(max_features=0.9000000000000001,
                                                        min_samples_leaf=18,
                                                        min_samples_split=13,
                                                        random_state=42))]))])
```

On procède à un seul test unitaire:

```python
test_df = pd.DataFrame([[65, "male", 12.5, 0, "yes", "southwest"]],
columns=["age", "sex", "bmi", "children", "smoker", "region"])
result = model.predict(test_df)
```

Le résultat:

```
array([24625.65374441])

```

On sérialise et sauvegarde le modèle afin de le réutiliser:

```python
joblib.dump(model, "model.joblib")
```






## Validation du Modèle


```python
import sys
import warnings
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from yellowbrick.regressor import ResidualsPlot, PredictionError
import os
current_directory = os.getcwd()
repertory_path = os.path.abspath(os.path.join(current_directory, "..", ".."))
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
```

Afin de valider le modèle généré par le processus AutoML, nous utiliserons la bibliothèque yellow_brick.

Tout d'abord, nous chargerons les ensembles d'entraînement et de test que nous avons précédemment sauvegardés sur le disque :

```python
training_set = pd.read_csv("../../data/training_set.csv")
testing_set = pd.read_csv("../../data/testing_set.csv")
```

```python
feature_columns = ["age", "sex", "bmi", "children", "smoker", "region"]
target_column = "charges"
X_train = training_set[feature_columns]
y_train = training_set[target_column]
X_test = testing_set[feature_columns]
y_test = testing_set[target_column]
```

Nous chargerons l'objet modèle ajusté qui a été sauvegardé à une étape précédente :

```python
model = joblib.load("model.joblib")
```

```python
predictions = model.predict(X_test)
```

Le coefficient de détermination (r²) du modèle et les erreurs sont calculés comme suit :

```python
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
```

les résulats :

```
Score r² : 0,8598880763704084
Erreur quadratique moyenne : 22 855 821,171577632
Erreur absolue moyenne : 2 651,672336749037
```

Ensuite, nous créerons un visualiseur yellow_brick pour le modèle :

```python
visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
```

Le visualiseur ResidualsPlot nous montre la différence entre la valeur observée et la valeur prédite de la variable cible. Cette visualisation est utile pour voir s'il existe des plages de valeurs pour la variable cible qui présentent plus ou moins d'erreur que d'autres plages de valeurs. 

Ensuite, nous générerons le graphique d'erreur de prédiction pour le modèle en utilisant le visualiseur PredictionError :

```python
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
```

Le graphique d'erreur de prédiction montre les valeurs réelles de la variable cible par rapport aux valeurs prédites générées par le modèle. Cela nous permet de voir quelle est la variance dans les prédictions faites par le modèle.


# Effectuer des prédictions

Le modèle de charges d'assurance est maintenant prêt à être utilisé pour faire des prédictions, donc maintenant nous devons le rendre disponible dans un format facile à utiliser. Le package ml_base définit une classe de base simple pour le code de prédiction de modèle qui nous permet de "envelopper" le code dans une classe qui suit l'interface MLModel. Cette interface publie les informations suivantes sur le modèle :

    Nom qualifié, un identifiant unique pour le modèle
    Nom d'affichage, un nom convivial pour le modèle utilisé dans les interfaces utilisateur
    Description, une description du modèle
    Version, version sémantique de la base de code du modèle
    Schéma d'entrée, un objet qui décrit les données d'entrée du modèle
    Schéma de sortie, un objet qui décrit le schéma de sortie du modèle

L'interface MLModel dicte également que la classe du modèle implémente deux méthodes :

    __init__, méthode d'initialisation qui charge les artefacts du modèle nécessaires pour effectuer des prédictions
    predict, méthode de prédiction qui reçoit les entrées du modèle, fait une prédiction et renvoie les sorties du modèle

En utilisant la classe de base MLModel, nous pourrons faire des choses plus intéressantes plus tard avec le modèle. Si vous souhaitez en savoir plus sur le package ml_base, il y a un article de blog à ce sujet.

Pour installer le package ml_base, exécutez cette commande :

```bash
pip install ml_base
```

## Création de schémas d'entrée et de sortie

Avant d'écrire la classe du modèle, nous devrons définir les schémas d'entrée et de sortie du modèle. Pour ce faire, nous utiliserons le [package pydantic](https://pydantic-docs.helpmanual.io/).

La caractéristique "sex" utilisée par le modèle est une caractéristique catégorielle qui peut être déclarée comme une énumération car elle a un nombre limité de valeurs autorisées :

```python
class SexEnum(str, Enum):
    male = "male"
    female = "female"
```

Nous utiliserons cette classe comme type dans le schéma d'entrée du modèle.

Nous aurons également besoin d'une autre énumération pour la caractéristique de région :

```python
class RegionEnum(str, Enum):
    southwest = "southwest"
    southeast = "southeast"
    northwest = "northwest"
    northeast = "northeast"
```

Maintenant, nous sommes prêts à créer le schéma d'entrée pour le modèle :

```python
class  InsuranceChargesModelInput(BaseModel):

"""Schema for input of the model's predict method."""

  

age: int  =  Field(None, title="Age", ge=18, le=65, description="Age of primary beneficiary in years.")

sex: SexEnum  =  Field(None, title="Sex", description="Gender of beneficiary.")

bmi: float  =  Field(None, title="Body Mass Index", ge=15.0, le=50.0, description="Body mass index of beneficiary.")

children: int  =  Field(None, title="Children", ge=0, le=5, description="Number of children covered by health "

"insurance.")

smoker: bool  =  Field(None, title="Smoker", description="Whether beneficiary is a smoker.")

region: RegionEnum  =  Field(None, title="Region", description="Region where beneficiary lives.")
```

Nous avons utilisé SexEnum et RegionEnum comme types pour les variables catégorielles, ajoutant des descriptions aux champs. Nous avons également ajouté les champs age, bmi, children et smoker. Ces champs sont de type entier, flottant, entier et booléen respectivement.

Nous pouvons utiliser la classe pour créer un objet comme ceci :

```python
from insurance_charges_model.prediction.schemas import InsuranceChargesModelInput

input = InsuranceChargesModelInput(age=22, sex="male", bmi=20.0, children=0, region="southwest")
```
Maintenant que nous avons défini l'entrée du modèle, passons à la sortie du modèle. Cette classe est beaucoup plus simple :

```python
class InsuranceChargesModelOutput(BaseModel):
    charges: float = Field(None, title="Charges", description="Individual medical costs billed by health insurance to customer in US dollars.")
```
Le modèle n'a qu'une seule sortie, les frais en dollars américains qui sont prédits, qui est un champ en virgule flottante. Les schémas du modèle se trouvent dans le le module schemas.py, dans le package de prédiction.

## Création de la classe du modèle

Maintenant que nous avons défini les schémas d'entrée et de sortie pour le modèle, nous pourrons créer la classe qui englobe le modèle.

Pour commencer, nous allons définir la classe et ajouter toutes les propriétés nécessaires :

```python
class InsuranceChargesModel(MLModel):
    @property
    def display_name(self) -> str:
        return "Insurance Charges Model"

    @property
    def qualified_name(self) -> str:
        return "insurance_charges_model"

    @property
    def description(self) -> str:
        return "Model to predict the insurance charges of a customer."

    @property
    def version(self) -> str:
        return __version__

    @property
    def input_schema(self):
        return InsuranceChargesModelInput

    @property
    def output_schema(self):
        return InsuranceChargesModelOutput
```


Les propriétés sont requises par la classe de base MLModel et sont utilisées pour accéder facilement aux métadonnées sur le modèle. Les classes de schéma d'entrée et de sortie sont renvoyées à partir des propriétés input_schema et output_schema.

La méthode \_\_init\_\_ de la classe ressemble à ceci :

```python
def __init__(self):
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    with open(os.path.join(dir_path, "model_files", "1", "model.joblib"), 'rb') as file:
        self._svm_model = joblib.load(file)
```

La méthode init est utilisée pour charger les paramètres du modèle depuis le disque et stocker l'objet modèle en tant qu'attribut d'objet. L'objet modèle sera utilisé pour faire des prédictions. Une fois que la méthode init est terminée, l'objet modèle devrait être initialisé et prêt à faire des prédictions.

La méthode de prédiction de la classe du modèle ressemble à ceci :

```python
def predict(self, data: InsuranceChargesModelInput) -> InsuranceChargesModelOutput:
    X = pd.DataFrame([[data.age, data.sex.value, data.bmi, data.children, data.smoker, data.region.value]], 
                     columns=["age", "sex", "bmi", "children", "smoker", "region"])

    y_hat = round(float(self._svm_model.predict(X)[0]), 2)
    
    return InsuranceChargesModelOutput(charges=y_hat)

```
La méthode predict accepte un objet de type InsuranceChargesModelInput et renvoie un objet de type InsuranceChargesModelOutput. Tout d'abord, la méthode convertit les données entrantes en un dataframe pandas, puis le dataframe est utilisé pour faire une prédiction, et le résultat est converti en un nombre à virgule flottante et arrondi à deux décimales. Enfin, l'objet de sortie est créé en utilisant la prédiction et renvoyé à l'appelant.

La classe du modèle est définie dans le module model.py, dans le package prediction.

# Ajout du décorateur de "Logging" à notre modèle

Nous devons, en premier lieu, télécharger le modèle de Logging depuis GitHub (https://github.com/schmidtbri/logging-for-ml-models). Le modèle inclut les fichiers filters et logging_decorator qui permettent d'effectuer le logging. Si vous désirez en savoir plus sur le logging, vous pouvez suivre le blog original de Brian Schmidt en anglais sur le sujet (https://www.tekhnoal.com/logging-for-ml-models)

Dans la section suivante, nous utiliserons Le package rest_model_service pour créer une API RESTful. Si ce n'est pas déjà fait, il est possible de télécharger le package comme suit:
```
pip install rest_model_service>=0.3.0
```
Le package rest_model_service est capable d'héberger des modèles d'apprentissage automatique et de créer une API RESTful pour chaque modèle individuel. Nous n'avons pas besoin d'écrire de code pour cela, car le service peut décorer les modèles qu'il héberge avec les décorateurs que nous fournissons. Vous pouvez en savoir plus sur le package dans cet article de blog. Vous pouvez apprendre comment le package rest_model_service peut être configuré pour ajouter des décorateurs à un modèle dans cet article de blog. 

Voici un exemple de fonctionnement de "Log" :
```
{"asctime": "2023-11-13 13:44:37,243", "node_ip": "123.123.123.123", "name": "rest_model_service.helpers", "levelname": "INFO", "message": "Created endpoint for insurance_charges_model model."}
```
Les "Logs" du logging package incluent systématiquement l'heure (asctime), le "node_ip" (qui sera utile plus tard lors du déploiement) et le nom du niveau du log (INFO par exemple) et un message.

Le décorateur ajoute également quelques champs au message de journalisation :

-   action : l'action que le modèle effectue, dans ce cas "prédiction"
-   model_qualified_name : le nom qualifié du modèle effectuant l'action
-   model_version : la version du modèle effectuant l'action
-   status : le résultat de l'action, pouvant être soit "succès" soit "erreur"
-   error_info : un champ facultatif qui ajoute des informations d'erreur lorsqu'une exception est levée

Afin d`intégrer ces logs à notre modèle, il suffit d'ajouter le fichier rest_configuration.yaml au répertoire "configuration" de notre package. Le voici:

```Bash
service_title: Insurance Charges Model Service
models:
  - class_path: insurance_charges_model.prediction.model.InsuranceChargesModel
    create_endpoint: true
    decorators:
      - class_path: ml_model_logging.logging_decorator.LoggingDecorator
        configuration:
          input_fields: ["age", "sex", "bmi", "children", "smoker", "region"]
          output_fields: ["charges"]
logging:
    version: 1
    disable_existing_loggers: false
    formatters:
      json_formatter:
        class: pythonjsonlogger.jsonlogger.JsonFormatter
        format: "%(asctime)s %(node_ip)s %(name)s %(levelname)s %(message)s"
    filters:
      environment_info_filter:
        "()": ml_model_logging.filters.EnvironmentInfoFilter
        env_variables:
        - NODE_IP
    handlers:
      stdout:
        level: INFO
        class: logging.StreamHandler
        stream: ext://sys.stdout
        formatter: json_formatter
        filters:
        - environment_info_filter
    loggers:
      root:
        level: INFO
        handlers:
        - stdout
        propagate: true
```

En utilisant le fichier de configuration, nous sommes en mesure de créer un fichier de spécification OpenAPI pour le service modèle en exécutant cette commande :

```bash
export PYTHONPATH=./
generate_openapi \--output_file=service_contract.yaml
```

# Création d'un service RESTful

Maintenant que nous avons une classe de modèle définie et le logging d'inclus, nous sommes enfin en mesure de construire le service RESTful qui hébergera le modèle lorsqu'il sera déployé. Heureusement, nous n'avons en fait besoin d'écrire aucun code pour cela car nous utiliserons le [package rest_model_service](https://pypi.org/project/rest-model-service/). Si vous souhaitez en savoir plus sur le package rest_model_service, Brian Schmidt a un [article de blog](https://schmidtbri.github.io/rest-model-service/articles/basic_usage/basic_usage.html) à ce sujet.

Le fichier service_contract.yaml sera généré et il contiendra la spécification qui a été générée pour le service modèle. Le point de terminaison insurance_charges_model est celui que nous appellerons pour faire des prédictions avec le modèle. Les schémas d'entrée et de sortie du modèle ont été automatiquement extraits et ajoutés à la spécification.

Pour exécuter le service localement, exécutez ces commandes :

```bash
export NODE_IP=123.123.123.123
export PYTHONPATH=./
export REST_CONFIG=./configuration/rest_configuration.yaml
uvicorn rest_model_service.main:app --reload
```
La variable d'environnement NODE_IP est définie de sorte que la valeur puisse être ajoutée aux messages de journal via le filtre que nous avons construit ci-dessus. Le service devrait démarrer et peut être accessible dans un navigateur web à l'adresse [http://127.0.0.1:8000](http://127.0.0.1:8000). Lorsque vous accédez à cette URL, vous serez redirigé vers la page de documentation générée par le package FastAPI :

![Service Documentation](https://www.tekhnoal.com/service_documentation_lfmlm.png)
La documentation vous permet de faire des requêtes contre l'API afin de l'essayer. Voici une requête de prédiction pour le modèle des frais d'assurance :

![Prediction Request](https://www.tekhnoal.com/prediction_request_lfmlm.png)

Le résultat:

![Prediction Response](https://www.tekhnoal.com/prediction_response_lfmlm.png)
La prédiction effectuée par le modèle a dû passer par le décorateur de journalisation que nous avons configuré dans le service, nous avons donc obtenu ces deux enregistrements de journal du processus :

![Prediction Log](https://www.tekhnoal.com/prediction_log_lfmlm.png)
Le processus du service web local émet les journaux vers stdout, exactement comme nous l'avons configuré.

Nous pouvons maintenant effectuer des prédictions via la méthode "POST", via l'interface utilisateur de notre fureteur ou avec une commande curl. 

Dans un nouveau terminal, on soumet la commande suivante: 
```
curl -X 'POST' \
'http://localhost:8000/api/models/insurance_charges_model/prediction' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{
"age": 65,
"sex": "male",
"bmi": 50,
"children": 5,
"smoker": true,
"region": "southwest"
}'

```
Ce qui devrait retourner : 
```
{"charges":46737.29}
```
Il est à noter que dépendamment de l'entraînement fait, la prédiction "charges" est sujette à varier. Elle devrait cependant être très proche de la valeure ci-haut. 

D'un autre côté, dans notre terminal original, on peut voir les "Logs" de notre modèle:
```
INFO:     127.0.0.1:50374 - "GET / HTTP/1.1" 307 Temporary Redirect
INFO:     127.0.0.1:50374 - "GET /docs HTTP/1.1" 200 OK
INFO:     127.0.0.1:50374 - "GET /openapi.json HTTP/1.1" 200 OK
{"asctime": "2023-11-13 14:22:36,382", "node_ip": "123.123.123.123", "name": "insurance_charges_model_logger", "levelname": "INFO", "message": "Prediction requested.", "action": "predict", "model_qualified_name": "insurance_charges_model", "model_version": "0.1.0", "age": 65, "sex": "male", "bmi": 50.0, "children": 5, "smoker": true, "region": "southwest"}
/home/nicolas/.local/lib/python3.10/site-packages/featuretools/entityset/entityset.py:1910: UserWarning: index index not found in dataframe, creating new integer column
  warnings.warn(
{"asctime": "2023-11-13 14:22:36,444", "node_ip": "123.123.123.123", "name": "insurance_charges_model_logger", "levelname": "INFO", "message": "Prediction created.", "action": "predict", "model_qualified_name": "insurance_charges_model", "model_version": "0.1.0", "status": "success", "charges": 46737.29}
INFO:     127.0.0.1:59350 - "POST /api/models/insurance_charges_model/prediction HTTP/1.1" 200 OK
```

# Déploiement du service

Maintenant que nous avons un service fonctionnel qui s'exécute localement, nous pouvons travailler sur son déploiement sur Docker/Kubernetes.

## Création d'une image Docker

Kubernetes doit disposer d'une image Docker pour déployer quelque chose, nous allons construire une image en utilisant le Dockerfile suivant :

```dockerfile
# Stage 1: Build Stage
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10 as base

# Creating and activating a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Installing dependencies
COPY ./service_requirements.txt ./service_requirements.txt
RUN pip install --no-cache -r service_requirements.txt



# Stage 2: Runtime Stage
FROM base as runtime

ARG DATE_CREATED
ARG REVISION
ARG VERSION

LABEL org.opencontainers.image.title="Logging for ML Models"
LABEL org.opencontainers.image.description="Logging for machine learning models."
LABEL org.opencontainers.image.created=$DATE_CREATED
LABEL org.opencontainers.image.authors="6666331+schmidtbri@users.noreply.github.com"
LABEL org.opencontainers.image.source="https://github.com/NicolasRichard1997/Insurance_Charges_Model"
LABEL org.opencontainers.image.version=$VERSION
LABEL org.opencontainers.image.revision=$REVISION
LABEL org.opencontainers.image.licenses="MIT License"
LABEL org.opencontainers.image.base.name="tiangolo/uvicorn-gunicorn-fastapi:python3.10"

WORKDIR /service

# Copy necessary files
COPY ./insurance_charges_model ./insurance_charges_model
COPY ./rest_configuration.yaml ./rest_configuration.yaml
COPY ./service_requirements.txt ./service_requirements.txt
COPY ./kubernetes_rest_config.yaml ./kubernetes_rest_config.yaml
COPY ./configuration ./configuration

# Install any dependencies
RUN pip install -r service_requirements.txt

# Expose the port your application runs on
EXPOSE 8000

# Install packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=base /opt/venv ./venv

COPY ./ml_model_logging ./ml_model_logging
COPY ./LICENSE ./LICENSE

ENV PATH /service/venv/bin:$PATH
ENV PYTHONPATH="${PYTHONPATH}:/service"

WORKDIR /service

CMD ["uvicorn", "rest_model_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Maintenant, nous pouvons utiliser les valeurs pour construire l'image. Nous fournirons également la version en tant qu'argument de construction (build argument).

```python
docker build \
  --build-arg DATE_CREATED="$DATE_CREATED" \
  --build-arg VERSION="0.1.0" \
  --build-arg REVISION="$REVISION" \
  -t insurance_charges_model_service:0.1.0 ..\
```
Pour trouver l'image que nous venons de construire, nous allons rechercher parmi les images Docker locales :

```python
docker images ls
```
Par exemple, on retrouve l'image ainsi:
```

REPOSITORY                        TAG       IMAGE ID       CREATED        SIZE
insurance_charges_model_service   0.1.0     59628036f039   23 hours ago   7.65GB
insurance_charges_model           0.1.0     8f8acb2943d6   4 days ago     4.9GB
gcr.io/k8s-minikube/kicbase       v0.0.42   dbc648475405   6 days ago     1.2GB
```
Maintenant, on peut démarrer un container avec cette image:

```python
docker run -d \
    -p 8000:8000 \
    -e REST_CONFIG=./configuration/rest_configuration.yaml \
    -e NODE_IP="123.123.123.123" \
    --name insurance_charges_docker \
    insurance_charges_model_service:0.1.0
```
Remarquez que nous avons ajouté une variable d'environnement appelée NODE_IP, c'est simplement pour avoir une valeur à extraire ultérieurement dans les journaux, ce n'est pas l'adresse IP réelle du nœud.

Le service est opérationnel dans le conteneur Docker. Pour afficher les "logs" émis par le processus, nous utiliserons la commande docker logs :

```python
docker logs insurance_charges_docker
```
```
{"asctime": "2023-11-13 19:33:11,253", "node_ip": "123.123.123.123", "name": "rest_model_service.helpers", "levelname": "INFO", "message": "Creating FastAPI app for: 'Insurance Charges Model Service'."}
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
Loading model from /service/insurance_charges_model/training/model.joblib
{"asctime": "2023-11-13 19:33:15,329", "node_ip": "123.123.123.123", "name": "rest_model_service.helpers", "levelname": "INFO", "message": "Loaded insurance_charges_model model."}
{"asctime": "2023-11-13 19:33:15,329", "node_ip": "123.123.123.123", "name": "rest_model_service.helpers", "levelname": "INFO", "message": "Added LoggingDecorator decorator to insurance_charges_model model."}
{"asctime": "2023-11-13 19:33:15,332", "node_ip": "123.123.123.123", "name": "rest_model_service.helpers", "levelname": "INFO", "message": "Created endpoint for insurance_charges_model model."}
INFO:     172.17.0.1:61206 - "GET / HTTP/1.1" 307 Temporary Redirect
INFO:     172.17.0.1:61206 - "GET /docs HTTP/1.1" 200 OK
INFO:     172.17.0.1:61206 - "GET /openapi.json HTTP/1.1" 200 OK
```

Comme prévu, les logs sont affichés au format JSON, bien qu'il y en ait certains qui ne le sont pas. Ces logs sont émis par des objets de logging qui ont été initialisés avant que le package rest_model_service n'ait eu la chance d'être initialisé.

Le service devrait être accessible sur le port 8000 de localhost, donc nous pouvons effectuer une prédiction avec la même commande curl que ci-haut. Le résultat obtenu 

On peut stopper et supprimer le conteneur. 

## Création d'un cluster Kubernetes

Pour montrer le système en action, nous allons déployer le service sur un cluster Kubernetes. Un cluster local peut être facilement démarré en utilisant [minikube](https://minikube.sigs.k8s.io/docs/). Les instructions d'installation peuvent être trouvées [ici](https://minikube.sigs.k8s.io/docs/start/).

Pour démarrer le cluster minikube, exécutez cette commande :


```python
minikube start
```
```
😄  minikube v1.32.0 on Debian bookworm/sid
✨  Using the docker driver based on existing profile
👍  Starting control plane node minikube in cluster minikube
🚜  Pulling base image ...
🏃  Updating the running docker "minikube" container ...
🐳  Preparing Kubernetes v1.28.3 on Docker 24.0.7 ...
🔎  Verifying Kubernetes components...
    ▪ Using image gcr.io/k8s-minikube/storage-provisioner:v5
🌟  Enabled addons: storage-provisioner, default-storageclass
🏄  Done! kubectl is now configured to use "minikube" cluster and "default" namespace by default
```
Affichons tous les pods en cours d'exécution dans le cluster minikube pour nous assurer que nous pouvons nous y connecter en utilisant la commande kubectl.

```python
kubectl get pods -A
```
```
NAMESPACE        NAME                                                  READY   STATUS             RESTARTS       AGE
default          insurance-charges-model-deployment-7dd4fc997b-2pp7x   0/1     ImagePullBackOff   0              42h
kube-system      coredns-5dd5756b68-6kdqt                              1/1     Running            7 (106s ago)   43h
kube-system      etcd-minikube                                         1/1     Running            3 (111s ago)   24h
kube-system      kube-apiserver-minikube                               1/1     Running            3 (101s ago)   24h
kube-system      kube-controller-manager-minikube                      1/1     Running            5 (111s ago)   43h
kube-system      kube-proxy-4gmz5                                      1/1     Running            5 (111s ago)   43h
kube-system      kube-scheduler-minikube                               1/1     Running            5 (111s ago)   43h
kube-system      storage-provisioner                                   1/1     Running            11 (97s ago)   43h
```
On dirait que nous pouvons nous connecter, nous sommes prêts à commencer le déploiement du service modèle sur le cluster.

### Création d'un namespace

Maintenant que nous avons un cluster et que nous y sommes connectés, nous allons créer un espace de noms pour contenir les ressources de notre déploiement de modèle. La définition de la ressource se trouve dans le fichier kubernetes/namespace.yaml. Pour appliquer le manifeste au cluster, exécutez cette commande :

```python
kubectl create -f kubernetes/namespace.yaml
```
```
    namespace/model-services created
```
Pour examiner les espaces de noms, exécutez cette commande :
```python
kubectl get namespace
```
```
NAME              STATUS   AGE
default           Active   43h
kube-node-lease   Active   43h
kube-public       Active   43h
kube-system       Active   43h
model-services    Active   6h18
```
### La création du "service modèle"

Le "service modèle" est déployé en utilisant des ressources Kubernetes. Il s'agit de :

-   ConfigMap : un ensemble d'options de configuration, dans ce cas, c'est un fichier YAML simple qui sera chargé dans le conteneur en cours d'exécution en tant que montage de volume. Cette ressource nous permet de modifier la configuration du service modèle sans avoir à modifier l'image Docker.
-   Deployment : une manière déclarative de gérer un ensemble de Pods, les pods du service modèle sont gérés via le Deployment.
-   Service : une manière d'exposer un ensemble de Pods dans un Deployment, le service modèle est rendu disponible à l'extérieur grâce au Service.

Ces ressources sont définies dans le fichier kubernetes/model_service.yaml. Voici son contenu: 

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: insurance-charges-model-deployment
  labels:
    app: insurance-charges-model-service
    app.kubernetes.io/name: insurance-charges-model-service
    app.kubernetes.io/version: "0.1.0"
    app.kubernetes.io/component: model-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: insurance-charges-model-service
  template:
    metadata:
      labels:
        app: insurance-charges-model-service
    spec:
      containers:
        - name: insurance-charges-model
          image: insurance_charges_model_service:0.1.0
          ports:
          - containerPort: 8000
            protocol: TCP
          imagePullPolicy: Never
          livenessProbe:
            httpGet:
              scheme: HTTP
              path: /api/health
              port: 8000
            initialDelaySeconds: 0
            periodSeconds: 10
            timeoutSeconds: 2
            failureThreshold: 5
            successThreshold: 1
          readinessProbe:
            httpGet:
              scheme: HTTP
              path: /api/health/ready
              port: 8000
            initialDelaySeconds: 0
            periodSeconds: 10
            timeoutSeconds: 2
            failureThreshold: 5
            successThreshold: 1
          startupProbe:
            httpGet:
              scheme: HTTP
              path: /api/health/startup
              port: 8000
            initialDelaySeconds: 2
            periodSeconds: 5
            timeoutSeconds: 2
            failureThreshold: 5
            successThreshold: 1
          resources:
            requests:
              cpu: "100m"
              memory: "250Mi"
            limits:
              cpu: "200m"
              memory: "250Mi"
          env:
            - name: REST_CONFIG
              value: rest_config.yaml
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
            - name: APP_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.labels['app']
          volumeMounts:
            - name: config-volume
              mountPath: /service/configuration
      volumes:
        - name: config-volume
          configMap:
            name: model-service-configuration
            items:
              - key: rest_config.yaml
                path: rest_config.yaml
---
apiVersion: v1
kind: Service
metadata:
  name: insurance-charges-model-service
  labels:
    app.kubernetes.io/name: insurance-charges-model-service
    app.kubernetes.io/version: "0.1.0"
    app.kubernetes.io/component: model-service
spec:
  type: NodePort
  selector:
    app: insurance-charges-model-service
  ports:
    - name: http
      protocol: TCP
      port: 8000
      targetPort: 8000
```


La définition du pod utilise la [API descendante fournie par Kubernetes](https://kubernetes.io/docs/tasks/inject-data-application/downward-api-volume-expose-pod-information/) pour accéder au nom du nœud, au nom du pod et au contenu de l'étiquette 'app'. Ces informations sont mises à disposition sous forme de variables d'environnement. Nous ajouterons ces informations au journal en ajoutant les noms des variables d'environnement à la configuration du journal que nous fournirons au service modèle. Nous avons construit une classe de contexte de journalisation ci-dessus dans le but d'ajouter des variables d'environnement aux enregistrements de journal.

Nous sommes presque prêts à déployer le service modèle, mais avant de le lancer, nous devrons envoyer l'image Docker du démon local Docker vers le cache d'images de minikube :


```python
minikube image load insurance_charges_model_service:0.1.0
```
Le chargement prend un moment. La commande suivante confirme que l'image docker a bien été chargée :
```python
minikube image ls
```
```
registry.k8s.io/pause:3.9
registry.k8s.io/kube-scheduler:v1.28.3
registry.k8s.io/kube-proxy:v1.28.3
registry.k8s.io/kube-controller-manager:v1.28.3
registry.k8s.io/kube-apiserver:v1.28.3
registry.k8s.io/etcd:3.5.9-0
registry.k8s.io/coredns/coredns:v1.10.1
gcr.io/k8s-minikube/storage-provisioner:v5
docker.io/library/insurance_charges_model_service:0.1.
```
On retrouve notre image à la dernière ligne.

Le service modèle devra accéder au fichier de configuration YAML que nous avons utilisé pour le service local ci-dessus. Ce fichier se trouve dans le dossier /configuration et s'appelle "kubernetes_rest_config.yaml", il est personnalisé pour l'environnement Kubernetes que nous sommes en train de construire.

Pour créer un [ConfigMap](https://kubernetes.io/docs/concepts/configuration/configmap/) pour le service, exécutez cette commande :


```python
kubectl create configmap model-service-configuration -n model-services --from-file=./configuration/kubernetes_rest_config.yaml
```
```
    configmap/model-service-configuration created
```

Le service est déployé sur le cluster Kubernetes avec cette commande :

```python
kubectl apply -n model-services -f ./kubernetes/model_service.yaml
```
```
    deployment.apps/credit-risk-model-deployment created
    service/credit-risk-model-service created
```

Le déploiement et le service pour le service modèle ont été créés simultanément. Voyons le déploiement pour vérifier s'il est déjà disponible :

```python
kubectl get deployments -n model-services 
```
```
NAME                                 READY   UP-TO-DATE   AVAILABLE   AGE
insurance-charges-model-deployment   1/1     1            1           5h57m
```
Vous pouvez également afficher les pods qui exécutent le service :

```python
kubectl get pods -n model-services -l app=insurance-charges-model-service
```
```
RESTARTS      AGE
insurance-charges-model-deployment-77b7d76c85-77p45   1/1     Running   3 (15m ago)   5h58m
```

The Kubernetes Service details look like this:


```python
kubectl get services -n model-services 
```
```
NAME                              TYPE       CLUSTER-IP       EXTERNAL-IP   PORT(S)          AGE
insurance-charges-model-service   NodePort   10.103.161.252   <none>        8000:32391/TCP   5h59m
```
Nous allons exécuter un processus proxy localement pour pouvoir accéder à l'extrémité du service modèle :

```bash
minikube service insurance-charges-model-service --url -n model-services
```
Voici le résulat:

```
minikube service insurance-charges-model-service --url -n model-services
http://127.0.0.1:34659
❗  Because you are using a Docker driver on linux, the terminal needs to be open to run it.
```
Encore une fois on peut interargir avec le modèle avec un commande curl, comme ci-haut, à l'exception qu'on doive respecter l'adresse locale donnée par minikube (http://127.0.0.1:34659 dans le cas présent).

Le modèle est déployé sur Kubernetes !

### Accéder aux Logs

Kubernetes dispose d'un système intégré qui reçoit les sorties stdout et stderr des conteneurs en cours d'exécution et les enregistre sur le disque dur du nœud pendant une durée limitée. Vous pouvez consulter les logs émis par les conteneurs en utilisant cette commande :

```python
kubectl logs -n model-services insurance-charges-model-deployment-77b7d76c85-77p45 -c insurance-charges-model | grep "\"action\": \"predict\""
```
```
{"asctime": "2023-11-13 20:03:51,360", "pod_name": "insurance-charges-model-deployment-77b7d76c85-77p45", "node_name": "minikube", "app_name": "insurance-charges-model-service", "name": "insurance_charges_model_logger", "levelname": "INFO", "message": "Prediction requested.", "action": "predict", "model_qualified_name": "insurance_charges_model", "model_version": "0.1.0", "age": 20, "sex": "male", "bmi": 15.0, "children": 0, "smoker": true, "region": "southwest"}
{"asctime": "2023-11-13 20:03:52,241", "pod_name": "insurance-charges-model-deployment-77b7d76c85-77p45", "node_name": "minikube", "app_name": "insurance-charges-model-service", "name": "insurance_charges_model_logger", "levelname": "INFO", "message": "Prediction created.", "action": "predict", "model_qualified_name": "insurance_charges_model", "model_version": "0.1.0", "status": "success", "charges": 17751.17}
```
