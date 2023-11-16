# Construction et déploiement d'un modèle de régression en apprentissage automatique

[Répertoire Github](https://github.com/NicolasRichard1997/Insurance_Charges_Model/)

[Modèle sur DockerHub](https://hub.docker.com/repository/docker/nicolasrichard1997/insurance_charges_model_service/general)

### Les instructions pour déployer le modèle localement, sur Docker et sur Kubernetes suivent dans cet ordre

Ce projet est une compilation, une adaptation et une traduction de deux projets par Brian Schmidt, [Training and Deploying an ML Model](https://www.tekhnoal.com/regression-model) et [Logging for ML Model Deployments](https://www.tekhnoal.com/logging-for-ml-models), visant à intégrer plusieurs techniques de science des données et de génie logiciel à un modèle d'apprentissage automatique. Dans ce blog, j'utiliserai des packages Python open source pour effectuer une exploration de données automatisée, une ingénierie de fonction automatisée, un apprentissage automatique automatisé et une validation de modèle. Dans un second temps, j'incorporerai les "logs" du second blog afin de pouvoir utiliser et dépanner le modèle de manière plus efficace. J'utiliserai également Docker et Kubernetes pour déployer le modèle localement. Le blog couvrira l'ensemble de la base de code du modèle, de l'exploration initiale des données au déploiement du modèle derrière une API RESTful sur Kubernetes.

Le [blog suivant]( https://github.com/NicolasRichard1997/Insurance_Charges_Model/blob/main/blog_post/post.md) contient le processus détaillé des étapes suivies, modifiées et mises-à-jour en vue de la réalisation du projet. 

## Prérequis

Python 3.10

## Installation

Le fichier Makefile inclus dans ce projet contient des cibles qui facilitent l'automatisation de plusieurs tâches.

Pour télécharger le code source, exécutez la commande :

```bash
git clone https://github.com/NicolasRichard1997/Insurance_Charges_Model/
```

Ensuite, créez un environnement virtuel et activez-le :

```bash
# go into the project directory
cd Insurance_Charges_Model

make venv

source venv/bin/activate
```

Installez les dépendances :

```bash
make dependencies
```

Le fichier requirements.txt ne contient que les dépendances nécessaires pour effectuer des prédictions avec le modèle. Pour entraîner le modèle, vous devrez installer les dépendances du fichier train_requirements.txt :

```bash
make train-dependencies
```

## Exécution des tests unitaires

Pour exécuter la suite de tests unitaires, exécutez les commandes suivantes :

```python
# first install the test dependencies
make test-dependencies
```
```python
# run the test suite
make test
```
#### Les 5 tests devraient passés aves succès. Dans la version originale, des "warnings" subsistent toujours

```python
# clean up the unit tests
make clean-test
```

## Exécution du Service

Pour démarrer le service localement, exécutez ces commandes :

```bash
uvicorn rest_model_service.main:app --reload 
```

## Génération d'une Spécification OpenAPI

Pour générer le fichier de spécification OpenAPI pour le service REST qui héberge le modèle, exécutez ces commandes :

```bash
`export PYTHONPATH=./
generate_openapi --output_file=service_contract.yaml` 
```

## Docker

Pour construire une image Docker pour le service local seulement, exécutez cette commande :

**Veuillez suivre les insructions dans le blog afin de générer une image déployable sur Kubernetes**

```
`docker build -t insurance_charges_model:0.1.0 .` 
```
Pour exécuter l'image, utilisez cette commande :

```

`docker run -d -p 8000:8000 insurance_charges_model:0.1.0` 
```

Pour surveiller les journaux provenant de l'image, exécutez cette commande :

```
docker logs $(docker ps -lq)` 
```

Pour arrêter l'image Docker, utilisez cette commande :
```
`docker kill $(docker ps -lq)`
```

## Déployer le service localement sur Kubernetes avec Minikube

On démarre Minikube et le ssh:

```
minikube start
minikube ssh
```

On procède ensuite à chargée l'image dans le cluster Minikube avant de quitter le service SSH:

```
docker pull nicolasrichard1997/insurance_charges_model_service:0.1.0
```
l'image devrait éventuellement télécharger. On quitte la session SSH avec la commande:
```
exit
```
Les commandes suivantes sont nécessaire pour créer un pod:

```python
kubectl create -f kubernetes/namespace.yaml
```
```p ython
kubectl create configmap model-service-configuration -n model-services --from-file=./configuration/kubernetes_rest_config.yaml
```
```python
kubectl apply -n model-services -f ./kubernetes/model_service.yaml
```
Finalement, on peut accéder au modèle comme suit:
```bash
minikube service insurance-charges-model-service --url -n model-services
```
Il est aussi possible d'accéder aux logs du modèle comme suit, en prenant soin d'y mentionner le nom spécifique du dépoloiement:

```python
kubectl get deployments -n model-services 
```
avec le nom du déploiement:
```python
kubectl logs -n model-services insurance-charges-model-deployment-XXXXXXXXXXXXXX -c insurance-charges-model | grep "\"action\": \"predict\""
```
# Remerciements


Merci à Brian Schmidt pour les ressources et les blogs. 
