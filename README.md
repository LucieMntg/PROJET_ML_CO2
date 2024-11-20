# Projet de Machine Learning : Le CO2 CALCULATOR

## Présentation

Ce répertoire contient le code de notre projet développé pendant notre formation de [Data Analyst](https://datascientest.com/formation-data-analyst) avec DataScientest sous Python 1.5.1.

L’objectif principal de ce projet est de développer un modèle de prédiction d'émission de CO2 des véhicules en utilisant les techniques de Machine Learning, sur la base de données officielles françaises fournies par le site [Car Labelling](https://carlabelling.ademe.fr/) de l'ADEME, en provenance des services d'homologation des constructeurs. 

Destiné aux constructeurs automobiles, le CO2 CALCULATOR est un modèle de régression capable de prédire l'émission de CO2 de véhicules à partir de caractéristiques renseignées. Il est accessible [ici](https://co2-calculator-project.streamlit.app/), dans la rubrique 'Démonstration'.

Les étapes principales du projet sont : l’acquisition et la préparation de données, la datavisualisation, le features engineering et pré-processing, la recherche, la conception et l'entraînement de six modèles de régression en Machine Learning, la sélection de trois modèles pour optimisation ainsi que l’évaluation approfondie de leur performance à l'aide de plusieurs métriques et pour finir, le développement du "CO2 CALCULATOR" avec le meilleur modèle.



---
Ce projet a été développé en novembre 2024 par l'équipe suivante :

Ndèye Oumy DIAGNE ([LinkedIn](https://www.linkedin.com/in/ndeyeoumy-diagne/)) ;

Justine GAUTERO ([LinkedIn](https://www.linkedin.com/in/justine-gautero-4251aa130/)) ;

Lucie MONTAGNE ([LinkedIn](https://www.linkedin.com/in/lucie-montagne-ab93b465/)) ;


Sous le mentorat d'Eliott Douieb de DataScientest.






## Rapport du projet

Le rapport complet de notre projet, au format PDF, est disponible dans le dossier 'Rapport' de ce répertoire GitHub.




## Organisation du répertoire
    
    ├───.streamlit                                                <- Code renfermant le thème graphique de l'application Streamlit
    │
    ├───Dataset                                                   <- Dataset avec les données brutes non transformées
    │
    ├───Images                                                    <- Visuels utilisés dans l'application Streamlit
    │
    ├───Modèles                                                   <- Modèles entraînés et prédictions
    │
    ├───Notebooks                                                 <- Jupyter notebooks contenant le code de préprocessing des données et de modélisation
    │       CODE_PROJET_CO2_EXPLORATION_DONNEES.ipynb             <- Exploration des données et visualisation
    │       CODE_PROJET_CO2_PREPROCESSING_MODELISATION.ipynb      <- Feature engineering, pré-processing et modélisation
    │
    ├───Rapport                                                   <- Rapport détaillé du projet au format PDF 
    │
    │   .gitignore                                                <- Fichier Gitignore pour ignorer certains fichiers/dossiers
    │   LICENSE.txt                                               <- Licence du projet
    │   README.md                                                 <- Fichier README avec présentation du projet et lien de l'application streamlit
    │   requirements.txt                                          <- Liste des dépendances Python nécessaires pour reproduire l'environnement
    │   streamlit_project_code_final.py                           <- Code source pour l'application Streamlit








