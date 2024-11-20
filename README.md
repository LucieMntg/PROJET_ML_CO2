# Projet de Machine Learning : Le CO2 CALCULATOR

## Présentation

Ce répertoire contient le code de notre projet développé pendant notre formation de [Data Analyst](https://datascientest.com/formation-data-analyst) avec DataScientest sous Python 1.5.1.

L’objectif principal de ce projet est de développer un modèle de prédiction d'émission de CO2 des véhicules en utilisant les techniques de Machine Learning, sur la base de données officielles françaises fournies par le site [Car Labelling](https://carlabelling.ademe.fr/) de l'ADEME, en provenance des services d'homologation des constructeurs. 

Destiné aux constructeurs automobiles, le CO2 CALCULATOR est un modèle de régression capable de prédire l'émission de CO2 de véhicules à partir de caractéristiques renseignées. Il est accessible [ici](https://co2-calculator-project.streamlit.app/), dans la rubrique 'Démonstration'.

Les étapes principales du projet sont : l’acquisition et la préparation de données (features engineering et pré-processing), la datavisualisation, la recherche, la conception et l'entraînement de six modèles de régression en Machine Learning, la sélection de trois modèles pour optimisation ainsi que l’évaluation approfondie de leur performance à l'aide de plusieurs métriques et pour finir, le développement du "CO2 CALCULATOR" avec le meilleur modèle.

Le rapport complet de notre projet est disponible dans le dossier 'Rapport' de ce répertoire GitHub.



Ce projet a été développé en novembre 2024 par l'équipe suivante :

Ndèye Oumy DIAGNE ([LinkedIn](https://www.linkedin.com/in/ndeyeoumy-diagne/)) ;

Justine GAUTERO ([LinkedIn](https://www.linkedin.com/in/justine-gautero-4251aa130/)) ;

Lucie MONTAGNE ([LinkedIn](https://www.linkedin.com/in/lucie-montagne-ab93b465/)) ;


Sous le mentorat d'Eliott Douieb de DataScientest.




## Organisation du répertoire
|   config.toml
|   et_regressor
|   LICENSE.txt
|   model_et_regressor
|   model_knn
|   model_lr
|   model_regressor
|   model_rf_regressor
|   model_xgb_regressor
|   README.md
|   requirements.txt
|   streamlit_project_code.py
|   X_train
|   X_train_concat
|
+---.devcontainer
|       devcontainer.json
|
+---.streamlit
|       config.toml
|
+---Autres
|       X_train
|       X_train_concat
|
+---Dataset
|       ADEME-CarLabelling-2023-local.csv
|       ADEME-CarLabelling-2023-local.csv5dwm75ls.part
|       ADEME-CarLabelling-2023-local.csvdj5fbyos.part
|       ADEME-CarLabelling-2023-local.csvlpklo94a.part
|       ADEME-CarLabelling-2023-local.csvq52pcmfh.part
|
+---Images
|       Entonnoir.png
|       IF_ETRegressor.jpg
|       IF_RFRegressor.jpg
|       IF_XGBRegressor.jpg
|       Image_conclu_gif.gif
|       Image_dataset_gif.gif
|       Image_dataviz_gif.gif
|       Image_demo_gif.gif
|       Image_features_gif.gif
|       Image_intro_gif.gif
|       Image_lazyp.jpg
|       Image_model_gif.gif
|       Image_transition_2_gif.gif
|       Image_transition_gif.gif
|       logo_co2_calculator.png
|       shape_2_pre-pross_chang_type_gif.gif
|       Tableau_top_features.jpg
|
\---Notebooks
    |   CODE_PROJET_CO2_EXPLORATION_DONNEES.ipynb
    |   CODE_PROJET_CO2_PREPROCESSING_MODELISATION.ipynb






