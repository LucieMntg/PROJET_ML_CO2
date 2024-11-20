# Import des librairies et modules
import streamlit as st
import random
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import plotly.tools as tls
import plotly.express as px
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score



# Cache
@st.cache_data
def generate_random_value(x): 
  return random.uniform(0, x) 

a = generate_random_value(10) 
b = generate_random_value(20) 





# Chargement du df
file_path = 'Dataset/ADEME-CarLabelling-2023-local.csv'

df=pd.read_csv(file_path, encoding='utf-8', sep=';')





# Sidebar
from pathlib import Path

# Ajout du logo dans le sidebar
image_path = Path("Images/logo_co2_calculator.png")
image_path_str = str(image_path) 

st.sidebar.image(image_path_str)

# Titre du sidebar
st.sidebar.title("LES ÉTAPES DU PROJET")

# Icone Google flèche
google_svg = """
<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#000000">
  <path d="m480-340 180-180-57-56-123 123-123-123-57 56 180 180Zm0 260q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q134 0 227-93t93-227q0-134-93-227t-227-93q-134 0-227 93t-93 227q0 134 93 227t227 93Zm0-320Z"/>
</svg>
"""
st.sidebar.markdown(google_svg, unsafe_allow_html=True)

# Pages du projet
pages = ["Introduction", "Dataset", "Dataviz", "Modélisation", "Démonstration", "Conclusion"]
page = st.sidebar.radio("", pages)

# Saut ligne
st.sidebar.markdown("<br>", unsafe_allow_html=True) 

# Utiliser markdown avec HTML pour personnaliser le fond et l'encadré
st.sidebar.markdown("""
    <div style="background-color:#5930F2; padding: 10px; border-radius: 10px;">
        <p style="color:white; font-size:16px;">
            <strong>Ndèye Oumy DIAGNE (<a href="https://www.linkedin.com/in/ndeyeoumy-diagne/" style="color:white; text-decoration:none;" target="_blank">LinkedIn</a>)</strong><br>
            <strong>Justine GAUTERO (<a href="https://www.linkedin.com/in/justine-gautero-4251aa130/" style="color:white; text-decoration:none;" target="_blank">LinkedIn</a>)</strong><br>
            <strong>Lucie MONTAGNE (<a href="https://www.linkedin.com/in/lucie-montagne-ab93b465/" style="color:white; text-decoration:none;" target="_blank">LinkedIn</a>)</strong><br>
        </p>
    </div>
""", unsafe_allow_html=True)









# Page 1 INTRO
if page == pages[0]:
    
   # Import de l'image gif
   gif_path = Path("Images/Image_intro_gif.gif")
   gif_path_str = str(gif_path)
   st.image(gif_path_str)

   # Titre 
   st.markdown("""
    <h3 style="color: #5930f2;">Introduction</h3>
    """, unsafe_allow_html=True)
   
   # Contenu
   st.write ("Ce projet représente une opportunité de contribuer à un enjeu environnemental majeur tout en mettant en pratique les connaissances acquises lors de notre formation.")
   st.success ("Dans le cadre de cette étude, nous avons pour **objectif principal l’analyse de l’émission de CO2 au niveau national d’un échantillon de véhicules variés**.")
   if st.checkbox("Technique"):
      st.write("D’un point de vue technique, le projet implique :")
      st.write( "- la compréhension des normes environnementales pour sélectionner les variables pertinentes et identifier les caractéristiques techniques liées aux émissions du CO2. Elle a un rôle important dans la construction du dataset ;")
      st.write ("- l’utilisation des outils d’analyse de données et de prédiction pour résoudre la problématique.")
   if st.checkbox("Économique et écologique"):
      st.write ("D’un point de vue économique et écologique, le projet a pour but :")
      st.write("- de réduire l'empreinte carbone en réduisant les émissions de CO2 des véhicules ;")
      st.write("- de proposer aux constructeurs des solutions permettant de répondre à des besoins clients soucieux de l’environnement mais aussi de réduire les taxes en optimisant le Malus écologique des véhicules concernés.")

   # Saut ligne
   st.markdown("<br><br>", unsafe_allow_html=True) 


   import time

   # Texte à afficher, divisé en paragraphes
   paragraphes = [
      "Pour atteindre ces objectifs nous allons essayer de répondre à la problématique en suivant les étapes suivantes :",
      "- **Identifier les modèles de véhicules qui émettent le plus de CO2** (on se focalise sur ce type de pollution, le dioxyde de carbone,  sachant qu’il existe aussi la pollution aux particules fines, dioxyde d’azote (NO) et les hydrocarbures non brûlés. Le CO2 > joue sur le changement climatique)",
      "- **Identifier les caractéristiques techniques qui influencent le plus les émissions de CO2**",
      "- **Estimer et anticiper la pollution future des nouveaux véhicules**",
      "- **Déterminer comment réduire l’impact carbone des voitures**",
      "- **Faire des recommandations aux constructeurs pour réduire les émissions de CO2 des véhicules**",
      "Ce domaine d’activité nous est totalement inconnu, nous avons effectué des recherches et consulté les documentations existantes pour nous imprégner du sujet."
   ]

   # Fonction pour afficher chaque paragraphe avec une animation de type "slide-in"
   def slide_in_paragraphs(paragraphs, speed=0.8):
      for paragraph in paragraphs:
         st.write(paragraph)
         time.sleep(speed)
         st.empty()  # Efface le texte précédent pour créer l'effet de slide-in


   # Bouton pour démarrer l'animation
   if st.button("Les étapes pour atteindre cet objectif"):
      slide_in_paragraphs(paragraphes)









# Page 2 DATASET
if page == pages[1] : 

   # Import de l'image gif
   gif_path = Path("Images/Image_dataset_gif.gif")
   gif_path_str = str(gif_path)
   st.image(gif_path_str)

   # Titre 
   st.markdown("""
    <h3 style="color: #5930f2;">1. Choix de la source de données</h3>
    """, unsafe_allow_html=True)
   
   # Contenu
   st.write("Nous avions à disposition deux sources à partir desquelles trouver les données concernant l’émission de CO2 des véhicules :\n\n")
   st.markdown("- **Dataset de l’Etat français (données issues de l’ADEME)** datant de 2014 :  Émissions de CO2 et de polluants des véhicules commercialisés en France.\n") 
   st.markdown("- **Dataset de l’Union Européenne à travers l’agence Européenne de l’environnement (EEA)** datant de 2023 (mais dont les données ne sont pas encore finales)  : Monitoring of CO2 emissions from passenger cars, 2023 - Provisional data.\n\n")
   st.markdown("- En parallèle, nous avons trouvé un **troisième dataset pour le territoire français (données également issues de l’ADEME)** mais plus récent puisque mis à jour en septembre 2024 : “ADEME-CarLabelling”. Il contient donc les derniers véhicules commercialisés dotés des dernières technologies.\n\n")
   st.success("Nous avons fait le choix de sélectionner ce dernier dataset, car au-delà du fait d’être le plus récent, il s'avère être beaucoup plus complet en termes de variables que le dataset français de 2014. Il contient notamment **la nouvelle norme WLTP (WorldWide harmonized Light vehicle Test Procedures)**.")
   
   st.dataframe(df.head())


   # Saut ligne
   st.markdown("<br><br>", unsafe_allow_html=True) 


   # Titre 
   st.markdown("""
    <h3 style="color: #5930f2;">2. Description du jeu de données</h3>
    """, unsafe_allow_html=True)

   data1 = {
      "Territoire": ["France"],
      "Nombre lignes": [3604],
      "Nombre colonnes": [52],
      "Types de données": ["object, float, interger"],
      "Temporalité": [2024],
      "Valeurs manquantes (%)": ["19"]        
   }

   dataset_info_base = pd.DataFrame(data1)
   dataset_info = dataset_info_base
   st.markdown(dataset_info.style.hide(axis="index").to_html(), unsafe_allow_html=True)

   # Saut ligne
   st.markdown("<br><br>", unsafe_allow_html=True) 







# Page 3 DATAVIZ
df['Rapport poids-puissance'] = df['Rapport poids-puissance'].str.replace(',', '.').astype(float)
df['Conso basse vitesse Max'] = df['Conso basse vitesse Max'].str.replace(',', '.').astype(float)
df['Conso moyenne vitesse Min'] = df['Conso moyenne vitesse Min'].str.replace(',', '.').astype(float)
df['Conso moyenne vitesse Max'] = df['Conso moyenne vitesse Max'].str.replace(',', '.').astype(float)
df['Conso haute vitesse Min'] = df['Conso haute vitesse Min'].str.replace(',', '.').astype(float)
df['Conso haute vitesse Max'] = df['Conso haute vitesse Max'].str.replace(',', '.').astype(float)
df['Conso T-haute vitesse Max'] = df['Conso T-haute vitesse Max'].str.replace(',', '.').astype(float)
df['Conso T-haute vitesse Min'] = df['Conso T-haute vitesse Min'].str.replace(',', '.').astype(float)
df['Conso vitesse mixte Min'] = df['Conso vitesse mixte Min'].str.replace(',', '.').astype(float)
df['Conso vitesse mixte Max'] = df['Conso vitesse mixte Max'].str.replace(',', '.').astype(float)
df['CO2 basse vitesse Min'] = df['CO2 basse vitesse Min'].str.replace(',', '.').astype(float)
df['CO2 moyenne vitesse Min'] = df['CO2 moyenne vitesse Min'].str.replace(',', '.').astype(float)
df['CO2 basse vitesse Max'] = df['CO2 basse vitesse Max'].str.replace(',', '.').astype(float)
df['CO2 moyenne vitesse Max'] = df['CO2 moyenne vitesse Max'].str.replace(',', '.').astype(float)
df['CO2 haute vitesse Min'] = df['CO2 haute vitesse Min'].str.replace(',', '.').astype(float)
df['CO2 vitesse mixte Min'] = df['CO2 vitesse mixte Min'].str.replace(',', '.').astype(float)
df['CO2 vitesse mixte Max'] = df['CO2 vitesse mixte Max'].str.replace(',', '.').astype(float)
df['CO2 haute vitesse Max'] = df['CO2 haute vitesse Max'].str.replace(',', '.').astype(float)
df['CO2 T-haute vitesse Min'] = df['CO2 T-haute vitesse Min'].str.replace(',', '.').astype(float)
df['CO2 T-haute vitesse Max'] = df['CO2 T-haute vitesse Max'].str.replace(',', '.').astype(float)
df['Essai CO2 type 1'] = df['Essai CO2 type 1'].str.replace(',', '.').astype(float)


if page == pages[2] : 

   # Import de l'image gif
   gif_path = Path("Images/Image_dataviz_gif.gif")
   gif_path_str = str(gif_path)
   st.image(gif_path_str)
   
   # Saut ligne
   st.markdown("<br>", unsafe_allow_html=True) 

   # Titre 
   st.markdown("""
    <h3 style="color: #5930f2;">Les principales visualisations</h3>
    """, unsafe_allow_html=True)
   
   # Contenu 
   st.markdown("""
   Nous utilisons tout d’abord la data visualisation dans le but d’obtenir une **description complète du jeu de données**, de voir les relations entre les variables et de **déterminer des corrélations**. Cette analyse va nous aider dans le **choix des caractéristiques de notre jeu de données final**. """)

   # Saut ligne
   st.markdown("<br>", unsafe_allow_html=True) 

   # Selectbox pour choisir le graphique à afficher
   option = st.selectbox(
      'Choisissez une visualisation à afficher',
      ('Nombre de valeurs nulles par colonne', 'Corrélation entre les valeurs quantitatives', "Relation entre la puissance fiscale et l'émission de CO2","Distribustion de la valeur cible"))

   if option == "Nombre de valeurs nulles par colonne":
      st.subheader('Nombre de valeurs nulles par colonne') 
   
      import plotly.express as px
      # Affichage des valeurs nulles par colonne
      null_counts = df.isnull().sum()

      fig=plt.figure( figsize = (20, 7))
      null_counts.plot(kind='bar', color='#5930F2')
      plt.xlabel('Colonnes')
      plt.ylabel('Nombre de valeurs nulles')
      plt.title('Nombre de valeurs nulles par colonne')
      st.pyplot(fig)
   

   elif option == "Corrélation entre les valeurs quantitatives":
      st.subheader("Corrélation entre les valeurs quantitatives")
      # Tracer la heatmap avec Plotly
      # Heatmap des variables quantitatives
      cor = df[['Cylindrée', 'Puissance fiscale', 'Poids à vide', 'Rapport poids-puissance', 'Masse OM Max', 'Conso vitesse mixte Max', 'Nombre rapports', 'CO2 vitesse mixte Max', 'Essai CO2 type 1', 'Prix véhicule']].corr() # sélection des variables quantitatives

      fig1 = ff.create_annotated_heatmap(
         z=cor.values,
         x=list(cor.columns),
         y=list(cor.index),
         annotation_text=cor.round(2).values,
         colorscale='Viridis',  # Utilisation d'une échelle de couleurs valide
         showscale=True)

      # Mettre à jour la taille du graphique
      fig1.update_layout(
         width=800,
         height=700)
      
      # Afficher la heatmap dans Streamlit
      st.plotly_chart(fig1)
   

   elif option == "Relation entre la puissance fiscale et l'émission de CO2":
      st.subheader("Relation entre la puissance fiscale et l\'émission de CO2 d\'un véhicule")
      # Tracer le graphique de la relation entre la puissance fiscale et l'émission de CO2
      fig2 = px.scatter(df, x="Puissance fiscale", y="CO2 vitesse mixte Max", trendline="lowess",
                     title="") #vide
      # Mettre la ligne de tendance en rouge
      fig2.update_traces(line=dict(color='#fdee00'))
      # Mettre les points en couleur #5930F2
      fig2.update_traces(marker=dict(color='#5930F2'))
      # Afficher le graphique dans Streamlit
      st.plotly_chart(fig2)

   elif option == "Distribustion de la valeur cible":
      st.subheader("Distribution de la valeur cible : Émission de CO2")
   
   # Distribution des émissions de CO2 (g/km)
      fig3 = sns.displot(df['CO2 vitesse mixte Max'], kde = True, rug = True, bins = 10, color = '#5930F2')
      st.pyplot(fig3)



 



# Page 4 MODELISATION
if page == pages[3] : 
   # Import de l'image gif
   gif_path = Path("Images/Image_model_gif.gif")
   gif_path_str = str(gif_path)
   st.image(gif_path_str)

   # Titre
   st.markdown("""
    <h3 style="color: #5930f2;">La partie modélisation se compose de plusieurs étapes :</h3>
    """, unsafe_allow_html=True)
   
   # Contenu
   st.write("""
   1. **Feature Engineering**
   2. **Pré-processing**
   3. **Overview des modèles possibles**
   4. **Modélisation et choix du modèle final**
   5. **Interprétation**
   """)


   from pathlib import Path

   # Import de l'image gif
   gif_path = Path("Images/Image_transition_gif.gif")
   gif_path_str = str(gif_path)
   if gif_path.exists():
      st.image(gif_path_str)
   else:
      st.error("Le fichier GIF n'a pas été trouvé.")



   # Section 1. Feature Engineering
  
   st.markdown("""
    <h3 style="color: #5930f2;">1. Feature Engineering</h3>
    """, unsafe_allow_html=True)


   st.write("""
   Lors de cette étape, nous avons réalisé 2 actions :
   - **Le choix de la variable cible**
      """)

   st.success("Nous avons sélectionné la variable ‘CO2 vitesse mixte Max’ pour deux raisons :")
   st.markdown("✅ **Sa fiabilité** : elle provient d’un test du cycle WLTP dont l’objectif est de mesurer les émissions polluantes en circulation réelle.")
   st.markdown("✅ **Sa qualité** : elle contient peu de valeurs manquantes.")

   # Saut ligne
   st.markdown("<br>", unsafe_allow_html=True) 


   st.write("""
   - **La sélection des variables**
   
   Afin de réduire le nombre de variables pour la partie modélisation, **nous nous sommes appuyées sur la matrice de corrélation vu précédemment afin de voir la relation entre les variables explicatives et la variable cible**.
   """)


   # Contenu
   st.write("""Nous avons décidé de ne pas conserver les variables :
   - mesurant les autres types de pollution ;
   - concernant les mesures propres aux véhicules électriques ;
   - de type ‘administratives’ ;
   - fortement corrélées (cf. tests statistiques).""")

   st.success("Nous avons donc choisi de conserver 6 variables sur les 52 variables totales.")

   # Créer un dictionnaire avec les noms des variables et leurs descriptions
   data2 = {
      "Nom de la variable": [
         "Energie", "Carrosserie", "Puissance Fiscale", 
         "Type de boîte", "Masse OM Max", "CO2 Vitesse mixte Max"
      ],
      "Description": [
         "Type d’énergie de la voiture (Essence, gazole…)", 
         "Type de carrosserie (Berline, Suv…)", 
         "Puissance en CV fiscaux", 
         "Boite de vitesse (Auto, mécanique…)", 
         "Masse en ordre de marche maxi (en Kg), ou masse maximale d'un véhicule prêt à circuler (poids du véhicule, conducteur, passagers, cargaison, fluides)", 
         "CO2 mixte combiné - maximum (en g/Km)"
      ]
   }

   dataset_variables = pd.DataFrame(data2)
   st.markdown(dataset_variables.style.hide(axis="index").to_html(), unsafe_allow_html=True)

   # Saut ligne
   st.markdown("<br><br>", unsafe_allow_html=True) 



   # Paragraphe de texte
   st.write("""
   - **La suppression de lignes concernant les véhicules électriques**
   
   Ensuite, nous nous sommes intéressées aux **véhicules non émetteurs de CO2**, soit les voitures électriques. Celles-ci ne sont pas concernées par notre problématique, à savoir la réduction des émissions de CO2, car elles n'en émettent pas.
   Nous pouvons donc les enveler de notre jeu de données.
   """)

   st.write("""
   ➡️ Suppression des lignes concernant les véhicules électriques
            
   ➡️ Modifier le type de la variable CO2_vitesse_mixte_Max
            
   ➡️ Supprimer les doublons
            
   ➡️ Séparer le jeu de données en jeu d'entrainement et de test
            
   ➡️ Vérifier les valeurs manquantes
            
   ➡️ Attribuer les colonnes numériques et catégorielles
            
   ➡️ Encoder les variables catégorielles
            
   ➡️ Mettre à l'échelle les variables numériques
   """)


   # Sauts de ligne 
   st.markdown("<br><br><br><br>", unsafe_allow_html=True)  




# Section 3. Overview des modèles possibles
   st.markdown("""
      <h3 style="color: #5930f2;">3. Overview des modèles possibles</h3>
      """, unsafe_allow_html=True)
   st.write("""
   Pour la sélection des modèles, notre stratégie a été la suivante :
   """)

   st.write("""
   -	Utilisation de **LazyPredict** pour avoir un aperçu des modèles d’apprentissage supervisés adéquats et de leur performance :
   """)


   # Expender
   image_path = "Images/Image_lazyp.jpg"

   with st.expander("Afficher les résultats du Lazy Predict"):
      st.image(image_path, caption="")


   # Sauts de ligne 
   st.markdown("<br><br>", unsafe_allow_html=True)  


   st.write("""
   -	Choix de **3 modèles dits “naïfs”**, pour leur simplicité et rapidité d’implémentation, puis choix de **3 modèles plus complexes** offrant potentiellement de meilleures performances :
   """)

   # Liste des modèles disponibles
   available_models = [
      "Linear Regression", 
      "Decision Tree Regressor", 
      "Random Forest Regressor", 
      "ExtraTrees Regressor", 
      "XGB Regressor", 
      "KNeighbors"
   ]

   # Sélection des modèles via st.multiselect avec des modèles pré-sélectionnés
   default_models = [
    "Linear Regression",
    "Decision Tree Regressor",  
    "KNeighbors",
    "ExtraTrees Regressor",
    "XGB Regressor", 
    "Random Forest Regressor"
   ]

   # Sélection des modèles via st.multiselect
   selected_models = st.multiselect("Sélection de 6 modèles", available_models, default=default_models)




   # Sauts de ligne 
   st.markdown("<br><br><br><br>", unsafe_allow_html=True)  
   






   # Section 4. Modélisation
   st.markdown("""
   <h3 style="color: #5930f2;">4. Modélisation</h3>
   """, unsafe_allow_html=True)
   
   from PIL import Image
   import base64

   # Expander 
   with st.expander("Voir le processus de la modélisation"):
    st.write("""Pour la partie modélisation, nous avons fonctionné selon la méthode de l'entonnoir :""")
    st.image("Images/Entonnoir.png")

   # Sous-section : Entrainement de tous les modèles avec leur hyperparamètre par défaut

   st.markdown(
      "<p style='background-color:#5930F2; color: white; padding: 8px; border-radius: 5px;'>Entrainement de tous les modèles avec leur hyperparamètre par défaut</p>",
      unsafe_allow_html=True
   )

   
   # Contenu
   st.success("""   
   Nous avons commencé par l'entrainement des 6 modèles de régression sélectionnés **avec leurs hyperparamètres par défaut** afin de comparer leur performance (classés selon LazyPredict) :
   1. ExtraTrees Regressor
   2. XGB Regressor
   3. Random Forest Regressor
   4. Linear Regression
   5. Decision Tree Regressor
   6. KNeighbors
   """)
   
   # Sauts de ligne
   st.markdown("<br>", unsafe_allow_html=True)  


   # Contenu
   st.write("""Aussi, nous avons basé notre analyse sur les métriques suivantes :
   - Coefficient de détermination (R2)
   - MAE
   - MSE
   - RMSE
   - Validation croisée
   """)



   # Sauts de ligne 
   st.markdown("<br><br>", unsafe_allow_html=True)  


   #  Sous-section : Sélectuib des 3 meilleurs modèles
   st.markdown(
      "<p style='background-color:#5930F2; color: white; padding: 8px; border-radius: 5px;'>Sélection des 3 meilleurs modèles</p>",
      unsafe_allow_html=True
   )


   # Contenu
   st.write("""
   Pour poursuivre notre modélisation, **nous avons choisi les 3 modèles qui affichent les meilleures performances** :
   1. ExtraTrees Regressor
   2. XGB Regressor
   3. Random Forest Regressor
   """)
   

   # Expender performances de chaque modèle
   data = {
      "Modèle": ["ExtraTrees Regressor", "XGB Regressor", "RandomForest Regressor"],
      "Score R2 (train)": [0.9989545616198728, 0.9973664922627589, 0.9876028297190369],
      "Score R2 (test)": [0.9678605197212603, 0.9486532220962143, 0.9532433786183869],
      "MAE train": [0.72, 1.73, 3.98],
      "MAE test": [7.04, 8.10, 7.94],
      "MSE train": [3.94, 9.92, 46.72],
      "MSE test": [108.21, 172.89, 157.43],
      "RMSE train": [1.98, 3.15, 6.83],
      "RMSE test": [10.40, 13.15, 12.55],
      "Validation croisée": [0.923307091061502, 0.906473055091632, 0.8875790095246996]
   }

   # Créer un DataFrame avec les valeurs
   df_metrics = pd.DataFrame(data)

   # Expander Afficher les performances des modèles
   with st.expander("Afficher les performances des modèles"):
      st.dataframe(df_metrics)
      st.write("""
      **Focus sur les 3 principales métriques :**

      ✅ Plus le **score de détermination (R2)** sur les données d'entraînement se rapproche de 1, plus le modèle capture la variabilité des valeurs cibles, indiquant qu'il est capable de bien expliquer la relation entre les caractéristiques (features) et les émissions de CO2 (target). 

      ✅ Plus le R2 sur les données de test se rapproche de 1, plus le modèle fonctionne sur les nouvelles données sans trop perdre de performance.

      ✅ La métrique **MAE** est la plus interprétable et adaptée pour notre problématique, car elle exprime l'erreur moyenne en unités de la cible.

      - **Interprétation de la valeur de la MAE test :**

         > Soit x la valeur MAE test :
         > Cela signifie qu'en moyenne, le modèle se trompe de x g/km dans ses prédictions des émissions du CO2 par kilomètre.
         > Il faut ensuite rapporter ce résultat (x) à l’échelle des données de la target pour savoir si l'erreur est modérée ou significative.

         > Exemple avec l'ExtraTreesRegressor :
         > Une MAE de 7 g/km sur le jeu de test correspond à environ 2% de l’amplitude totale des valeurs, soit une erreur que nous pouvons considérer comme modérée, notamment pour les véhicules à fortes émissions, mais pas pour l'inverse car l'erreur devient significative en proportion.

         > En effet, pour une voiture n’émettant que 30 g/km, une erreur de 7 g/km représente environ 23% de la valeur réelle. La prédiction est donc moins fiable pour ce type de véhicule.

      ✅ La **validation croisée** est intéressante car elle permet d’évaluer la robustesse et la généralisation du modèle.
      """)




   # Sauts de ligne 
   st.markdown("<br><br>", unsafe_allow_html=True)  



   # Sous-section : Optimisations et réentrainement
   st.markdown(
      "<p style='background-color:#5930F2; color: white; padding: 8px; border-radius: 5px;'>Optimisations et réentrainement</p>",
      unsafe_allow_html=True
   )


   # Contenu
   st.write("""
   Pour **améliorer au maximum nos modèles**, nous avons effectué plusieurs manipulations :
            
    ✅ Modification de la standardisation : nous avons **remplacé le RobustScaler par le StandardScaler** ;
            
    ✅ Modification de la taille du jeu de test : nous sommes passées **de 10 à 20%** ;
            
    ✅ Recherche des meilleurs hyperparamètres avec : les techniques **GridSearchCV et RandomSearchCV**.
      """)
   
   # Expander 
   with st.expander("En savoir plus sur le GridSearchCV et RandomSearchCV"):
    st.write("""
    Nous sommes appuyées sur les deux techniques les plus connues :
    - **La technique GridSearch** consiste à effectuer une recherche sur
      une grille prédéfinie d'hyperparamètres pour identifier la combinaison
      qui produit les meilleures performances du modèle.
    - **La technique RandomSearch** va échantillonner des combinaisons
      d'hyperparamètres de manière aléatoire. Elle va ensuite évaluer ces
      combinaisons pour déterminer celles qui offrent les meilleures
      performances du modèle. Cette technique a l’avantage d’être plus rapide.
    """)


   # Hyperparamètres par défaut d'ExtraTreesRegressor
   default_params = {
      'n_estimators': [100, 200, 300, 500],
      'max_depth': [None, 3, 5, 6, 7, 15, 20],
      'min_samples_split': [2, 3, 4, 5, 10, 15, 20],
      'min_samples_leaf': [1, 2, 3, 4, 6, 8, 10],
      'max_features': ['auto', 'sqrt', 'log2'],
      'bootstrap': [True, False]
   }

   # Meilleure combinaison d'hyperparamètres trouvée
   best_params = {
      'n_estimators': 300,
      'min_samples_split': 5,
      'min_samples_leaf': 1,
      'max_features': 'auto',
      'max_depth': None,
      'bootstrap': False
   }

   # Définition des hyperparamètres
   param_definitions = {
      'n_estimators': "Nbr d’arbres à construire dans le modèle",
      'max_depth': "Profondeur max de chaque arbre",
      'min_samples_split': "Nbr min d’échantillons requis pour diviser les noeuds",
      'min_samples_leaf': "Nbr min d’échantillons requis pour être une feuille",
      'max_features': "Nbr max de features à considérer pour trouver la meilleure division à chaque noeud",
      'bootstrap': "Entraînement de chaque arbre sur un échantillon aléatoire"
   }

   # Convertir toutes les valeurs en chaînes de caractères pour les afficher correctement
   default_params_str = {key: [str(val) for val in value] for key, value in default_params.items()}
   best_params_str = {key: str(value) for key, value in best_params.items()}

   # Créer un DataFrame pour regrouper toutes les informations
   data = []
   for param in default_params_str:
      # Ajouter une ligne pour chaque hyperparamètre avec sa définition, ses valeurs possibles et modifiées
      data.append([
         param, 
         param_definitions[param], 
         ', '.join(default_params_str[param]), 
         best_params_str.get(param, 'Non modifié')
      ])

   # Créer un DataFrame final avec toutes les informations
   df_final = pd.DataFrame(data, columns=['Hyperparamètres', 'Définitions', 'Valeurs possibles', 'Valeurs modifiées'])

   # Expander 
   with st.expander("ExtraTrees Regressor avec les hyperparamètres modifiés"):
      st.write("""Pour une question de temps de calcul, nous avons opté pour la technique **RandomSearchCV**. """)
      st.dataframe(df_final)




   # Hyperparamètres par défaut de XGB Regressor
   default_params = {
      'n_estimators': [100, 200, 300, 400, 500],
      'max_depth': [3, 5, 6, 7],
      'learning_rate': [0.01, 0.1, 0.2],
      'subsample': [0.5, 0.7, 1],
      'colsample_bytree': [0.5, 0.7, 1],
      'gamma': [0, 0.1, 0.2]
   }

   # Meilleure combinaison d'hyperparamètres trouvée par GridSearchCV
   best_params = {
      'n_estimators': 500,
      'max_depth': 3,
      'learning_rate': 0.2,
      'subsample': 1,
      'colsample_bytree': 0.5,
      'gamma': 0
   }

   # Définition des hyperparamètres
   param_definitions = {
      'n_estimators': "Nbr d’arbres à construire dans le modèle",
      'max_depth': "Profondeur max de chaque arbre",
      'learning_rate': "Taux d'apprentissage qui détermine la contribution de chaque arbre",
      'subsample': "Proportion d'échantillons à utiliser pour chaque arbre",
      'colsample_bytree': "Proportion de caractéristiques à utiliser pour chaque arbre",
      'gamma': "Contrôle la réduction de la perte min pour faire une nouvelle séparation"
   }

   # Convertir toutes les valeurs en chaînes de caractères pour les afficher correctement
   default_params_str = {key: [str(val) for val in value] for key, value in default_params.items()}
   best_params_str = {key: str(value) for key, value in best_params.items()}

   # Créer un DataFrame pour regrouper toutes les informations
   data = []
   for param in default_params_str:
      # Ajouter une ligne pour chaque hyperparamètre avec sa définition, ses valeurs possibles et modifiées
      data.append([
         param, 
         param_definitions[param], 
         ', '.join(default_params_str[param]), 
         best_params_str.get(param, 'Non modifié')
      ])

   # Créer un DataFrame final avec toutes les informations
   df_final = pd.DataFrame(data, columns=['Hyperparamètres', 'Définitions', 'Valeurs possibles', 'Valeurs modifiées'])

   # Expander 
   with st.expander("XGB Regressor avec les hyperparamètres modifiés"):
      st.write("""Parmi les deux techniques testées, GridSearchCV et RandomSearchCV, la technique **GridSearchCV** a été retenue pour ses meilleures performances. """)
      st.dataframe(df_final)




   # Hyperparamètres par défaut de RandomForest Regressor
   default_params = {
      'n_estimators': [50, 60, 70, 80, 90, 100, 150],
      'max_depth': [40, 50, 60, 80, 100],
      'min_samples_split': [2, 3, 4, 5],
      'min_samples_leaf': [1, 2, 3, 4],
      'max_features': ['sqrt', 5, 10, 15, 18],
      'bootstrap': [True, False]
   }

   # Meilleure combinaison d'hyperparamètres trouvée 
   best_params = {
      'n_estimators': 68,
      'min_samples_split': 2,
      'min_samples_leaf': 1,
      'max_features': 14,
      'max_depth': 68,
      'criterion': 'squared_error',
      'bootstrap': False
   }

   # Définition des hyperparamètres
   param_definitions = {
      'n_estimators': "Nbr d’arbres à construire dans le modèle",
      'max_depth': "Profondeur max de chaque arbre",
      'min_samples_split': "Nbr min d’échantillons requis pour diviser les noeuds",
      'min_samples_leaf': "Nbr min d’échantillons requis pour être une feuille",
      'max_features': "Nbr max de features à considérer pour trouver la meilleure division à chaque noeud",
      'bootstrap': "Entraînement de chaque arbre sur un échantillon aléatoire plutôt que sur l’ensemble des données"
   }

   # Convertir toutes les valeurs en chaînes de caractères pour les afficher correctement
   default_params_str = {key: [str(val) for val in value] for key, value in default_params.items()}
   best_params_str = {key: str(value) for key, value in best_params.items()}

   # Créer un DataFrame pour regrouper toutes les informations
   data = []
   for param in default_params_str:
      # Ajouter une ligne pour chaque hyperparamètre avec sa définition, ses valeurs possibles et modifiées
      data.append([
         param, 
         param_definitions[param], 
         ', '.join(default_params_str[param]), 
         best_params_str.get(param, 'Non modifié')
      ])

   # Créer un DataFrame final avec toutes les informations
   df_final = pd.DataFrame(data, columns=['Hyperparamètres', 'Définitions', 'Valeurs possibles', 'Valeurs modifiées'])

   # Expander 
   with st.expander("RandomForest Regressor avec les hyperparamètres modifiés"):
      st.write("""Pour une question de temps de calcul, nous avons opté pour la technique **RandomSearchCV**. """)
      st.dataframe(df_final)



   # Sauts de ligne 
   st.markdown("<br><br>", unsafe_allow_html=True)  

   # Sous-section : Choix du meilleur modèle
   st.markdown(
      "<p style='background-color:#5930F2; color: white; padding: 8px; border-radius: 5px;'>Choix du meilleur modèle</p>",
      unsafe_allow_html=True
   )

   # DataFrame avec les données fournies
   data = {
      'Modèles': ['ExtraTrees Regressor', 'XGB Regressor', 'Random Forest Regressor'],
      'MAE Train / Test': ['0,72 / 7,04', '1,73 / 8,10', '3,98 / 7,94'],
      'MSE Train / Test': ['3,93 / 108,2', '9,92 / 172,88', '46,71 / 157,43'],
      'RMSE Train / Test': ['1,98 / 10,40', '3,15 / 13,14', '6,83 / 12,54'],
      'R2 Train / Test': ['0,99 / 0,96', '0,99 / 0,94', '0,98 / 0,95'],
      'Validation Croisée': ['0,9233', '0,9064', '0,8875'],
      'MAE Train / Test (après)': ['3,45 / 7,22', '3,57 / 8,54', '0,72 / 7,24'],
      'MSE Train / Test (après)': ['31,30 / 116,15', '22,99 / 142,54', '3,93 / 110,53'],
      'RMSE Train / Test (après)': ['5,59 / 10,67', '4,79 / 11,93', '1,98 / 10,51'],
      'R2 Train / Test (après)': ['0,99 / 0,96', '0,99 / 0,95', '0,99 / 0,96'],
      'Validation Croisée (après)': ['0,9307', '0,9200', '0,9012']
   }

   df = pd.DataFrame(data)

   # DataFrame pour les performances avant et après
   df_display = pd.DataFrame({
      'Modèles': df['Modèles'],
      'MAE Train / Test (avant)': df['MAE Train / Test'],
      'MSE Train / Test (avant)': df['MSE Train / Test'],
      'RMSE Train / Test (avant)': df['RMSE Train / Test'],
      'R2 Train / Test (avant)': df['R2 Train / Test'],
      'Validation Croisée (avant)': df['Validation Croisée'],
      'MAE Train / Test (après)': df['MAE Train / Test (après)'],
      'MSE Train / Test (après)': df['MSE Train / Test (après)'],
      'RMSE Train / Test (après)': df['RMSE Train / Test (après)'],
      'R2 Train / Test (après)': df['R2 Train / Test (après)'],
      'Validation Croisée (après)': df['Validation Croisée (après)']
   })

   # DataFrame pour les performances avant et après
   df_display = pd.DataFrame({
      'Modèles': df['Modèles'],
      'MAE Train / Test (avant)': df['MAE Train / Test'],
      'MSE Train / Test (avant)': df['MSE Train / Test'],
      'RMSE Train / Test (avant)': df['RMSE Train / Test'],
      'R2 Train / Test (avant)': df['R2 Train / Test'],
      'Validation Croisée (avant)': df['Validation Croisée'],
      'MAE Train / Test (après)': df['MAE Train / Test (après)'],
      'MSE Train / Test (après)': df['MSE Train / Test (après)'],
      'RMSE Train / Test (après)': df['RMSE Train / Test (après)'],
      'R2 Train / Test (après)': df['R2 Train / Test (après)'],
      'Validation Croisée (après)': df['Validation Croisée (après)']
   })

   # Fonction de style pour les cellules spécifiques de Validation Croisée et des noms des modèles
   def highlight_cells(val):
      # Définir les couleurs pour les valeurs spécifiques
      colors = {
         '0,9233': '#5930F2',
         '0,9200': '#7B5CF2',
         '0,9012': '#9B84F0',
         'ExtraTrees Regressor': '#5930F2',
         'XGB Regressor': '#7B5CF2',
         'Random Forest Regressor': '#9B84F0'
      }
      
      # Vérifier si la valeur est dans le dictionnaire, sinon aucune couleur
      color = colors.get(val, '')
      if color:
         return f'background-color: {color}; color: white;'
      else:
         return ''

   # Appliquer le style avec la fonction 'highlight_cells' sur les colonnes ciblées
   styled_df = df_display.style.applymap(
      highlight_cells, 
      subset=['Validation Croisée (avant)', 'Validation Croisée (après)', 'Modèles']  # Ajouter la colonne 'Modèles' contenant les noms des modèles
   )


   # Contenu
   st.markdown("""
   Voici un tableau récapitulatif des 3 modèles les plus performants **avant et après l'ajustement des hyperparamètres**.
   
   Nous nous sommes basées sur les valeurs de la **validation croisée** pour faire notre choix :
   """)

   # Affichage du tableau formaté dans Streamlit avec un style pour mettre en valeur les colonnes Validation Croisée
   st.dataframe(styled_df)

   
   # Contenu
   st.success("Le modèle le plus robuste et le plus performant est l’**Extra Trees Regressor avec ses hyperparamètres par défaut**, bien que sa validation croisée soit supérieure après modification des hyperparamètres.  \
   Après plusieurs tests, le modèle s’est montré plus fiable (lors de la conception du CO2 CALCULATOR).  \
   En seconde position le XGB Regressor avec les hyperparamètres modifiés et enfin le Random Forest Regressor avec les hyperparamètres modifiés.")




   # Sauts de ligne 
   st.markdown("<br><br>", unsafe_allow_html=True)  



   # Sous-section : Interprétation
   st.markdown(
      "<p style='background-color:#5930F2; color: white; padding: 8px; border-radius: 5px;'>Interprétation</p>",
      unsafe_allow_html=True
   )

   st.write("""Pour **interpréter concrètement nos modèles**, c'est à dire pour savoir **quelles caractéristiques influent le plus sur l'émission de CO2**, nous nous sommes appuyées sur la technique **Features Importance**. 
   
   > Celle-ci a mesuré l'influence de chaque caractéristique d'entrée sur la prédiction de chacun de nos modèles de machine learning. Grâce à elle, nous avons pu clairement identifier les variables les plus significatives. 
            
   Afin d'avoir des résultats plus fiables, **nous avons comparé les caractéristiques et leur importance pour les 3 modèles les plus performants**. """)

   # Afficher les Importance Features de chacun des 3 modèles
   with st.expander("Voir les diagrammes des Features Importance"):
      # Chemins des images 
      img_path1 = "Images/IF_ETRegressor.jpg"
      img_path2 = "Images/IF_XGBRegressor.jpg"
      img_path3 = "Images/IF_RFRegressor.jpg"

      # Affichage des images dans l'expander
      st.image([img_path1, img_path2, img_path3], caption=["ExtraTrees Regressor", "XGB Regressor", "Random Forest Regressor"])


   # Afficher les Features Importance de chacun des 3 modèles
   with st.expander("Voir le tableau compararif"):
      # Chemins de l'image
      img_path1 = "Images/Tableau_top_features.jpg"

      # Affichage des images dans l'expander
      st.image([img_path1])


   st.write("""Avec le modèle Extra Trees Regressor, le Features Importance met en évidence la **Puissance fiscale** qui est la caractéristique qui influe le plus dans les émissions de CO2 des véhicules (poids supérieur de 40%).
            """)

   st.success("""
   En cumulant l’importance de ces caractéristiques sur les 3 modèles, les 3 caractéristiques les plus importantes sont : 
   - **la Puissance fiscale** (exprimée en chevaux fiscaux/CV pour taxation) ;
   - la **Carrosserie Minibus** (forme du véhicule) ;
   - la **Masse OM Max** (poids maximal du véhicule chargé : poids à vide + fluides + passagers + équipements divers) ;
   - l'**Energie Essence** (carburant).
   """)


   # Import de l'image gif
   gif_path = Path("Images/Image_features_gif.gif")
   gif_path_str = str(gif_path)
   st.image(gif_path_str)







# Page 5 DEMONSTRATION
if page == pages[4] : 
   # Import de l'image gif
   gif_path = Path("Images/Image_demo_gif.gif")
   gif_path_str = str(gif_path)
   st.image(gif_path_str)


   # DEMONSTRATION
   # Chargement des objets joblib 
   X_train = joblib.load(Modèles/'X_train') 
   X_train_concat = joblib.load(Modèles/'X_train_concat')  # Chargement des données d'entraînement
   et_regressor = joblib.load(Modèles/'et_regressor')  # Chargement du modèle Extra Trees

   # Calcul de la moyenne et de l'écart-type de X_train
   mean_puissance = X_train['Puissance_fiscale'].mean()
   std_puissance = X_train['Puissance_fiscale'].std()
   mean_masse = X_train['Masse_OM_Max'].mean()
   std_masse = X_train['Masse_OM_Max'].std()

   # Interface Streamlit pour l'utilisateur

   # Titre 
   st.markdown("""
    <h3 style="color: #5930f2;">Puissance</h3>
    """, unsafe_allow_html=True)
   
   # Slider pour Puissance Fiscale
   puissance_fiscale = st.slider('Puissance fiscale (en CV)', min_value=1, max_value=100, value=10)

   
   # Titre 
   st.markdown("""
    <h3 style="color: #5930f2;">Poids</h3>
    """, unsafe_allow_html=True)
   

   # Champ pour entrer la Masse OM Max
   masse_om_max = st.number_input('Masse OM Max (en kg)', min_value=800, max_value=3000, value=1915)

   # Pour les autres caractéristiques, on utilise des boutons radio (0 ou 1)
   # Titre 
   st.markdown("""
    <h3 style="color: #5930f2;">Carburant</h3>
    """, unsafe_allow_html=True)
   
   energie_essence = st.radio('Essence', [0, 1], index=0)
   energie_gazole = st.radio('Gazole', [0, 1], index=0)
   energie_essence_elec_hnr = st.radio("Hybride essence et électrique HNR", [0, 1], index=0)
   energie_gaz_elec_hnr = st.radio('Hybride gazole et électrique HNR', [0, 1], index=0)   
   energie_essence_gpl = st.radio('Essence et G.P.L.', [0, 1], index=0)
   energie_superethanol = st.radio('Superethanol', [0, 1], index=0)
   st.write("*Si toutes les variables énergie ci-dessus sont = 0, alors l'énergie 'Hybride essence et électrique HR' = 1*")

   # Titre 
   st.markdown("""
    <h3 style="color: #5930f2;">Carrosserie</h3>
    """, unsafe_allow_html=True)
    
   carrosserie_break = st.radio('BREAK', [0, 1], index=0)
   carrosserie_cabriolet = st.radio('CABRIOLET', [0, 1], index=0)
   carrosserie_combispace = st.radio('COMBISPACE', [0, 1], index=0)
   carrosserie_coupe = st.radio('COUPE', [0, 1], index=0)
   carrosserie_minibus = st.radio('MINIBUS', [0, 1], index=0)
   carrosserie_monospace = st.radio('MONOSPACE', [0, 1], index=0)
   carrosserie_monospace_compact = st.radio('MONOSPACE COMPACT', [0, 1], index=0)
   carrosserie_ts_terrains_chem = st.radio('TS TERRAINS/CHEMINS', [0, 1], index=0)
   st.write("*Si toutes les variables carrosserie ci-dessus sont = 0, alors la carrosserie 'Berline' = 1*")

   # Titre 
   st.markdown("""
    <h3 style="color: #5930f2;">Transmission</h3>
    """, unsafe_allow_html=True)
   
   type_boite_mecanique = st.radio('Mécanique', [0, 1], index=0)
   type_boite_variation_continue = st.radio('Variation continue', [0, 1], index=0)
   st.write("*Si toutes les variables carrosserie ci-dessus sont = 0, alors la transmission 'Automatique' = 1*")


   # Créer un dictionnaire avec les valeurs sélectionnées
   vehicule_A = {
      'Puissance_fiscale': puissance_fiscale,
      'Masse_OM_Max': masse_om_max,
      'Energie_ESS+ELEC HNR': energie_essence_elec_hnr,
      'Energie_ESS+G.P.L.': energie_essence_gpl,
      'Energie_ESSENCE': energie_essence,
      'Energie_GAZ+ELEC HNR': energie_gaz_elec_hnr,
      'Energie_GAZOLE': energie_gazole,
      'Energie_SUPERETHANOL': energie_superethanol,
      'Carrosserie_BREAK': carrosserie_break,
      'Carrosserie_CABRIOLET': carrosserie_cabriolet,
      'Carrosserie_COMBISPACE': carrosserie_combispace,
      'Carrosserie_COUPE': carrosserie_coupe,
      'Carrosserie_MINIBUS': carrosserie_minibus,
      'Carrosserie_MONOSPACE': carrosserie_monospace,
      'Carrosserie_MONOSPACE COMPACT': carrosserie_monospace_compact,
      'Carrosserie_TS TERRAINS/CHEMINS': carrosserie_ts_terrains_chem,
      'Type_de_boite_MECANIQUE': type_boite_mecanique,
      'Type_de_boite_VARIATION CONTINUE': type_boite_variation_continue
   }

   # Création du DataFrame
   vehicules_df = pd.DataFrame([vehicule_A])

   # Normalisation des données
   vehicules_df['Puissance_fiscale'] = (vehicules_df['Puissance_fiscale'] - mean_puissance) / std_puissance
   vehicules_df['Masse_OM_Max'] = (vehicules_df['Masse_OM_Max'] - mean_masse) / std_masse

   # Faire la prédiction
   predictions = et_regressor.predict(vehicules_df)

   # Afficher les prédictions
   st.markdown(
      f"""
      <div style="background-color: #5930F2; padding: 20px; border-radius: 10px;">
         <h4 style="color: white;">Prédiction des émissions de CO2 pour ce véhicule : {predictions[0]:.2f} g/km</h4>
      </div>
      """, 
      unsafe_allow_html=True
   )







# Page 6 CONCLUSION
if page == pages[5] : 

   # Import de l'image gif
   gif_path = Path("Images/Image_conclu_gif.gif")
   gif_path_str = str(gif_path)
   st.image(gif_path_str)

   # Texte
   st.write("""
      Par le biais du CO2 CALCULATOR, nous recommandons aux constructeurs d’apporter une attention particulière à :
      - La **puissance fiscale** ;
      - Le **niveau d’équipement du véhicule** (Masse OM MAX - en ordre de marche maximum), avec une attention particulière aux **véhicules de gros volume**, tels que ceux à **carrosserie Minibus** ;
      - L'**énergie utilisée**, en particulier l'**essence**.
               
      Ces caractéristiques doivent être prises en compte, car elles ne favorisent pas une réduction optimale des émissions de CO2.
   """)

   # Sauts de ligne 
   st.markdown("<br>", unsafe_allow_html=True)  

   # Texte encadré avec un fond violet
   st.markdown(
      """
      <div style="background-color: #5930F2; padding: 20px; border-radius: 10px; color: white;">
         <p><b>Ainsi, le CO2 CALCULATOR</b> permettra aux constructeurs désireux de maîtriser au maximum l'émission de CO2 de leur nouveau véhicule, d’ajuster avec précision chaque caractéristique influente.</p>
         <p>L’enjeu est important car il leur permettra d’éviter un potentiel Malus écologique, qui nous le savons, impacte le prix du véhicule et donc son achat par les consommateurs.</p>
      </div>
      """, 
      unsafe_allow_html=True
   )

   # Sauts de ligne 
   st.markdown("<br>", unsafe_allow_html=True)  


   # Expander AXES D'AMELIORATION
   with st.expander("Axe d'amélioration du modèle"):
      st.markdown(
         """
         En guise d’amélioration du modèle, nous pourrions augmenter le volume du dataset avec des données futures, 
         par exemple de 2024 et au-delà, afin d’améliorer sa robustesse.
         """,
         unsafe_allow_html=True
      )


   # Expander CONCLUSION PERSONNELLE
   with st.expander("Conclusion personnelle"):
      st.markdown(
      """
      Enfin, nous souhaitons terminer ce rapport avec une conclusion plus
      personnelle. Tout d’abord, nous avons été surprises par la qualité des
      prédictions de notre modèle et plus largement impressionnées par la
      puissance du Machine Learning. Ce projet nous a permis de passer de
      concepts très abstraits, abordés lors des différents sprints, à une application
      très concrète et à des réalités professionnelles.
      
      Nous nous sommes plus que jamais rendues compte que la conduite d’un tel
      projet nécessite une solide base de compétences techniques, de la rigueur
      mais aussi beaucoup de curiosité, d’assiduité et de ténacité pour arriver à des
      résultats concluants. 
      
      Par ailleurs, nous tenons à souligner l’importance de la
      dimension humaine pour nous toutes. Le projet fut l’occasion de nous exercer
      à la conduite de projets data en groupe, et en distanciel, mais aussi
      d’échanger sur la formation, de confronter nos idées et de nous encourager
      mutuellement.
      
      Pour finir, nous tenons à remercier vivement Eliott Douieb pour la qualité
      de son accompagnement tout au long de ce projet.
      """,
      unsafe_allow_html=True
   )



