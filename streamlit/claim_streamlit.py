#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:03:05 2021

@author: M.Benrhouma
"""
#import des librairies
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

import numpy as np
import seaborn as sns
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import association_metrics as am



#fonction pour le calcul de la metric 'gini_normalized'
def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

score=make_scorer(gini_normalized,greater_is_better = True)


features_name = ('feats')
target_name=('target')
feats_enrichies=('feats_enrichies')   


#import des données
@st.cache(allow_output_mutation=True)
def load_features(features_name): 
    feats=pd.read_csv('streamlit/feats_maj.csv', sep=';',index_col='Identifiant')
    return (feats)   

@st.cache(allow_output_mutation=True)
def load_target(target_name): 
    target=pd.read_csv('streamlit/target.csv', sep=';',index_col='Identifiant')
    return (target)     

feats=load_features('feats')
target = load_target('target')


X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2,random_state=42)


#### mes fonctions#########
@st.cache(allow_output_mutation=True)
def load_features_enrichie(feats_enrichies): 
    feats_enrichies=pd.read_csv('streamlit/feats_enrichies.csv', sep=';')
    return (feats_enrichies)   

feats_enrichies=load_features_enrichie('feats_enrichies')




X_train, X_test, y_train, y_test = train_test_split(feats_enrichies, target, test_size=0.2,random_state=123)

models=("GB","RF","XGB","LR","SVM")

@st.cache(allow_output_mutation=True)
def crossValidation(models):
    #on  prépare la  configuration pour la méthode de  cross validation 
    seed = 42
    # prepare models
    models = []
    models.append(('GB', GradientBoostingClassifier()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('XGB', XGBClassifier()))
    models.append(('LR', LogisticRegression()))
    models.append(('SVM', SVC()))

            
    # evaluer  chaque modèle 
    results = []
    names = []
    scoring = score
    x=[1,2,3,4,5]
    kfold = model_selection.KFold(n_splits=10, random_state=seed,shuffle=True)
    for name, model in models:
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=score)
        results.append(cv_results)
        names.append(name)
        # boxplot algorithm comparison
    fig = plt.figure()
    plt.boxplot(results)
    plt.xticks(x, names)
    plt.title('Boxplot algorithm comparison')
    return fig




#titre de l'appli###########
html_temp = """
    <div style="background-color:SteelBlue;padding:10px">
    <h2 style="color:white;text-align:center;"> Claim prediction ML App </h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)



#creation du menu et mise en forme
st.sidebar.image('streamlit/logo.png')
rad=st.sidebar.radio("Menu", ["Home","Présentation du projet", "Les jeux de données","Cleaning et préparation des données ","Analyse visuelle des données","Méthodologie","Modélisation","Conclusion"])

if rad == "Home":
    st.write(
            """ 
            ##
            ##
            """)     
   
    st.image("streamlit/site_expose.png")
    html_temp = """ 
     <h2 style="color:black;text-align:center;"> Data Analyst : Promotion Bootcamp  Mars 2021</h2>
    
    """

    st.markdown(html_temp,unsafe_allow_html=True)
    break_line = '<hr style="border:2px solid gray"> </hr>'

    st.sidebar.markdown(break_line, unsafe_allow_html = True)
    st.sidebar.markdown("### Background")
    st.sidebar.markdown("Ce projet a été réalise sur un jeu de données de 10 114 observations proposé par Generali dans le cadre du site Challenge")
    st.sidebar.markdown(break_line, unsafe_allow_html = True)
    # Info box
    st.sidebar.markdown("### Réalisé par :")
     
    
    st.sidebar.info('''
		
        Ghayet El Mouna Ghazouani, [Linkedin](https://www.linkedin.com/in/ghayet-el-mouna-ghazouani-64454b15b/)
        
        
        Claude Haik, [Linkedin](https://www.linkedin.com/in/claudehaik/)
        
        Edouard Delannoy
        
        Michael Abergel,[Linkedin] (https://fr.linkedin.com/in/michaelabergel)
        ''')
        
    st.sidebar.markdown("### Sous la supervision de: ")
    st.sidebar.info("Thibault VENET de Datascientest.")
    st.sidebar.markdown(break_line, unsafe_allow_html = True)
    st.sidebar.markdown("### Sources: ")
    st.sidebar.write(
    """
    [![](https://img.shields.io/badge/GitHub%20-Claimpy-informational)](https://github.com/DataScientest-Studio/Claimpy.git)
    """)  
                                                                             

            

#partie presentation projet

if rad == "Présentation du projet":
    
    #creation des sous section presentation
    st.write(
            """ 
            ##
            """)     
    titre=st.checkbox("Présentation du projet")
    if titre:
            st.write('''
            - Il s’agit d’un projet dans le cadre du site Challenge Data ENS.''')
            st.write('''Nous avons un jeu de données d'immeubles de l’assureur Generali sans sinistre ou avec au moins un sinistre reporté sur une période donnée.''')
            st.write('''La police d’assurance d’un bâtiment est une assurance proposée à des bailleurs pour 
            protéger la structure de l’habitation ainsi que tous les événements extérieurs
            susceptibles de provoquer un sinistre.
            ''')
            st.write(''' - Pour les assurances, obtenir ce type d’indicateur permettrait d’adapter leur grille tarifaire au risque réel, au coût réel qu’il peut y avoir.''')
            st.write('''Ce type d’indicateur permet également de prédire le nombre de réclamations qu’un assureur recevra dans un futur proche.''')
            
    titre2=st.checkbox("Contexte")
    if titre2:
                st.write(''' - Le projet est challenging au regard du temps disponible et de la qualité du jeu de données fourni.''')
                st.write('''Il fait appel à une partie conséquente d’analyse et d'enrichissement des données et la mise en place d’une série d’évaluations sur différents
                modèles de Machine Learning en privilégiant leur benchmark sur le modèle xg boost testé
                par Générali avec un score NGC normalized gini coefficient de 0.41.''')
    titre3=st.checkbox("Objectif de l'étude")
    if titre3:  
                st.write(' - L’objectif de cette étude est de prédire le risque de sinistre sur un bâtiment pendant une certaine période.')
                st.write('''Le problème consiste, à partir d’un jeu de données fourni par Generali de prédire avec un modèle basé sur certaines caractéristiques du bâtiment si ce dernier fera l'objet d'une
réclamation d'assurance pendant une période donnée.''')
                st.write('''La métrique normalized gini est imposée pour ce projet.''')
                st.write('\n')

    
#partie datasets                
if rad == "Les jeux de données":
    st.write(
            """ 
            ##
            """)     

    titre=st.checkbox("Description des données")
    if titre:
            st.write('''
            - Le jeu de données disponible est constitué de 10.114 observations ; chacune identifiée par un numéro INSEE et correspond à un bâtiment assuré par Generali ''')
            st.write('''- Chaque bâtiment est décliné avec 25 variables dont 19 anonymisées
                     représentant une caractéristique du bâtiment assuré par la compagnie.''')
                     
    titre=st.checkbox("Description des variables")
    if titre:
            st.write('''
            - Le jeu de données ne présente aucun doublon.''')
            st.write('''- Chaque bâtiment assuré est identifié par un INSEE unique''')
            st.write('''- Plusieurs variables ont des descriptions manquantes (NaN)''')
            
            st.image("streamlit/variables.png")
            
            my_expander = st.expander(label='Informations importantes')
            with my_expander:
                '''- Le fichier source compte plus de 10.000 observations mais 1.500 INSEE uniques.'''
                '''- Cette partie a clairement monopolisé plusieurs ressources  une grande partie de la durée du projet, mais a permis de finaliser  et arriver à un enrichissement très qualitatif du jeu d’origine grâce aux sources externes sur les catastrophes naturelles.'''
    
    titre=st.checkbox("Les jeux de données externes")   
    if titre: 
            my_expander = st.expander(label='Informations importantes')
            with my_expander:
                '''Nous nous sommes heurtés à un choix limité de sources pouvant apporter un complément utile au set de base'''
                '''- à la nécessité de matcher sur une seule variable possible : le code INSEE'''
                '''- à la qualité des données trouvées, incomplètes ou sans codes INSEE'''
                '''- en quantité insuffisante faisant courir le risque de matcher avec peu de codes INSEE'''
                '''de notre fichier source (perte de données) ou générant des NaN'''
                
            st.write('''3 jeux de données ont été utilisés :''')
            st.write('''- la fréquence des déclarations de sinistres par les assureurs
                         en fonction du code INSEE, de 1995 à 2016 (source Géorisques)''')
            st.write('''- La correspondance code Insee et données géographiqes (source DataGouv)''')
            st.write('''- Les “arrêtés de catastrophes naturelles” en fonction du code INSEE de 1982 à 2015 (source DataGouv)''')
                
    titre=st.checkbox("La variable cible")   
    if titre: 
            st.write('''- La variable cible ‘target’ comporte deux modalités correspondant à l’absence de
                     réclamation (0) ou à la constatation d’au moins une réclamation (1) sur la période assurée.''')
            st.write('''- Elle ne nécessite aucun traitement spécifique pour faciliter son traitement.''')
            st.write('''- La distribution des sinistres est déséquilibrée (80/20) comme en témoigne le graphique
                     suivant.''')
            st.image("streamlit/cible.png")

##########partie Analyse exploratoire#########            

if rad == "Cleaning et préparation des données ":
    
    st.markdown("Le dataset de départ présente des colonnes incomplètes et des données à regrouper avant de passer à l'étape de modélisation.")
    st.markdown("On distingue ci-après des actions de nettoyage usuelles mais aussi de transformation pour faciliter le traitement par les algorithmes.")

    titre_clean1=st.checkbox("Principales actions de cleaning")
    if titre_clean1:
    
        st.write('''1- variable **EXPO** (durée d'assurance en cours d'année): suppression des ponctuations et transformation en **variable continue**.''')
        st.write('''2- l'information **INSEE** est absente de plusieurs lignes. Sachant qu'elle sera indispensable pour l'apport de données externes, nous supprimons ces quelques lignes.''')
        st.write('''3- l'information **Superficie** est absente de nombreuses lignes. Voir le paragraphe Analyse visuelle sur le remplacement pour les valeurs manquantes par la **médiane**.''')
        st.write('''4- la variable ft_24_categ présente quelques **'.'** et des valeurs extrêmes, que l'on remplace respectivement par des 0 ou des 10.''')  
    
    titre_clean2=st.checkbox("Préparation du jeu de données avant le pré-processing")
    if titre_clean2:
        st.write('''
        * Création d’une nouvelle variable “Age du bâtiment"
         
        L’âge du bâtiment est une nouvelle variable créée selon la description suivante:
        
        Age bâtiment = Année observée par la police d’assurance - Année de construction''')
        
        my_expander = st.expander(label='** Détail important  **')
        
        with my_expander:('''-Face à un taux de NaN supérieur à 10% sur la variable Année de construction (ft_22_categ) on a utilisé la méthode knn_imputer pour les remplacer.''')
        
        st.write('''
        * Discrétiser ‘superficief’
        La superficie du bâtiment a été ventilé selon la correspondance suivante :
        
        Superficie : [0,500,1002,2190,30745],
        labels = ['petit','moyen','trés-grand','Géant'])
    
        ### ==> Une fois les variables dummisées, les données sont prêtes à être utilisées.
            ''')

    
    
    titre_clean3=st.checkbox("Préparation des données externes 'Périls' avant intégration avec les données de base")
    if titre_clean3:  
        st.markdown('''Il s'agit préparer le jeu de données sur les **périls liés aux catastrophes naturelles** qui produira la variable **'Péril'** mise en oeuvre plus loin dans les étapes de modélisation et interprétation.''')
                
        st.write('''La base des 'Arrêtés de catastrophes naturelles' (source officielle .gouv) est totalement factuelle et fiable pour connaître l'intensité du péril au niveau d'un bâtiment.''')       
        st.subheader("Résumé des étapes de preprocessing sur les catastrophes naturelles")
        st.write('''1- On regroupe d'abord 40 types de périls différents - d'inondation à tornades... -, en trois catégories : action de la Méteo, du Sol, et de l'Eau.''')
        st.write('''2- On label-encode ces informations et on cumule ces périls sur chaque ligne ou code INSEE. Une même commune peut en effet avoir subi plusieurs catastrophes naturelles.''')
        st.write('''3- le dataframe résultant permet de dresser des cartes géographiques de périls et il peut être intégré avec le dataset sur les bâtiments sur la base du code INSEE.''')

        



if rad=="Analyse visuelle des données":
    st.markdown("## *Pourquoi commencer par une analyse visuelle ?*")
    st.write('''
            Les techniques graphiques offrent l'avantage d'identifier
            rapidement des relations entre variables.
            On les confirme ensuite par les calculs.''')
    
    rad2=st.radio("",["1- Sur les données numériques du dataset d'origine","2- Recherche de corrélations entre variables à l'aide de **heatmaps**","3- Analyse visuelle à l'aide de données externes"])
    
    
    if rad2=="1- Sur les données numériques du dataset d'origine":
        
        
            
    # 1- SUR LES DONNEES DE BASE
        
            st.subheader('''Superficie et age du bâtiment ont-ils une relation avec les sinistres ?''')
            st.write('''
    Année de construction et superficie sont les seules données numériques 
    du dataset qui intuitivement auraient un impact sur la probabilité d'un sinistre.''')
    
        # Chargement du dataframe X_train et fusion avec y_train (target)
            df = pd.read_csv('streamlit/X_train.csv', sep = ',')
            df=df.drop(['Identifiant'],axis=1)
            df_y = pd.read_csv('streamlit/y_train_saegPGl.csv', sep = ',')
            data = pd.concat([df, df_y], axis = 1)
    
    # Scatterplot combiné de tous les bâtiments avec age et superficie pour axes. Et hue = target
    
            df = pd.read_csv('streamlit/X_train.csv', sep = ',')
            plt.figure(figsize = (3,3))
            fig, ax = plt.subplots()
            plt.xlabel("Année de construction")
            plt.ylabel("Superficie")
            sns.scatterplot(x = data.ft_22_categ, y = data.superficief, hue=data.target)
            plt.title("Chaque point est un bâtiment assuré", fontsize= 18) 
            plt.legend(title = 'Point jaune = sinistré', fontsize = 'small', title_fontsize = 10, loc='upper right' )           
            plt.xlim(1960,2016)
            plt.ylim(0, 20000);

            fig=plt.gcf()
            st.pyplot(fig)
            st.write('''**_En première analyse_**, les bâtiments de grande surface présenteraient un plus grand nombre de sinistres. Mais surtout, on observe des valeurs de superficie très élevées.''')

            st.subheader("Etude de la variable superficie des bâtiments")
            st.write ("Un aperçu des données d'origine montre beaucoup de valeurs manquantes (scroller la liste)")
            st.write(data['superficief'])
            st.write ('''nombre de valeurs manquantes''')
            st.write (df['superficief'].isnull().sum(axis=0))
    
            st.write("**_Mais par quoi remplacer ces valeurs manquantes ?_**")

            st.write('''Comparons les valeurs moyennes et médianes''')
            Bouton_surface1=st.checkbox("Valeur moyenne de la superficie (m2)")
            if Bouton_surface1:
                st.write(round(df.superficief.mean()))

            Bouton_surface2=st.checkbox("Valeur médiane de la superficie(m2)")
            if Bouton_surface2:
                st.write(round(df.superficief.median()))

                st.write('''Sachant que la valeur moyenne est presque **le double de la valeur médiane**,''')
                st.write('''Il y a donc des valeurs **très extrêmes** parmi les superficies.''') 
                st.write('''Nous remplacerons les valeurs manquantes par la médiane, plus représentative.''')

                st.subheader("Confirmons le phénomène par un découpage en quartiles")
                plt.figure(figsize = (4,4))
                sns.boxplot(y = df.superficief[df.superficief < 14000]);
                plt.ylabel("Superficie (m2)")
                fig=plt.gcf()
                st.pyplot(fig)
                st.markdown("**_Attention_**: ces valeurs extrêmes de superficie ne sont pas forcément des valeurs aberrantes. A conserver dans le dataset.")


                st.subheader('''Analysons l'influence directe de la superficie sur les sinistres''')
                plt.figure(figsize= (10, 10))
                plt.ylim(0,10000)
                sns.boxplot(y = data.superficief, x = data.target );
                plt.title("Distribution des superficies: avec et sans sinistre", fontsize=20)
                plt.xticks([0,1], ['sans sinistre déclaré', 'avec sinistre déclaré'])

                fig=plt.gcf()
                st.pyplot(fig)

                st.write('''Le graphique montre que les bâtiments sinistrés sont plus grands que les non-sinistrés.''')

            st.subheader('''Enfin, validons cette hypothèse par un calcul de type **ANOVA**''')
            st.write('''La fonction 'F_oneway'de scipy.stat teste l'hypothèse *H0 = les 2 variables superficief et target sont indépendantes*''')  
            st.write('''F_onewayResult : statistic=6472.082859614605, **pvalue < 5%**.
            Il y a donc bien **une dépendance entre la superficie et les sinistres**''')  
    
    
    if rad2=="2- Recherche de corrélations entre variables à l'aide de **heatmaps**":
# 2- LES HEATMAPS V-CRAMER ET ANOVA
        
        col1, col2=st.columns(2)
    
        res_bouton1 = st.button('1- Entre variables numériques')
    
        if res_bouton1:
            st.write('''## *Analyse de corrélation entre les 4 variables numériques.* ''')

            df = pd.read_csv('streamlit/claim_c.csv',sep=';',index_col='Identifiant')
       #suppression de 'Unnamed: 0'
            df.drop('Unnamed: 0',axis=1,inplace=True)

       #selection des variables numériques
            df_num=df.select_dtypes(include=['float'])

       # calcul de corrélation entre les variables numériques
            corr=df_num.corr()

       #graphique de corrélation
       #plt.figure(figsize = (5,5))
            fig, ax = plt.subplots(figsize=(2,2))
            sns.set(font_scale = 0.4)
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            sns.heatmap(corr, annot= True, ax= ax, annot_kws= {'size': 5}, cmap="viridis",center=0)
      
            fig=plt.gcf()
            st.pyplot(fig)
   
        res_bouton2 = st.button('2- Entre variables catégorielles', help = 'attention ce calcul prend quelques secondes')
        if res_bouton2:
       
            st.markdown("Il y a un petit temps de calcul du **_V de Cramer_** qui compare ici de nombreuses variables...")
           
            df = pd.read_csv('streamlit/claim_c.csv',sep=';',index_col='Identifiant')
       #selection des variables catégorielles
            df_cat=df.select_dtypes(exclude=['float'])

         #suppression de 'Insee'       
            df_cat.drop('Insee',axis=1,inplace=True)
            df_cat.drop('Unnamed: 0',axis=1,inplace=True)


       #Test de corrélation via' V de cramer' entre les variables catégorielles

       
       # Convertir  les colonnes qui ont des str en  Category 
            df_cat = df_cat.apply(
            lambda x: x.astype("category") if( x.dtype == "O") | (x.dtype == "int" ) else x)

       # Initializer les objets de CamresV en utilisant df_cramer
            cramersv = am.CramersV(df_cat) 

       #retournera une matrice par paires remplie du V de Cramer, où les colonnes et l’index sont les variables 
       #catégorielles de df_cat
            corr_cramer=cramersv.fit()

       #graphique de corrélation
            st.write('''## *Analyse de corrélation entre variables catégorielles*''')
            sns.set(font_scale = 1.2)
            fig, ax = plt.subplots(figsize=(15,15))
            sns.heatmap(corr_cramer, annot= True, annot_kws= {'size': 11}, ax= ax, cmap="viridis")
            ax.set_title("Test de corrélation via' V de cramer' entre les variables catégorielles");
            fig=plt.gcf()
            st.pyplot(fig)
       
            st.write(''' On remarque une forte corrélation de l'ordre de 1 entre ft_15 et ft_16 et avec les variables suivantes:
                    de ft_6 jusqu'à ft_17 et ft_24.''')
            st.write ('''## *Notre décision* : ''')
            st.write ('''Ces variables semblent apporter la même information et seront remplacées par la variable ft_24_categ.
                    Nous procédons donc à la suppression des variables suivantes :
                    ft_17_categ','ft_9_categ','ft_7_categ','ft_15_categ','ft_16_categ','ft_8_categ','ft_10_categ','ft_11_categ','ft_12_categ','ft_13_categ','ft_14_categ'1 ''')
   
    
            st.write('''## *Nouvelle heatmap après supppression des variables redondantes*''')
            df_cat_drop=df_cat.drop(['ft_17_categ','ft_9_categ','ft_7_categ','ft_15_categ','ft_16_categ','ft_8_categ','ft_10_categ','ft_11_categ','ft_12_categ','ft_13_categ','ft_14_categ',]
                    ,axis=1)
       # Initializer les objets de CamresV en utilisant df_cramer
            cramersv = am.CramersV(df_cat_drop) 

       #retournera une matrice par paires remplie du V de Cramer, où les colonnes et l’index sont les variables 
       #catégorielles de df_cat
            corr_cramer=cramersv.fit()

       #graphique de corrélation
            sns.set(font_scale = 1.2)
            fig, ax = plt.subplots(figsize=(15,15))
            sns.heatmap(corr_cramer, annot= True, annot_kws= {'size': 11}, ax= ax, cmap="viridis")
            ax.set_title("Test de corrélation via' V de cramer' entre les variables catégorielles");
            fig=plt.gcf()
            st.pyplot(fig)
   

# 3 - LES DONNEES EXTERNES CATASTROPHES NATURELLES SUR LA CARTE  
    if rad2=="3- Analyse visuelle à l'aide de données externes":
        st.markdown('''Les données externes utilisées : les coordonnées géographiques - latitude & longitude par code INSEE - et les arrêtés de catastrophes naturelles.''')
   # chargement des observations du dataset Claimpy d'origine, après cleaning
        claim_c=pd.read_csv('streamlit/claim_c.csv', sep=';')
   # chargement des coordonnées géographiques et régions des codes INSEE
        geocoord=pd.read_csv('streamlit/Codes geo_gps.csv', sep=';')
   # Ajout des coordonnées géographiques longitude - latitude, département, région  
        claim_geo= claim_c.merge(geocoord, on='Insee', how='left')
   # Tous les codes INSEE du dataset d'origine n'ont pas matché avec les codes INSEE
   # d'où nécessité de supprimer les lignes avec des NaN (10% Région absente)
        claim_geo=claim_geo.dropna(axis=0, how='all', subset= ['Region'])
   # Il reste 9.000 observations dans le dataframe à analyser

   # Regroupement/compilation des données par région (somme et quantité)
   # nota: 'sum'sur la variable 'target'correspond à la somme des sinistres déclarés
        function_to_apply= {'target' : ['sum', 'count']}
        par_region2 = claim_geo.groupby('Region', as_index= False).agg(function_to_apply)
   # L'information qui nous intéresse est le ratio de déclaration de sinistres 
   # sur l'ensemble des assurés d'une région. On crée donc une variable 'ratio'
        par_region2['ratio']= par_region2[('target','sum')] / par_region2[('target','count')]
   # et on trie sur la variable 'ratio' par ordre décroissant
        par_region2_sorted=par_region2.sort_values(by = 'ratio', ascending= False)
   # Les 22 régions sont ainsi analysées et triées

   # Affichage graphique des 22 régions par ordre décroissant de taux de sinistres
        st.subheader('''1- Pourcentage de sinistres par région en fonction du nombre de bâtiments assurés''')
        plt.figure(figsize = (12,9))
        plt.title("Sources : données d'origine du projet Claimpy et coordonnées GPS & région")
        plt.xticks(rotation=90)
        plt.ylabel('% de sinistres déclarés par assuré')
        plt.ylim(0.15, 0.31)
        sns.barplot(x ='Region',y = 'ratio', data= par_region2_sorted);
        fig=plt.gcf()
        st.pyplot(fig)

        st.write('''Que se passe-t-il dans ces régions Centre ou Ile-de-France ?''')

# Affichage des cartes de catastrophes naturelles
        st.subheader("2- Cartographie des catastrophes naturelles en France")
        st.markdown("Source de données externes : les Arrêtés de Catastrophes naturelles (source gouv.fr) depuis 1982.")

        col1, col2 = st.columns(2)
        from PIL import Image
        meteo=Image.open("Arretes catnat meteo.png")
        all_catnat=Image.open("Arretes catnat all.png")
        col1.header("Toutes les catastrophes")
        col1.image(all_catnat, use_column_width= True)
        col2.header("Exemple : les tempêtes")
        col2.image(meteo, use_column_width= True)

        st.subheader("3- Cartographie des catastrophes naturelles les plus graves en France")
   
        catnat_graves=Image.open("Arretes catnat majeures all.png")
        st.image(catnat_graves)
   
        my_expander = st.expander(label='** Provenance des données  **')
        with my_expander:'''- Données issues du travail de discretisation et compilation présenté dans la section 'préparation des données.'''

        my_expander = st.expander(label='** Solution utilisée pour afficher des cartes **')
        with my_expander:'''- Les modules **cartopy.crs** pour l'affichage des points et **cartopy.features** pour afficher les côtes,rivières et frontières.'''
 
   
        st.subheader('''4- Sachant que la superficie du bâtiment est déterminante, examinons où se trouvent les grands bâtiments.''')
                                                
   #Grands bâtiments assurés
        img = plt.imread("streamlit/50 plus grands batiments.jpg")
        st.image(img)
        st.write(''' On observe une relative concordance entre l'implantation des bâtiments de grande surface et les périls de type catastrophe naturelle qui peuvent les sinistrer.
            #Observation à confirmer bien sûr par la modélisation.''')





            
################partie Méthodologie#############

elif  rad=="Méthodologie":
    st.write(
            """ 
            ##
            """)     
               
    titre=st.checkbox("Description du problème")
    if titre:
        st.write('''
    Il s’agit d’un problème d’apprentissage supervisé dans un contexte de classification.
Toutes nos données sont étiquetées et les algorithmes apprennent à prédire le résultat des
données d’entrée.''')
    titre2=st.checkbox("Métrique utilisée")
    if titre2:
        
        st.write('''
Le gini normalized est la métrique imposée par Generali pour ce challenge et donc pour la
mesure de performance afin de choisir un modèle.
            ''')

    titre3=st.checkbox("Démarche")
    if titre3:
        st.write('''
Les étapes de modélisation qu’on a suivies restent les mêmes sans ou avec l’ajout des 
données externes.
            ''')
        img = plt.imread("streamlit/Screen demarche.jpg")
        st.image(img)


#partie modelisation #################### 
elif  rad == "Modélisation" :
    
    rad1=st.radio("",["Lazy predict","Cross validation","Optimisation des hyperparamètres","Prédiction"," Rapport d'évaluation et Interprétabilité du modèle retenu"])
    
    #creation des sous section Modélisation
    if rad1=="Lazy predict"  :
        
            
                st.markdown("### Lazy predict")
                st.write('''
                 Afin d’avoir un aperçu rapide des modèles prometteurs sans optimisation préalable,
nous avons testé notre jeu de données avec le module ‘Lazy Predict’ sur un
ensemble non exhaustif de modèles.

Ci_dessous le résultat du lazy predict :                
                ''')
                img2 = plt.imread("streamlit/lazy.jpg")
                st.image(img2)
         
                st.write('''
                 ○ Le gini normalized reste faible même avec les modèles les plus basiques tels que ‘GaussianNB’ et ‘BernouilliNB’. 


○ Nous avons alors décidé d'approfondir notre analyse avec d'autres modèles plus puissants tels que le Gradient boosting classifier
et le Xgboost qui ont une très bonne précision en général… 
Ces modèles sont très populaires vu qu'ils ont un temps de traitement plus rapide et moins de complexité !

                ''')
       
    elif rad1=="Cross validation" :
        
            st.markdown("### Cross validation")
            
            st.write('''
                     
                    Pour la comparaison et la sélection des hyperparamètres, on a choisis la validation croisée qui est le meilleur outil
                    puisqu'elle permet de diviser l'ensemble d'entraînement en K ensembles plus petits.
                    
                    Ci_dessous un box plot des résultats obtenus.
                    ''')
            crossValidationResult = crossValidation(models)
            st.pyplot(crossValidationResult)
            st.write('''
                     
                    À partir de ces résultats, on a pu conclure que le Gradient Boosting Classifier ‘GB’, le
random forest ‘RF’ et le XGboost ‘XGB’ méritent d’être étudiés plus pour notre problème de
classification même si les scores moyens obtenus par validation croisée s'avérer être une estimation pessimiste du modèle.


                     ''') 
    elif rad1=="Optimisation des hyperparamètres" :
    
            st.markdown( "### **Optimisation des hyperparamètres**")
            st.markdown("#### *la fonction compute sample weight :*")
            st.write('''
             Pour cette étape d'optimisation , on a commencé par corriger le déséquilibre de notre jeu
de données, ceci en utilisant la fonction ‘ compute_sample_weight’ qui permet d'attribuer
des poids à chaque observation et non aux classes quand il s'agit d'un modèle déséquilibré
comme le nôtre .

Ci dessous un exemple illustrant cette fonction:
            ''')
            img = plt.imread("streamlit/sample_w.jpg")
            st.image(img)
            
            url = "https://www.researchgate.net/publication/288001323_Struck_Structured_Output_Tracking_with_Kernels"
            st.write("source : [https://www.researchgate.net/publication/288001323_Struck_Structured_Output_Tracking_with_Kernels](%s)" % url)
            
                   
            
            chek=st.checkbox("Optimisation des hyperparamètres pour  Gradient boosting classifier")
            if chek :
                
                models_param=st.selectbox("Choisir un hyperparamètre ", ["n_estimators","learning_rates"]) 
        
                if models_param=="n_estimators":
        #graphique pour comparer l'evolution du gini score en fonction  des n_estimators
                    n_estimator = [1, 2, 4, 8, 16, 32, 64, 100, 200,300]
                    train_results = []
                    test_results = [] 
                    for estimator in n_estimator:
                        GB = GradientBoostingClassifier(n_estimators=estimator)
                        GB.fit(X_train, y_train,compute_sample_weight(class_weight='balanced', y=(y_train)))
                        train_pred = GB.predict(X_train)
                        score_gini_train=gini_normalized(y_train, train_pred)
                        train_results.append(score_gini_train)
                        y_pred = GB.predict(X_test)
                        score_gini_test = gini_normalized(y_test, y_pred)
                        test_results.append(score_gini_test)
        #plotting
                    line1, = plt.plot(n_estimator, train_results, 'b', label='Train gini_normalized')
                    line2, = plt.plot(n_estimator, test_results, 'r', label='Test gini_normalized') 
                    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
                    plt.ylabel('gini_normalized')
                    plt.xlabel('n_estimators')
                    plt.title('Evolution du gini score en fonction  des n_estimators pour GB model ') 
                
                    st.set_option('deprecation.showPyplotGlobalUse', False)
       
                    st.pyplot()
                    st.markdown(" D'après le graphique ci dessus on remarque que le fait d’augmenter le nombre d'estimateurs va engendrer un overfitting ce qui est pas bien pour la qualité du modèle.")
            
                else:
                    ###learning rates##########
                    learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
                    train_results = []
                    test_results = []
                    for eta in learning_rates:
                        model = GradientBoostingClassifier(learning_rate=eta)
                        model.fit(X_train, y_train,compute_sample_weight(class_weight='balanced', y=(y_train)))
                        train_pred = model.predict(X_train)
                        score_gini_train=gini_normalized(y_train, train_pred)
                        train_results.append(score_gini_train)
                        y_pred = model.predict(X_test)
                        score_gini_test = gini_normalized(y_test, y_pred)
                        test_results.append(score_gini_test)

                    line1, = plt.plot(learning_rates, train_results, 'b', label='Train gini_normalized')
                    line2, = plt.plot(learning_rates, test_results, 'r', label='Test gini_normalized')
                    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
                    plt.ylabel('gini_normalized')
                    plt.xlabel('learning_rates')
                    plt.title('Evolution du gini score en fonction  des learning_rates : GB')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
       
                    st.pyplot() 
                    
                    st.markdown(" #### Conclusion ")
                    st.write('''
                    
                    * L'optimisation des hyperparamètres est une étape cruciale dans la modélisation.
                    
                    * En fait, elle nous a permis non selement d'améliorer le score mais aussi de gérer les problèmes de sous apprentissage et sur apprentissage
                            
                             ''')

###############Prédiction##################
        
    elif rad1== "Prédiction" :
            st.markdown("### Prédiction")
         
            st.write('''
                  ### Explorer les trois modèles de classification
                  Quel est le meilleur ?
                  ''')
            classifier_names=st.selectbox(' * Choisir un modèle de classification' ,("GB","RF","XGb"))      
            
            dataset_name=st.sidebar.selectbox('Select Dataset' ,("Données origines","Données enrichies"))
            
            if dataset_name=="Données origines":
                
                load_features(feats)
                load_target(target)
                st.sidebar.write('Informations sur le dataframe  :') 
                st.sidebar.write( "* Shape:",feats.shape)
                st.sidebar.write(" * Nombre de classes :",len(np.unique(target)))
                if classifier_names=="GB":
                    
                    GB_optimised = GradientBoostingClassifier(n_estimators=25,learning_rate=0.1)
                    GB_optimised.fit(X_train, y_train, compute_sample_weight(class_weight='balanced', y=(y_train))) 
                    y_test_predictions_op=GB_optimised.predict(X_test)
                    st.write("Gini_normalized test  : %.2g" % gini_normalized(y_test, y_test_predictions_op))
                
                elif classifier_names=="RF":
                    rf = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini',
                    max_depth=None, max_features='auto', max_leaf_nodes=None,
                     min_samples_leaf=1,
                     min_samples_split=2, min_weight_fraction_leaf=0.0,
                     n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
                     verbose=0, warm_start=False)
                    rf.fit(X_train, y_train)
                    y_test_predictions=rf.predict(X_test)
                    st.write("Gini_normalized test : %.2g" % gini_normalized(y_test, y_test_predictions))
                    
                else :
        
                    xgb_model = XGBClassifier(
                        objective = 'binary:logistic',
                        colsample_bytree = 0.5,
                        learning_rate = 0.05,
                        max_depth =6 ,
                        min_child_weight = 1,
                        n_estimators = 1000,
                        subsample = 0.7,
                        scale_pos_weight=99)

                    xgb_model.fit(X_train,y_train)
                    y_pred_xgb = xgb_model.predict(X_test)
                    gini_score = gini_normalized(y_test, y_pred_xgb)
                    st.write("Gini_normalized test  : %.2g " %  gini_score)
                    
                    st.write('''
                             Après l’optimisation des hyperparamètres pour chacun des modèles sélectionnés lors de
                             l'étape précédente , on a pu choisir le Gradient Boosting Classifier comme meilleur modèle
                             étant donné le faible écart entre le gini normalized test et le gini normalized train
                             ''')
                
            else :
                load_features_enrichie(feats_enrichies)
                load_target(target)
                st.sidebar.write('Informations sur le dataframe  :') 
                st.sidebar.write( "* Shape:",feats_enrichies.shape)
                st.sidebar.write(" * Nombre de classes :",len(np.unique(target)))    
                if classifier_names=="GB":
                    GB_optimised = GradientBoostingClassifier(n_estimators=150,learning_rate=0.1)
                    GB_optimised.fit(X_train, y_train, compute_sample_weight(class_weight='balanced', y=(y_train)))        
                    y_test_predictions_op=GB_optimised.predict(X_test)
                    st.write("Gini_normalized test  : %.2g" % gini_normalized(y_test, y_test_predictions_op))
                elif  classifier_names=="RF"   :
                    rf = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini',
                             max_depth=None, max_features='auto', max_leaf_nodes=None,
                              min_samples_leaf=1,
                             min_samples_split=2, min_weight_fraction_leaf=0.0,
                             n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
                             verbose=0, warm_start=False)
                    rf.fit(X_train, y_train)
                    y_test_predictions=rf.predict(X_test)
                    st.write("Gini_normalized test : %.2g" % gini_normalized(y_test, y_test_predictions))
                    
                else :
                    model = XGBClassifier(scale_pos_weight=99)
                    model.fit(X_train, y_train)
                    y_pred_test = model.predict(X_test)
                    st.write("Gini_normalized : %.2f" % gini_normalized(y_test, y_pred_test))
                    
                    
                    st.write('''
                             L'ajout des données externes nous a permis d'améliorer le Gini_normalized score
                             pour le modèle Gradient Boosting Classifier qui son  score de test passe de 0.37 à 0.39.
                             
                             ''')
            
                    
    
    else:
        titre=st.checkbox("Matrice de confusion")
        if titre:
            st.markdown(" ### *Matrice de confusion pour Gradient Boosting Classifier* ") 
        
            img1 = plt.imread("streamlit/cm.jpg")
            st.image(img1)
        
        titre2=st.checkbox("Rapport de classification")
        if titre2:
            
            
        
            img2 = plt.imread("streamlit/classification_imbalanced.jpg")
            st.image(img2)
            st.write(''' 
                     * Le tableau précedent montre que le rappel et le f1-score de la classe 1 sont pas 
                      élevés par contre pour la classe 0, ils sont meilleurs.
                     
                     * En outre, la moyenne géométrique est également pas très élevée mais acceptable.
                     
                     * Le modèle est donc plus ou moins acceptable  pour notre problème.
                     
                     ''')
        
        titre3=st.checkbox("Interprétabilité ")
        if titre3:
            st.markdown(" ### Interprétabilité ")
            st.write('''
                 l'interprétabilité est la mesure dans laquelle un être humain peut prédire de manière cohérente le
                 résultat d'un modèle.Elle reflète la logique derrière les décisions ou prédictions issues d’un
                 modèle.
                 
                 Pour notre , cas cette étape n'était pas complexe étant donné que de nombreux modèles
                 proposent de retourner l'importance de chaque variable (feature importance) y compris le
                 Gradient Boosting Classifier . Ci- dessous un graphique qui illustre le score d’importance pour
                 chaque variables explicatives dans notre modèle.
                 
                ''')   
                
            img3 = plt.imread("streamlit/features_Imp.jpg")
            st.image(img3)            
            st.write('''
                     
                     
                     * On voit dans ce graphique l’apparition de la variable ‘Périls’ parmi les variables les plus importantes 
                     dans le modèle ce qui justifie qu’on a réussi  à enrichir notre jeu de données.
                     
                     * On peut voir également que la variable ‘superficief_Géant’ est deux fois plus importante que
                     les variables ’EXPO’ et ‘âge du bâtiment’. 
                     
                     * Pour la ‘ft_21_categ’ et ‘superficie très grand’ elles ont un score d’importance presque égale.
                     
                     ### Conclusion : 
                     
                     * Ces résultats confirment  nos intuitions pour la relation superficie - target comme pour la relation
                     Périls-target.
                     
                     
                     * On peut conclure que plus la superficie d’un bâtiment est importante , plus ça augmente la probabilité
                     d’avoir un sinistre .
                     
                     
                     ==> Ces résultats viennent confirmer nos intuitions et nos hypothèses émises sur l’existence d’une relation positive
                     entre la variable cible et les variables explicatives ‘superficief’ et ‘Périls’.
                             
                     
                    
                     ''')                 
            
            
#partie conclusion
if rad == "Conclusion":
    st.write(
            """ 
            ##
            """)     

    titre=st.checkbox("Limite du projet")
    if titre:
            st.write('''Le jeu de données est insuffisant pour bien prédire notre variable cible - il aurait été  intéressant  d'avoir plus d'informations
                     sur la variable ft-24_categ et ft-21_categ par exemple..''')
            st.write('''L’exploration des données du jeu initial a monopolisé l'intégralitéde l'équipe car
                     il a fallu bien comprendre les données pour mieux les analyser et supprimer les informations redondantes.''')
            st.write(''' Nous avons identifié et travaillé avec plusieurs autres sources de données externes qui n'ont
                     pû être intégré car trop de déperdition de données par rapport aux codes INSEE. ''')

    titre=st.checkbox("Conclusion")
    if titre:
            st.write('''
            - Nous pouvons conclure que plus la superficie d’un bâtiment est importante , plus ça augmente la probabilité
d’avoir un sinistre .(existence d’une relation positive entre
la variable cible et la variable explicative ‘superficief’)''')
            st.write('''- Le choix du gradient boosting classifier comme modèle de
classification qui nous a permis de souligner les variables les plus importantes et qui vient
confirmer nos hypothèses pour la relation superficie - target comme pour la relation
Périls-target..''')
            st.write('''Ce projet pourra être poursuivi dans l’entreprise vu qu’il est
intéressant et très ouvert , étant donné qu’on peut toujours enrichir le jeu de données initial
via les codes Insee avec le fichier des crimes et délits et les fichiers comportant Les critères sociaux.''')


            
            # Info box
            st.sidebar.markdown("### Réalisé par :")
            st.sidebar.info('''
		
        Ghayet El Mouna Ghazouani,[Linkedin](https://www.linkedin.com/in/ghayet-el-mouna-ghazouani-64454b15b/)
        
        Claude Haik,[Linkedin](https://www.linkedin.com/in/claudehaik/)
        
        Edouard Delannoy
        
        Michael Abergel,[Linkedin] (https://fr.linkedin.com/in/michaelabergel)
        ''')
        
            st.sidebar.markdown("### Sous la supervision de: ")
            st.sidebar.info("Thibault VENET de Datascientest.")
            
            st.sidebar.markdown("### Sources: ")
            st.sidebar.write("""
               
                [![](https://img.shields.io/badge/GitHub%20-Claimpy-informational)](https://github.com/gelamick/claim_insurance.git)
                   
                            """)   
        
            
                    
                    
            
            
                
       
                
                
                
                
         
         
         
                  
    
            
    
        

                
                 
        
    
    
