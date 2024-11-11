#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:31:24 2024

@author: jhodi
"""

import numpy as np
import pandas as pd
import os 
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

os.chdir("/home/jhodi/Documents/Rstudio/Python/MLP_sec_str")

from Exctract_features_V2 import *
pwd = "/home/jhodi/Documents/Rstudio/Brute/513_distribute" 

df = create_dataset(pwd) # Depuis le script Exctract_features

# Convertir les étiquettes en valeurs numériques
df['secondary_structure'] = df['secondary_structure'].map({'H': 0, 'E': 1, 'C': 2})

# Séparer les caractéristiques et les étiquettes
X = df[['feature1', 'feature2', 'feature3']]
y = df['secondary_structure']

# Diviser le jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un pipeline avec un standardiseur et un MLP
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalisation des données
    ('mlp', MLPClassifier(hidden_layer_sizes=(3,), max_iter=1000, random_state=42))  # MLP avec 3 neurones en sortie
])

# Entraîner le modèle
pipeline.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = pipeline.predict(X_test)

# Évaluer le modèle
print(classification_report(y_test, y_pred, target_names=['Hélice', 'Feuillet', 'Random Coil']))

    
    