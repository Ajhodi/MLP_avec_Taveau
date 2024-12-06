#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:08:06 2024

@author: jhodi
"""

import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

length = 13

pwd = "513_distribute" 

def extract_first_(line, length, from_):
    # Enlève les virgules et retourne les premiers caractères jusqu'à la longueur spécifiée
    return line.replace(',', '')[from_ : from_ + length]

def DSS_translate(str_):
    # Remplace 'H' et 'E' par elles-mêmes, toutes les autres lettres par 'C'
    return ''.join('C' if letter not in 'HE' else letter for letter in str_)

length = 13

def read_files(filepath):
    RES = []
    DSSP = []
    
    with open(filepath, 'r') as f1:
        for line in f1:
            if line.startswith('RES:'):
                for i in range(len(line)):
                    res = extract_first_(line.split(":")[1], length, i)
                    if len(res) >= length:
                        RES.append(res)
            elif line.startswith("DSSP:"):
                for i in range(len(line)):
                    dssp = DSS_translate(extract_first_(line.split(":")[1], length, i))
                    if len(dssp) >= length:
                        DSSP.append(dssp)

    # Vérification si les listes ne sont pas vides avant d'accéder à l'index [0]
    if RES and DSSP:
        return {
            'RES': (RES), 
            'DSSP': DSSP
        }
    else:
        return None  # Retourne None si les listes sont vides

# test = read_files("/net/cremi/javizara/Downloads/PhilBi/513_distribute/1adeb-2-AUTO.1.all")

# One Hot encoding
def DSSP_ohe(str_):
    if str_ in 'HG':
        return 0
    elif str_ in 'EB':
        return 1
    else:
        return 2


def _freq(seq):
    Residues = "ARNDCQEGHILKMFPSTWYV"
    freq = np.zeros(len(Residues))  # Tableau pour stocker les fréquences

    # Compter les occurrences de chaque résidu
    for residue in seq:
        if residue in Residues:
            index = Residues.index(residue)
            freq[index] += 1

    # Calculer les fréquences relatives
    freq_relative = freq / len(seq)  # Diviser par la longueur de la séquence
    return freq_relative

# Test de la fonction
# test = _freq('TDPIADMLTAIRN')


def freq_for_column(col):
    results = []
    for seq in col:
        results.append(_freq(seq))
    return pd.DataFrame(results)

def midle(dssp):
    for seq in dssp:
        l = seq[len(seq)//2]
        return l
    
def resample(df):
    # Séparer les labels et les features
    X, y = df.drop(columns='DSSP'), df['DSSP']

    # Appliquer le resampling 
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    # Recréer le dataframe
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df['DSSP'] = y_resampled

    # Réinitialiser l'index
    resampled_df.reset_index(drop=True, inplace=True)

    return resampled_df



def create_dataset(pwd):
    RES = []
    DSSP = []
    for file in os.listdir(pwd):
        try:
            file_data = read_files(os.path.join(pwd, file))
            if file_data is not None:  # Vérifie si les données du fichier ne sont pas None
                for item in file_data['RES']:
                    RES.append(item)  
                for item in file_data['DSSP']:
                    DSSP.append(item)  
                
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {file}: {e}")

    # Convertit la liste de dictionnaires en DataFrame
    data = pd.DataFrame({
        'DSSP': DSSP,
        'RES': RES
        })
    
    # Garder que les séquence composant une seule et unique structure 
    data['DSSP'] = data['DSSP'].apply(midle)
    
    # DSSP OneHot Encoding
    data['DSSP'] = data['DSSP'].apply(DSSP_ohe)

    # RES frequence encoding 
    RES_encoded = freq_for_column(data['RES'])
    
    # Reinitialisation des index
    data = data.reset_index(drop=True)
    RES_encoded = RES_encoded.reset_index(drop=True)
    
    df = pd.concat([data.drop(columns = 'RES'), RES_encoded], axis = 1).set_index(data['RES'])
    
    return resample(df)
    



df2 = create_dataset(pwd)
