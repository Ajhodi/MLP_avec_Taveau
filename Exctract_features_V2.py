#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:29:42 2024

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

def aa_ohe(aa):
    # Initialiser le dictionnaire pour l'encodage one-hot
    Residues = "ARNDCQEGHILKMFPSTWYV"
    residue_ohe = {aa: 0 for aa in Residues}
    # Remplir le dictionnaire en fonction de la séquence
    if aa in residue_ohe:
        residue_ohe[aa] = 1
    else:
        print(f"Amino acid '{aa}' not recognized and will be ignored.")
    # Retourner un DataFrame avec une seule ligne
    return list(residue_ohe.values())

#test = aa_ohe('A')

def seq_ohe(seq):
    arr = []
    for aa in seq:
        arr.append(aa_ohe(aa))
    return np.array(arr)

#test = seq_ohe('TDPIADMLTAIRN')    

def RES_ohe(column):
    # Créer une liste pour stocker les résultats
    results = []
    
    # Appliquer seq_ohe à chaque séquence et aplatir le résultat
    for seq in column:
        ohe_result = seq_ohe(seq).flatten()  # Aplatir le tableau 2D
        results.append(ohe_result)  # Ajouter le résultat à la liste
    # Créer un DataFrame à partir de la liste de résultats
    values_df = pd.DataFrame(results)
    
    return values_df

#test = RES_ohe(df['RES'])

def clear(df, column):
    # Remplacer les listes par l'élément du milieu
    df[column] = df[column].apply(lambda x: x[len(x) // 2] if isinstance(x, list) and len(x) > 0 else x)
    return df

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
    
    # RES OneHot Encoding 
    RES_encoded = RES_ohe(data['RES'])
    
    # Reinitialisation des index
    data = data.reset_index(drop=True)
    RES_encoded = RES_encoded.reset_index(drop=True)
    
    df = pd.concat([data.drop(columns = 'RES'), RES_encoded], axis = 1).set_index(data['RES'])
    
    return resample(df)
    

df = create_dataset(pwd)

