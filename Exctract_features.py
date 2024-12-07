#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:41:48 2024

@author: jhodi
"""

import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import time


# Testing parameters
length = 13

pwd = "513_distribute" 

def extract_first_(line, length, from_):
    # Enlève les virgules, les retourns à la lignes et retourne les premiers caractères jusqu'à la longueur spécifiée
    r = line.replace(",", "")[from_ : from_ + length]
    if "\n" in r:
        return r.replace("\n", "")
    else:
        return r

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
    # else:
    #     print(f"Amino acid '{aa}' not recognized and will be ignored.")
    # Retourner un DataFrame avec une seule ligne
    return list(residue_ohe.values())

# test = aa_ohe('X')

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
    for seq in tqdm(column, desc="Processing files"):
        ohe_result = seq_ohe(seq).flatten()  # Aplatir le tableau 2D
        results.append(ohe_result)  # Ajouter le résultat à la liste
    # Créer un DataFrame à partir de la liste de résultats
    values_df = pd.DataFrame(results)
    
    return values_df

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
    for seq in tqdm(col, desc="Processing files"):
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

    # Afficher un message d'attente
    print("Processing... Please wait.")
    
    # Effectuer le resampling
    X_resampled, y_resampled = sm.fit_resample(X, y)

    # Recréer le dataframe
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df['DSSP'] = y_resampled

    # Réinitialiser l'index
    resampled_df.reset_index(drop=True, inplace=True)

    print("Resampling complete!")
    return resampled_df


def create_dataset(pwd, method = None, rsp = False):
    """
    method = 
        ohe : (onehote enconding) 
        freq : (frequence encoding)
        None : (unmodified sequence)
    rsp = Respampling via SMOTE 
        True : performe a resampling to balance classes 
        False : classes distribution remains the same 
    """
    
    RES = []
    DSSP = []
    print("Precessing files ...")
    for file in tqdm(os.listdir(pwd), desc="Processing files"):
        try:
            file_data = read_files(os.path.join(pwd, file))
            if file_data is not None:  # Vérifie si les données du fichier ne sont pas None
                for item in file_data['RES']:
                    RES.append(item)  
                for item in file_data['DSSP']:
                    DSSP.append(item)  
                
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {file}: {e}")

    print("Encoding ...")    
    # Convertit la liste de dictionnaires en DataFrame
    data = pd.DataFrame({
        'DSSP': DSSP,
        'RES': RES
        })
    
    # Garder que les séquence composant une seule et unique structure 
    data['DSSP'] = data['DSSP'].apply(midle)
    
    # DSSP OneHot Encoding
    data['DSSP'] = data['DSSP'].apply(DSSP_ohe)
    
    if method == 'ohe':
        # RES OneHot Encoding 
        print("OneHot Encoding ...")
        RES_encoded = RES_ohe(data['RES'])
    elif method == 'freq':
        # RES frequence encoding 
        print("Frequences calculation ...")
        RES_encoded = freq_for_column(data['RES'])
    else:
        # return resample(data) # retourner la séquence non modifiée ## La fonction resample ne marche pas pour les tableaux de str
        # Reinitialisation des index
        data = data.reset_index(drop=True)
        return data # Retourne un df tel quel sans re
    
    # Reinitialisation des index
    data = data.reset_index(drop=True)
    RES_encoded = RES_encoded.reset_index(drop=True)
    
    df = pd.concat([data.drop(columns = 'RES'), RES_encoded], axis = 1).set_index(data['RES'])
        
    if rsp == True:
        print("Resampling ...")
        return resample(df)
    else:
        return df

# # TEST ########################################################################

# # Charger les data avec du onehot encoding
# df1 = create_dataset(pwd, 'ohe', False) # False pour ne pas faire de resampling
# # Charger les data avec la fréquence d'aa
# df2 = create_dataset(pwd, 'freq', False) # False pour ne pas faire de resampling
# # Charger les data sans encodage
# df3 = create_dataset(pwd, None)

# # Combinener les df
# combine_df = pd.concat([df1, df2], axis = 1)
