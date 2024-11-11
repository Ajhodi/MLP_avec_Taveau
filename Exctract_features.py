#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:54:08 2024

@author: jhodi
"""

import os
import pandas as pd

# One Hot encoding

Residues = "ARNDCQEGHILKMFPSTWYV"

sequence = "MTTKEKDEFNIGS"



def one_hot(seq):
    residue_to_index = {residue: index for index, residue in enumerate(Residues)}
    seq_df = []
    count = {aa: 0 for aa in Residues}
    for aa in seq:
        encoded = [0]*len(Residues)
        if aa in Residues:
            count[aa] += 1
            encoded[residue_to_index[aa]] = 1
            seq_df.append(encoded)
    return seq_df

test = one_hot("MTTKEKDEFNIGS")

# Parceur

pwd = "/home/jhodi/Documents/Rstudio/Brute/513_distribute"    

def extract_first_(line, length):
    # Enlève les virgules et retourne les premiers caractères jusqu'à la longueur spécifiée
    return line.replace(',', '')[:length]

def DSS_translate(str_):
    # Remplace 'H' et 'E' par elles-mêmes, toutes les autres lettres par 'C'
    return ''.join('C' if letter not in 'HE' else letter for letter in str_)

length = 13

def read_files(filepath):
    RES = []
    DSSP = []
    DSSPACC = []
    STRIDE = []
    
    with open(filepath, 'r') as f1:
        for line in f1:
            if line.startswith('RES:'):
                RES.append(extract_first_(line.split(":")[1], length))
            elif line.startswith("DSSP:"):
                DSSP.append(extract_first_(line.split(":")[1], length))
            elif line.startswith("DSSPACC:"):
                DSSPACC.append(extract_first_(line.split(":")[1], length))
            elif line.startswith("STRIDE:"):
                STRIDE.append(extract_first_(line.split(":")[1], length))

    # Vérification si les listes ne sont pas vides avant d'accéder à l'index [0]
    if RES and DSSP and DSSPACC and STRIDE:
        return {
            'RES': one_hot(RES[0]),  # Assurez-vous que one_hot est défini
            'DSSP': DSS_translate(DSSP[0]),
            'DSSPACC': DSSPACC[0],
            'STRIDE': STRIDE[0]
        }
    else:
        return None  # Retourne None si les listes sont vides

def expand_column(dataset, column):
    new_df = pd.DataFrame(dataset[column].tolist(), index=dataset.index)  # Crée un DataFrame à partir de la colonne 'RES'
    new_df.columns = [f'RES_{i}' for i in range(new_df.shape[1])]  # Renomme les colonnes
    dataset = pd.concat([dataset.drop(columns=[column]), new_df], axis=1)  # Concatène les nouveaux colonnes avec le DataFrame d'origine
    return dataset

def create_dataset(pwd):
    data = []  # Liste pour stocker les données de chaque fichier
    filenames = []  # Liste pour stocker les noms de fichiers
    for file in os.listdir(pwd):
        filename = file.split('.')[0]
        try:
            file_data = read_files(os.path.join(pwd, file))
            if file_data is not None:  # Vérifie si les données du fichier ne sont pas None
                data.append(file_data)  # Ajoute les données à la liste
                filenames.append(filename)  # Ajoute le nom du fichier à la liste
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {file}: {e}")

    # Convertit la liste de dictionnaires en DataFrame
    dataset = pd.DataFrame(data, index=filenames)  # Utilise les noms de fichiers comme index
    dataset = expand_column(dataset, 'RES')
    return dataset


