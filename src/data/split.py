"""
Script pour diviser les données en ensembles d'entraînement et de test.
La variable cible est silica_concentrate (dernière colonne du dataset).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

# Ajouter le répertoire racine au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def split_data():
    """
    Charge les données brutes, les divise en ensembles d'entraînement et de test,
    et sauvegarde les 4 datasets (X_train, X_test, y_train, y_test) dans data/processed.
    """
    # Définir les chemins
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_path = os.path.join(project_root, 'data', 'raw_data', 'raw.csv')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    
    # Créer le répertoire processed s'il n'existe pas
    os.makedirs(processed_dir, exist_ok=True)
    
    # Charger les données
    print(f"Chargement des données depuis {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    
    print(f"Shape du dataset: {df.shape}")
    print(f"Colonnes: {df.columns.tolist()}")
    
    # Séparer les features (X) de la variable cible (y)
    # La variable cible est dans la dernière colonne
    X = df.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
    y = df.iloc[:, -1]   # Dernière colonne (silica_concentrate)
    
    print(f"\nNombre de features: {X.shape[1]}")
    print(f"Variable cible: {y.name}")
    
    # Diviser en ensembles d'entraînement et de test (80/20 par défaut)
    # Utiliser un random_state pour la reproductibilité
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"\nShape X_train: {X_train.shape}")
    print(f"Shape X_test: {X_test.shape}")
    print(f"Shape y_train: {y_train.shape}")
    print(f"Shape y_test: {y_test.shape}")
    
    # Sauvegarder les datasets
    X_train_path = os.path.join(processed_dir, 'X_train.csv')
    X_test_path = os.path.join(processed_dir, 'X_test.csv')
    y_train_path = os.path.join(processed_dir, 'y_train.csv')
    y_test_path = os.path.join(processed_dir, 'y_test.csv')
    
    print(f"\nSauvegarde des datasets dans {processed_dir}...")
    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    
    print("✓ X_train.csv sauvegardé")
    print("✓ X_test.csv sauvegardé")
    print("✓ y_train.csv sauvegardé")
    print("✓ y_test.csv sauvegardé")
    
    print(f"\nSplit terminé avec succès!")

if __name__ == "__main__":
    split_data()

