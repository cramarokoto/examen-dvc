"""
Script pour normaliser les données d'entraînement et de test.
Les données sont normalisées à l'aide de StandardScaler pour gérer les différentes échelles.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def normalize_data():
    """
    Charge X_train et X_test, les normalise avec StandardScaler,
    et sauvegarde X_train_scaled et X_test_scaled dans data/processed.
    
    La colonne "date" est exclue de la normalisation et conservée telle quelle
    dans les datasets finaux.
    """
    # Définir les chemins
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_data_dir = os.path.join(project_root, 'data', 'processed_data')
    processed_dir = os.path.join(project_root, 'data', 'processed_data')
    
    # Créer le répertoire processed s'il n'existe pas
    os.makedirs(processed_dir, exist_ok=True)
    
    # Chemins des fichiers d'entrée
    X_train_path = os.path.join(processed_data_dir, 'X_train.csv')
    X_test_path = os.path.join(processed_data_dir, 'X_test.csv')
    
    # Charger les données
    print(f"Chargement de X_train depuis {X_train_path}...")
    X_train = pd.read_csv(X_train_path)
    
    print(f"Chargement de X_test depuis {X_test_path}...")
    X_test = pd.read_csv(X_test_path)
    
    print(f"\nShape X_train: {X_train.shape}")
    print(f"Shape X_test: {X_test.shape}")
    print(f"Features: {X_train.columns.tolist()}")
    
    # Séparer la colonne "date" des autres features à normaliser
    # Extraire les colonnes date
    X_train_date = X_train[['date']].copy()
    X_test_date = X_test[['date']].copy()
    
    # Extraire les features à normaliser (toutes sauf date)
    features_to_scale = [col for col in X_train.columns if col != 'date']
    X_train_features = X_train[features_to_scale].copy()
    X_test_features = X_test[features_to_scale].copy()
    
    print("\nColonne exclue de la normalisation: 'date'")
    print(f"Features à normaliser ({len(features_to_scale)}): {features_to_scale}")

    
    # Initialiser le StandardScaler
    scaler = StandardScaler()
    
    # Ajuster le scaler sur les données d'entraînement et transformer X_train
    print("\nAjustement du scaler sur X_train (features uniquement)...")
    X_train_features_scaled = scaler.fit_transform(X_train_features)
    X_train_features_scaled = pd.DataFrame(
        X_train_features_scaled, 
        columns=features_to_scale,
        index=X_train_features.index
    )
    
    # Transformer X_test avec le scaler ajusté sur X_train
    print("Transformation de X_test avec le scaler ajusté sur X_train...")
    X_test_features_scaled = scaler.transform(X_test_features)
    X_test_features_scaled = pd.DataFrame(
        X_test_features_scaled,
        columns=features_to_scale,
        index=X_test_features.index
    )
    
    # Réassembler les datasets avec la colonne date (si elle existe)
    X_train_scaled = pd.concat([X_train_date, X_train_features_scaled], axis=1)
    X_test_scaled = pd.concat([X_test_date, X_test_features_scaled], axis=1)
    
    print(f"\nNormalisation terminée!")
    print(f"Shape X_train_scaled: {X_train_scaled.shape}")
    print(f"Shape X_test_scaled: {X_test_scaled.shape}")
    
    # Statistiques avant normalisation (features uniquement, sans date)
    print("\n=== Statistiques avant normalisation (features uniquement) ===")
    print("X_train_features - Moyennes:")
    print(X_train_features.mean())
    print("\nX_train_features - Écart-types:")
    print(X_train_features.std())
    
    # Statistiques après normalisation (features uniquement, sans date)
    print("\n=== Statistiques après normalisation (features uniquement) ===")
    print("X_train_features_scaled - Moyennes (devrait être ~0):")
    print(X_train_features_scaled.mean().round(6))
    print("\nX_train_features_scaled - Écart-types (devrait être ~1):")
    print(X_train_features_scaled.std().round(6))
    
    if X_train_date is not None:
        print("\nColonne 'date' conservée sans normalisation")
        print(f"Exemple de valeurs date: {X_train_date['date'].iloc[:3].tolist()}")
    
    # Sauvegarder les datasets normalisés
    X_train_scaled_path = os.path.join(processed_dir, 'X_train_scaled.csv')
    X_test_scaled_path = os.path.join(processed_dir, 'X_test_scaled.csv')
    
    print(f"\nSauvegarde des datasets normalisés dans {processed_dir}...")
    X_train_scaled.to_csv(X_train_scaled_path, index=False)
    X_test_scaled.to_csv(X_test_scaled_path, index=False)
    
    # Sauvegarder le scaler pour utilisation future (inference)
    scaler_path = os.path.join(processed_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    print("✓ X_train_scaled.csv sauvegardé")
    print("✓ X_test_scaled.csv sauvegardé")
    print(f"✓ scaler.pkl sauvegardé (pour utilisation future)")
    
    print(f"\nNormalisation terminée avec succès!")

if __name__ == "__main__":
    normalize_data()

