"""
Script pour entraîner le meilleur modèle identifié par GridSearch.
Charge les meilleurs paramètres depuis best_params.pkl et entraîne le modèle
avec ces paramètres sur les données d'entraînement.
"""

import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

def load_best_params():
    """
    Charge les meilleurs paramètres depuis best_params.pkl.
    
    Returns:
        dict: Dictionnaire contenant model_name, best_params, et les scores
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, 'models')
    best_params_path = os.path.join(models_dir, 'best_params.pkl')
    
    if not os.path.exists(best_params_path):
        raise FileNotFoundError(
            f"Le fichier {best_params_path} n'existe pas. "
            "Exécutez d'abord le script gridsearch.py pour générer les meilleurs paramètres."
        )
    
    print(f"Chargement des meilleurs paramètres depuis {best_params_path}...")
    best_params = joblib.load(best_params_path)
    
    print(f"\nModèle identifié: {best_params['model_name'].upper()}")
    print(f"Score R2 (CV): {best_params['best_score_r2']:.4f}")
    print(f"Paramètres: {best_params['best_params']}")
    
    return best_params

def create_model(model_name, params):
    """
    Crée une instance du modèle avec les paramètres spécifiés.
    
    Args:
        model_name: Nom du modèle ('random_forest', 'gradient_boosting', 'ridge')
        params: Dictionnaire des paramètres du modèle
    
    Returns:
        Modèle sklearn instancié
    """
    # Ajouter random_state pour la reproductibilité
    if 'random_state' not in params:
        params['random_state'] = 42
    
    if model_name == 'random_forest':
        # n_jobs=-1 pour utiliser tous les processeurs
        if 'n_jobs' not in params:
            params['n_jobs'] = -1
        model = RandomForestRegressor(**params)
    elif model_name == 'gradient_boosting':
        model = GradientBoostingRegressor(**params)
    elif model_name == 'ridge':
        model = Ridge(**params)
    else:
        raise ValueError(
            f"Modèle non reconnu: {model_name}. "
            "Modèles supportés: 'random_forest', 'gradient_boosting', 'ridge'"
        )
    
    return model

def load_training_data():
    """
    Charge les données d'entraînement normalisées.
    
    Returns:
        tuple: (X_train, y_train) - Features et variable cible
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(project_root, 'data', 'processed_data')
    
    X_train_path = os.path.join(processed_dir, 'X_train_scaled.csv')
    y_train_path = os.path.join(processed_dir, 'y_train.csv')
    
    print(f"\nChargement de X_train_scaled depuis {X_train_path}...")
    X_train = pd.read_csv(X_train_path)
    
    print(f"Chargement de y_train depuis {y_train_path}...")
    y_train = pd.read_csv(y_train_path).iloc[:, 0]  # Prendre la première colonne
    
    # Exclure la colonne 'date' des features pour l'entraînement
    if 'date' in X_train.columns:
        X_train_features = X_train.drop(columns=['date'])
        print(f"\nColonne 'date' exclue de l'entraînement.")
    else:
        X_train_features = X_train
    
    print(f"Shape X_train_features: {X_train_features.shape}")
    print(f"Shape y_train: {y_train.shape}")
    print(f"Features utilisées: {X_train_features.columns.tolist()}")
    
    return X_train_features, y_train

def train_model(model, X_train, y_train):
    """
    Entraîne le modèle sur les données d'entraînement.
    
    Args:
        model: Modèle sklearn à entraîner
        X_train: Features d'entraînement
        y_train: Variable cible d'entraînement
    
    Returns:
        Modèle entraîné
    """
    print("\n" + "="*70)
    print("Entraînement du modèle...")
    print("="*70)
    
    model.fit(X_train, y_train)
    
    print("✓ Modèle entraîné avec succès!")
    
    return model

def save_model(model, model_name):
    """
    Sauvegarde le modèle entraîné dans le dossier models/.
    
    Args:
        model: Modèle entraîné à sauvegarder
        model_name: Nom du modèle pour le nom de fichier
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, 'models')
    
    # Créer le répertoire models s'il n'existe pas
    os.makedirs(models_dir, exist_ok=True)
    
    # Nom du fichier
    filename = 'trained_model.pkl'
    filepath = os.path.join(models_dir, filename)
    
    # Sauvegarder le modèle
    joblib.dump(model, filepath)
    
    print(f"\n✓ Modèle sauvegardé dans {filepath}")
    print(f"  - Type: {model_name.upper()}")
    print(f"  - Fichier: {filename}")

def main():
    """
    Fonction principale qui orchestre le processus d'entraînement.
    """
    print("="*70)
    print("ENTRAÎNEMENT DU MODÈLE")
    print("="*70)
    
    # Charger les meilleurs paramètres
    best_params = load_best_params()
    
    # Créer le modèle avec les meilleurs paramètres
    model = create_model(best_params['model_name'], best_params['best_params'])
    
    # Charger les données d'entraînement
    X_train, y_train = load_training_data()
    
    # Entraîner le modèle
    trained_model = train_model(model, X_train, y_train)
    
    # Sauvegarder le modèle entraîné
    save_model(trained_model, best_params['model_name'])
    
    print("\n" + "="*70)
    print("Entraînement terminé avec succès!")
    print("="*70)

if __name__ == "__main__":
    main()

