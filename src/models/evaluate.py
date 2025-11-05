"""
Script pour évaluer le modèle entraîné sur les données de test.
Calcule les métriques de performance et sauvegarde les prédictions.
"""

import pandas as pd
import joblib
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def load_model():
    """
    Charge le modèle entraîné depuis trained_model.pkl.
    
    Returns:
        Modèle sklearn entraîné
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, 'models')
    model_path = os.path.join(models_dir, 'trained_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Le fichier {model_path} n'existe pas. "
            "Exécutez d'abord le script train.py pour entraîner le modèle."
        )
    
    print(f"Chargement du modèle depuis {model_path}...")
    model = joblib.load(model_path)
    
    print(f"✓ Modèle chargé: {type(model).__name__}")
    
    return model

def load_test_data():
    """
    Charge les données de test normalisées.
    
    Returns:
        tuple: (X_test, y_test) - Features et variable cible de test
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(project_root, 'data', 'processed_data')
    
    X_test_path = os.path.join(processed_dir, 'X_test_scaled.csv')
    y_test_path = os.path.join(processed_dir, 'y_test.csv')
    
    print(f"\nChargement de X_test_scaled depuis {X_test_path}...")
    X_test = pd.read_csv(X_test_path)
    
    print(f"Chargement de y_test depuis {y_test_path}...")
    y_test = pd.read_csv(y_test_path).iloc[:, 0]  # Prendre la première colonne
    
    # Exclure la colonne 'date' des features pour la prédiction
    if 'date' in X_test.columns:
        X_test_features = X_test.drop(columns=['date'])
        date_column = X_test[['date']].copy()
        print(f"\nColonne 'date' exclue de la prédiction.")
    else:
        X_test_features = X_test
        date_column = None
    
    print(f"Shape X_test_features: {X_test_features.shape}")
    print(f"Shape y_test: {y_test.shape}")
    print(f"Features utilisées: {X_test_features.columns.tolist()}")
    
    return X_test, X_test_features, y_test, date_column

def make_predictions(model, X_test_features):
    """
    Effectue les prédictions sur les données de test.
    
    Args:
        model: Modèle entraîné
        X_test_features: Features de test (sans la colonne date)
    
    Returns:
        array: Prédictions
    """
    print("\n" + "="*70)
    print("Prédictions sur les données de test...")
    print("="*70)
    
    predictions = model.predict(X_test_features)
    
    print(f"✓ {len(predictions)} prédictions générées")
    
    return predictions

def calculate_metrics(y_test, predictions):
    """
    Calcule les métriques d'évaluation du modèle.
    
    Args:
        y_test: Valeurs réelles
        predictions: Prédictions du modèle
    
    Returns:
        dict: Dictionnaire contenant les métriques
    """
    print("\n" + "="*70)
    print("Calcul des métriques d'évaluation...")
    print("="*70)
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'rmse': float(rmse)
    }
    
    print(f"✓ MSE (Mean Squared Error): {mse:.4f}")
    print(f"✓ MAE (Mean Absolute Error): {mae:.4f}")
    print(f"✓ R² (R-squared): {r2:.4f}")
    print(f"✓ RMSE (Root Mean Squared Error): {rmse:.4f}")
    
    return metrics

def save_predictions(predictions):
    """
    Sauvegarde les prédictions dans un fichier csv.
    
    Args:
        predictions: Prédictions du modèle
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(project_root, 'models')
    predictions_path = os.path.join(model_dir, 'prediction.csv')
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Prédictions sauvegardées dans {predictions_path}")

def save_metrics(metrics):
    """
    Sauvegarde les métriques dans metrics/scores.json.
    
    Args:
        metrics: Dictionnaire contenant les métriques
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metrics_dir = os.path.join(project_root, 'metrics')
    
    # Créer le répertoire metrics s'il n'existe pas
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Sauvegarder les métriques
    scores_path = os.path.join(metrics_dir, 'scores.json')
    
    with open(scores_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*70)
    print("Sauvegarde des métriques")
    print("="*70)
    print(f"✓ Métriques sauvegardées dans {scores_path}")

def main():
    """
    Fonction principale qui orchestre le processus d'évaluation.
    """
    print("="*70)
    print("ÉVALUATION DU MODÈLE")
    print("="*70)
    
    # Charger le modèle entraîné
    model = load_model()
    
    # Charger les données de test
    X_test, X_test_features, y_test, date_column = load_test_data()
    
    # Effectuer les prédictions
    predictions = make_predictions(model, X_test_features)
    
    # Calculer les métriques
    metrics = calculate_metrics(y_test, predictions)
    
    # Sauvegarde les prédictions dans un fichier csv
    save_predictions(predictions)
    
    # Sauvegarder les métriques
    save_metrics(metrics)
    
    print("\n" + "="*70)
    print("Évaluation terminée avec succès!")
    print("="*70)

if __name__ == "__main__":
    main()

