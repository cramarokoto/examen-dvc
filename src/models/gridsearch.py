"""
Script pour effectuer une GridSearch sur plusieurs modèles de régression
et trouver les meilleurs paramètres pour la prédiction de silica_concentrate.
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib
import os
import json

def load_data():
    """
    Charge les données d'entraînement normalisées et la variable cible.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(project_root, 'data', 'processed_data')
    
    # Charger les données
    X_train_path = os.path.join(processed_dir, 'X_train_scaled.csv')
    y_train_path = os.path.join(processed_dir, 'y_train.csv')
    
    print(f"Chargement de X_train_scaled depuis {X_train_path}...")
    X_train = pd.read_csv(X_train_path)
    
    print(f"Chargement de y_train depuis {y_train_path}...")
    y_train = pd.read_csv(y_train_path).iloc[:, 0]  # Prendre la première colonne
    
    # Exclure la colonne 'date' des features pour l'entraînement
    if 'date' in X_train.columns:
        X_train_features = X_train.drop(columns=['date'])
        print(f"\nColonne 'date' exclue de l'entraînement.")
    else:
        X_train_features = X_train
    
    print(f"\nShape X_train_features: {X_train_features.shape}")
    print(f"Shape y_train: {y_train.shape}")
    print(f"Features utilisées: {X_train_features.columns.tolist()}")
    
    return X_train_features, y_train

def grid_search_models(X_train, y_train):
    """
    Effectue une GridSearch sur 3 modèles de régression:
    1. Random Forest Regressor
    2. Gradient Boosting Regressor
    3. Ridge Regression
    
    Retourne un dictionnaire avec les meilleurs paramètres pour chaque modèle.
    """
    # Définir les métriques de scoring
    scoring = {
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }
    
    # Utiliser R2 comme métrique principale pour le scoring
    refit_metric = 'r2'
    
    results = {}
    
    # ============================================
    # 1. Random Forest Regressor
    # ============================================
    print("\n" + "="*70)
    print("1. GridSearch pour Random Forest Regressor")
    print("="*70)
    
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    rf_grid = GridSearchCV(
        estimator=rf_model,
        param_grid=rf_param_grid,
        cv=5,
        scoring=scoring,
        refit=refit_metric,
        n_jobs=-1,
        verbose=1
    )
    
    rf_grid.fit(X_train, y_train)
    
    print(f"\nMeilleurs paramètres pour Random Forest:")
    print(json.dumps(rf_grid.best_params_, indent=2))
    print(f"\nMeilleur score R2: {rf_grid.best_score_:.4f}")
    print(f"Meilleur score MSE: {-rf_grid.cv_results_['mean_test_neg_mse'][rf_grid.best_index_]:.4f}")
    print(f"Meilleur score MAE: {-rf_grid.cv_results_['mean_test_neg_mae'][rf_grid.best_index_]:.4f}")
    
    results['random_forest'] = {
        'best_params': rf_grid.best_params_,
        'best_score_r2': float(rf_grid.best_score_),
        'best_score_mse': float(-rf_grid.cv_results_['mean_test_neg_mse'][rf_grid.best_index_]),
        'best_score_mae': float(-rf_grid.cv_results_['mean_test_neg_mae'][rf_grid.best_index_]),
        'best_estimator': rf_grid.best_estimator_
    }
    
    # ============================================
    # 2. Gradient Boosting Regressor
    # ============================================
    print("\n" + "="*70)
    print("2. GridSearch pour Gradient Boosting Regressor")
    print("="*70)
    
    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 1.0]
    }
    
    gb_model = GradientBoostingRegressor(random_state=42)
    
    gb_grid = GridSearchCV(
        estimator=gb_model,
        param_grid=gb_param_grid,
        cv=5,
        scoring=scoring,
        refit=refit_metric,
        n_jobs=-1,
        verbose=1
    )
    
    gb_grid.fit(X_train, y_train)
    
    print(f"\nMeilleurs paramètres pour Gradient Boosting:")
    print(json.dumps(gb_grid.best_params_, indent=2))
    print(f"\nMeilleur score R2: {gb_grid.best_score_:.4f}")
    print(f"Meilleur score MSE: {-gb_grid.cv_results_['mean_test_neg_mse'][gb_grid.best_index_]:.4f}")
    print(f"Meilleur score MAE: {-gb_grid.cv_results_['mean_test_neg_mae'][gb_grid.best_index_]:.4f}")
    
    results['gradient_boosting'] = {
        'best_params': gb_grid.best_params_,
        'best_score_r2': float(gb_grid.best_score_),
        'best_score_mse': float(-gb_grid.cv_results_['mean_test_neg_mse'][gb_grid.best_index_]),
        'best_score_mae': float(-gb_grid.cv_results_['mean_test_neg_mae'][gb_grid.best_index_]),
        'best_estimator': gb_grid.best_estimator_
    }
    
    # ============================================
    # 3. Ridge Regression
    # ============================================
    print("\n" + "="*70)
    print("3. GridSearch pour Ridge Regression")
    print("="*70)
    
    ridge_param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
    
    ridge_model = Ridge(random_state=42)
    
    ridge_grid = GridSearchCV(
        estimator=ridge_model,
        param_grid=ridge_param_grid,
        cv=5,
        scoring=scoring,
        refit=refit_metric,
        n_jobs=-1,
        verbose=1
    )
    
    ridge_grid.fit(X_train, y_train)
    
    print(f"\nMeilleurs paramètres pour Ridge Regression:")
    print(json.dumps(ridge_grid.best_params_, indent=2))
    print(f"\nMeilleur score R2: {ridge_grid.best_score_:.4f}")
    print(f"Meilleur score MSE: {-ridge_grid.cv_results_['mean_test_neg_mse'][ridge_grid.best_index_]:.4f}")
    print(f"Meilleur score MAE: {-ridge_grid.cv_results_['mean_test_neg_mae'][ridge_grid.best_index_]:.4f}")
    
    results['ridge'] = {
        'best_params': ridge_grid.best_params_,
        'best_score_r2': float(ridge_grid.best_score_),
        'best_score_mse': float(-ridge_grid.cv_results_['mean_test_neg_mse'][ridge_grid.best_index_]),
        'best_score_mae': float(-ridge_grid.cv_results_['mean_test_neg_mae'][ridge_grid.best_index_]),
        'best_estimator': ridge_grid.best_estimator_
    }
    
    return results

def save_best_parameters(results):
    """
    Sauvegarde uniquement les meilleurs paramètres du meilleur modèle dans un fichier .pkl
    dans le dossier models/.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, 'models')
    
    # Créer le répertoire models s'il n'existe pas
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Sauvegarde des meilleurs paramètres")
    print("="*70)
    
    # Trouver le meilleur modèle basé sur le score R2
    best_model_name, best_model_results = max(results.items(), key=lambda x: x[1]['best_score_r2'])
    
    # Sauvegarder uniquement les paramètres du meilleur modèle
    params_to_save = {
        'model_name': best_model_name,
        'best_params': best_model_results['best_params'],
        'best_score_r2': best_model_results['best_score_r2'],
        'best_score_mse': best_model_results['best_score_mse'],
        'best_score_mae': best_model_results['best_score_mae']
    }
    
    # Nom du fichier
    filename = 'best_params.pkl'
    filepath = os.path.join(models_dir, filename)
    
    # Sauvegarder
    joblib.dump(params_to_save, filepath)
    
    print(f"\n✓ Meilleur modèle: {best_model_name.upper()}")
    print(f"✓ {filename} sauvegardé dans {models_dir}")
    print(f"  - R2 Score: {params_to_save['best_score_r2']:.4f}")
    print(f"  - MSE: {params_to_save['best_score_mse']:.4f}")
    print(f"  - MAE: {params_to_save['best_score_mae']:.4f}")
    print(f"\n{'='*70}")

def main():
    """
    Fonction principale qui orchestre le processus de GridSearch.
    """
    print("="*70)
    print("GRIDSEARCH POUR MODÈLES DE RÉGRESSION")
    print("="*70)
    
    # Charger les données
    X_train, y_train = load_data()
    
    # Effectuer la GridSearch
    results = grid_search_models(X_train, y_train)
    
    # Sauvegarder les meilleurs paramètres
    save_best_parameters(results)
    
    print("\n" + "="*70)
    print("GridSearch terminé avec succès!")
    print("="*70)

if __name__ == "__main__":
    main()

