import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_model():
    processed_path = "data/processed_matches.csv"
    model_dir = "data"
    
    if not os.path.exists(processed_path):
        print(f"Error: No se encontró el archivo {processed_path}. Por favor ejecuta feature_engineering.py primero.")
        return
        
    df = pd.read_csv(processed_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Selección de características (Features) y Target
    feature_cols = [
        'elo_home', 'elo_away', 'elo_diff',
        'form_goals_scored_home', 'form_goals_conceded_home', 'form_win_rate_home',
        'form_goals_scored_away', 'form_goals_conceded_away', 'form_win_rate_away',
        'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_draw_rate',
        'market_value_home', 'market_value_away', 'market_value_diff',
        'neutral'
    ]
    
    X = df[feature_cols]
    y = df['result']
    dates = df['date']
    is_wc = df['is_world_cup']
    
    # 2. División temporal de los datos
    # Entrenamiento: hasta finales de 2017
    train_mask = dates < '2018-01-01'
    # Validación: todo 2018 (incluye Rusia 2018)
    val_mask = (dates >= '2018-01-01') & (dates < '2019-01-01')
    # Prueba: de 2019 en adelante (incluye Qatar 2022)
    test_mask = dates >= '2019-01-01'
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Guardar máscaras de copa del mundo para evaluación específica
    val_wc_mask = val_mask & (is_wc == 1)
    test_wc_mask = test_mask & (is_wc == 1)
    
    X_val_wc, y_val_wc = X[val_wc_mask], y[val_wc_mask]
    X_test_wc, y_test_wc = X[test_wc_mask], y[test_wc_mask]
    
    print(f"Set de Entrenamiento: {X_train.shape[0]} partidos")
    print(f"Set de Validación (2018): {X_val.shape[0]} partidos (World Cup: {X_val_wc.shape[0]} partidos)")
    print(f"Set de Pruebas (2019+): {X_test.shape[0]} partidos (World Cup: {X_test_wc.shape[0]} partidos)")
    
    # 3. Normalización de variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_test_wc_scaled = scaler.transform(X_test_wc)
    
    # Guardar el scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler guardado en {scaler_path}")
    
    # 4. Diseño y Configuración de la Red Neuronal (MLPClassifier)
    # Aumentamos la profundidad de la red a (128, 64, 32, 16)
    # Ajustamos alpha=0.005 (mayor regularización para evitar sobreajuste con red más grande)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.005,
        batch_size=64,
        learning_rate_init=0.0005,
        early_stopping=True,
        validation_fraction=0.1,  # 10% del set de entrenamiento reservado para early stopping
        n_iter_no_change=12,      # paciencia ligeramente mayor para detenerse
        random_state=42,
        verbose=True
    )
    
    # 5. Entrenamiento del modelo
    print("\nEntrenando la red neuronal (MLPClassifier)...")
    model.fit(X_train_scaled, y_train)
    
    # Guardar modelo
    model_path = os.path.join(model_dir, "world_cup_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModelo guardado en {model_path}")
    
    # 6. Graficar Curvas de Aprendizaje (Pérdida y Precisión de Validación)
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(12, 4))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(model.loss_curve_, label='Pérdida Entrenamiento')
        plt.title('Curva de Pérdida (Loss Curve)')
        plt.xlabel('Iteraciones')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True)
        
        # Validation accuracy (si early stopping está activo)
        if hasattr(model, 'validation_scores_'):
            plt.subplot(1, 2, 2)
            plt.plot(model.validation_scores_, label='Precisión Validación (10% Train)', color='orange')
            plt.title('Precisión de Validación')
            plt.xlabel('Iteraciones')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
        curves_path = os.path.join(model_dir, "learning_curves.png")
        plt.tight_layout()
        plt.savefig(curves_path)
        plt.close()
        print(f"Gráficas de aprendizaje guardadas en {curves_path}")
    
    # 7. Evaluación y Comparación contra Línea Base (Basado en ELO)
    def elo_baseline_predict(x_df):
        preds = []
        for _, row in x_df.iterrows():
            diff = row['elo_diff']
            if row['neutral'] == 0:
                diff += 100.0  # Ventaja de localía
            
            if diff > 100.0:
                preds.append(0)  # Local
            elif diff < -100.0:
                preds.append(2)  # Visitante
            else:
                preds.append(1)  # Empate
        return np.array(preds)
        
    print("\n" + "="*50)
    print(" EVALUACIÓN DEL MODELO ")
    print("="*50)
    
    # Predicciones
    y_pred = model.predict(X_test_scaled)
    y_pred_probs = model.predict_proba(X_test_scaled)
    
    print("\n--- Resultados generales en el Set de Prueba (2019-2026) ---")
    acc_t = accuracy_score(y_test, y_pred)
    print(f"Precisión (Accuracy) en Test General: {acc_t:.2%}")
    
    # Línea Base ELO en todo el Test set
    elo_preds_all = elo_baseline_predict(X_test)
    elo_acc_all = np.mean(elo_preds_all == y_test)
    print(f"Precisión de Línea Base ELO (Test General): {elo_acc_all:.2%}")
    
    print("\n--- Resultados específicos de la COPA MUNDIAL (Qatar 2022, etc.) ---")
    # Filtrar predicciones del set de prueba para partidos del mundial
    wc_test_indices = test_wc_mask.values[test_mask.values]
    y_test_wc_preds = y_pred[wc_test_indices]
    acc_wc = accuracy_score(y_test_wc, y_test_wc_preds)
    print(f"Precisión (Accuracy) en la Copa del Mundo: {acc_wc:.2%}")
    
    # Línea Base ELO en Copa del Mundo
    elo_preds_wc = elo_baseline_predict(X_test_wc)
    elo_acc_wc = np.mean(elo_preds_wc == y_test_wc)
    print(f"Precisión de Línea Base ELO (Copa del Mundo): {elo_acc_wc:.2%}")
    
    # Reporte de clasificación detallado para Copa del Mundo
    print("\nReporte de Clasificación (Copa del Mundo):")
    print(classification_report(y_test_wc, y_test_wc_preds, target_names=['Local', 'Empate', 'Visitante'], zero_division=0))
    
if __name__ == "__main__":
    train_model()
