import os
import math
import pickle
import numpy as np
from feature_engineering import estimate_market_value


def find_team_suggestions(query, all_teams):
    """Devuelve nombres de equipos similares si no se encuentra coincidencia exacta."""
    query_lower = query.lower()
    matches = [t for t in all_teams if query_lower in t.lower()]
    return matches[:5]

def run_predictor():
    model_path = "data/world_cup_model.pkl"
    scaler_path = "data/scaler.pkl"
    stats_path = "data/latest_stats.pkl"
    
    # Verificar que los archivos necesarios existen
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(stats_path)):
        print("Error: Faltan archivos del modelo entrenado o estadísticas.")
        print("Asegúrate de haber ejecutado en orden:")
        print("  1. download_data.py")
        print("  2. feature_engineering.py")
        print("  3. train.py")
        return
        
    print("Cargando modelo de red neuronal y base de datos de selecciones...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    with open(stats_path, 'rb') as f:
        latest_stats = pickle.load(f)
        
    elo_dict = latest_stats['elo_dict']
    team_history = latest_stats['team_history']
    h2h_history = latest_stats.get('h2h_history', {})
    all_teams = list(elo_dict.keys())
    
    print("\n" + "="*50)
    print("  PREDICTOR DE PARTIDOS DE FÚTBOL (NEURAL NETWORK) ")
    print("="*50)
    print(f"Base de datos cargada con {len(all_teams)} selecciones.")
    print("Nota: Escribe los nombres de los equipos en inglés (ej. 'Argentina', 'Brazil', 'Germany', 'France').")
    print("Escribe 'salir' para cerrar el programa.\n")
    
    while True:
        # 1. Solicitar Equipo Local / Equipo 1
        home_input = input("Equipo 1 (Local/A): ").strip()
        if home_input.lower() == 'salir':
            break
        if not home_input:
            continue
            
        # Buscar coincidencia exacta o sugerencias
        home_team = next((t for t in all_teams if t.lower() == home_input.lower()), None)
        if not home_team:
            sugs = find_team_suggestions(home_input, all_teams)
            print(f"[-] No encontré '{home_input}'.")
            if sugs:
                print(f"¿Quisiste decir? {', '.join([f'\"{t}\"' for t in sugs])}")
            continue
            
        # 2. Solicitar Equipo Visitante / Equipo 2
        away_input = input("Equipo 2 (Visitante/B): ").strip()
        if away_input.lower() == 'salir':
            break
        if not away_input:
            continue
            
        away_team = next((t for t in all_teams if t.lower() == away_input.lower()), None)
        if not away_team:
            sugs = find_team_suggestions(away_input, all_teams)
            print(f"[-] No encontré '{away_input}'.")
            if sugs:
                print(f"¿Quisiste decir? {', '.join([f'\"{t}\"' for t in sugs])}")
            continue
            
        if home_team == away_team:
            print("[WARN] Un equipo no puede jugar contra sí mismo. Elige otro rival.\n")
            continue
            
        # 3. Solicitar Campo Neutral
        print("\n[INFO] Campo Neutral:")
        print("  - 'S' (Sí): Se juega en sede neutral (como el Mundial). Ningún equipo tiene ventaja de localía.")
        print("  - 'N' (No): El Equipo 1 juega en su propio estadio y recibe ventaja deportiva de local.")
        neutral_input = input("¿El partido es en Campo Neutral? (S/N) [Por defecto: S]: ").strip().lower()
        if neutral_input == 'salir':
            break
        neutral = 1
        if neutral_input in ['n', 'no']:
            neutral = 0
            
        # 4. Obtener estadísticas para las features
        # ELO
        elo_h = elo_dict[home_team]
        elo_a = elo_dict[away_team]
        elo_diff = elo_h - elo_a
        
        # Historial de Racha (Form)
        def extract_form_stats(team):
            history = team_history.get(team, [])
            if not history:
                return 0.0, 0.0, 0.0
            avg_scored = sum(h[0] for h in history) / len(history)
            avg_conceded = sum(h[1] for h in history) / len(history)
            win_rate = sum(1.0 if h[2] == 3 else (0.5 if h[2] == 1 else 0.0) for h in history) / len(history)
            return avg_scored, avg_conceded, win_rate
            
        gs_h, gc_h, wr_h = extract_form_stats(home_team)
        gs_a, gc_a, wr_a = extract_form_stats(away_team)
        
        # Historial de H2H
        pair = tuple(sorted([home_team, away_team]))
        prev_matches = h2h_history.get(pair, [])
        
        h_wins = 0
        a_wins = 0
        draws = 0
        total_h2h = len(prev_matches)
        
        for m in prev_matches:
            if m['winner'] == 'draw':
                draws += 1
            elif m['winner'] == home_team:
                h_wins += 1
            elif m['winner'] == away_team:
                a_wins += 1
                
        h2h_home_wr = h_wins / total_h2h if total_h2h > 0 else 0.33
        h2h_away_wr = a_wins / total_h2h if total_h2h > 0 else 0.33
        h2h_draw_r = draws / total_h2h if total_h2h > 0 else 0.33
        
        # Obtener Valor de Mercado estimado
        val_home = estimate_market_value(home_team, elo_h)
        val_away = estimate_market_value(away_team, elo_a)
        val_diff = val_home - val_away
        
        import pandas as pd
        features_df = pd.DataFrame([[
            elo_h, elo_a, elo_diff,
            gs_h, gc_h, wr_h,
            gs_a, gc_a, wr_a,
            h2h_home_wr, h2h_away_wr, h2h_draw_r,
            val_home, val_away, val_diff,
            neutral
        ]], columns=[
            'elo_home', 'elo_away', 'elo_diff',
            'form_goals_scored_home', 'form_goals_conceded_home', 'form_win_rate_home',
            'form_goals_scored_away', 'form_goals_conceded_away', 'form_win_rate_away',
            'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_draw_rate',
            'market_value_home', 'market_value_away', 'market_value_diff',
            'neutral'
        ])
        
        # Normalizar características
        features_scaled = scaler.transform(features_df)
        
        # 5. Hacer predicción
        probs = model.predict_proba(features_scaled)[0]
        
        # 5.5 Calcular goles esperados y marcador exacto (Poisson)
        adj_elo_diff = elo_diff
        if neutral == 0:
            adj_elo_diff += 100.0
            
        base_h = gs_h if gs_h > 0 else 1.2
        base_a = gs_a if gs_a > 0 else 1.2
        lambda_h = max(0.2, min(5.0, base_h * (1.0 + adj_elo_diff / 1000.0)))
        lambda_a = max(0.2, min(5.0, base_a * (1.0 - adj_elo_diff / 1000.0)))
        
        def poisson_prob(lmbda, k):
            return (lmbda**k * math.exp(-lmbda)) / math.factorial(k)
            
        score_probs = []
        for i in range(6):
            for j in range(6):
                prob = poisson_prob(lambda_h, i) * poisson_prob(lambda_a, j)
                score_probs.append((f"{i}-{j}", prob))
        score_probs = sorted(score_probs, key=lambda x: x[1], reverse=True)
        
        # 6. Mostrar resultados
        print("\n" + "-"*40)
        print(f"ANÁLISIS DE ENFRENTAMIENTO:")
        print(f"Campo: {'Neutral (Mundial / Copa)' if neutral == 1 else f'Estadio de {home_team}'}")
        print(f"Stats: {home_team} (ELO: {elo_h:.1f}, Valor: {val_home:.1f}M) vs {away_team} (ELO: {elo_a:.1f}, Valor: {val_away:.1f}M)")
        print(f"   Diferencia de ELO: {elo_diff:+.1f} | Diferencia de Valor: {val_diff:+.1f}M")
        print(f"   Últimos 5 partidos globales:")
        print(f"     * {home_team}: Goles anotados: {gs_h:.1f}/partido, Concedidos: {gc_h:.1f}/partido, Victorias: {wr_h:.0%}")
        print(f"     * {away_team}: Goles anotados: {gs_a:.1f}/partido, Concedidos: {gc_a:.1f}/partido, Victorias: {wr_a:.0%}")
        print(f"   Historial cara a cara previo (Últimos 5 encuentros directos):")
        if total_h2h > 0:
            print(f"     * Victorias de {home_team}: {h_wins}, Victorias de {away_team}: {a_wins}, Empates: {draws}")
        else:
            print("     * No hay registro de enfrentamientos previos en el dataset.")
        print("-"*40)
        
        print("GOLES ESPERADOS (xG) Y MARCADORES MÁS PROBABLES:")
        print(f"   Goles esperados: {home_team} {lambda_h:.2f} vs {away_team} {lambda_a:.2f}")
        print("   Top 3 marcadores exactos:")
        for idx_s, (score, prob) in enumerate(score_probs[:3], 1):
            print(f"     {idx_s}. Marcador {score}: {prob:.2%}")
        print("-"*40)
        
        print("PROBABILIDADES CALCULADAS POR LA RED NEURONAL:")
        print(f"   Victoria {home_team}: {probs[0]:.2%}")
        print(f"   Empate: {probs[1]:.2%}")
        print(f"   Victoria {away_team}: {probs[2]:.2%}")
        
        # Pronóstico sugerido
        max_idx = np.argmax(probs)
        if max_idx == 0:
            pronostico = f"Victoria de {home_team}"
        elif max_idx == 1:
            pronostico = "Empate"
        else:
            pronostico = f"Victoria de {away_team}"
        print(f"\nPronóstico recomendado: {pronostico}")
        print("="*50 + "\n")

if __name__ == "__main__":
    run_predictor()
