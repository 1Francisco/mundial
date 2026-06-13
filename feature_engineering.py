import os
import pickle
import pandas as pd
import numpy as np

# Valoración económica aproximada de plantillas en millones de euros (Transfermarkt 2024-2026)
SQUAD_VALUES = {
    "England": 1470.0, "France": 1210.0, "Portugal": 980.0, "Spain": 960.0, "Brazil": 940.0,
    "Germany": 830.0, "Argentina": 810.0, "Italy": 705.0, "Netherlands": 620.0, "Belgium": 560.0,
    "Norway": 480.0, "Uruguay": 430.0, "Denmark": 410.0, "Morocco": 390.0, "Ukraine": 350.0,
    "Croatia": 330.0, "Japan": 320.0, "Turkey": 310.0, "United States": 310.0, "Colombia": 280.0,
    "Senegal": 270.0, "Sweden": 260.0, "Austria": 250.0, "Serbia": 230.0, "Poland": 220.0,
    "Switzerland": 210.0, "Ivory Coast": 200.0, "Scotland": 200.0, "Mexico": 170.0, "Nigeria": 170.0,
    "Ecuador": 160.0, "Wales": 150.0, "Hungary": 150.0, "Republic of Ireland": 140.0, "Georgia": 140.0,
    "Greece": 130.0, "Algeria": 120.0, "South Korea": 120.0, "Ghana": 110.0, "Czech Republic": 110.0,
    "Slovakia": 100.0, "Romania": 95.0, "Egypt": 85.0, "Cameroon": 140.0, "Mali": 160.0,
    "Slovenia": 75.0, "Burkina Faso": 75.0, "Guinea": 80.0, "DR Congo": 60.0, "Gabon": 45.0,
    "Angola": 35.0, "Zambia": 30.0, "Cape Verde": 25.0, "Israel": 45.0, "Peru": 45.0,
    "Paraguay": 40.0, "Chile": 40.0, "Australia": 40.0, "Tunisia": 40.0, "Canada": 40.0,
    "South Africa": 25.0, "Saudi Arabia": 20.0, "Costa Rica": 18.0, "Venezuela": 18.0,
    "New Zealand": 15.0, "Qatar": 15.0, "Bolivia": 12.0, "Honduras": 15.0, "Panama": 10.0,
    "Jamaica": 8.0, "China PR": 8.0, "El Salvador": 5.0, "Iraq": 5.0, "United Arab Emirates": 5.0,
    "Syria": 3.0, "Iceland": 30.0, "Albania": 60.0, "Finland": 25.0, "Northern Ireland": 30.0,
    "North Macedonia": 20.0, "Bosnia and Herzegovina": 45.0, "Montenegro": 20.0, "Bulgaria": 15.0
}

def estimate_market_value(team, elo):
    """Estima el valor de la plantilla en millones de euros."""
    # Intentar obtener de la base de datos fija de Transfermarkt
    base_val = SQUAD_VALUES.get(team)
    if base_val is not None:
        return base_val
        
    # Imputar valor utilizando una curva exponencial basada en el ELO
    if elo >= 2100:
        return 900.0
    elif elo >= 2000:
        return 450.0 + (elo - 2000) * 4.5
    elif elo >= 1900:
        return 220.0 + (elo - 1900) * 2.3
    elif elo >= 1800:
        return 90.0 + (elo - 1800) * 1.3
    elif elo >= 1700:
        return 35.0 + (elo - 1700) * 0.55
    elif elo >= 1600:
        return 15.0 + (elo - 1600) * 0.2
    elif elo >= 1500:
        return 5.0 + (elo - 1500) * 0.1
    else:
        return max(0.1, 0.1 + (elo - 1000) * 0.01)

def run_feature_engineering():
    input_path = "data/results.csv"
    output_path = "data/processed_matches.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: No se encontró el archivo {input_path}. Por favor ejecuta download_data.py primero.")
        return
        
    print("Cargando y ordenando dataset de partidos...")
    df = pd.read_csv(input_path)
    
    # Limpieza básica y ordenamiento cronológico
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score'])
    df = df.sort_values(by='date').reset_index(drop=True)
    
    print(f"Total de partidos cargados: {len(df)}")
    
    # Estructuras para ELO, Racha y H2H
    elo_dict = {}       # Mapeo de team -> ELO actual
    team_history = {}   # Mapeo de team -> lista de tuplas (goles_a_favor, goles_en_contra, puntos) de los últimos 5 partidos
    h2h_history = {}    # Mapeo de tuple(sorted(team1, team2)) -> lista de dicts con encuentros previos (max 5)
    
    # Listas para guardar las nuevas características
    elo_home_list = []
    elo_away_list = []
    
    form_goals_scored_home = []
    form_goals_conceded_home = []
    form_win_rate_home = []
    
    form_goals_scored_away = []
    form_goals_conceded_away = []
    form_win_rate_away = []
    
    # H2H Features
    h2h_home_wr_list = []
    h2h_away_wr_list = []
    h2h_draw_r_list = []
    
    # Market Value Features
    market_value_home_list = []
    market_value_away_list = []
    
    print("Calculando ELO, racha, H2H y valor de mercado (esto puede tomar un minuto)...")
    
    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        # 1. Obtener ELO actual (pre-partido)
        r_home = elo_dict.get(home, 1500.0)
        r_away = elo_dict.get(away, 1500.0)
        
        elo_home_list.append(r_home)
        elo_away_list.append(r_away)
        
        # 2. Obtener Racha actual (pre-partido)
        def get_form(team):
            history = team_history.get(team, [])
            if not history:
                return 0.0, 0.0, 0.0
            avg_scored = sum(h[0] for h in history) / len(history)
            avg_conceded = sum(h[1] for h in history) / len(history)
            win_rate = sum(1.0 if h[2] == 3 else (0.5 if h[2] == 1 else 0.0) for h in history) / len(history)
            return avg_scored, avg_conceded, win_rate
            
        g_scored_h, g_conceded_h, wr_h = get_form(home)
        g_scored_a, g_conceded_a, wr_a = get_form(away)
        
        form_goals_scored_home.append(g_scored_h)
        form_goals_conceded_home.append(g_conceded_h)
        form_win_rate_home.append(wr_h)
        
        form_goals_scored_away.append(g_scored_a)
        form_goals_conceded_away.append(g_conceded_a)
        form_win_rate_away.append(wr_a)
        
        # 3. Obtener H2H actual (pre-partido)
        pair = tuple(sorted([home, away]))
        prev_matches = h2h_history.get(pair, [])
        
        h_wins = 0
        a_wins = 0
        draws = 0
        total_h2h = len(prev_matches)
        
        for m in prev_matches:
            if m['winner'] == 'draw':
                draws += 1
            elif m['winner'] == home:
                h_wins += 1
            elif m['winner'] == away:
                a_wins += 1
                
        h2h_home_wr = h_wins / total_h2h if total_h2h > 0 else 0.33
        h2h_away_wr = a_wins / total_h2h if total_h2h > 0 else 0.33
        h2h_draw_r = draws / total_h2h if total_h2h > 0 else 0.33
        
        h2h_home_wr_list.append(h2h_home_wr)
        h2h_away_wr_list.append(h2h_away_wr)
        h2h_draw_r_list.append(h2h_draw_r)
        
        # 3.5 Calcular Valor de Mercado estimado
        val_home = estimate_market_value(home, r_home)
        val_away = estimate_market_value(away, r_away)
        market_value_home_list.append(val_home)
        market_value_away_list.append(val_away)
        
        # 4. Calcular resultado del partido actual
        h_score = int(row['home_score'])
        a_score = int(row['away_score'])
        gd = abs(h_score - a_score)
        
        w = 0.5  # Empate
        pts_home = 1
        pts_away = 1
        h2h_winner = 'draw'
        
        if h_score > a_score:
            w = 1.0  # Victoria local
            pts_home = 3
            pts_away = 0
            h2h_winner = home
        elif h_score < a_score:
            w = 0.0  # Victoria visitante
            pts_home = 0
            pts_away = 3
            h2h_winner = away
            
        # 5. Actualizar ELO para el futuro
        home_adv = 100.0 if not row['neutral'] else 0.0
        dr = r_home + home_adv - r_away
        w_e = 1.0 / (10**(-dr / 400.0) + 1.0)
        
        # Factor K basado en el tipo de torneo
        tourney = row['tournament']
        if tourney == 'FIFA World Cup':
            k = 60
        elif 'qualification' in tourney.lower() or 'qualifying' in tourney.lower():
            k = 40
        elif tourney == 'Friendly':
            k = 20
        else:
            k = 30
            
        # Multiplicador por diferencia de goles (G)
        if gd <= 1:
            g = 1.0
        elif gd == 2:
            g = 1.5
        else:
            g = 1.75 + (gd - 3) / 8.0
            
        delta = k * g * (w - w_e)
        
        # Actualizar diccionario ELO
        elo_dict[home] = r_home + delta
        elo_dict[away] = r_away - delta
        
        # 6. Actualizar historial de racha (post-partido)
        if home not in team_history:
            team_history[home] = []
        team_history[home].append((h_score, a_score, pts_home))
        if len(team_history[home]) > 5:
            team_history[home].pop(0)
            
        if away not in team_history:
            team_history[away] = []
        team_history[away].append((a_score, h_score, pts_away))
        if len(team_history[away]) > 5:
            team_history[away].pop(0)
            
        # 7. Actualizar historial H2H (post-partido)
        if pair not in h2h_history:
            h2h_history[pair] = []
        h2h_history[pair].append({
            'home': home,
            'away': away,
            'winner': h2h_winner
        })
        if len(h2h_history[pair]) > 5:
            h2h_history[pair].pop(0)

    # Añadir las nuevas características al DataFrame original
    df['elo_home'] = elo_home_list
    df['elo_away'] = elo_away_list
    df['elo_diff'] = df['elo_home'] - df['elo_away']
    
    df['form_goals_scored_home'] = form_goals_scored_home
    df['form_goals_conceded_home'] = form_goals_conceded_home
    df['form_win_rate_home'] = form_win_rate_home
    
    df['form_goals_scored_away'] = form_goals_scored_away
    df['form_goals_conceded_away'] = form_goals_conceded_away
    df['form_win_rate_away'] = form_win_rate_away
    
    df['h2h_home_win_rate'] = h2h_home_wr_list
    df['h2h_away_win_rate'] = h2h_away_wr_list
    df['h2h_draw_rate'] = h2h_draw_r_list
    
    df['market_value_home'] = market_value_home_list
    df['market_value_away'] = market_value_away_list
    df['market_value_diff'] = df['market_value_home'] - df['market_value_away']
    
    # Codificar variables categóricas sencillas
    df['neutral'] = df['neutral'].astype(int)
    df['is_world_cup'] = (df['tournament'] == 'FIFA World Cup').astype(int)
    
    # Definir el Target
    def get_result_class(row):
        h = int(row['home_score'])
        a = int(row['away_score'])
        if h > a:
            return 0
        elif h == a:
            return 1
        else:
            return 2
            
    df['result'] = df.apply(get_result_class, axis=1)
    
    print("Guardando dataset procesado...")
    df.to_csv(output_path, index=False)
    print(f"¡Hecho! Dataset con características guardado en {output_path}")

    # Guardar las estadísticas más recientes de todos los equipos para el predictor
    latest_stats = {
        'elo_dict': elo_dict,
        'team_history': team_history,
        'h2h_history': h2h_history
    }
    stats_path = "data/latest_stats.pkl"
    with open(stats_path, 'wb') as f:
        pickle.dump(latest_stats, f)
    print(f"Estadísticas recientes de selecciones (incluyendo H2H y valores) guardadas en {stats_path}")

    # Mostrar ranking ELO actual al final del dataset
    print("\nRanking ELO Top 15 Selecciones actual (al final del dataset):")
    sorted_elo = sorted(elo_dict.items(), key=lambda item: item[1], reverse=True)
    for i, (team, elo) in enumerate(sorted_elo[:15], 1):
        print(f"{i}. {team}: {elo:.1f}")

if __name__ == "__main__":
    run_feature_engineering()
