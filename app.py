import os
import math
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from feature_engineering import estimate_market_value


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Rutas de los archivos (relativas al directorio de app.py)
MODEL_PATH = os.path.join(BASE_DIR, "data", "world_cup_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "data", "scaler.pkl")
STATS_PATH = os.path.join(BASE_DIR, "data", "latest_stats.pkl")

# Variables globales para cargar en memoria al iniciar el servidor
model = None
scaler = None
elo_dict = {}
team_history = {}
h2h_history = {}
all_teams = []

def init_resources():
    global model, scaler, elo_dict, team_history, h2h_history, all_teams
    
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(STATS_PATH)):
        raise FileNotFoundError("Faltan archivos de modelo o estadísticas. Asegúrate de ejecutar train.py primero.")
        
    print("Cargando base de datos del modelo y estadísticas...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    with open(STATS_PATH, 'rb') as f:
        latest_stats = pickle.load(f)
        
    elo_dict = latest_stats['elo_dict']
    team_history = latest_stats['team_history']
    h2h_history = latest_stats.get('h2h_history', {})
    all_teams = sorted(list(elo_dict.keys()))
    print(f"Recursos cargados correctamente. {len(all_teams)} selecciones disponibles.")

def calculate_poisson_exact_scores(gs_h, gc_h, gs_a, gc_a, elo_diff):
    """Calcula los goles esperados y las probabilidades de marcador exacto usando Poisson."""
    # Promedio de goles por partido en el fútbol (aprox 1.2 o 1.35 por equipo)
    base_h = gs_h if gs_h > 0 else 1.2
    base_a = gs_a if gs_a > 0 else 1.2
    
    # Ajuste por diferencia de ELO (1000 puntos doblan/mitigan la expectativa)
    lambda_h = base_h * (1.0 + elo_diff / 1000.0)
    lambda_a = base_a * (1.0 - elo_diff / 1000.0)
    
    # Limitar para evitar valores extremos absurdos
    lambda_h = max(0.2, min(5.0, lambda_h))
    lambda_a = max(0.2, min(5.0, lambda_a))
    
    def poisson_prob(lmbda, k):
        return (lmbda**k * math.exp(-lmbda)) / math.factorial(k)
        
    score_probs = []
    # Evaluar marcadores del 0-0 al 5-5
    for i in range(6):
        for j in range(6):
            prob_h = poisson_prob(lambda_h, i)
            prob_a = poisson_prob(lambda_a, j)
            prob = prob_h * prob_a
            score_probs.append({
                "score": f"{i}-{j}",
                "prob": round(prob, 4)
            })
            
    # Ordenar por probabilidad descendente
    score_probs = sorted(score_probs, key=lambda x: x['prob'], reverse=True)
    
    return {
        "expected_goals": {
            "home": round(lambda_h, 2),
            "away": round(lambda_a, 2)
        },
        "exact_scores": score_probs[:3] # Top 3 más probables
    }

# Inicializar recursos
init_resources()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/teams', methods=['GET'])
def get_teams():
    return jsonify({
        "success": True,
        "teams": all_teams
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No se recibieron datos"}), 400
            
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        neutral = int(data.get('neutral', 1))
        
        if not home_team or not away_team:
            return jsonify({"success": False, "error": "Falta especificar el equipo local o visitante"}), 400
            
        # Validar existencia de equipos
        if home_team not in elo_dict or away_team not in elo_dict:
            return jsonify({"success": False, "error": "Uno o ambos equipos no existen en la base de datos"}), 404
            
        if home_team == away_team:
            return jsonify({"success": False, "error": "Un equipo no puede jugar contra sí mismo"}), 400
            
        # 1. Obtener ELO actual
        elo_h = elo_dict[home_team]
        elo_a = elo_dict[away_team]
        elo_diff = elo_h - elo_a
        
        # 2. Obtener Racha actual (Form)
        def get_form_stats(team):
            history = team_history.get(team, [])
            if not history:
                return 0.0, 0.0, 0.0
            avg_scored = sum(h[0] for h in history) / len(history)
            avg_conceded = sum(h[1] for h in history) / len(history)
            win_rate = sum(1.0 if h[2] == 3 else (0.5 if h[2] == 1 else 0.0) for h in history) / len(history)
            return avg_scored, avg_conceded, win_rate
            
        gs_h, gc_h, wr_h = get_form_stats(home_team)
        gs_a, gc_a, wr_a = get_form_stats(away_team)
        
        # 3. Obtener H2H actual
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
        
        # 4. Construir vector de características
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
        
        # Normalizar con el scaler
        features_scaled = scaler.transform(features_df)
        
        # 5. Predecir probabilidades
        probs = model.predict_proba(features_scaled)[0]
        
        # Formatear el resultado
        max_idx = np.argmax(probs)
        if max_idx == 0:
            recommended = f"Victoria de {home_team}"
        elif max_idx == 1:
            recommended = "Empate"
        else:
            recommended = f"Victoria de {away_team}"
            
        # 5.5 Calcular goles esperados y marcador exacto (Poisson)
        # Ajustar diferencia de ELO con ventaja de localía si no es neutral
        adj_elo_diff = elo_diff
        if neutral == 0:
            adj_elo_diff += 100.0
        poisson_stats = calculate_poisson_exact_scores(gs_h, gc_h, gs_a, gc_a, adj_elo_diff)
            
        return jsonify({
            "success": True,
            "prediction": {
                "home_prob": float(probs[0]),
                "draw_prob": float(probs[1]),
                "away_prob": float(probs[2]),
                "recommended": recommended
            },
            "stats": {
                "home": {
                    "name": home_team,
                    "elo": round(elo_h, 1),
                    "goals_scored": round(gs_h, 2),
                    "goals_conceded": round(gc_h, 2),
                    "win_rate": int(wr_h * 100),
                    "market_value": round(val_home, 1)
                },
                "away": {
                    "name": away_team,
                    "elo": round(elo_a, 1),
                    "goals_scored": round(gs_a, 2),
                    "goals_conceded": round(gc_a, 2),
                    "win_rate": int(wr_a * 100),
                    "market_value": round(val_away, 1)
                },
                "elo_diff": round(elo_diff, 1),
                "market_value_diff": round(val_diff, 1),
                "h2h": {
                    "total": total_h2h,
                    "home_wins": h_wins,
                    "away_wins": a_wins,
                    "draws": draws,
                    "home_wr": round(h2h_home_wr * 100, 1),
                    "away_wr": round(h2h_away_wr * 100, 1),
                    "draw_r": round(h2h_draw_r * 100, 1)
                },
                "neutral": neutral
            },
            "goals": poisson_stats
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

WORLD_CUP_GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Ecuador', 'Ivory Coast', 'Curaçao'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Uruguay', 'Saudi Arabia', 'Cape Verde'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Austria', 'Algeria', 'Jordan'],
    'K': ['Portugal', 'Colombia', 'Uzbekistan', 'DR Congo'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama']
}

@app.route('/api/simulate', methods=['POST'])
def simulate_tournament():
    try:
        # Recuperar lista de todos los equipos del mundial
        all_teams_list = []
        for g_name, t_list in WORLD_CUP_GROUPS.items():
            all_teams_list.extend(t_list)
            
        # 1. Pre-cálculo masivo de todos los enfrentamientos posibles
        matchup_list = []
        features_raw = []
        
        def get_form_stats(team):
            history = team_history.get(team, [])
            if not history:
                return 0.0, 0.0, 0.0
            avg_scored = sum(h[0] for h in history) / len(history)
            avg_conceded = sum(h[1] for h in history) / len(history)
            win_rate = sum(1.0 if h[2] == 3 else (0.5 if h[2] == 1 else 0.0) for h in history) / len(history)
            return avg_scored, avg_conceded, win_rate

        for t1 in all_teams_list:
            for t2 in all_teams_list:
                if t1 == t2:
                    continue
                    
                elo_1 = elo_dict.get(t1, 1500.0)
                elo_2 = elo_dict.get(t2, 1500.0)
                elo_diff = elo_1 - elo_2
                
                gs_1, gc_1, wr_1 = get_form_stats(t1)
                gs_2, gc_2, wr_2 = get_form_stats(t2)
                
                pair = tuple(sorted([t1, t2]))
                prev_matches = h2h_history.get(pair, [])
                h_wins = 0
                a_wins = 0
                draws = 0
                total_h2h = len(prev_matches)
                
                for m in prev_matches:
                    if m['winner'] == 'draw':
                        draws += 1
                    elif m['winner'] == t1:
                        h_wins += 1
                    elif m['winner'] == t2:
                        a_wins += 1
                        
                h2h_home_wr = h_wins / total_h2h if total_h2h > 0 else 0.33
                h2h_away_wr = a_wins / total_h2h if total_h2h > 0 else 0.33
                h2h_draw_r = draws / total_h2h if total_h2h > 0 else 0.33
                
                val_1 = estimate_market_value(t1, elo_1)
                val_2 = estimate_market_value(t2, elo_2)
                val_diff = val_1 - val_2
                
                matchup_list.append((t1, t2))
                features_raw.append([
                    elo_1, elo_2, elo_diff,
                    gs_1, gc_1, wr_1,
                    gs_2, gc_2, wr_2,
                    h2h_home_wr, h2h_away_wr, h2h_draw_r,
                    val_1, val_2, val_diff,
                    1 # neutral = 1
                ])
                
        # Batch scale y predicción
        import pandas as pd
        features_df = pd.DataFrame(features_raw, columns=[
            'elo_home', 'elo_away', 'elo_diff',
            'form_goals_scored_home', 'form_goals_conceded_home', 'form_win_rate_home',
            'form_goals_scored_away', 'form_goals_conceded_away', 'form_win_rate_away',
            'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_draw_rate',
            'market_value_home', 'market_value_away', 'market_value_diff',
            'neutral'
        ])
        
        features_scaled = scaler.transform(features_df)
        probs_batch = model.predict_proba(features_scaled)
        
        probs_lookup = {}
        lambdas_lookup = {}
        
        for idx, (t1, t2) in enumerate(matchup_list):
            probs_lookup[(t1, t2)] = probs_batch[idx]
            
            elo_1 = features_raw[idx][0]
            elo_2 = features_raw[idx][1]
            elo_diff = elo_1 - elo_2
            gs_1 = features_raw[idx][3]
            gs_2 = features_raw[idx][6]
            
            base_1 = gs_1 if gs_1 > 0 else 1.2
            base_2 = gs_2 if gs_2 > 0 else 1.2
            
            lambda_1 = base_1 * (1.0 + elo_diff / 1000.0)
            lambda_2 = base_2 * (1.0 - elo_diff / 1000.0)
            
            lambda_1 = max(0.2, min(5.0, lambda_1))
            lambda_2 = max(0.2, min(5.0, lambda_2))
            
            lambdas_lookup[(t1, t2)] = (lambda_1, lambda_2)
            
        # 2. Inicializar contadores de rondas y variables para encontrar la simulación más lógica (máxima verosimilitud)
        N_SIMS = 5000
        r32_counts = {t: 0 for t in all_teams_list}
        r16_counts = {t: 0 for t in all_teams_list}
        qf_counts = {t: 0 for t in all_teams_list}
        sf_counts = {t: 0 for t in all_teams_list}
        final_counts = {t: 0 for t in all_teams_list}
        champion_counts = {t: 0 for t in all_teams_list}
        
        best_run_score = -1.0
        best_bracket_details = None
        
        def simulate_match(t1, t2, record_round_list=None):
            lambda_1, lambda_2 = lambdas_lookup[(t1, t2)]
            g1 = int(np.random.poisson(lambda_1))
            g2 = int(np.random.poisson(lambda_2))
            
            winner = None
            home_pen = None
            away_pen = None
            
            p1, pd, p2 = probs_lookup[(t1, t2)]
            
            if g1 > g2:
                winner = t1
                match_prob = p1
            elif g2 > g1:
                winner = t2
                match_prob = p2
            else:
                match_prob = pd
                total_p = p1 + p2
                w_prob = p1 / total_p if total_p > 0 else 0.5
                winner = t1 if np.random.random() < w_prob else t2
                if winner == t1:
                    home_pen = 5
                    away_pen = int(np.random.choice([3, 4]))
                    match_prob += (w_prob * 0.5)
                else:
                    away_pen = 5
                    home_pen = int(np.random.choice([3, 4]))
                    match_prob += ((1.0 - w_prob) * 0.5)
                    
            if record_round_list is not None:
                record_round_list.append({
                    "home": t1,
                    "away": t2,
                    "home_score": g1,
                    "away_score": g2,
                    "winner": winner,
                    "home_pen": home_pen,
                    "away_pen": away_pen
                })
                
            return winner, match_prob

        # Bucle de simulación Monte Carlo
        for sim_idx in range(N_SIMS):
            current_run_score = 0.0
            current_bracket = {
                "groups": {},
                "r32": [],
                "r16": [],
                "qf": [],
                "sf": [],
                "final": None,
                "champion": None
            }
            
            # Fase de Grupos
            group_winners = {}
            group_runners_up = {}
            third_placed_teams = []
            
            # Registrar estadísticas de grupo de esta corrida
            group_run_stats = {}
            
            for g_name, teams in WORLD_CUP_GROUPS.items():
                g_stats = {t: {'points': 0, 'gd': 0, 'gf': 0, 'elo': elo_dict.get(t, 1500.0), 'name': t} for t in teams}
                
                # Partidos (6 por grupo)
                fixtures = [
                    (teams[0], teams[1]), (teams[2], teams[3]),
                    (teams[0], teams[2]), (teams[1], teams[3]),
                    (teams[0], teams[3]), (teams[1], teams[2])
                ]
                
                for h, a in fixtures:
                    l_h, l_a = lambdas_lookup[(h, a)]
                    gh = np.random.poisson(l_h)
                    ga = np.random.poisson(l_a)
                    
                    g_stats[h]['gf'] += gh
                    g_stats[h]['gd'] += (gh - ga)
                    g_stats[a]['gf'] += ga
                    g_stats[a]['gd'] += (ga - gh)
                    
                    p1, pd, p2 = probs_lookup[(h, a)]
                    if gh > ga:
                        g_stats[h]['points'] += 3
                        current_run_score += p1
                    elif ga > gh:
                        g_stats[a]['points'] += 3
                        current_run_score += p2
                    else:
                        g_stats[h]['points'] += 1
                        g_stats[a]['points'] += 1
                        current_run_score += pd
                        
                # Clasificar
                ranked = sorted(
                    g_stats.values(),
                    key=lambda x: (x['points'], x['gd'], x['gf'], x['elo']),
                    reverse=True
                )
                
                group_winners[g_name] = ranked[0]['name']
                group_runners_up[g_name] = ranked[1]['name']
                third_placed_teams.append(ranked[2])
                
                # Guardar estadísticas de grupo
                for team_data in ranked:
                    group_run_stats[team_data['name']] = team_data
                
                # Guardar standings para la simulación actual
                current_bracket["groups"][g_name] = [
                    {
                        "name": x['name'],
                        "points": x['points'],
                        "gd": x['gd'],
                        "gf": x['gf']
                    } for x in ranked
                ]
                    
            # 8 mejores terceros puestos
            sorted_thirds = sorted(
                third_placed_teams,
                key=lambda x: (x['points'], x['gd'], x['gf'], x['elo']),
                reverse=True
            )
            best_thirds_names = [x['name'] for x in sorted_thirds[:8]]
            
            # Rankear primeros, segundos y terceros clasificados
            winners_ranked = sorted(
                [group_run_stats[w] for w in group_winners.values()],
                key=lambda x: (x['points'], x['gd'], x['gf'], x['elo']),
                reverse=True
            )
            runners_up_ranked = sorted(
                [group_run_stats[r] for r in group_runners_up.values()],
                key=lambda x: (x['points'], x['gd'], x['gf'], x['elo']),
                reverse=True
            )
            best_thirds_ranked = sorted(
                [group_run_stats[t] for t in best_thirds_names],
                key=lambda x: (x['points'], x['gd'], x['gf'], x['elo']),
                reverse=True
            )
            
            winners_names = [x['name'] for x in winners_ranked]
            runners_up_names = [x['name'] for x in runners_up_ranked]
            thirds_names = [x['name'] for x in best_thirds_ranked]
            
            # Armar llaves de Dieciseisavos de Final (R32)
            r32_matchups = []
            
            # Top 8 ganadores de grupo vs 8 mejores terceros (cruzados)
            for i in range(8):
                r32_matchups.append((winners_names[i], thirds_names[7 - i]))
                
            # Siguientes 4 ganadores de grupo vs peores 4 segundos
            for i in range(4):
                r32_matchups.append((winners_names[8 + i], runners_up_names[11 - i]))
                
            # Top 8 segundos juegan entre sí
            for i in range(4):
                r32_matchups.append((runners_up_names[i], runners_up_names[7 - i]))
                
            # Registrar clasificación a R32
            for t in winners_names + runners_up_names + thirds_names:
                r32_counts[t] += 1
                
            # Simular Dieciseisavos de Final (Round of 32)
            r32_winners = []
            for t1, t2 in r32_matchups:
                w, m_p = simulate_match(t1, t2, record_round_list=current_bracket["r32"])
                current_run_score += m_p
                r32_winners.append(w)
                r16_counts[w] += 1
                
            # Octavos de Final (Round of 16)
            r16_matchups = [
                (r32_winners[0], r32_winners[1]),
                (r32_winners[2], r32_winners[3]),
                (r32_winners[4], r32_winners[5]),
                (r32_winners[6], r32_winners[7]),
                (r32_winners[8], r32_winners[9]),
                (r32_winners[10], r32_winners[11]),
                (r32_winners[12], r32_winners[13]),
                (r32_winners[14], r32_winners[15])
            ]
            
            r16_winners = []
            for t1, t2 in r16_matchups:
                w, m_p = simulate_match(t1, t2, record_round_list=current_bracket["r16"])
                current_run_score += m_p
                r16_winners.append(w)
                qf_counts[w] += 1
                
            # Cuartos de Final
            qf_matchups = [
                (r16_winners[0], r16_winners[1]),
                (r16_winners[2], r16_winners[3]),
                (r16_winners[4], r16_winners[5]),
                (r16_winners[6], r16_winners[7])
            ]
            
            qf_winners = []
            for t1, t2 in qf_matchups:
                w, m_p = simulate_match(t1, t2, record_round_list=current_bracket["qf"])
                current_run_score += m_p
                qf_winners.append(w)
                sf_counts[w] += 1
                
            # Semifinales
            sf_matchups = [
                (qf_winners[0], qf_winners[1]),
                (qf_winners[2], qf_winners[3])
            ]
            
            sf_winners = []
            for t1, t2 in sf_matchups:
                w, m_p = simulate_match(t1, t2, record_round_list=current_bracket["sf"])
                current_run_score += m_p
                sf_winners.append(w)
                final_counts[w] += 1
                
            # Final
            final_list = []
            champ, m_p = simulate_match(sf_winners[0], sf_winners[1], record_round_list=final_list)
            current_run_score += m_p
            champion_counts[champ] += 1
            
            current_bracket["final"] = final_list[0]
            current_bracket["champion"] = champ
            
            # Guardar la simulación que maximiza la verosimilitud de resultados (los favoritos ganan más)
            if current_run_score > best_run_score:
                best_run_score = current_run_score
                best_bracket_details = current_bracket
            
        # 3. Compilar resultados finales
        simulation_results = []
        for t in all_teams_list:
            simulation_results.append({
                "team": t,
                "elo": round(elo_dict.get(t, 1500.0), 1),
                "market_value": round(estimate_market_value(t, elo_dict.get(t, 1500.0)), 1),
                "r32_prob": round((r32_counts[t] / N_SIMS) * 100, 2),
                "r16_prob": round((r16_counts[t] / N_SIMS) * 100, 2),
                "qf_prob": round((qf_counts[t] / N_SIMS) * 100, 2),
                "sf_prob": round((sf_counts[t] / N_SIMS) * 100, 2),
                "final_prob": round((final_counts[t] / N_SIMS) * 100, 2),
                "champion_prob": round((champion_counts[t] / N_SIMS) * 100, 2)
            })
            
        # Ordenar por probabilidad de salir campeón descendente
        simulation_results = sorted(simulation_results, key=lambda x: x['champion_prob'], reverse=True)
        
        return jsonify({
            "success": True,
            "results": simulation_results,
            "bracket": best_bracket_details
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
