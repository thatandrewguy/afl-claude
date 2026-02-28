import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score
import warnings
import os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Page config – must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AFL Claude – Match Forecaster",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: rgba(255,255,255,0.6);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #e8ecf0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
    }

    .section-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6b7280;
        margin-bottom: 1rem;
    }

    .result-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        margin-top: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .result-winner {
        font-size: 2rem;
        font-weight: 700;
        color: #34d399;
        margin-bottom: 0.25rem;
    }

    .result-confidence {
        font-size: 1rem;
        color: rgba(255,255,255,0.6);
        margin-bottom: 1.5rem;
    }

    .score-display {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.5rem;
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
    }

    .score-team {
        text-align: center;
        flex: 1;
    }

    .score-team-name {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.5);
        margin-bottom: 0.25rem;
    }

    .score-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
    }

    .score-divider {
        font-size: 1.5rem;
        color: rgba(255,255,255,0.3);
        font-weight: 300;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .stat-item {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }

    .stat-label {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }

    .stat-value {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
    }

    .confidence-bar-container {
        margin: 1rem 0;
    }

    .confidence-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        margin-bottom: 0.4rem;
    }

    .confidence-bar {
        height: 8px;
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #34d399, #059669);
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s ease;
        letter-spacing: 0.3px;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.4);
    }

    .loading-text {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        padding: 1rem;
    }

    .error-card {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #dc2626;
        font-size: 0.9rem;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stDateInput"] label,
    div[data-testid="stTimeInput"] label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #374151;
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        border-radius: 8px;
        border-color: #d1d5db;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading – cached so it only runs once per session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load CSV, build rolling features (no leakage), train classifiers."""

    csv_path = os.path.join(os.path.dirname(__file__), "ReadyFor2026.csv")
    if not os.path.exists(csv_path):
        return None, "ReadyFor2026.csv not found. Please place it in the same directory as app.py."

    df = pd.read_csv(csv_path)
    df.columns = [col.replace('Possesions', 'Possessions') for col in df.columns]
    df['matchDate'] = pd.to_datetime(df['match.venueLocalStartTime'], errors='coerce')
    df = df.dropna(subset=['matchDate']).sort_values('matchDate').reset_index(drop=True)

    df['home_total'] = pd.to_numeric(df['homeTeamScore.matchScore.totalScore'], errors='coerce')
    df['away_total'] = pd.to_numeric(df['awayTeamScore.matchScore.totalScore'], errors='coerce')
    df['home_win'] = (df['home_total'] > df['away_total']).astype(int)

    numeric_team_cols = [
        col for col in df.columns
        if any(x in col for x in ['homeTeam', 'awayTeam', 'homeTeamScore', 'awayTeamScore',
                                   'homeTeamScoreChart', 'awayTeamScoreChart'])
        and pd.api.types.is_numeric_dtype(df[col])
        and 'brownlowVotes' not in col
    ]

    def clean_base(col):
        for prefix in ['homeTeam.', 'awayTeam.', 'homeTeamScore.', 'awayTeamScore.',
                        'homeTeamScoreChart.', 'awayTeamScoreChart.']:
            col = col.replace(prefix, '')
        return col

    base_stat_names = sorted({clean_base(c) for c in numeric_team_cols})

    teams = sorted(set(df['match.homeTeam.name'].dropna()) | set(df['match.awayTeam.name'].dropna()))
    team_home_history = {t: [] for t in teams}
    team_away_history = {t: [] for t in teams}
    team_results_history = {t: [] for t in teams}
    team_elo = {t: 1500.0 for t in teams}
    team_h2h = {}
    team_last_played = {}
    team_venue_wr = {t: {} for t in teams}

    ELO_K = 32
    HOME_BOOST = 55
    ROLLING_N = 8
    EMA_ALPHA = 0.35

    TEAM_STATES = {
        'Collingwood': 'VIC', 'Essendon': 'VIC', 'Carlton': 'VIC', 'Geelong Cats': 'VIC',
        'Hawthorn': 'VIC', 'North Melbourne': 'VIC', 'Richmond': 'VIC', 'St Kilda': 'VIC',
        'Western Bulldogs': 'VIC', 'Melbourne': 'VIC',
        'Sydney Swans': 'NSW', 'GWS GIANTS': 'NSW',
        'Brisbane Lions': 'QLD', 'Gold Coast SUNS': 'QLD',
        'Adelaide Crows': 'SA', 'Port Adelaide': 'SA',
        'West Coast Eagles': 'WA', 'Fremantle': 'WA',
    }

    def get_rolling(team, is_home=True, n=ROLLING_N):
        history = team_home_history[team] if is_home else team_away_history[team]
        if not history:
            return {name: 0.0 for name in base_stat_names}
        recent = history[-n:]
        result = {}
        for name in base_stat_names:
            vals = [g.get(name, 0) for g in recent]
            ema_val = vals[0]
            for v in vals[1:]:
                ema_val = EMA_ALPHA * v + (1 - EMA_ALPHA) * ema_val
            result[name] = ema_val
        return result

    def get_form(team, n=6):
        history = team_results_history[team]
        if not history:
            return 0.5, 85.0, 85.0, 0.0, 0
        recent = history[-n:]
        wins = sum(w for _, _, w in recent)
        wr = wins / len(recent)
        sf = np.mean([s for s, _, _ in recent])
        sa = np.mean([s for _, s, _ in recent])
        mg = np.mean([sf - sa for sf, sa, _ in recent])
        streak = 0
        for _, _, w in reversed(recent):
            if w and streak >= 0:
                streak += 1
            elif not w and streak <= 0:
                streak -= 1
            else:
                break
        return wr, sf, sa, mg, streak

    def get_h2h(home, away, n=6):
        key = (home, away)
        if key not in team_h2h or not team_h2h[key]:
            return 0.0
        return np.mean(team_h2h[key][-n:])

    def get_venue_wr(team, venue, min_games=3):
        vr = team_venue_wr[team].get(venue, [])
        if len(vr) < min_games:
            return 0.5
        return np.mean(vr)

    def expected_elo(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    round_effect = df.groupby('round.roundNumber')['home_win'].mean().to_dict()
    avg_temp = df['weather.tempInCelsius'].mean()

    feature_rows = []
    labels = []
    y_home_scores = []
    y_away_scores = []
    last_year = None

    for _, row in df.iterrows():
        ht = row['match.homeTeam.name']
        at = row['match.awayTeam.name']
        if pd.isna(ht) or pd.isna(at) or pd.isna(row['home_total']) or pd.isna(row['away_total']):
            continue

        mdt = row['matchDate']
        yr = mdt.year
        if yr != last_year and last_year is not None:
            for t in team_elo:
                team_elo[t] = 0.6 * team_elo[t] + 0.4 * 1500.0
        last_year = yr

        h_rest = min((mdt - team_last_played.get(ht, mdt - pd.Timedelta(days=8))).days, 30)
        a_rest = min((mdt - team_last_played.get(at, mdt - pd.Timedelta(days=8))).days, 30)

        vstate = row.get('venue.state', '')
        h_travel = int(TEAM_STATES.get(ht, '') != vstate)
        a_travel = int(TEAM_STATES.get(at, '') != vstate)

        h_roll = get_rolling(ht, is_home=True)
        a_roll = get_rolling(at, is_home=False)
        h_wr, h_sf, h_sa, h_mg, h_streak = get_form(ht)
        a_wr, a_sf, a_sa, a_mg, a_streak = get_form(at)

        h_elo_adj = team_elo[ht] + HOME_BOOST
        a_elo_val = team_elo[at]
        elo_diff = h_elo_adj - a_elo_val
        elo_exp = expected_elo(h_elo_adj, a_elo_val)
        h2h = get_h2h(ht, at)

        venue = row.get('venue.name', '')
        h_venue_wr = get_venue_wr(ht, venue)
        a_venue_wr = get_venue_wr(at, venue)

        start_dt = pd.to_datetime(row['match.venueLocalStartTime'], errors='coerce')
        start_hour = start_dt.hour if pd.notna(start_dt) else 19
        rnd = row.get('round.roundNumber', 0)
        rnd = rnd if isinstance(rnd, (int, float)) else 0

        h_i50_eff = h_roll.get('goalsTotal', 0) / max(h_roll.get('inside50sTotal', 1), 1)
        a_i50_eff = a_roll.get('goalsTotal', 0) / max(a_roll.get('inside50sTotal', 1), 1)

        feat = {
            'elo_diff': elo_diff, 'elo_expected': elo_exp,
            'home_elo': team_elo[ht], 'away_elo': team_elo[at],
            'home_win_rate': h_wr, 'away_win_rate': a_wr, 'win_rate_diff': h_wr - a_wr,
            'home_score_for': h_sf, 'home_score_against': h_sa,
            'away_score_for': a_sf, 'away_score_against': a_sa,
            'home_margin': h_mg, 'away_margin': a_mg, 'margin_diff': h_mg - a_mg,
            'home_streak': h_streak, 'away_streak': a_streak,
            'h2h_margin': h2h,
            'home_rest': h_rest, 'away_rest': a_rest, 'rest_diff': h_rest - a_rest,
            'home_interstate': h_travel, 'away_interstate': a_travel, 'travel_diff': a_travel - h_travel,
            'home_venue_wr': h_venue_wr, 'away_venue_wr': a_venue_wr, 'venue_wr_diff': h_venue_wr - a_venue_wr,
            'home_i50_eff': h_i50_eff, 'away_i50_eff': a_i50_eff, 'i50_eff_diff': h_i50_eff - a_i50_eff,
            'round_number': rnd, 'round_effect': round_effect.get(rnd, 0.5),
            'temp': float(row.get('weather.tempInCelsius', avg_temp)),
            'start_hour': start_hour,
            'home_games': len(team_results_history[ht]),
            'away_games': len(team_results_history[at]),
        }

        for name in base_stat_names:
            feat[f'diff_{name}'] = h_roll.get(name, 0) - a_roll.get(name, 0)

        feature_rows.append(feat)
        labels.append(row['home_win'])
        y_home_scores.append(row['home_total'])
        y_away_scores.append(row['away_total'])

        # Update AFTER features (no leakage)
        h_stats = {clean_base(c): row.get(c, 0) for c in numeric_team_cols
                   if c.startswith(('homeTeam', 'homeTeamScore', 'homeTeamScoreChart'))}
        a_stats = {clean_base(c): row.get(c, 0) for c in numeric_team_cols
                   if c.startswith(('awayTeam', 'awayTeamScore', 'awayTeamScoreChart'))}
        team_home_history[ht].append(h_stats)
        team_away_history[at].append(a_stats)

        h_won = row['home_win']
        team_results_history[ht].append((row['home_total'], row['away_total'], h_won))
        team_results_history[at].append((row['away_total'], row['home_total'], not h_won))

        team_venue_wr[ht].setdefault(venue, []).append(int(h_won))
        team_venue_wr[at].setdefault(venue, []).append(int(not h_won))

        team_last_played[ht] = mdt
        team_last_played[at] = mdt

        actual = 1.0 if h_won else 0.0
        margin_mult = np.log1p(abs(row['home_total'] - row['away_total'])) / np.log1p(40)
        k = ELO_K * (0.5 + 0.5 * margin_mult)
        team_elo[ht] += k * (actual - elo_exp)
        team_elo[at] += k * ((1 - actual) - (1 - elo_exp))

        key_ab = (ht, at)
        key_ba = (at, ht)
        team_h2h.setdefault(key_ab, []).append(row['home_total'] - row['away_total'])
        team_h2h.setdefault(key_ba, []).append(row['away_total'] - row['home_total'])

    X = pd.DataFrame(feature_rows).fillna(0)
    y = np.array(labels)
    y_home = np.array(y_home_scores)
    y_away = np.array(y_away_scores)

    MIN_HISTORY = 60
    X_model = X.iloc[MIN_HISTORY:]
    y_model = y[MIN_HISTORY:]
    y_home_model = y_home[MIN_HISTORY:]
    y_away_model = y_away[MIN_HISTORY:]

    split_idx = int(len(X_model) * 0.8)
    X_train, X_test = X_model.iloc[:split_idx], X_model.iloc[split_idx:]
    y_train, y_test = y_model[:split_idx], y_model[split_idx:]

    clf = HistGradientBoostingClassifier(
        max_iter=300, max_depth=2, learning_rate=0.05,
        min_samples_leaf=30, l2_regularization=1.5, random_state=42
    )
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    clf_final = HistGradientBoostingClassifier(
        max_iter=300, max_depth=2, learning_rate=0.05,
        min_samples_leaf=30, l2_regularization=1.5, random_state=42
    )
    clf_final.fit(X_model, y_model)

    home_reg = HistGradientBoostingRegressor(
        max_iter=300, max_depth=2, learning_rate=0.05,
        min_samples_leaf=30, l2_regularization=1.5, random_state=42
    )
    away_reg = HistGradientBoostingRegressor(
        max_iter=300, max_depth=2, learning_rate=0.05,
        min_samples_leaf=30, l2_regularization=1.5, random_state=42
    )
    home_reg.fit(X_model, y_home_model)
    away_reg.fit(X_model, y_away_model)

    # Lookups
    team_id_map = {}
    for team in teams:
        home_ids = df[df['match.homeTeam.name'] == team]['match.homeTeam.teamId'].dropna()
        if not home_ids.empty:
            team_id_map[team] = home_ids.iloc[0]
        else:
            away_ids = df[df['match.awayTeam.name'] == team]['match.awayTeam.teamId'].dropna()
            if not away_ids.empty:
                team_id_map[team] = away_ids.iloc[0]

    venue_map = df.groupby('venue.name').agg({
        'venue.venueId': 'first',
        'venue.state': 'first',
        'venue.timeZone': 'first'
    }).to_dict('index')

    venues = sorted(df['venue.name'].dropna().unique().tolist())

    model_data = {
        'clf_final': clf_final,
        'home_reg': home_reg,
        'away_reg': away_reg,
        'X_model': X_model,
        'teams': teams,
        'venues': venues,
        'team_home_history': team_home_history,
        'team_away_history': team_away_history,
        'team_results_history': team_results_history,
        'team_elo': team_elo,
        'team_h2h': team_h2h,
        'team_last_played': team_last_played,
        'team_venue_wr': team_venue_wr,
        'base_stat_names': base_stat_names,
        'round_effect': round_effect,
        'avg_temp': avg_temp,
        'venue_map': venue_map,
        'team_id_map': team_id_map,
        'TEAM_STATES': TEAM_STATES,
        'HOME_BOOST': HOME_BOOST,
        'ROLLING_N': ROLLING_N,
        'EMA_ALPHA': EMA_ALPHA,
        'train_acc': train_acc,
        'test_acc': test_acc,
    }
    return model_data, None


def predict_match(model_data, home_team, away_team, venue_name, temp, round_number, start_datetime_str):
    """Run prediction using the trained model."""
    clf_final = model_data['clf_final']
    home_reg = model_data['home_reg']
    away_reg = model_data['away_reg']
    X_model = model_data['X_model']
    team_home_history = model_data['team_home_history']
    team_away_history = model_data['team_away_history']
    team_results_history = model_data['team_results_history']
    team_elo = model_data['team_elo']
    team_h2h = model_data['team_h2h']
    team_last_played = model_data['team_last_played']
    team_venue_wr = model_data['team_venue_wr']
    base_stat_names = model_data['base_stat_names']
    round_effect = model_data['round_effect']
    avg_temp = model_data['avg_temp']
    venue_map = model_data['venue_map']
    TEAM_STATES = model_data['TEAM_STATES']
    HOME_BOOST = model_data['HOME_BOOST']
    ROLLING_N = model_data['ROLLING_N']
    EMA_ALPHA = model_data['EMA_ALPHA']

    if home_team not in team_home_history or away_team not in team_away_history:
        missing = home_team if home_team not in team_home_history else away_team
        return None, f"Team not found: {missing}"

    def get_rolling(team, is_home=True, n=ROLLING_N):
        history = team_home_history[team] if is_home else team_away_history[team]
        if not history:
            return {name: 0.0 for name in base_stat_names}
        recent = history[-n:]
        result = {}
        for name in base_stat_names:
            vals = [g.get(name, 0) for g in recent]
            ema_val = vals[0]
            for v in vals[1:]:
                ema_val = EMA_ALPHA * v + (1 - EMA_ALPHA) * ema_val
            result[name] = ema_val
        return result

    def get_form(team, n=6):
        history = team_results_history[team]
        if not history:
            return 0.5, 85.0, 85.0, 0.0, 0
        recent = history[-n:]
        wins = sum(w for _, _, w in recent)
        wr = wins / len(recent)
        sf = np.mean([s for s, _, _ in recent])
        sa = np.mean([s for _, s, _ in recent])
        mg = np.mean([sf - sa for sf, sa, _ in recent])
        streak = 0
        for _, _, w in reversed(recent):
            if w and streak >= 0:
                streak += 1
            elif not w and streak <= 0:
                streak -= 1
            else:
                break
        return wr, sf, sa, mg, streak

    def get_h2h(home, away, n=6):
        key = (home, away)
        if key not in team_h2h or not team_h2h[key]:
            return 0.0
        return np.mean(team_h2h[key][-n:])

    def get_venue_wr(team, venue, min_games=3):
        vr = team_venue_wr[team].get(venue, [])
        if len(vr) < min_games:
            return 0.5
        return np.mean(vr)

    def expected_elo(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    h_roll = get_rolling(home_team, is_home=True)
    a_roll = get_rolling(away_team, is_home=False)
    h_wr, h_sf, h_sa, h_mg, h_streak = get_form(home_team)
    a_wr, a_sf, a_sa, a_mg, a_streak = get_form(away_team)
    h_elo_adj = team_elo[home_team] + HOME_BOOST
    a_elo_val = team_elo[away_team]
    elo_diff = h_elo_adj - a_elo_val
    elo_exp = expected_elo(h_elo_adj, a_elo_val)
    h2h = get_h2h(home_team, away_team)

    venue = venue_name
    h_venue_wr = get_venue_wr(home_team, venue)
    a_venue_wr = get_venue_wr(away_team, venue)

    start_dt = pd.to_datetime(start_datetime_str, errors='coerce')
    start_hour = start_dt.hour if pd.notna(start_dt) else 19
    rnd = int(round_number) if str(round_number).isdigit() else 0

    h_rest = min(max((start_dt - team_last_played.get(home_team, start_dt - pd.Timedelta(days=7))).days, 0), 30)
    a_rest = min(max((start_dt - team_last_played.get(away_team, start_dt - pd.Timedelta(days=7))).days, 0), 30)

    vstate = venue_map.get(venue, {}).get('venue.state', 'VIC')
    h_travel = int(TEAM_STATES.get(home_team, '') != vstate)
    a_travel = int(TEAM_STATES.get(away_team, '') != vstate)

    h_i50_eff = h_roll.get('goalsTotal', 0) / max(h_roll.get('inside50sTotal', 1), 1)
    a_i50_eff = a_roll.get('goalsTotal', 0) / max(a_roll.get('inside50sTotal', 1), 1)

    feat = {
        'elo_diff': elo_diff, 'elo_expected': elo_exp,
        'home_elo': team_elo[home_team], 'away_elo': team_elo[away_team],
        'home_win_rate': h_wr, 'away_win_rate': a_wr, 'win_rate_diff': h_wr - a_wr,
        'home_score_for': h_sf, 'home_score_against': h_sa,
        'away_score_for': a_sf, 'away_score_against': a_sa,
        'home_margin': h_mg, 'away_margin': a_mg, 'margin_diff': h_mg - a_mg,
        'home_streak': h_streak, 'away_streak': a_streak,
        'h2h_margin': h2h,
        'home_rest': h_rest, 'away_rest': a_rest, 'rest_diff': h_rest - a_rest,
        'home_interstate': h_travel, 'away_interstate': a_travel, 'travel_diff': a_travel - h_travel,
        'home_venue_wr': h_venue_wr, 'away_venue_wr': a_venue_wr, 'venue_wr_diff': h_venue_wr - a_venue_wr,
        'home_i50_eff': h_i50_eff, 'away_i50_eff': a_i50_eff, 'i50_eff_diff': h_i50_eff - a_i50_eff,
        'round_number': rnd, 'round_effect': round_effect.get(rnd, 0.5),
        'temp': float(temp) if temp is not None else avg_temp,
        'start_hour': start_hour,
        'home_games': len(team_results_history[home_team]),
        'away_games': len(team_results_history[away_team]),
    }

    for name in base_stat_names:
        feat[f'diff_{name}'] = h_roll.get(name, 0) - a_roll.get(name, 0)

    X_new = pd.DataFrame([feat]).reindex(columns=X_model.columns, fill_value=0)

    proba = clf_final.predict_proba(X_new)[0]
    home_prob = proba[1]
    winner = home_team if home_prob > 0.5 else away_team
    confidence = max(home_prob, 1 - home_prob) * 100

    pred_home = round(home_reg.predict(X_new)[0])
    pred_away = round(away_reg.predict(X_new)[0])
    margin = abs(pred_home - pred_away)

    return {
        'winner': winner,
        'confidence': round(confidence, 1),
        'home_prob': round(home_prob * 100, 1),
        'away_prob': round((1 - home_prob) * 100, 1),
        'pred_home': pred_home,
        'pred_away': pred_away,
        'margin': margin,
        'home_elo': round(team_elo[home_team]),
        'away_elo': round(team_elo[away_team]),
        'h_wr': round(h_wr * 100, 1),
        'a_wr': round(a_wr * 100, 1),
        'h_streak': h_streak,
        'a_streak': a_streak,
        'h_travel': h_travel,
        'a_travel': a_travel,
        'h_rest': h_rest,
        'a_rest': a_rest,
    }, None


# ─────────────────────────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏉 AFL Claude</h1>
    <p>Machine-learning match forecaster · HistGradientBoosting · ELO + rolling stats</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Loading model and training on historical data…"):
    model_data, load_error = load_model()

if load_error:
    st.markdown(f'<div class="error-card">⚠️ {load_error}</div>', unsafe_allow_html=True)
    st.stop()

teams = model_data['teams']
venues = model_data['venues']
train_acc = model_data['train_acc']
test_acc = model_data['test_acc']

# Model accuracy badge
col_acc1, col_acc2, col_acc3 = st.columns(3)
with col_acc1:
    st.metric("Model Train Accuracy", f"{train_acc:.1%}")
with col_acc2:
    st.metric("Model Test Accuracy (last 20%)", f"{test_acc:.1%}")
with col_acc3:
    st.metric("Teams Available", len(teams))

st.markdown("---")

# ─── Input form ───────────────────────────────────────────────────────────────
with st.form("forecast_form"):
    st.markdown("### Match Details")

    # Teams
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">🏠 Home Team</div>', unsafe_allow_html=True)
        home_team = st.selectbox(
            "Home Team",
            options=teams,
            index=teams.index("Sydney Swans") if "Sydney Swans" in teams else 0,
            label_visibility="collapsed",
        )
    with col2:
        st.markdown('<div class="section-title">✈️ Away Team</div>', unsafe_allow_html=True)
        away_team = st.selectbox(
            "Away Team",
            options=teams,
            index=teams.index("Carlton") if "Carlton" in teams else 1,
            label_visibility="collapsed",
        )

    st.markdown("---")

    # Venue & Round
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-title">📍 Venue</div>', unsafe_allow_html=True)
        venue_name = st.selectbox(
            "Venue",
            options=venues,
            index=0,
            label_visibility="collapsed",
        )
    with col4:
        st.markdown('<div class="section-title">🔢 Round Number</div>', unsafe_allow_html=True)
        round_number = st.number_input(
            "Round Number",
            min_value=0,
            max_value=30,
            value=1,
            step=1,
            label_visibility="collapsed",
        )

    st.markdown("---")

    # Date, Time & Weather
    col5, col6, col7 = st.columns(3)
    with col5:
        st.markdown('<div class="section-title">📅 Match Date</div>', unsafe_allow_html=True)
        match_date = st.date_input(
            "Match Date",
            value=datetime.date(2026, 3, 14),
            label_visibility="collapsed",
        )
    with col6:
        st.markdown('<div class="section-title">🕐 Local Start Time</div>', unsafe_allow_html=True)
        match_time = st.time_input(
            "Local Start Time",
            value=datetime.time(19, 50),
            step=60,
            label_visibility="collapsed",
        )
    with col7:
        st.markdown('<div class="section-title">🌡️ Temperature (°C)</div>', unsafe_allow_html=True)
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10,
            max_value=50,
            value=21,
            step=1,
            label_visibility="collapsed",
        )

    st.markdown("---")

    submitted = st.form_submit_button("🔮  Generate Forecast", use_container_width=True)


# ─── Results ──────────────────────────────────────────────────────────────────
if submitted:
    if home_team == away_team:
        st.markdown('<div class="error-card">⚠️ Home and away teams must be different.</div>', unsafe_allow_html=True)
    else:
        start_datetime_str = f"{match_date}T{match_time.strftime('%H:%M:%S')}"

        with st.spinner("Running forecast…"):
            result, err = predict_match(
                model_data,
                home_team,
                away_team,
                venue_name,
                temperature,
                round_number,
                start_datetime_str,
            )

        if err:
            st.markdown(f'<div class="error-card">⚠️ {err}</div>', unsafe_allow_html=True)
        else:
            winner = result['winner']
            confidence = result['confidence']
            home_prob = result['home_prob']
            away_prob = result['away_prob']
            pred_home = result['pred_home']
            pred_away = result['pred_away']
            margin = result['margin']

            # Winner highlight
            winner_is_home = winner == home_team
            home_label = f"🏠 {home_team}"
            away_label = f"✈️ {away_team}"

            st.markdown("### 📊 Forecast Result")

            res_col1, res_col2 = st.columns([3, 2])

            with res_col1:
                st.markdown(f"""
<div class="result-card">
    <div class="result-winner">{'🏆 ' + winner}</div>
    <div class="result-confidence">Predicted winner · {confidence:.1f}% confidence</div>

    <div class="score-display">
        <div class="score-team">
            <div class="score-team-name">{home_label}</div>
            <div class="score-value" style="color: {'#34d399' if winner_is_home else '#ffffff'}">{pred_home}</div>
        </div>
        <div class="score-divider">vs</div>
        <div class="score-team">
            <div class="score-team-name">{away_label}</div>
            <div class="score-value" style="color: {'#ffffff' if winner_is_home else '#34d399'}">{pred_away}</div>
        </div>
    </div>

    <div style="text-align:center; color: rgba(255,255,255,0.5); font-size:0.85rem; margin-bottom:1rem;">
        Predicted margin: <strong style="color:white">{margin} points</strong>
    </div>

    <div class="confidence-bar-container">
        <div class="confidence-bar-label">
            <span>{home_team} {home_prob}%</span>
            <span>{away_team} {away_prob}%</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{home_prob}%"></div>
        </div>
    </div>

    <div class="stat-grid">
        <div class="stat-item">
            <div class="stat-label">Home ELO</div>
            <div class="stat-value">{result['home_elo']}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Away ELO</div>
            <div class="stat-value">{result['away_elo']}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Margin</div>
            <div class="stat-value">{margin} pts</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Home Win %</div>
            <div class="stat-value">{result['h_wr']}%</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Away Win %</div>
            <div class="stat-value">{result['a_wr']}%</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Venue</div>
            <div class="stat-value" style="font-size:0.75rem">{venue_name}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

            with res_col2:
                st.markdown("#### Match Context")

                streak_h = result['h_streak']
                streak_a = result['a_streak']
                streak_h_str = f"{'🔥 ' if streak_h > 0 else '❄️ '}{abs(streak_h)} {'W' if streak_h > 0 else 'L'} streak"
                streak_a_str = f"{'🔥 ' if streak_a > 0 else '❄️ '}{abs(streak_a)} {'W' if streak_a > 0 else 'L'} streak"

                travel_h = "✈️ Interstate" if result['h_travel'] else "🏠 Home state"
                travel_a = "✈️ Interstate" if result['a_travel'] else "🏠 Home state"

                st.markdown(f"""
| | {home_team} | {away_team} |
|---|---|---|
| **ELO** | {result['home_elo']} | {result['away_elo']} |
| **Win Rate** | {result['h_wr']}% | {result['a_wr']}% |
| **Streak** | {streak_h_str} | {streak_a_str} |
| **Travel** | {travel_h} | {travel_a} |
| **Rest (days)** | {result['h_rest']} | {result['a_rest']} |
""")

                st.markdown("#### Match Info")
                st.markdown(f"""
- 📅 **Date:** {match_date.strftime('%d %b %Y')}
- 🕐 **Start:** {match_time.strftime('%H:%M')} local
- 📍 **Venue:** {venue_name}
- 🌡️ **Temp:** {temperature}°C
- 🔢 **Round:** {round_number}
""")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.8rem;'>"
    "AFL Claude · HistGradientBoosting · ELO ratings · Rolling 8-game EMA stats"
    "</p>",
    unsafe_allow_html=True,
)
