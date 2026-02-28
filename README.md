# 🏉 AFL Claude – Match Forecaster

A modern Streamlit web application that uses a machine-learning model (HistGradientBoosting + ELO ratings + rolling EMA stats) to forecast AFL match outcomes before a game is played.

Served at: `https://andrewsayer.me/afl-claude`

---

## How It Works

The model is trained on historical AFL match data from `ReadyFor2026.csv`. For each match it builds features including:

- **ELO ratings** (with home-ground advantage boost of +55)
- **Rolling 8-game EMA statistics** (separate home/away histories)
- **Form** – win rate, score for/against, margin, win/loss streak
- **Head-to-head** margin history
- **Rest days** between games
- **Interstate travel** flag
- **Venue win rate** history
- **Inside-50 efficiency**
- **Round number** and historical round effect
- **Temperature** and **start hour**

Two models are trained:
1. **Win classifier** – `HistGradientBoostingClassifier` predicts which team wins and with what probability
2. **Score regressors** – separate `HistGradientBoostingRegressor` models predict the home and away scores

---

## Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) (recommended) **or** `pip`
- `ReadyFor2026.csv` placed in the project root directory

---

## Setup & Deployment

### 1. Clone the repository

```bash
git clone https://github.com/andrewsayer/afl-claude.git
cd afl-claude
```

### 2. Add the data file

Place your `ReadyFor2026.csv` file in the project root (same directory as `app.py`).

```
afl-claude/
├── app.py
├── requirements.txt
├── README.md
└── ReadyFor2026.csv   ← place here
```

### 3. Create a virtual environment and install dependencies

#### Using `uv` (recommended)

```bash
uv venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

uv pip install -r requirements.txt
```

#### Using standard `pip`

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` by default.

---

## Serving at a Sub-path (e.g. `/afl-claude`)

To serve the app at `https://andrewsayer.me/afl-claude`, configure your reverse proxy (nginx/Caddy) to forward requests and set the base URL:

```bash
streamlit run app.py \
  --server.baseUrlPath="/afl-claude" \
  --server.port=8501
```

### Example nginx config

```nginx
location /afl-claude {
    proxy_pass         http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header   Upgrade $http_upgrade;
    proxy_set_header   Connection "upgrade";
    proxy_set_header   Host $host;
    proxy_set_header   X-Real-IP $remote_addr;
    proxy_read_timeout 86400;
}
```

### Running as a systemd service

Create `/etc/systemd/system/afl-claude.service`:

```ini
[Unit]
Description=AFL Claude Streamlit App
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/afl-claude
ExecStart=/path/to/afl-claude/.venv/bin/streamlit run app.py \
          --server.baseUrlPath="/afl-claude" \
          --server.port=8501 \
          --server.headless=true
Restart=always

[Install]
WantedBy=multi-user.target
```

Then enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable afl-claude
sudo systemctl start afl-claude
```

---

## Using the Forecaster

When the app loads it will automatically:
1. Read `ReadyFor2026.csv`
2. Build rolling features for all historical matches (no data leakage)
3. Train the win classifier and score regressors
4. Display model accuracy metrics at the top of the page

### Input Form Fields

| Field | Description | Example |
|-------|-------------|---------|
| **Home Team** | Select the home team from the dropdown | `Sydney Swans` |
| **Away Team** | Select the away team from the dropdown | `Carlton` |
| **Venue** | Select the venue from the dropdown | `SCG` |
| **Round Number** | Integer round number (0 = Opening Round) | `1` |
| **Match Date** | Date of the match | `2026-03-14` |
| **Local Start Time** | Venue local start time (HH:MM) | `20:05` |
| **Temperature (°C)** | Forecast temperature as an integer | `21` |

> **Note on start time:** The time picker supports minute-level precision. For a 20:05 start, simply set the time to `20:05`. The model uses the hour component to capture day/night game effects.

> **Note on temperature:** Enter a whole number (integer). The model uses this to account for weather effects on scoring.

### Interpreting Results

After clicking **Generate Forecast**, the app displays:

- **Predicted winner** with confidence percentage
- **Predicted scoreline** (home score vs away score)
- **Predicted margin** in points
- **Win probability bar** showing the split between home and away
- **Comparison table** with ELO ratings, win rates, streaks, travel status, and rest days
- **Match context** summary (date, time, venue, temperature, round)

#### Understanding the metrics

| Metric | What it means |
|--------|---------------|
| **ELO** | Team strength rating (1500 = average; higher = stronger) |
| **Win Rate** | Recent form win percentage (last 6 games) |
| **Streak** | Consecutive wins (🔥) or losses (❄️) |
| **Travel** | Whether the team is playing interstate |
| **Rest days** | Days since the team's last match |
| **Confidence** | How strongly the model favours the predicted winner |

A confidence of **50%** means the model sees the match as a coin flip. A confidence of **75%+** indicates a strong favourite.

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor` |
| Max iterations | 300 |
| Max depth | 2 |
| Learning rate | 0.05 |
| Min samples leaf | 30 |
| L2 regularisation | 1.5 |
| ELO K-factor | 32 |
| Home advantage boost | +55 ELO points |
| Rolling window | 8 games (EMA α=0.35) |
| Min history before training | 60 matches |
| Train/test split | 80% / 20% (chronological) |

---

## File Structure

```
afl-claude/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── ReadyFor2026.csv    # Historical match data (not included in repo)
```

---

## Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
```

---

## Notes

- `ReadyFor2026.csv` is **not** included in the repository (add it to `.gitignore` if it contains sensitive data, or include it if it is public).
- The model is retrained from scratch each time the app starts (cached per session via `@st.cache_resource`).
- ELO ratings are reset 40% toward 1500 at the start of each new season to account for off-season changes.
