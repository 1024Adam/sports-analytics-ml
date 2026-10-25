---
name: nhl-data-integration
description: "Use when: implementing, parsing, debugging, or creating test fixtures for NHL data ingestion, including Official NHL Web API (api-web.nhle.com/v1) schedules/boxscores/rosters, MoneyPuck CSV datasets, 2-Tier pre-game starting goalie/lineup/injury feeds, team tricode normalization, or Silver Parquet reconciliation."
argument-hint: "Specify the NHL endpoint, MoneyPuck dataset, or reconciliation logic to implement or debug..."
user-invocable: true
disable-model-invocation: false
---

# NHL Data Integration Playbook

## When to Use
- Implementing or updating `src/sports_ml/sports/hockey/ingestion/` (`nhl_api.py`, `moneypuck.py`, `lineups.py`, `reconciler.py`) or `constants.py`.
- Creating deterministic offline JSON/CSV fixtures in `tests/sports/hockey/test_ingestion.py`.
- Debugging schema mismatches, missing advanced stats, or team/player ID mapping issues.

---

## 1. Critical Rule: Never Use the Deprecated NHL API
- **DEAD API (Do NOT use)**: `https://statsapi.web.nhl.com/api/v1/...` (shut down in late 2023).
- **ACTIVE API (Always use)**: `https://api-web.nhle.com/v1/...`

---

## 2. Official NHL Web API (`api-web.nhle.com/v1/`) Reference

### 2.1 Endpoints & Caching Policy
| Endpoint | Purpose | Cache Policy (`data/raw/nhl_api/`) |
| :--- | :--- | :--- |
| `GET /v1/schedule/{YYYY-MM-DD}` | Returns 7-day `gameWeek` array starting on `{YYYY-MM-DD}`. | Cache permanently only if all games in week are `FINAL`/`OFF`; refresh if any game is `FUT`/`PRE`/`LIVE`. |
| `GET /v1/schedule/now` | Active season detection and today's slate. | Never cache permanently (TTL $\le 15$ min). |
| `GET /v1/club-schedule-season/{TEAM}/{SEASON}` | Full season schedule for `{TEAM}` (e.g., `BOS`, `20242025`). | Refresh for active season; permanent for past seasons. |
| `GET /v1/gamecenter/{GAME_ID}/boxscore` | Official skater & goalie boxscore stats, TOI, starter flags. | Cache permanently once `gameState in {"FINAL", "OFF"}`. |
| `GET /v1/gamecenter/{GAME_ID}/landing` | Period-by-period scoring, shootout details, pre-game matchup context. | Cache permanently once `FINAL`/`OFF`. |
| `GET /v1/roster/{TEAM}/current` | Current active 23-man NHL roster (`forwards`, `defensemen`, `goalies`). | Short TTL (1–6 hours) during active season. |

### 2.2 NHL API Payload Parsing Rules
1. **Game ID & Game Type Filtering**:
   - `id`: 10-digit integer (e.g., `2024020105`).
   - `season`: 8-digit integer (e.g., `20242025`).
   - `gameType`: `2` = Regular Season, `3` = Playoffs (`1` = Preseason — exclude from training/Silver tables by default).
   - `gameState`: `"FINAL"` or `"OFF"` means completed; `"FUT"` or `"PRE"` means upcoming; `"LIVE"` or `"CRIT"` means in progress.
2. **Regulation/OT Goals vs. Shootout Winner Goal**:
   - When `periodDescriptor.periodType == "SO"` (or `gameOutcome.lastPeriodType == "SO"`), the NHL adds `+1` to the winning team's final `score` (`home_goals` / `away_goals`).
   - Always compute `home_reg_goals` and `away_reg_goals` by subtracting `1` from the shootout winner's final score so team goal-rate models are not distorted by the shootout skills competition.
3. **Boxscore Player Nesting (`playerByGameStats`)**:
   - Located at `boxscore["playerByGameStats"]["awayTeam"]` and `boxscore["playerByGameStats"]["homeTeam"]`.
   - Each team object contains arrays: `forwards`, `defense`, and `goalies`.
   - **Time on Ice (`toi`) Parsing**: Returned as `"MM:SS"` string (e.g., `"19:45"`). Convert to decimal minutes via:
     $$\text{toi\_min} = \text{int}(\text{MM}) + \frac{\text{int}(\text{SS})}{60.0}$$
   - **Starting Goalie Flag**: In `goalies`, check `starter: bool` (fallback: goalie with highest `toi` and earliest first-period shift if `starter` key is absent on legacy games).

---

## 3. MoneyPuck CSV Integration (`moneypuck.py`)

### 3.1 Data Structure & Situations
MoneyPuck game-by-game records contain 4 rows per team/player per game corresponding to `situation in {"all", "5on5", "5on4", "4on5"}`:
- **`5on5`**: Extract `xGoalsFor` ($\text{xGF}_{5v5}$), `xGoalsAgainst` ($\text{xGA}_{5v5}$), `shotAttemptsFor` ($\text{CF}_{5v5}$), `shotAttemptsAgainst` ($\text{CA}_{5v5}$), `unblockedShotAttemptsFor` ($\text{FF}_{5v5}$), `unblockedShotAttemptsAgainst` ($\text{FA}_{5v5}$), `highDangerxGoalsFor`, `highDangerxGoalsAgainst`, and `iceTime` (Note: MoneyPuck `iceTime` is in **seconds**—divide by `60.0` to get `toi_5v5_minutes`!).
- **`5on4` (Power Play)** & **`4on5` (Penalty Kill)**: Extract `xGoalsFor`, `goalsFor`, `iceTime / 60.0`.
- **Goalies (`all` situation)**: Extract `xGoals` ($\text{xGA}_{\text{all}}$), `goals` ($\text{GA}$), and compute $\text{GSAx} = \text{xGA}_{\text{all}} - \text{GA}$.

### 3.2 Canonical Team Tricode Normalization (`constants.py`)
Always normalize team codes before joining NHL API and MoneyPuck tables:
```python
TRICODE_ALIASES: dict[str, str] = {
    "S.J": "SJS",
    "L.A": "LAK",
    "N.J": "NJD",
    "T.B": "TBL",
    "ARI": "UTA",  # Historical Arizona Coyotes mapped to Utah franchise continuity (configurable)
    "PHX": "UTA",
    "ATL": "WPG",
}
```

---

## 4. Two-Tier Pre-Game Lineup & Goalie Waterfall (`lineups.py`)

When `sports-ml predict` runs for an upcoming matchup:
1. **Tier 1 (Official NHL Warmup/Pre-Game Lineup)**:
   - Query `/v1/gamecenter/{GAME_ID}/boxscore` and `/landing`.
   - If dressed skaters (`forwards` + `defense`) and a `starter == True` goalie are populated, return them with `goalie_status = "CONFIRMED_NHL_OFFICIAL"`.
2. **Tier 2 (Daytime Morning-Skate Feed + Active Roster)**:
   - Fetch `/v1/roster/{TEAM}/current` for valid active `player_id`s.
   - Fetch daytime starting goalie & line combinations / injury list (e.g., DailyFaceoff starting goalies & team line combinations).
   - Match player names to NHL 7-digit `player_id`s (using normalized case/diacritic stripping + team filter), exclude injured/scratched players, and tag goalie status as `"CONFIRMED_MORNING_SKATE"` or `"EXPECTED_MORNING_SKATE"`.
   - If predicting future days before morning skates are posted, fall back to the active roster's most recent game lineup and primary non-B2B starter (`"PROJECTED_HEURISTIC"`).
