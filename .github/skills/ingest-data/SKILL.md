---
name: ingest-data
description: "Use when: running data ingestion (`sports-ml ingest`), verifying upstream feed connectivity (`sports-ml check-sources`), backfilling historical NHL or EPL seasons from a starting year (`--from-year`), syncing the active season to date (`--current-season`), fetching a specific date window (`--start-date` / `--end-date`), or reconciling raw caches into Silver Parquet tables for `--sport hockey` or `--sport soccer`."
argument-hint: "Specify --sport (hockey|soccer), --from-year (e.g., 2023), --current-season, --seasons (e.g., 20232024,20242025 or 2023-2026), or check-sources..."
user-invocable: true
disable-model-invocation: false
---

# Ingest Data Workflow (`/ingest-data`)

## When to Use
- When the user invokes `/ingest-data` or asks to pull, backfill, refresh, or reconcile NHL or EPL schedules, boxscores, rosters, and advanced stats into `data/silver/<sport>/`.
- When verifying live connectivity and schema contracts across external feeds (`sports-ml check-sources`).
- Before initial model training or when backfilling historical seasons from a starting year through the active season.

---

## Step-by-Step Execution Procedure

### Step 1: Select Ingestion Scope
Choose the appropriate ingestion mode based on the user's request:
1. **Upstream Connectivity & Schema Check (`sports-ml check-sources`)**:
   - Probes live HTTP reachability and Polars/JSON schema contracts across the Official NHL Web API (`/v1/schedule/now`, `/v1/roster/{TEAM}/current`), MoneyPuck CSVs, and DailyFaceoff `__NEXT_DATA__` (or FPL/Understat/Football-Data for `--sport soccer`).
2. **All Seasons From Year to Current (`--from-year YYYY` or `--since YYYY`)**:
   - Expands all seasons from the given start year (e.g., `--from-year 2023` or `--from-year 20232024`) through the active season, downloads/caches all historical seasons, and automatically includes the `--current-season` incremental sync.
3. **Current Season Incremental Sync (`--current-season`)**:
   - Pulls recent schedules and newly completed `FINAL` boxscores for the active season, refreshes current-season advanced stat feeds, and upserts into Silver Parquet tables.
4. **Date Range Sync (`--start-date YYYY-MM-DD --end-date YYYY-MM-DD`)**:
   - Fetches all NHL schedule weeks and completed boxscores within `[start_date, end_date]` and upserts them into `data/silver/hockey/`.
5. **Explicit Season List or Range (`--seasons 20232024,20242025` or `--seasons 2023-2026` / `2023+`)**:
   - Downloads/caches the specified seasons or season range and reconciles them into Silver tables.
6. **Offline Cache Reconciliation (`--offline`)**:
   - Re-parses and reconciles existing raw JSON and CSV files into Silver Parquet tables without making network requests.

### Step 2: Run `sports-ml check-sources` or `sports-ml ingest`
Execute the appropriate command via bash:

```bash
# Verify live connectivity and schema contracts (NHL Hockey or EPL Soccer)
uv run sports-ml check-sources --sport hockey
uv run sports-ml check-sources --sport soccer

# Ingest all seasons from a starting year through the current active season
uv run sports-ml ingest --sport hockey --from-year 2023
uv run sports-ml ingest --sport soccer --from-year 2023

# Sync active season to date only
uv run sports-ml ingest --sport hockey --current-season
uv run sports-ml ingest --sport soccer --current-season

# Sync a specific date window (Hockey)
uv run sports-ml ingest --sport hockey --start-date 2024-10-10 --end-date 2024-10-24

# Backfill explicit seasons or a season range
uv run sports-ml ingest --sport hockey --seasons 2023-2026 --current-season
uv run sports-ml ingest --sport soccer --seasons 20232024,20242025 --current-season

# Reconcile local raw cache into Silver tables offline
uv run sports-ml ingest --sport hockey --offline
uv run sports-ml ingest --sport soccer --offline
```

### Step 3: Verify Silver Lakehouse Tables
After ingestion completes, verify the 4 canonical Silver Parquet tables under `data/silver/hockey/`:
- `games.parquet` (unique on `game_id`)
- `team_game_logs.parquet` (unique on `(game_id, team)`)
- `skater_game_logs.parquet` (unique on `(game_id, player_id)`)
- `goalie_game_logs.parquet` (unique on `(game_id, player_id)`)

Report the row counts, date range (`min(game_date)` to `max(game_date)`), and whether the user should run `/train-models` or `/predict-matchup` next.
