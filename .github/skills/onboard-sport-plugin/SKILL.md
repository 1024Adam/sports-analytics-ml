---
name: onboard-sport-plugin
description: "Use when: designing or scaffolding a new sport plugin (such as Baseball MLB in sports/baseball/ or Golf PGA in sports/golf/) onto the sport-agnostic core engine (src/sports_ml/core/)."
argument-hint: "Sport name (e.g., baseball / mlb, golf / pga) to design or scaffold..."
user-invocable: true
disable-model-invocation: false
---

# Onboard New Sport Plugin Playbook

## When to Use
- When extending `sports-analytics-ml` beyond NHL Hockey to a new sport such as **Soccer (EPL)**, **Baseball (MLB)**, or **Golf (PGA)**.
- Use with `sports-ml-architect` when designing the new sport's data/feature spec in `docs/`, or with `sports-ml-developer` when scaffolding `src/sports_ml/sports/<sport>/`.

---

## 1. Pre-Onboarding Core Compatibility Check
Verify which competition topology and exposure basis the new sport uses in `src/sports_ml/core/`:

| Sport | Topology (`contracts.py`) | Temporal Ordering Key | Exposure Basis (`pregame_exposure_rate`) | Specialist / Venue Hooks |
| :--- | :--- | :--- | :--- | :--- |
| **Soccer (EPL)** | `head_to_head` (3-way `1X2`: Home Win, Draw, Away Win) | `(start_time_utc, event_seq, event_id)` | Outfield & GK: per 90 min (`scale=90.0`) | Starting Goalkeeper ($\text{GSAA}/90$, $\text{PSxG}-\text{GA}$), Promoted-Club Elo Prior, Set-Piece/Penalty Takers |
| **Baseball (MLB)** | `head_to_head` | `(start_time_utc, event_seq, event_id)` (`event_seq=1,2` for doubleheaders) | Batters: per PA (`scale=1.0`); Pitchers: per 9 IP (`scale=9.0`) | Starting Pitcher ($xFIP$, Stuff+, K-BB%), Bullpen 3-day workload, Ballpark & Weather factors |
| **Golf (PGA)** | `multi_entrant_field` (tournament) & `head_to_head` (matchups) | `(start_time_utc, event_seq, event_id)` (`event_seq=1..4` for Rounds 1–4) | Per 18-hole round (`scale=1.0`) for Strokes Gained (`SG:OTT`, `SG:APP`, `SG:ATG`, `SG:PUTT`) | Course-fit weights (driving distance/accuracy, approach buckets, turf type) & Field Strength |

---

## 2. Standard Plugin Directory Skeleton
Create the plugin under `src/sports_ml/sports/<sport>/` mirroring the hockey layout:

```text
src/sports_ml/sports/<sport>/
├── __init__.py
├── adapter.py           # Implements SportAdapter protocol from core/contracts.py
├── constants.py         # Team/player registries, venue coordinates, park/course factors
├── ingestion/
│   ├── api_client.py    # Cached HTTP client (e.g., MLB StatsAPI / Baseball Savant / PGA feeds)
│   ├── lineups.py       # 2-Tier live pre-event starters/lineups/tee-times & injury resolver
│   └── reconciler.py    # Normalizes raw feeds into data/silver/<sport>/*.parquet
├── features/
│   ├── entity.py        # Team or golfer rolling efficiency & schedule features
│   ├── specialist.py    # Starting pitcher / bullpen (MLB) or course-fit (PGA) features
│   └── participant.py   # Stage 2 individual player stat projection features
└── models/
    ├── stage1_event.py  # Mandatory calibrated win / tournament finish probabilities
    ├── stage2_player.py # Optional participant stat projections conditioned on Stage 1
    └── predictor.py     # Unified two-stage predictor registered in cli.py
```

---

## 3. Onboarding Checklist
1. **Documentation First**: Create `docs/<sport>-data-and-features.md` defining Silver Parquet schemas, Shrinkage constants $k$, and Stage 1 / Stage 2 targets.
2. **Implement `SportAdapter`**: Register the new sport slug (`--sport baseball` or `--sport golf`) in `src/sports_ml/cli.py`.
3. **Mandatory Offline Tests (`tests/sports/<sport>/`)**:
   - Future-Mutation Anti-Leakage test (including MLB doubleheader Game 1 $\rightarrow$ Game 2 or PGA Round $r \rightarrow r+1$).
   - Stage 1 probability sum-to-one test ($\sum P = 1.0$ per game or across tournament field).
   - Stage 2 non-negativity and Stage 1 environment conditioning test.
