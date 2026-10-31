# MLB & NPB Baseball Data Contracts & Feature Engineering Specification

> **Related Documents**: [System Architecture Blueprint](./architecture.md) | [Hockey Data & Feature Specification](./hockey-data-and-features.md) | [EPL Soccer Data & Feature Specification](./soccer-data-and-features.md) | [Implementation Roadmap](./roadmap.md)

## 1. Unified Multi-League Baseball Architecture (`MLB` + `NPB`)

The baseball plugin (`src/sports_ml/sports/baseball/`) models both **Major League Baseball (`league = "MLB"`, 30 clubs across AL & NL)** and **Nippon Professional Baseball (`league = "NPB"`, 12 clubs across Central & Pacific Leagues)** within a single cohesive domain architecture (`--sport baseball --league mlb|npb|all`), while respecting fundamental structural differences between the two leagues:

| Dimension | Major League Baseball (`MLB`) | Nippon Professional Baseball (`NPB`) | Architectural Handling |
| :--- | :--- | :--- | :--- |
| **Season Format** |Single-year calendar (`season = 2026`, 162 games/club, late March–Sept + Postseason) | Single-year calendar (`season = 2026`, 143 games/club, late March–Oct + Climax Series/Japan Series) | 4-digit calendar year (`2023`, `2024`, `2025`, `2026`) or normalized `Int32` season key partitioned by `league` |
| **Extra-Inning / Tie Rule** | Unlimited extra innings with automatic runner on 2B from 10th inning (**0% ties** in regular season) | **Hard 12-inning cap**: game ends in a **Tie/Draw** if tied after 12 innings (~2.5%–3.5% of games) | **League-Aware Stage 1 Head**: Binary 2-Way (`P(Home)`, `P(Away)`) for `MLB`; 3-Way (`P(Home)`, `P(Draw)`, `P(Away)`) + RPS for `NPB` |
| **Sub-Leagues & DH Rules** | American League (`AL`) & National League (`NL`); Universal DH since 2022 | Pacific League (`PL`, DH) & Central League (`CL`, traditional pitcher batting historically / DH rule context during Interleague _Koryusen_) | Explicit `sub_league`, `uses_dh`, and `is_interleague` flags on every game |
| **Run Environment & Ball** | Higher barrel/HR rate ($\sim 4.4\text{--}4.6$ R/G per team) | Lower scoring / higher contact environment ($\sim 3.4\text{--}3.9$ R/G per team; tackling "dead ball" / high-pitching eras) | League-partitioned priors ($\mu_{\text{MLB}}, \mu_{\text{NPB}}$), separate Negative-Binomial dispersion $\alpha_{\text{league}}$, and league-specific park factors |
| **Venue & Schedule Topology** | Fixed 30 home ballparks + frequent doubleheaders (`event_seq = 1, 2`) + cross-continent 3-timezone travel | 12 primary home stadiums **plus** regional/provincial neutral games (Miyazaki, Matsuyama, Toyama, Tokyo Dome) & Hanshin Tigers' August _Koshien_ road trip; Monday off-days & 6-man starting rotations | Dynamic `(venue_id, venue_lat, venue_lon, is_neutral_regional_venue, is_dome)` lookup per game rather than static home-team stadium assumption |
| **Player Movement** | Domestic call-ups + international signings (`NPB -> MLB` posting/free agency) | Domestic roster (`1st-gun` / `ichi-gun` vs. `ni-gun` farm) + foreign signings (`MLB/AAA -> NPB` _suketto_) | **Cross-League Translation Priors (`translation.py`)** mapping prior-season NPB stats to MLB equivalents (and vice versa) for cold-start Bayesian shrinkage |

---

## 2. Hybrid MLB & NPB Data Source Strategy

### 2.1 MLB Sources (`league = "MLB"`)
1. **Source A1 — Official MLB StatsAPI (`https://statsapi.mlb.com/api/v1/`)**:
   * **Role**: Canonical source of truth for the 30 MLB clubs, schedules (including split/traditional doubleheaders via `gameNumber` $\in \{1, 2\}$ and rescheduled suspended games), probable starting pitchers (`hydrate=probablePitcher,lineups,venue,weather`), confirmed 1–9 batting orders, batter/pitcher handedness (`batSide`, `pitchHand`), boxscores (`outs_pitched`, `pitches_thrown`, `strikes`, `bf`, `hits`, `runs`, `earned_runs`, `bb`, `k`, `hr`, `sb`), and active 26-man rosters / injured list (`10-day`, `15-day`, `60-day IL`).
   * **Key Endpoints**:
     * `GET /api/v1/schedule?sportId=1&startDate={YYYY-MM-DD}&endDate={YYYY-MM-DD}&hydrate=probablePitcher,lineups,team,venue,weather,decisions`
     * `GET /api/v1/game/{gamePk}/boxscore` & `GET /api/v1/game/{gamePk}/feed/live`
     * `GET /api/v1/teams/{teamId}/roster/active`
2. **Source A2 — Baseball Savant / Statcast & FanGraphs Exports (`baseballsavant.mlb.com`)**:
   * **Role**: Advanced contact-quality and pitch-modeling metrics: batter & pitcher Expected wOBA (`xwOBA`), Expected ERA (`xERA`), Barrel%, Hard-Hit% ($\ge 95\text{ mph}$), Sweet-Spot% ($8^\circ\text{--}32^\circ$), Chase/Whiff% (`O-Swing%`, `SwStr%`), Sprint Speed, and pitcher arsenal velocity/stuff indicators.
   * **Graceful Fallback**: When Statcast post-game processing lags, the reconciler computes FIP, $x\text{FIP}$ (using league-average HR/FB%), linear-weight `wOBA`, and `K-BB%` directly from MLB StatsAPI boxscores and sets `has_statcast_metrics = False`.

### 2.2 NPB Sources (`league = "NPB"`)
1. **Source B1 — Official NPB BIS / English & Japanese Game Feeds (`https://npb.jp/bis/eng/` & `https://npb.jp/scores/`)**:
   * **Role**: Canonical schedule, stadium assignment (capturing regional/provincial neutral-site games and dome venues), 12-inning final scores (including ties), official batting & pitching boxscores, and active first-team (`ichi-gun`) registration / de-registration transactions (`公示 / Roster Announcements`).
2. **Source B2 — NPB Pre-Game Probable Starter (`予告先発`) & Lineup Feeds (`npb.jp/announcement/starter/` & Yahoo! Japan / SportsNavi Baseball)**:
   * **Role**: In NPB, official next-day probable starting pitchers (`予告先発 / Yokoku Senpatsu`) are formally announced by both Central and Pacific League offices the afternoon before every game. Starting 1–9 batting orders (`スタメン`) are published ~45–60 minutes before first pitch.
3. **Source B3 — Historical NPB Sabermetric & DIPS Archives (ProYakyu / Baseball-Reference / FanGraphs International)**:
   * **Role**: Multi-season historical NPB club, batter, and pitcher logs with league-specific linear weights (`wOBA`, `wRAA`), Defense-Independent Pitching Statistics (`FIP`, `tRA`, `K-BB%`, `WHIP`, `GO/AO`), and multi-year rolling NPB stadium park factors (e.g., high-HR Meiji Jingu Stadium & Tokyo Dome vs. pitcher-friendly Vantelin Dome Nagoya & Hanshin Koshien Stadium).

---

## 3. Canonical Polars Parquet Schemas (Silver Layer)

All Silver tables are stored under `data/silver/baseball/` and strictly sorted by `(game_date, start_time_utc, event_seq, game_id)`. Every table includes `league` $\in \{\text{"MLB"}, \text{"NPB"}\}$.

### 3.1 `games.parquet` (1 row per MLB or NPB game)
| Column | Polars Type | Description |
| :--- | :--- | :--- |
| `sport` | `pl.Utf8` | Always `"baseball"` |
| `league` | `pl.Utf8` | `"MLB"` or `"NPB"` |
| `sub_league` | `pl.Utf8` | `"AL"`, `"NL"`, `"INTERLEAGUE_MLB"`, `"PL"` (Pacific), `"CL"` (Central), or `"KORYUSEN"` (NPB Interleague) |
| `season` | `pl.Int32` | Calendar season year (e.g., `2024`, `2025`, `2026`) |
| `game_id` | `pl.Int64` | Canonical unique game ID (`event_id` alias; prefixed/namespaced so MLB `gamePk` and NPB IDs never collide) |
| `event_seq` | `pl.Int32` | Sub-day sequence: `1` for single games or Game 1 of a doubleheader; `2` for Game 2 of a same-day doubleheader |
| `game_date` | `pl.Date` | Local venue calendar date |
| `start_time_utc` | `pl.Datetime("us", "UTC")` | Exact first-pitch UTC timestamp (canonical temporal sort key) |
| `home_team` | `pl.Utf8` | Canonical 3-letter club code (MLB: `"LAD"`, `"NYY"`, `"BOS"`, etc.; NPB: `"YOM"`, `"HAN"`, `"SOF"`, `"ORX"`, `"CHU"`, `"HIR"`, `"YOK"`, `"YAK"`, `"FUK"`, `"LOT"`, `"RAK"`, `"SEI"`) |
| `away_team` | `pl.Utf8` | Canonical 3-letter club code |
| `venue_id` | `pl.Utf8` | Canonical ballpark ID (supports NPB provincial/regional stadiums & MLB international series) |
| `venue_name` | `pl.Utf8` | Standardized ballpark name |
| `venue_timezone` | `pl.Utf8` | IANA timezone (e.g., `"America/New_York"`, `"America/Los_Angeles"`, `"Asia/Tokyo"`) |
| `is_dome_or_retractable` | `pl.Boolean` | `True` for fixed domes (Tokyo Dome, Kyocera Dome, Vantelin Dome, PayPay Dome, Tropicana) or closed retractable roofs |
| `is_neutral_or_regional_venue` | `pl.Boolean` | `True` for NPB provincial games, Koshien road displacements, or MLB London/Tokyo/Seoul series |
| `uses_dh` | `pl.Boolean` | `True` if Designated Hitter rule is active for this game |
| `scheduled_innings` | `pl.Int8` | `9` (or `7` for historical 2020–2021 MLB doubleheaders) |
| `innings_played` | `pl.Int8` | Actual innings completed (`null` if `"FUT"`, `10+` for extras, up to `12` in NPB regular season) |
| `game_state` | `pl.Utf8` | `"FINAL"` for completed official games; `"FUT"` for upcoming scheduled games |
| `home_sp_id`, `away_sp_id` | `pl.Int64` | Starting pitcher canonical `player_id` (probable pre-game or actual post-game) |
| `home_sp_hand`, `away_sp_hand` | `pl.Utf8` | Starting pitcher throwing hand (`"L"` or `"R"`) |
| `home_runs`, `away_runs` | `pl.Int16` | Final runs scored (`null` if `"FUT"`) |
| `outcome_1x2` | `pl.Int8` | `0 = Home Win`, `1 = Draw/Tie` (NPB 12-inning ties), `2 = Away Win` (`null` if `"FUT"`) |
| `home_win` | `pl.Int8` | `1` if `home_runs > away_runs`, else `0` (`null` if `"FUT"`) |
| `is_tie` | `pl.Int8` | `1` if `home_runs == away_runs` (NPB 12-inning draw), else `0` (`null` if `"FUT"`) |
| `market_home_decimal_odds` | `pl.Float64` | Nullable evaluation-only closing home decimal odds (strictly excluded from feature matrices `X`) |
| `market_draw_decimal_odds` | `pl.Float64` | Nullable evaluation-only closing draw odds (for NPB 3-way markets; `null` for 2-way moneyline) |
| `market_away_decimal_odds` | `pl.Float64` | Nullable evaluation-only closing away decimal odds |
| `market_prob_home`, `market_prob_draw`, `market_prob_away` | `pl.Float64` | Power-de-vigged market closing probabilities (`market_*` firewall) |

### 3.2 `team_game_logs.parquet` (2 rows per game — 1 Home, 1 Away)
| Column | Polars Type | Description |
| :--- | :--- | :--- |
| `game_id`, `event_seq`, `game_date`, `start_time_utc`, `season`, `league` | — | Temporal & league keys |
| `team`, `opponent`, `is_home` | — | Club context |
| `runs_for`, `runs_against` | `pl.Int16` | Runs scored and allowed |
| `hits_for`, `hits_against` | `pl.Int16` | Hits recorded and allowed |
| `doubles_for`, `triples_for`, `hr_for`, `hr_against` | `pl.Int16` | Extra-base hit and home run counts |
| `bb_for`, `bb_against`, `k_for`, `k_against` | `pl.Int16` | Walks and strikeouts for batting and pitching units |
| `hbp_for`, `sf_for` | `pl.Int16` | Hit-by-pitch and sacrifice flies (for exact `wOBA` denominator) |
| `pa_for`, `outs_pitched` | `pl.Int16` | Total offensive Plate Appearances (`PA`) and defensive outs recorded ($3 \times \text{IP}$) |
| `lob_for`, `errors_committed` | `pl.Int16` | Left on base and defensive fielding errors |
| `bullpen_outs_pitched`, `bullpen_pitches`, `bullpen_er` | `pl.Int16` | Relief corps usage and earned runs allowed in this game |
| `team_woba`, `opp_woba` | `pl.Float64` | Single-game linear-weight `wOBA` for and against |
| `team_xwoba`, `opp_xwoba` | `pl.Float64` | Statcast expected `xwOBA` (MLB) or DIPS-estimated `xwOBA` proxy (NPB) |
| `has_statcast_metrics` | `pl.Boolean` | `True` when MLB Statcast tracking metrics were joined |

### 3.3 `pitcher_game_logs.parquet` (1 row per pitcher appearance — Starters & Relievers)
| Column | Polars Type | Description |
| :--- | :--- | :--- |
| `game_id`, `event_seq`, `game_date`, `start_time_utc`, `season`, `league` | — | Temporal & league keys |
| `player_id` | `pl.Int64` | Canonical pitcher ID |
| `player_name` | `pl.Utf8` | Normalized pitcher name |
| `team`, `opponent`, `is_home` | — | Club context |
| `pitch_hand` | `pl.Utf8` | `"L"` or `"R"` |
| `is_starter` | `pl.Boolean` | `True` for starting pitcher (`SP`), `False` for relief pitcher (`RP`) |
| `outs_pitched` | `pl.Int16` | Exact outs recorded ($3 \times \text{full IP} + \text{partial outs}$; avoids base-10 `.1`/`.2` math errors!) |
| `innings_pitched` | `pl.Float64` | True fractional innings: $\text{outs\_pitched} / 3.0$ |
| `batters_faced` | `pl.Int16` | Total batters faced (`BF` / `TBF`) |
| `pitches_thrown`, `strikes_thrown` | `pl.Int16` | Pitch count and strike count |
| `hits_allowed`, `runs_allowed`, `er_allowed`, `hr_allowed` | `pl.Int16` | Official pitching boxscore counts |
| `bb_allowed`, `ibb_allowed`, `hbp_allowed`, `k_recorded` | `pl.Int16` | Walks, intentional walks, hit batters, and strikeouts |
| `gb_count`, `fb_count`, `ld_count` | `pl.Int16` | Ground balls, fly balls, and line drives allowed |
| `quality_start` | `pl.Int8` | `1` if `is_starter == True`, `outs_pitched >= 18` ($6.0+\text{ IP}$), and `er_allowed <= 3`, else `0` |
| `xwoba_against`, `barrel_pct_against`, `swstr_pct`, `avg_fastball_velo` | `pl.Float64` | Micro pitch-quality metrics (Statcast for MLB; DIPS/whiff equivalents when available for NPB) |

### 3.4 `batter_game_logs.parquet` (1 row per batter appearance)
| Column | Polars Type | Description |
| :--- | :--- | :--- |
| `game_id`, `event_seq`, `game_date`, `start_time_utc`, `season`, `league` | — | Temporal & league keys |
| `player_id` | `pl.Int64` | Canonical batter ID |
| `player_name` | `pl.Utf8` | Normalized batter name |
| `team`, `opponent`, `is_home` | — | Club context |
| `bat_side` | `pl.Utf8` | `"L"`, `"R"`, or `"S"` (Switch) |
| `batting_order_slot` | `pl.Int8` | `1..9` for starting lineup batters; `0` for pinch-hitters / defensive subs |
| `position_group` | `pl.Utf8` | `"C"`, `"INF"`, `"OF"`, `"DH"` |
| `is_starter` | `pl.Boolean` | `True` if in starting 1–9 batting order |
| `pa`, `ab` | `pl.Int16` | Plate Appearances (`PA`) and At-Bats (`AB`) |
| `hits`, `singles`, `doubles`, `triples`, `hr` | `pl.Int16` | Hit breakdown |
| `total_bases` | `pl.Int16` | $\text{1B} + 2\cdot\text{2B} + 3\cdot\text{3B} + 4\cdot\text{HR}$ |
| `runs`, `rbi` | `pl.Int16` | Runs scored and Runs Batted In |
| `bb`, `ibb`, `hbp`, `k`, `sf`, `sb`, `cs` | `pl.Int16` | Discipline, sacrifice fly, and baserunning counts |
| `woba_num` | `pl.Float64` | Single-game linear-weight wOBA numerator: $w_{BB}\text{uBB} + w_{HBP}\text{HBP} + w_{1B}\text{1B} + w_{2B}\text{2B} + w_{3B}\text{3B} + w_{HR}\text{HR}$ |
| `xwoba`, `hard_hit_rate`, `barrel_rate` | `pl.Float64` | Contact quality micro-metrics |

---

## 4. Temporal Firewall, Exposure Basis & Cross-League Shrinkage

### 4.1 Strict Sub-Day Doubleheader Ordering (`event_seq = 1, 2`)
Baseball features frequent same-day doubleheaders (`event_seq = 1` for Game 1, `event_seq = 2` for Game 2).
* All rolling, EWMA, Elo, and bullpen workload windows enforce `.shift(1)` over `(start_time_utc, event_seq, game_id)` via `src/sports_ml/core/temporal.py`.
* **Zero Leakage Guarantee**: Game 1 of a doubleheader cannot see any pitches, runs, or bullpen usage from Game 1 or Game 2; Game 2 only sees completed games with timestamp $< T_{\text{G2}}$ (and when predicting pre-day doubleheaders before Game 1 starts, both games use strictly pre-doubleheader state).

### 4.2 Exposure Normalization (`pregame_exposure_rate`)
Unlike clock-based sports (NHL per-60 TOI, EPL per-90 minutes), baseball is an **out-and-plate-appearance** game:
1. **Pitching Per-9-Innings Rates (`scale_factor = 27.0` over `outs_pitched`)**:
   Because `1 IP = 3 outs_pitched`, all per-9-inning pitching rates (`FIP`, `xFIP`, `ERA`, `K/9`, `BB/9`, `HR/9`) are computed over exact `outs_pitched` with `scale_factor = 27.0`:
   $$\text{Rate}_{\text{per 9 IP}}(x) = 27.0 \cdot \frac{\sum_{t-W}^{t-1} x}{\sum_{t-W}^{t-1} \text{outs\_pitched}}$$
2. **Pitching Per-Batter-Faced Percentages (`scale_factor = 1.0` over `batters_faced`)**:
   $$\text{K\%} = \frac{\sum K}{\sum \text{BF}}, \quad \text{BB\%} = \frac{\sum (\text{BB} - \text{IBB})}{\sum \text{BF}}, \quad \text{K-BB\%} = \text{K\%} - \text{BB\%}$$
3. **Batting Per-Plate-Appearance Rates (`scale_factor = 1.0` over `pa`)**:
   $$\text{wOBA} = \frac{\sum \text{wOBA\_num}}{\sum (\text{AB} + \text{BB} - \text{IBB} + \text{SF} + \text{HBP})}, \quad \text{ISO} = \text{SLG} - \text{AVG}, \quad \text{K\%} = \frac{\sum K}{\sum \text{PA}}, \quad \text{BB\%} = \frac{\sum \text{uBB}}{\sum \text{PA}}$$

### 4.3 Defense-Independent Pitching (`FIP` & `xFIP`)
For both MLB and NPB starting pitchers and bullpens, rolling defense-independent skill is computed strictly from pre-game cumulative sums:
$$\text{FIP} = \frac{13 \cdot \text{HR} + 3 \cdot (\text{BB} + \text{HBP} - \text{IBB}) - 2 \cdot K}{\text{outs\_pitched} / 3.0} + c_{\text{FIP}(\text{league, season})}$$
$$\text{xFIP} = \frac{13 \cdot (\text{FB} \cdot \mu_{\text{HR/FB, league}}) + 3 \cdot (\text{BB} + \text{HBP} - \text{IBB}) - 2 \cdot K}{\text{outs\_pitched} / 3.0} + c_{\text{FIP}(\text{league, season})}$$
where $c_{\text{FIP}(\text{league, season})}$ centers league-average FIP on league-average ERA ($\approx 3.15$ in MLB; $\approx 3.05$ in NPB).

### 4.4 Bayesian Shrinkage Constants ($k$) & Cross-League (`NPB ↔ MLB`) Translation Priors
1. **Standard Intra-League Shrinkage Constants ($k$)**:
   * **Team Batting & Bullpen Rolling Rates**: $k_{\text{team}} = 22.0$ games (reflecting baseball's larger 143/162-game schedule and higher single-game variance).
   * **Starting Pitcher Rates (`FIP`, `xFIP`, `K-BB%`, `GB%`)**: $k_{\text{sp}} = 180.0$ batters faced ($\approx 7\text{--}8$ starts or $42\text{ IP}$), regressed toward the pitcher's prior-season baseline blended with league starter average.
   * **Batter Rates (`wOBA`, `xwOBA`, `ISO`, `K%`, `BB%`)**: $k_{\text{batter}} = 240.0$ Plate Appearances, regressed toward prior-season + positional baseline.
   * **Handedness Platoon Splits (`vs. LHP` / `vs. RHP`)**: Platoon split deltas have high small-sample noise and require heavy shrinkage ($k_{\text{platoon\_LHB}} = 1,000\text{ PA}$, $k_{\text{platoon\_RHB}} = 2,200\text{ PA}$) toward empirical league platoon advantages ($+0.022\text{ wOBA}$ for opposite-hand matchups).
2. **Cross-League Player Translation Priors (`sports/baseball/features/translation.py`)**:
   When a player transfers from **NPB to MLB** (e.g., via posting or international free agency) or from **MLB to NPB** (`prev_league != curr_league` and `curr_league_exposure < k`), `translation.py` transforms their prior-season stats into the destination league's context before applying `bayesian_shrinkage`:
   * **NPB $\rightarrow$ MLB Batter Translation**:
     $$\hat{\mu}_{\text{MLB}}(\text{wOBA}) = \mu_{\text{MLB, lg}} + 0.78 \cdot (\text{wOBA}_{\text{NPB}} - \mu_{\text{NPB, lg}}) - 0.018$$
     $$\hat{\mu}_{\text{MLB}}(\text{K\%}) = 1.24 \cdot \text{K\%}_{\text{NPB}}, \quad \hat{\mu}_{\text{MLB}}(\text{BB\%}) = 0.86 \cdot \text{BB\%}_{\text{NPB}}, \quad \hat{\mu}_{\text{MLB}}(\text{ISO}) = 0.82 \cdot \text{ISO}_{\text{NPB}}$$
   * **NPB $\rightarrow$ MLB Pitcher Translation**:
     $$\hat{\mu}_{\text{MLB}}(\text{K\%}) = 0.91 \cdot \text{K\%}_{\text{NPB}}, \quad \hat{\mu}_{\text{MLB}}(\text{BB\%}) = 1.16 \cdot \text{BB\%}_{\text{NPB}}, \quad \hat{\mu}_{\text{MLB}}(\text{FIP}) = \mu_{\text{MLB, FIP}} + 0.85 \cdot (\text{FIP}_{\text{NPB}} - \mu_{\text{NPB, FIP}}) + 0.35$$
   * **MLB $\rightarrow$ NPB Translation**: Applies the calibrated inverse equivalency multipliers so foreign signings in NPB start with realistic power/strikeout expectations rather than generic NPB rookie priors.

---

## 5. Domain Feature Suite (`sports/baseball/features/`)

### 5.1 Starting Pitcher Specialist Features (`features/pitcher.py`)
In baseball, the starting pitcher accounts for ~55–65% of defensive match variation. For both `home_sp_id` and `away_sp_id`:
* **Shrunk Skill Metrics (Rolling 5, 10 starts, EWMA & Season-to-Date)**: `sp_xfip_shrunk`, `sp_fip_shrunk`, `sp_era_shrunk`, `sp_k_minus_bb_pct_shrunk`, `sp_whip_shrunk`, `sp_hr_per_9_shrunk`, `sp_gb_pct_shrunk`, `sp_xwoba_against_shrunk`, and `sp_swstr_pct_shrunk`.
* **Workload, Pitch Count Ceiling & Rest**:
  * `sp_days_rest`: Days since the starter's last appearance (handling MLB 5-man rotation = 4 days rest vs. NPB 6-man rotation = 5–6 days rest).
  * `sp_rolling_pitches_per_start` & `sp_expected_outs_ceiling`: Estimates how deep into the game the starter is stretched out to pitch (critical for early-season ramp-up or starters returning from the IL / farm).
* **Starter Elo Adjustment (`StarterModifierHook`)**:
  $$\Delta E_{\text{SP}} = \kappa_{\text{SP}} \cdot \left( \mu_{\text{TeamSP, xFIP}} - \text{xFIP}_{\text{today's SP}} \right)$$
  where $\kappa_{\text{SP}} \approx 32.0$ Elo points per $1.00$ run of `xFIP` better than the team's baseline rotation average (translating a true ace like Skubal/Skenes/Yamamoto or an NPB Sawamura Award contender into an immediate pre-game Elo boost on the day they pitch).

### 5.2 Bullpen Fatigue & High-Leverage Availability (`features/bullpen.py`)
* **Rolling Team Bullpen Quality**: Shrunk bullpen `bp_xfip_shrunk`, `bp_k_minus_bb_pct_shrunk`, and `bp_lob_pct_shrunk` over the last 15 and 30 team games.
* **3-Day & Back-to-Back Bullpen Fatigue (`strictly < T_game`)**:
  * `bp_pitches_last_1d`, `bp_pitches_last_3d`, and `bp_outs_last_3d` (including same-day doubleheader Game 1 bullpen depletion when predicting live before Game 2).
  * `high_leverage_arms_unavailable_count`: Number of the team's top-3 relievers (by rolling high-leverage usage/save share) who pitched on consecutive prior days (`pitched_d_minus_1` AND `pitched_d_minus_2`) or threw $\ge 28$ pitches yesterday.

### 5.3 Platoon-Adjusted Active Lineup & Bench Aggregation (`features/lineup.py`)
* **PA-Weighted 1–9 Lineup Strength**:
  Batting order slots `1..9` do not receive equal Plate Appearances: each slot up the order receives $\approx 18\text{ PA}$ more per 162 games ($\sim 4.65\text{ PA}$ for Slot 1 down to $\sim 3.75\text{ PA}$ for Slot 9).
  The lineup aggregator weights the 9 starting batters by expected slot PA share $w_{\text{slot}}$ AND adjusts each batter's shrunk `wOBA`, `xwOBA`, `ISO`, and `K%` for the **opposing starting pitcher's handedness (`opp_sp_hand` $\in \{\text{"L"}, \text{"R"}\}$)**:
  $$\text{LineupPlatoon-wOBA} = \sum_{s=1}^{9} w_s \cdot \widehat{\text{wOBA}}_{i(s) \mid \text{opp\_sp\_hand}}$$
* **Missing Regular Production (`missing_regular_pa_woba`)**: Quantifies offensive drop-off when primary lineup regulars are on the Injured List, deactivated from the NPB first-team (`ichi-gun`) roster, or resting (e.g., catcher day-game-after-night-game rest).

### 5.4 Ballpark Factors, Weather & League Schedule Context (`features/team.py`)
* **Multi-Year Rolling Ballpark Factors (`constants.py` & `team.py`)**:
  * `venue_run_factor` (centered at `1.00`; e.g., Coors Field $\approx 1.28$, Meiji Jingu $\approx 1.12$, Vantelin Dome Nagoya $\approx 0.86$, Koshien $\approx 0.91$).
  * `venue_hr_factor_lhb` and `venue_hr_factor_rhb` (capturing asymmetrical dimensions like Yankee Stadium's short right-field porch or Fenway's Green Monster).
* **Dome / Weather Context**: `is_dome_or_retractable`, temperature/wind run-scoring multiplier for outdoor ballparks, and `is_day_after_night_game`.
* **NPB & MLB Schedule Nuances**:
  * `is_doubleheader_g1`, `is_doubleheader_g2`
  * `is_interleague` (MLB AL vs. NL; NPB Central vs. Pacific _Koryusen_)
  * `is_neutral_or_regional_venue` & `is_koshien_summer_road_trip` (capturing Hanshin Tigers' annual August displacement during the Summer High School Baseball Championship).

---

## 6. Two-Stage Modeling & Evaluation Specification

### 6.1 Stage 1 — League-Aware Calibrated Game Outcome & Run Environment (`stage1_game.py`)
1. **Auxiliary Team-Runs Environment Model**:
   * Fits Negative-Binomial / Tweedie regressors estimating pre-game expected runs $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$ conditioned on starting pitcher `xFIP`/`K-BB%`, bullpen availability, platoon lineup `wOBA`, ballpark run factor, and DH status.
2. **League-Aware Probability Heads**:
   * **MLB (`2-Way Binary`)**:
     * Combines Starter-Adjusted Elo probability, Negative-Binomial implied win probability, and a binary LightGBM classifier calibrated out-of-fold via Platt/Isotonic calibration (`BinaryProbabilityCalibrator`) so $P(\text{Home Win}) + P(\text{Away Win}) = 1.0$.
     * Evaluated via **Log Loss**, **Brier Score**, **ECE**, and **De-Vigged Closing Moneyline CLV**.
   * **NPB (`3-Way 12-Inning 1X2`)**:
     * Explicitly models $P(\text{Home Win})$, $P(\text{Draw / 12-Inning Tie})$, and $P(\text{Away Win})$ summing strictly to $1.0$.
     * Derives structural 12-inning tie probability from the bivariate Negative-Binomial / Skellam run distribution ($P(R_H = R_A \text{ at 9 IP}) \times P(\text{scoreless/tied through innings 10--12})$) blended with a 3-class LightGBM classifier and calibrated via `MulticlassProbabilityCalibrator`.
     * Evaluated via **Ranked Probability Score (RPS)**, **3-Class Log Loss**, **Multi-Class Brier Score**, and **ECE**, plus conditional 2-way Draw-No-Bet (`DNB`) accuracy for direct moneyline comparison.

### 6.2 Stage 2 — Starting Pitcher & Lineup Batter Matchup Projections (`stage2_player.py`)
Conditioned on Stage 1's $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$ and win/tie script:
1. **Starting Pitcher (`SP`) Projections**:
   * **Exposure Model**: Predicts `proj_outs_pitched` ($\in [0, 27]$) and `proj_batters_faced` ($\widehat{\text{BF}}$) based on starter pitch-count ceiling, rest days, and opponent patience (`BB%`, pitches/PA).
   * **Count & Milestone Models**: Predicts `proj_ip` ($\text{outs}/3$), `proj_k`, `proj_bb`, `proj_hits_allowed`, `proj_er`, `prob_5plus_k`, `prob_6plus_k`, `prob_7plus_k`, and `prob_quality_start` ($P(\text{IP} \ge 6.0 \land \text{ER} \le 3)$).
2. **Lineup Batter (`1..9`) Projections**:
   * **Plate Appearance Model**: Predicts `proj_pa` ($\in [0, 6.0]$) conditioned on `batting_order_slot` (`1..9`), home/away status (home batters bat in the 9th inning less often when leading!), and Stage 1 $\hat{\lambda}_{\text{team}}$ (higher scoring teams turn the lineup over more often).
   * **Count & Milestone Models**: Predicts `proj_hits`, `proj_total_bases`, `proj_hr`, `proj_runs`, `proj_rbi`, `proj_bb`, `proj_k`, `prob_1plus_hit` ($1 - \prod (1 - p_{\text{hit/PA}})$), `prob_2plus_tb`, and `prob_1plus_hr`.
3. **Cross-Stage Coherence Reconciliation (`reconcile_baseball_stage2_projections`)**:
   * Soft-scales batter `proj_runs` and `proj_rbi` toward Stage 1's $\hat{\lambda}_{\text{team}}$ ($\gamma = 0.50$), and reconciles starting pitcher `proj_k` against opposing lineup projected strikeouts (accounting for expected starter share of game BF).
