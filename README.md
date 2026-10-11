# World Cup AI Predictor

A machine-learning powered FIFA World Cup prediction tool that runs in terminal. It predicts match outcomes (win / draw / loss probabilities), compares teams head-to-head, and scores how well its own predictions matched the real 2026 tournament.

> Data is based on the real, completed **FIFA World Cup 2026** (104 matches, 48 teams) plus historical tournaments (2014 / 2018 / 2022).

---

## ✨ Features

- **ML ensemble prediction** — Gradient Boosting + Random Forest + Logistic Regression (soft voting), blended with a heuristic "power score".
- **104 matches** with full results, plus historical data for training.
- **Interactive terminal UI** with 10 commands (predict, compare, team stats, accuracy check, retrain…).
- **Probability bars, expected goals, and an ML-vs-heuristic breakdown** for every prediction.
- **Honest accuracy reporting** — the tool checks its own picks against the actual results.

## 🧱 Tech Stack

- **Language:** Python 3.11+
- **ML:** scikit-learn, pandas, numpy, joblib
- **Interface:** Pure terminal (ANSI colors, Windows-compatible)

---

## Structure

```text
├── data.py             # All match & team data (single source of truth)
├── ml_model.py         # ML engine: features, augmentation, ensemble, blending
├── terminal_app.py     # Interactive CLI (main entry point)
├── demo.py             # Non-interactive demo → generates sample output
├── sample_output.txt   # Static copy of demo output (for GitHub preview)
└── requirements.txt  
```

---


On first run the model trains automatically (a `model.joblib` file is saved). You then get an interactive menu:

| # | Command | What it does |
|---|---------|--------------|
| 1 | Predict a match | Enter any two teams + stage → win/draw/loss probabilities, expected goals, ML vs heuristic split |
| 2 | View all 2026 predictions | Every fixture with the AI pick vs the real result (✓ / ✗) |
| 3 | List all teams | All teams sorted by FIFA rank with key stats |
| 4 | Team detail | One team's stats + its full 2026 match history |
| 5 | Compare two teams | Side-by-side stats + head-to-head + AI prediction |
| 6 | Model info & accuracy | Model type, cross-validated accuracy, training history |
| 7 | 2026 WC final results | All 104 real results, grouped by stage |
| 8 | Prediction accuracy check | How many of the 2026 matches the model called correctly |
| 9 | Retrain model | Re-train on current data and refresh the cache |
| 0 | Exit | Quit |

---

## How It Works

1. **Feature engineering** — each match becomes a 26-value vector: FIFA rank, win rate, goals scored/conceded, form, World Cup titles, continent strength, head-to-head record, plus the *difference* between the two teams (which helps the model judge who is stronger).
2. **Data augmentation** — every historical match is "mirror-flipped" (teams swapped, score reversed, result reversed), doubling the training set so the model learns more stably from limited football data.
3. **Ensemble + adaptive blend** — three classifiers vote on probabilities; their output is then blended with a heuristic power score. The blend weight shifts with data size: the more training data, the more the model is trusted (30 % → 75 % ML). This prevents the all-draw overfitting bug that pure-ML setups hit on small datasets.
4. **Expected goals** — a lightweight Poisson-style estimate from each team's average goals for/against.

---

## Sample Output

The tool is a terminal app, so it can't "run" inside a GitHub preview. Below is **real output** captured by `demo.py` (which runs the same functions with no user input). The full file is committed as [`sample_output.txt`](sample_output.txt).

```text
>> (1) Single match prediction  -  2026 Final

                    Spain  vs  Argentina

  Stage:       KNOCKOUT
  Expected:    1.2 - 0.5
  Verdict:     Argentina WIN
  Confidence:  39.4%

  Win Probabilities:
    Spain                ███████░░░░░░░░░░░░░░░░░░░░░░░  24.6%
    Draw                 ██████████░░░░░░░░░░░░░░░░░░░░  35.9%
    Argentina            ███████████░░░░░░░░░░░░░░░░░░░  39.4%

  ML vs Heuristic breakdown:
    Source           Win1     Draw     Win2
    ML              17.1%    45.0%    37.9%
    Heuristic       47.2%     8.9%     43.9%
────────────────────────────────────────────────────────────
>> (2) Head-to-head comparison  -  France vs England

                    France  vs  England

  Head-to-Head Record:
    Played: 34  |  France: 13W  |  Draw: 8  |  England: 13W

  Metric               France             England
────────────────────────────────────────────────────────────
  FIFA Rank            2                  4
  Win Rate             60%                63%
  Avg Goals Scored     2.5                2.5
  Avg Goals Conceded   1.3                1.5
  WC Titles            2                  1
  Form (last 5)        3/5                4/5
  Continent            UEFA               UEFA
  Star Player          Kylian Mbappe      Jude Bellingham

  AI Prediction:
    France             ███████████░░░░░░░░░░░░░░░░░░░  39.9%
    Draw               ████████░░░░░░░░░░░░░░░░░░░░░░  28.5%
    England            █████████░░░░░░░░░░░░░░░░░░░░░  31.5%
    → France WIN (39.9%)
────────────────────────────────────────────────────────────
>> (3) Team detail  -  Spain

                        Team: Spain

  FIFA Rank:      3
  Win Rate:       80.0%
  Avg Goals:      2.0 scored / 0.3 conceded
  WC Titles:      2
  Continent:      UEFA
  Star Player:    Rodri
  Recent Form:    ● ● ● ● ● (5/5)

  2026 World Cup Matches:
    [D] vs Cabo Verde       0-0  Group H
    [W] vs Saudi Arabia     4-0  Group H
    [W] vs Uruguay          1-0  Group H
    [W] vs Austria          3-0  Round of 32
    [W] vs Portugal         1-0  Round of 16
    [W] vs Belgium          2-1  Quarter-final
    [W] vs France           2-0  Semi-final
    [W] vs Argentina        1-0  Final
────────────────────────────────────────────────────────────
>> (4) Model accuracy on the real 2026 tournament
  Overall accuracy on 2026 results: 67/104 = 64.4%
```

*(The model is honest: it picked Argentina in the Final, but Spain actually won 1–0 — that miss is reflected in the 64.4 % accuracy.)*
