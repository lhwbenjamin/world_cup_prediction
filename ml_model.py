"""
World Cup AI Predictor - ML Prediction Engine
  1. Head-to-head features added to feature vector
  2. Continent familiarity / home-continent advantage
  3. Ensemble: GradientBoosting + RandomForest + LogisticRegression (soft voting)
  4. Cross-validated accuracy instead of single split
  5. Adaptive blending: ML weight grows as training data increases
  6. Goal prediction (expected goals via Poisson-like estimation)
  7. Prediction confidence calibration
"""
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from data import HISTORICAL_MATCHES, TEAM_STATS, WC2026_FIXTURES, HEAD_TO_HEAD

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
ACCURACY_LOG_PATH = os.path.join(os.path.dirname(__file__), "accuracy_log.json")

LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}
LABEL_REVERSE = {0: "home_win", 1: "draw", 2: "away_win"}

# Continent strength multiplier (historical WC performance bias)
CONTINENT_STRENGTH = {
    "UEFA": 1.0, "CONMEBOL": 0.95, "CONCACAF": 0.75, "CAF": 0.70, "AFC": 0.68, "OFC": 0.60,
}


def get_team_features(team_name: str) -> dict:
    """Extract feature dict for a team, with safe defaults."""
    stats = TEAM_STATS.get(team_name, {
        "fifa_rank": 50, "avg_goals_scored": 1.0, "avg_goals_conceded": 1.5,
        "win_rate": 0.40, "wc_titles": 0, "continent": "UEFA",
        "star_player": "Unknown", "form": [0, 0, 0, 0, 0],
    })
    form = stats.get("form", [0, 0, 0, 0, 0])
    form_score = sum(form) / max(len(form), 1)
    # Recent form (last 3 games) weighted heavier
    recent_form = sum(form[:3]) / max(min(len(form), 3), 1)
    return {
        "fifa_rank": stats["fifa_rank"],
        "avg_goals_scored": stats["avg_goals_scored"],
        "avg_goals_conceded": stats["avg_goals_conceded"],
        "win_rate": stats["win_rate"],
        "wc_titles": stats["wc_titles"],
        "form_score": form_score,
        "recent_form": recent_form,
        "continent": stats.get("continent", "UEFA"),
        "continent_strength": CONTINENT_STRENGTH.get(stats.get("continent", "UEFA"), 0.7),
    }


def get_h2h_features(team1: str, team2: str) -> dict:
    """Head-to-head record features."""
    h2h = HEAD_TO_HEAD.get((team1, team2)) or HEAD_TO_HEAD.get((team2, team1))
    if not h2h:
        return {"h2h_played": 0, "h2h_team1_win_rate": 0.5, "h2h_draw_rate": 0.33, "h2h_advantage": 0.0}

    played = h2h["played"]
    # Determine which team is team1 in the H2H record
    is_reversed = (team2, team1) in HEAD_TO_HEAD
    t1_wins = h2h["team2_wins"] if is_reversed else h2h["team1_wins"]
    t2_wins = h2h["team1_wins"] if is_reversed else h2h["team2_wins"]
    draws = h2h["draws"]

    if played == 0:
        return {"h2h_played": 0, "h2h_team1_win_rate": 0.5, "h2h_draw_rate": 0.33, "h2h_advantage": 0.0}

    return {
        "h2h_played": min(played, 50),  # cap to avoid extreme values
        "h2h_team1_win_rate": t1_wins / played,
        "h2h_draw_rate": draws / played,
        "h2h_advantage": (t1_wins - t2_wins) / played,
    }


def build_feature_row(team1: str, team2: str, round_type: str) -> list:
    """Build full feature vector: team stats + H2H + context."""
    f1 = get_team_features(team1)
    f2 = get_team_features(team2)
    h2h = get_h2h_features(team1, team2)
    is_knockout = 1 if round_type == "knockout" else 0

    return [
        f1["fifa_rank"],
        f2["fifa_rank"],
        f1["fifa_rank"] - f2["fifa_rank"],
        f1["avg_goals_scored"],
        f2["avg_goals_scored"],
        f1["avg_goals_conceded"],
        f2["avg_goals_conceded"],
        f1["avg_goals_scored"] - f2["avg_goals_scored"],
        f1["win_rate"],
        f2["win_rate"],
        f1["win_rate"] - f2["win_rate"],
        f1["wc_titles"],
        f2["wc_titles"],
        f1["form_score"],
        f2["form_score"],
        f1["form_score"] - f2["form_score"],
        f1["recent_form"],
        f2["recent_form"],
        f1["continent_strength"],
        f2["continent_strength"],
        f1["continent_strength"] - f2["continent_strength"],
        h2h["h2h_played"],
        h2h["h2h_team1_win_rate"],
        h2h["h2h_draw_rate"],
        h2h["h2h_advantage"],
        is_knockout,
    ]


FEATURE_NAMES = [
    "team1_rank", "team2_rank", "rank_diff",
    "team1_goals_scored", "team2_goals_scored",
    "team1_goals_conceded", "team2_goals_conceded",
    "goals_scored_diff",
    "team1_win_rate", "team2_win_rate", "win_rate_diff",
    "team1_wc_titles", "team2_wc_titles",
    "team1_form", "team2_form", "form_diff",
    "team1_recent_form", "team2_recent_form",
    "team1_continent_str", "team2_continent_str", "continent_diff",
    "h2h_played", "h2h_team1_win_rate", "h2h_draw_rate", "h2h_advantage",
    "is_knockout",
]


def prepare_training_data(extra_matches: list = None):
    """Prepare feature matrix and label vector with augmentation."""
    matches = list(HISTORICAL_MATCHES)
    if extra_matches:
        matches += extra_matches

    X, y = [], []
    for m in matches:
        if m["result"] is None:
            continue
        features = build_feature_row(m["team1"], m["team2"], m.get("round", "group"))
        label = LABEL_MAP.get(m["result"])
        if label is not None:
            X.append(features)
            y.append(label)

    # Data augmentation: swap perspectives to double training set
    X_aug, y_aug = [], []
    # Swap pairs: (team1_feature_index, team2_feature_index)
    swap_pairs = [(0,1), (3,4), (5,6), (8,9), (11,12), (13,14), (16,17), (18,19)]
    diff_indices = [2, 7, 10, 15, 20, 24]  # difference features to negate

    for features, label in zip(X, y):
        X_aug.append(features)
        y_aug.append(label)
        # Swap perspective
        swapped = features[:]
        for a, b in swap_pairs:
            swapped[a], swapped[b] = swapped[b], swapped[a]
        for d_idx in diff_indices:
            swapped[d_idx] = -swapped[d_idx]
        # h2h_team1_win_rate becomes h2h_team2_win_rate = 1 - original
        swapped[22] = 1.0 - swapped[22]
        # h2h_advantage negated (already in diff_indices)
        # Flip result
        if label == 0:
            y_aug.append(2)
        elif label == 2:
            y_aug.append(0)
        else:
            y_aug.append(1)
        X_aug.append(swapped)

    return np.array(X_aug), np.array(y_aug)


def _build_ensemble():
    """Create ensemble of diverse models for robust prediction."""
    gb = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3,
        min_samples_leaf=4, subsample=0.8, random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=3,
        random_state=42, class_weight="balanced",
    )
    lr = LogisticRegression(
        max_iter=2000, C=0.5, class_weight="balanced",
        random_state=42,
    )
    return VotingClassifier(
        estimators=[("gb", gb), ("rf", rf), ("lr", lr)],
        voting="soft",
        weights=[2, 2, 1],
    )


def train_model(extra_matches: list = None) -> dict:
    """Train the ensemble model, evaluate with CV, and save."""
    X, y = prepare_training_data(extra_matches)

    if len(X) < 10:
        return {"error": "Not enough training data", "accuracy": 0}

    model = _build_ensemble()

    # Cross-validated accuracy (more robust than single split)
    n_splits = min(5, len(set(y)), len(X) // 10)
    n_splits = max(n_splits, 2)
    try:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        cv_accuracy = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
    except Exception:
        cv_accuracy = 0.0
        cv_std = 0.0

    # Fit on full data
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    # Feature importance (from GradientBoosting component)
    try:
        gb_model = model.named_estimators_["gb"]
        feat_importance = dict(zip(FEATURE_NAMES, gb_model.feature_importances_.tolist()))
        top_features = sorted(feat_importance.items(), key=lambda x: -x[1])[:10]
    except Exception:
        top_features = []

    # Log
    log = []
    if os.path.exists(ACCURACY_LOG_PATH):
        with open(ACCURACY_LOG_PATH, "r") as f:
            log = json.load(f)
    log.append({
        "timestamp": datetime.now().isoformat(),
        "accuracy": round(cv_accuracy, 4),
        "cv_std": round(cv_std, 4),
        "cv_splits": n_splits,
        "train_samples": len(X),
        "model_type": "ensemble_v2",
    })
    with open(ACCURACY_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    return {
        "accuracy": round(cv_accuracy, 4),
        "cv_std": round(cv_std, 4),
        "cv_splits": n_splits,
        "train_samples": len(X),
        "top_features": top_features,
        "model_type": "GradientBoosting + RandomForest + LogisticRegression (soft voting)",
    }


def load_model():
    """Load saved model or train new."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    train_model()
    return joblib.load(MODEL_PATH)


def _power_score(f: dict) -> float:
    """Composite team strength score (0-1 scale)."""
    rank_score = max(0, 1 - f["fifa_rank"] / 60)
    win_score = f["win_rate"]
    form_score = f["form_score"]
    recent = f["recent_form"]
    goals_score = min(f["avg_goals_scored"] / 3.0, 1.0)
    defense_score = max(0, 1 - f["avg_goals_conceded"] / 3.0)
    title_bonus = min(f["wc_titles"] * 0.04, 0.12)
    continent_bonus = (f["continent_strength"] - 0.7) * 0.3
    return (
        rank_score * 0.28
        + win_score * 0.22
        + form_score * 0.10
        + recent * 0.10
        + goals_score * 0.10
        + defense_score * 0.08
        + title_bonus
        + continent_bonus
    )


def _heuristic_probs(team1: str, team2: str, round_type: str) -> dict:
    """Feature-based probability estimate (no ML)."""
    f1 = get_team_features(team1)
    f2 = get_team_features(team2)
    h2h = get_h2h_features(team1, team2)

    p1 = _power_score(f1)
    p2 = _power_score(f2)

    # H2H adjustment: if we have data, blend in
    if h2h["h2h_played"] > 0:
        h2h_weight = min(h2h["h2h_played"] / 30.0, 0.3)
        p1 = p1 * (1 - h2h_weight) + h2h["h2h_team1_win_rate"] * h2h_weight
        p2 = p2 * (1 - h2h_weight) + (1 - h2h["h2h_team1_win_rate"] - h2h["h2h_draw_rate"]) * h2h_weight

    draw_baseline = 0.28
    if round_type == "knockout":
        draw_baseline = 0.15

    total = p1 + p2 + draw_baseline
    home_prob = p1 / total
    draw_prob = draw_baseline / total
    away_prob = p2 / total

    return {"home_win": home_prob, "draw": draw_prob, "away_win": away_prob}


def _adaptive_blend(ml_probs: dict, heur_probs: dict, train_samples: int) -> dict:
    """
    Blend ML and heuristic. ML weight grows with more data.
    <50 samples: 30% ML; 50-100: 45%; 100-200: 60%; >200: 75%.
    """
    if train_samples < 50:
        ml_weight = 0.30
    elif train_samples < 100:
        ml_weight = 0.45
    elif train_samples < 200:
        ml_weight = 0.60
    else:
        ml_weight = 0.75

    hw = 1 - ml_weight
    blended = {}
    for key in ["home_win", "draw", "away_win"]:
        blended[key] = ml_weight * ml_probs.get(key, 0.33) + hw * heur_probs.get(key, 0.33)

    # Normalize
    total = sum(blended.values())
    if total > 0:
        for k in blended:
            blended[k] /= total
    return blended


def predict_match(team1: str, team2: str, round_type: str = "group") -> dict:
    """Predict match outcome with probabilities."""
    model = load_model()
    features = build_feature_row(team1, team2, round_type)
    X = np.array([features])

    # ML probabilities
    try:
        probs = model.predict_proba(X)[0]
        classes = model.classes_
        ml_probs = {}
        for cls, prob in zip(classes, probs):
            outcome = LABEL_REVERSE[cls]
            ml_probs[outcome] = float(prob)
        for outcome in ["home_win", "draw", "away_win"]:
            if outcome not in ml_probs:
                ml_probs[outcome] = 0.0
    except Exception:
        ml_probs = {"home_win": 0.33, "draw": 0.34, "away_win": 0.33}

    # Heuristic
    heur_probs = _heuristic_probs(team1, team2, round_type)

    # Determine training data size for adaptive blend
    X_train, _ = prepare_training_data()
    train_samples = len(X_train)

    # Blend
    blended = _adaptive_blend(ml_probs, heur_probs, train_samples)

    home_prob = blended["home_win"]
    draw_prob = blended["draw"]
    away_prob = blended["away_win"]

    probs_map = {"home_win": home_prob, "draw": draw_prob, "away_win": away_prob}
    predicted_outcome = max(probs_map, key=probs_map.get)
    confidence = round(max(home_prob, draw_prob, away_prob), 4)

    # Expected goals (simple Poisson-like estimate from avg goals)
    f1 = get_team_features(team1)
    f2 = get_team_features(team2)
    exp_goals_1 = round(f1["avg_goals_scored"] * f2["avg_goals_conceded"] / 1.5, 2)
    exp_goals_2 = round(f2["avg_goals_scored"] * f1["avg_goals_conceded"] / 1.5, 2)

    return {
        "team1": team1,
        "team2": team2,
        "home_win_prob": round(float(home_prob), 4),
        "draw_prob": round(float(draw_prob), 4),
        "away_win_prob": round(float(away_prob), 4),
        "predicted_outcome": predicted_outcome,
        "confidence": confidence,
        "expected_goals": {"team1": exp_goals_1, "team2": exp_goals_2},
        "ml_probs": {k: round(v, 4) for k, v in ml_probs.items()},
        "heuristic_probs": {k: round(v, 4) for k, v in heur_probs.items()},
    }


def get_accuracy_log() -> list:
    if not os.path.exists(ACCURACY_LOG_PATH):
        return []
    with open(ACCURACY_LOG_PATH, "r") as f:
        return json.load(f)


# Prediction cache
_prediction_cache = {}


def get_cached_prediction(match_id: str, team1: str, team2: str, round_type: str) -> dict:
    if team1 == "TBD" or team2 == "TBD":
        return {
            "team1": team1, "team2": team2,
            "home_win_prob": 0.33, "draw_prob": 0.34, "away_win_prob": 0.33,
            "predicted_outcome": "draw", "confidence": 0.34,
            "expected_goals": {"team1": 0, "team2": 0},
        }
    key = f"{match_id}_{team1}_{team2}"
    if key not in _prediction_cache:
        _prediction_cache[key] = predict_match(team1, team2, round_type)
    return _prediction_cache[key]


def invalidate_prediction_cache():
    global _prediction_cache
    _prediction_cache = {}
