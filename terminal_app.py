"""
World Cup AI Predictor - Terminal Interface 
Interactive CLI for match predictions, team stats, and tournament results.
Run: python terminal_app.py
"""
import sys
import os
import io

# Fix Windows terminal encoding for Unicode chars
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.system("")  # enable ANSI escape codes on Windows

# Add backend to path so we can import ml_model and data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_model import train_model, predict_match, get_accuracy_log, load_model, invalidate_prediction_cache
from data import HISTORICAL_MATCHES, TEAM_STATS, WC2026_FIXTURES, HEAD_TO_HEAD

# ANSI colors
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def bar(prob, width=30, color=C.GREEN):
    """Draw a horizontal probability bar."""
    filled = int(prob * width)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{C.RESET} {prob*100:5.1f}%"


def outcome_label(outcome, team1, team2):
    if outcome == "home_win":
        return f"{C.GREEN}{team1} WIN{C.RESET}"
    elif outcome == "away_win":
        return f"{C.RED}{team2} WIN{C.RESET}"
    return f"{C.YELLOW}DRAW{C.RESET}"


def print_header(title):
    width = 60
    print()
    print(f"{C.BG_BLUE}{C.BOLD}{' ' * width}{C.RESET}")
    print(f"{C.BG_BLUE}{C.BOLD}  {title:^{width-4}}  {C.RESET}")
    print(f"{C.BG_BLUE}{C.BOLD}{' ' * width}{C.RESET}")
    print()


def print_divider():
    print(f"{C.DIM}{'─' * 60}{C.RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Screens
# ─────────────────────────────────────────────────────────────────────────────

def show_prediction(team1, team2, round_type="group"):
    """Show full prediction for a match."""
    pred = predict_match(team1, team2, round_type)

    print_header(f"{team1}  vs  {team2}")
    print(f"  {'Stage:':<12} {round_type.upper()}")
    print(f"  {'Expected:':<12} {pred['expected_goals']['team1']:.1f} - {pred['expected_goals']['team2']:.1f}")
    print(f"  {'Verdict:':<12} {outcome_label(pred['predicted_outcome'], team1, team2)}")
    print(f"  {'Confidence:':<12} {pred['confidence']*100:.1f}%")
    print()
    print(f"  {C.BOLD}Win Probabilities:{C.RESET}")
    print(f"    {team1:<20} {bar(pred['home_win_prob'], color=C.GREEN)}")
    print(f"    {'Draw':<20} {bar(pred['draw_prob'], color=C.YELLOW)}")
    print(f"    {team2:<20} {bar(pred['away_win_prob'], color=C.RED)}")
    print()
    print(f"  {C.DIM}ML vs Heuristic breakdown:{C.RESET}")
    ml = pred.get("ml_probs", {})
    hr = pred.get("heuristic_probs", {})
    print(f"    {'Source':<12} {'Win1':>8} {'Draw':>8} {'Win2':>8}")
    print(f"    {'ML':<12} {ml.get('home_win',0)*100:>7.1f}% {ml.get('draw',0)*100:>7.1f}% {ml.get('away_win',0)*100:>7.1f}%")
    print(f"    {'Heuristic':<12} {hr.get('home_win',0)*100:>7.1f}% {hr.get('draw',0)*100:>7.1f}% {hr.get('away_win',0)*100:>7.1f}%")
    print_divider()


def show_all_predictions():
    """Show predictions for all 2026 WC fixtures."""
    print_header("2026 World Cup - All Match Predictions")
    print(f"  {'Match':<8} {'Stage':<16} {'Teams':<30} {'Prediction':<12} {'Conf':>6}")
    print_divider()

    for f in WC2026_FIXTURES:
        if f["team1"] == "TBD" or f["team2"] == "TBD":
            continue
        teams = f"{f['team1']} vs {f['team2']}"
        pred = predict_match(f["team1"], f["team2"], f.get("round", "group"))
        outcome = pred["predicted_outcome"]
        if outcome == "home_win":
            winner = f["team1"]
        elif outcome == "away_win":
            winner = f["team2"]
        else:
            winner = "Draw"
        actual = f.get("result")
        actual_str = ""
        if actual:
            if actual == "home_win":
                actual_str = f"  → {C.GREEN}{f['team1']} won {f['score1']}-{f['score2']}{C.RESET}"
            elif actual == "away_win":
                actual_str = f"  → {C.RED}{f['team2']} won {f['score1']}-{f['score2']}{C.RESET}"
            else:
                actual_str = f"  → {C.YELLOW}Draw {f['score1']}-{f['score2']}{C.RESET}"

        correct = ""
        if actual:
            if actual == outcome:
                correct = f"  {C.GREEN}✓{C.RESET}"
            else:
                correct = f"  {C.RED}✗{C.RESET}"

        print(f"  {f['match_id']:<8} {f['stage']:<16} {teams:<30} {winner:<12} {pred['confidence']*100:>5.1f}%{correct}{actual_str}")
    print_divider()


def show_team_list():
    """List all teams sorted by FIFA rank."""
    print_header("All Teams (by FIFA Ranking)")
    print(f"  {'#':<4} {'Team':<20} {'Rank':>5} {'Win%':>6} {'Goals':>6} {'Conc':>6} {'WC':>3} {'Continent':<10} {'Star Player'}")
    print_divider()

    sorted_teams = sorted(TEAM_STATS.items(), key=lambda x: x[1]["fifa_rank"])
    for i, (name, s) in enumerate(sorted_teams, 1):
        print(f"  {i:<4} {C.CYAN}{name:<20}{C.RESET} {s['fifa_rank']:>5} "
              f"{s['win_rate']*100:>5.0f}% {s['avg_goals_scored']:>5.1f} "
              f"{s['avg_goals_conceded']:>5.1f} {s['wc_titles']:>3} "
              f"{s['continent']:<10} {s['star_player']}")

    print()
    print(f"  {C.DIM}Total: {len(sorted_teams)} teams{C.RESET}")
    print_divider()


def show_team_detail(team_name):
    """Show detailed stats for one team."""
    # Fuzzy match
    matched = None
    for key in TEAM_STATS:
        if key.lower() == team_name.lower():
            matched = key
            break
    if not matched:
        for key in TEAM_STATS:
            if team_name.lower() in key.lower():
                matched = key
                break
    if not matched:
        print(f"\n  {C.RED}Team '{team_name}' not found.{C.RESET}")
        return

    s = TEAM_STATS[matched]
    print_header(f"Team: {matched}")
    print(f"  {'FIFA Rank:':<15} {C.BOLD}{s['fifa_rank']}{C.RESET}")
    print(f"  {'Win Rate:':<15} {s['win_rate']*100:.1f}%")
    print(f"  {'Avg Goals:':<15} {s['avg_goals_scored']} scored / {s['avg_goals_conceded']} conceded")
    print(f"  {'WC Titles:':<15} {s['wc_titles']}")
    print(f"  {'Continent:':<15} {s['continent']}")
    print(f"  {'Star Player:':<15} {C.MAGENTA}{s['star_player']}{C.RESET}")
    print(f"  {'Recent Form:':<15} ", end="")
    for g in s["form"]:
        if g == 1:
            print(f"{C.GREEN}●{C.RESET}", end=" ")
        else:
            print(f"{C.RED}○{C.RESET}", end=" ")
    print(f"({sum(s['form'])}/5)")
    print()

    # WC 2026 matches
    wc2026 = [f for f in WC2026_FIXTURES if f["team1"] == matched or f["team2"] == matched]
    if wc2026:
        print(f"  {C.BOLD}2026 World Cup Matches:{C.RESET}")
        for f in wc2026:
            is_home = f["team1"] == matched
            opp = f["team2"] if is_home else f["team1"]
            s1, s2 = f["score1"], f["score2"]
            my_score = s1 if is_home else s2
            opp_score = s2 if is_home else s1
            if my_score > opp_score:
                result_color = C.GREEN
                result_str = "W"
            elif my_score < opp_score:
                result_color = C.RED
                result_str = "L"
            else:
                result_color = C.YELLOW
                result_str = "D"
            print(f"    {result_color}[{result_str}]{C.RESET} vs {opp:<16} {my_score}-{opp_score}  {f['stage']}")
    print_divider()


def show_compare(team1, team2):
    """Compare two teams side by side."""
    t1 = TEAM_STATS.get(team1)
    t2 = TEAM_STATS.get(team2)
    if not t1 or not t2:
        print(f"\n  {C.RED}One or both teams not found.{C.RESET}")
        print(f"  Available: {', '.join(TEAM_STATS.keys())}")
        return

    print_header(f"{team1}  vs  {team2}")

    # H2H
    h2h = HEAD_TO_HEAD.get((team1, team2)) or HEAD_TO_HEAD.get((team2, team1))
    if h2h:
        is_rev = (team2, team1) in HEAD_TO_HEAD
        t1_wins = h2h["team2_wins"] if is_rev else h2h["team1_wins"]
        t2_wins = h2h["team1_wins"] if is_rev else h2h["team2_wins"]
        print(f"  {C.BOLD}Head-to-Head Record:{C.RESET}")
        print(f"    Played: {h2h['played']}  |  {team1}: {t1_wins}W  |  Draw: {h2h['draws']}  |  {team2}: {t2_wins}W")
        print()

    # Stats comparison
    print(f"  {'Metric':<20} {C.CYAN}{team1:<18}{C.RESET} {C.CYAN}{team2:<18}{C.RESET}")
    print_divider()
    rows = [
        ("FIFA Rank", t1["fifa_rank"], t2["fifa_rank"], "lower"),
        ("Win Rate", f"{t1['win_rate']*100:.0f}%", f"{t2['win_rate']*100:.0f}%", "higher"),
        ("Avg Goals Scored", t1["avg_goals_scored"], t2["avg_goals_scored"], "higher"),
        ("Avg Goals Conceded", t1["avg_goals_conceded"], t2["avg_goals_conceded"], "lower"),
        ("WC Titles", t1["wc_titles"], t2["wc_titles"], "higher"),
        ("Form (last 5)", f"{sum(t1['form'])}/5", f"{sum(t2['form'])}/5", "higher"),
        ("Continent", t1["continent"], t2["continent"], None),
        ("Star Player", t1["star_player"], t2["star_player"], None),
    ]
    for label, v1, v2, better in rows:
        if better == "lower":
            winner_color = C.GREEN if v1 < v2 else (C.RED if v1 > v2 else C.YELLOW)
            print(f"  {label:<20} {winner_color}{str(v1):<18}{C.RESET} {str(v2):<18}")
        elif better == "higher":
            try:
                n1, n2 = float(str(v1).rstrip("%")), float(str(v2).rstrip("%"))
                winner_color = C.GREEN if n1 > n2 else (C.RED if n1 < n2 else C.YELLOW)
            except ValueError:
                winner_color = C.RESET
            print(f"  {label:<20} {winner_color}{str(v1):<18}{C.RESET} {str(v2):<18}")
        else:
            print(f"  {label:<20} {str(v1):<18} {str(v2):<18}")
    print()

    # Prediction
    print(f"  {C.BOLD}AI Prediction:{C.RESET}")
    pred = predict_match(team1, team2, "group")
    print(f"    {team1:<18} {bar(pred['home_win_prob'], color=C.GREEN)}")
    print(f"    {'Draw':<18} {bar(pred['draw_prob'], color=C.YELLOW)}")
    print(f"    {team2:<18} {bar(pred['away_win_prob'], color=C.RED)}")
    print(f"    → {outcome_label(pred['predicted_outcome'], team1, team2)} ({pred['confidence']*100:.1f}%)")
    print_divider()


def show_model_info():
    """Show model training info and accuracy."""
    print_header("ML Model Information")
    log = get_accuracy_log()
    if log:
        latest = log[-1]
        print(f"  {'Model Type:':<15} {C.CYAN}{latest.get('model_type', 'N/A')}{C.RESET}")
        print(f"  {'Accuracy:':<15} {C.GREEN}{latest['accuracy']*100:.1f}%{C.RESET} (CV std: {latest.get('cv_std', 'N/A')})")
        print(f"  {'CV Splits:':<15} {latest.get('cv_splits', 'N/A')}")
        print(f"  {'Train Samples:':<15} {latest['train_samples']}")
        print(f"  {'Total Retrains:':<15} {len(log)}")
        print()
        print(f"  {C.BOLD}Training History:{C.RESET}")
        for entry in log[-5:]:
            ts = entry["timestamp"][:19].replace("T", " ")
            print(f"    {ts}  acc={entry['accuracy']*100:.1f}%  samples={entry['train_samples']}")
    else:
        print(f"  {C.YELLOW}No model trained yet. Training now...{C.RESET}")
        result = train_model()
        print(f"  Result: {result}")
    print_divider()


def show_wc2026_results():
    """Show all 2026 World Cup results."""
    print_header("2026 World Cup - Final Results")
    current_stage = ""
    for f in WC2026_FIXTURES:
        if f["stage"] != current_stage:
            current_stage = f["stage"]
            print(f"\n  {C.MAGENTA}{C.BOLD}=== {current_stage} ==={C.RESET}")

        s1, s2 = f["score1"], f["score2"]
        r = f["result"]
        if r == "home_win":
            line = f"    {C.GREEN}{f['team1']:<16} {s1}-{s2} {f['team2']:<16}{C.RESET}"
        elif r == "away_win":
            line = f"    {f['team1']:<16} {s1}-{s2} {C.GREEN}{f['team2']:<16}{C.RESET}"
        else:
            line = f"    {C.YELLOW}{f['team1']:<16} {s1}-{s2} {f['team2']:<16}{C.RESET}"
        print(line)
    print()
    print_divider()


def show_accuracy_check():
    """Check how well the model predicted the 2026 WC results."""
    print_header("2026 Prediction Accuracy Check")
    correct = 0
    total = 0
    correct_strict = 0  # exact outcome match
    results_by_stage = {}

    for f in WC2026_FIXTURES:
        if f["team1"] == "TBD" or f["team2"] == "TBD":
            continue
        if f["result"] is None:
            continue

        pred = predict_match(f["team1"], f["team2"], f.get("round", "group"))
        actual = f["result"]
        predicted = pred["predicted_outcome"]

        total += 1
        stage = f["stage"]
        if stage not in results_by_stage:
            results_by_stage[stage] = {"correct": 0, "total": 0}
        results_by_stage[stage]["total"] += 1

        if actual == predicted:
            correct += 1
            correct_strict += 1
            results_by_stage[stage]["correct"] += 1
            mark = f"{C.GREEN}✓{C.RESET}"
        else:
            mark = f"{C.RED}✗{C.RESET}"

        # Also check if actual winner was in top-2 predicted
        actual_winner = actual  # home_win/away_win/draw

    print(f"  {'Match':<8} {'Teams':<30} {'Predicted':<12} {'Actual':<12} {'Result'}")
    print_divider()

    for f in WC2026_FIXTURES:
        if f["team1"] == "TBD" or f["team2"] == "TBD" or f["result"] is None:
            continue
        pred = predict_match(f["team1"], f["team2"], f.get("round", "group"))
        predicted = pred["predicted_outcome"]
        actual = f["result"]
        if predicted == actual:
            mark = f"{C.GREEN}✓{C.RESET}"
        else:
            mark = f"{C.RED}✗{C.RESET}"
        teams = f"{f['team1']} vs {f['team2']}"
        print(f"  {f['match_id']:<8} {teams:<30} {predicted:<12} {actual:<12} {mark}")

    print_divider()
    print(f"\n  {C.BOLD}Overall Accuracy: {C.GREEN}{correct}/{total} = {correct/total*100:.1f}%{C.RESET}")
    print()
    print(f"  {C.BOLD}By Stage:{C.RESET}")
    for stage in results_by_stage:
        r = results_by_stage[stage]
        pct = r["correct"]/r["total"]*100 if r["total"] > 0 else 0
        color = C.GREEN if pct >= 50 else C.YELLOW if pct >= 33 else C.RED
        print(f"    {stage:<20} {color}{r['correct']}/{r['total']} = {pct:.0f}%{C.RESET}")
    print_divider()


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Menu
# ─────────────────────────────────────────────────────────────────────────────

def menu_predict():
    """Interactive match prediction."""
    print_header("Match Prediction")
    print(f"  Available teams: {', '.join(sorted(TEAM_STATS.keys()))}")
    print()
    t1 = input(f"  {C.CYAN}Enter Team 1: {C.RESET}").strip()
    t2 = input(f"  {C.CYAN}Enter Team 2: {C.RESET}").strip()
    r = input(f"  {C.CYAN}Round type (group/knockout) [group]: {C.RESET}").strip() or "group"

    # Fuzzy match
    def fuzzy(name):
        for k in TEAM_STATS:
            if k.lower() == name.lower():
                return k
        for k in TEAM_STATS:
            if name.lower() in k.lower():
                return k
        return name

    t1 = fuzzy(t1)
    t2 = fuzzy(t2)
    if t1 not in TEAM_STATS or t2 not in TEAM_STATS:
        print(f"\n  {C.RED}Team not found. Check spelling.{C.RESET}")
        input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")
        return

    show_prediction(t1, t2, r)
    input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")


def menu():
    """Main menu loop."""
    while True:
        clear()
        print(f"\n{C.BG_BLUE}{C.BOLD}{' ' * 60}{C.RESET}")
        print(f"{C.BG_BLUE}{C.BOLD}{'    World Cup AI Predictor  ':^56}{C.RESET}")
        print(f"{C.BG_BLUE}{C.BOLD}{' ' * 60}{C.RESET}")
        print()
        print(f"  {C.BOLD}Menu:{C.RESET}")
        print(f"    {C.CYAN}1.{C.RESET} Predict a match (custom)")
        print(f"    {C.CYAN}2.{C.RESET} View all 2026 WC predictions")
        print(f"    {C.CYAN}3.{C.RESET} List all teams")
        print(f"    {C.CYAN}4.{C.RESET} Team detail view")
        print(f"    {C.CYAN}5.{C.RESET} Compare two teams")
        print(f"    {C.CYAN}6.{C.RESET} Model info & accuracy")
        print(f"    {C.CYAN}7.{C.RESET} 2026 WC final results")
        print(f"    {C.CYAN}8.{C.RESET} Prediction accuracy check")
        print(f"    {C.CYAN}9.{C.RESET} Retrain model")
        print(f"    {C.RED}0.{C.RESET} Exit")
        print()

        choice = input(f"  {C.BOLD}Choice: {C.RESET}").strip()

        if choice == "1":
            menu_predict()
        elif choice == "2":
            show_all_predictions()
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")
        elif choice == "3":
            show_team_list()
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")
        elif choice == "4":
            print_header("Team Detail")
            name = input(f"  {C.CYAN}Team name: {C.RESET}").strip()
            show_team_detail(name)
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")
        elif choice == "5":
            print_header("Compare Teams")
            t1 = input(f"  {C.CYAN}Team 1: {C.RESET}").strip()
            t2 = input(f"  {C.CYAN}Team 2: {C.RESET}").strip()
            show_compare(t1, t2)
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")
        elif choice == "6":
            show_model_info()
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")
        elif choice == "7":
            show_wc2026_results()
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")
        elif choice == "8":
            show_accuracy_check()
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")
        elif choice == "9":
            print_header("Retraining Model...")
            invalidate_prediction_cache()
            result = train_model()
            print(f"  {C.GREEN}Done!{C.RESET}")
            print(f"  Accuracy: {result.get('accuracy', 'N/A')}")
            print(f"  Train samples: {result.get('train_samples', 'N/A')}")
            if result.get("top_features"):
                print(f"\n  {C.BOLD}Top Features:{C.RESET}")
                for name, imp in result["top_features"][:5]:
                    print(f"    {name:<25} {imp*100:.1f}%")
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")
        elif choice == "0":
            print(f"\n  {C.CYAN}Goodbye!{C.RESET}\n")
            break
        else:
            input(f"  {C.RED}Invalid choice.{C.RESET} Press Enter...")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Train model on first run
    print(f"{C.DIM}Initializing model...{C.RESET}")
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "model.joblib")):
        print(f"{C.DIM}Training model (first run)...{C.RESET}")
        train_model()
    else:
        load_model()
    print(f"{C.GREEN}Model ready!{C.RESET}\n")

    menu()
