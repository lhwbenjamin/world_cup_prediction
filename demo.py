from terminal_app import (
    show_prediction,
    show_compare,
    show_team_detail,
)
from data import WC2026_FIXTURES
from ml_model import predict_match


def accuracy_summary():
    """Compute overall prediction accuracy on the real 2026 results."""
    correct = total = 0
    for f in WC2026_FIXTURES:
        if f["team1"] == "TBD" or f["team2"] == "TBD" or f["result"] is None:
            continue
        pred = predict_match(f["team1"], f["team2"], f.get("round", "group"))
        if pred["predicted_outcome"] == f["result"]:
            correct += 1
        total += 1
    pct = (correct / total * 100) if total else 0
    print(f"  Overall accuracy on 2026 results: {correct}/{total} = {pct:.1f}%")


def main():
    print()
    print("=" * 64)
    print("  WORLD CUP AI PREDICTOR - DEMO OUTPUT (no user input required)")
    print("=" * 64)
    print()

    print(">> (1) Single match prediction  -  2026 Final")
    show_prediction("Spain", "Argentina", "knockout")

    print(">> (2) Head-to-head comparison  -  France vs England")
    show_compare("France", "England")

    print(">> (3) Team detail  -  Spain")
    show_team_detail("Spain")

    print(">> (4) Model accuracy on the real 2026 tournament")
    accuracy_summary()
    print()


if __name__ == "__main__":
    main()
