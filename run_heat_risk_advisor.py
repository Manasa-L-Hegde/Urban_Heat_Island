from pathlib import Path

import pandas as pd

from src.heat_risk_advisor import answer_questions, build_context


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "outputs" / "city_ward_dataset.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    context = build_context(df)

    questions = [
        "Which areas are most dangerous for elderly people?",
        "Which wards need urgent cooling measures?",
        "Where should trees be planted first?",
    ]

    print("=== Context Summary ===")
    print(context)
    print("\n=== Answers ===\n")

    answers = answer_questions(df, questions)
    for question, answer in zip(questions, answers):
        print(f"Q: {question}")
        print(f"A: {answer}")
        print()


if __name__ == "__main__":
    main()
