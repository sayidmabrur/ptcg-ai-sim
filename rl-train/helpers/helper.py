import csv
import os
from pathlib import Path

RL_TRAIN_ROOT = Path(__file__).resolve().parent.parent


def build_deck(path: str | Path) -> list[int]:
    deck_path = Path(path)
    if not deck_path.is_file() and not deck_path.is_absolute():
        deck_path = RL_TRAIN_ROOT / deck_path
    if not deck_path.is_file():
        raise FileNotFoundError(f"no decklist at {path}")

    ids = []
    for lineno, line in enumerate(deck_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(int(line))
        except ValueError:
            raise ValueError(f"{deck_path}:{lineno}: not a card ID: {line!r}") from None

    if len(ids) != 60:
        raise ValueError(f"{deck_path}: a deck must be 60 cards, found {len(ids)}")
    return ids


def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck


def count_supported_cards(csv_file_path) -> int:
    with open(csv_file_path, encoding="utf-8-sig", newline="") as f:
        return len({row["Card ID"] for row in csv.DictReader(f)})
