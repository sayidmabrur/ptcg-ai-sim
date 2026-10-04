import os
from pathlib import Path

# Deck paths in the training code are written relative to rl-train/ (e.g.
# "./decks/main_agent.csv"), which only resolves when that is also the working
# directory. Keeping the root here means a probe run from anywhere still finds
# its decks instead of failing on a path that looks correct in the source.
RL_TRAIN_ROOT = Path(__file__).resolve().parent.parent


def build_deck(path: str | Path) -> list[int]:
    """Read a decklist CSV into the 60 card IDs ``battle_start`` expects.

    The files are one card ID per line, no header. Blank lines and stray
    whitespace are ignored so a trailing newline is not an error.

    Args:
        path: Decklist location, absolute or relative to the working directory;
            a relative path is also tried against rl-train/ before giving up.

    Returns:
        list[int]: The 60 card IDs, in file order.

    Raises:
        FileNotFoundError: No decklist at that path.
        ValueError: The file does not hold exactly 60 integer card IDs.
    """
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

    # The engine rejects a short deck with a bare error code, so the count is
    # checked here where the file name is still around to name in the message.
    if len(ids) != 60:
        raise ValueError(f"{deck_path}: a deck must be 60 cards, found {len(ids)}")
    return ids


def read_deck_csv() -> list[int]:
    """Read deck.csv.

    Returns:
        list[int]: A list of card IDs in the deck.
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck
