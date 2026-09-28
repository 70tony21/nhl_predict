import pandas as pd
from nhl_model.load import load_all
from nhl_model.canonical import build_matches
from nhl_model.goalies import attach_starters
from nhl_model.features import team_5v5_rolling, merge_team_features
from nhl_model.config import OUT_DIR


def main():
    data = load_all()
    for k in data:
        if "game_id" in data[k].columns:
            data[k]["game_id"] = data[k]["game_id"].astype(str)

    matches = build_matches(data["scores"], data["pbp"], shifts=data["shifts"])
    matches = attach_starters(matches, data["shifts"], data["pbp"])

    roll = team_5v5_rolling(data["team_stats"], window=10)
    dataset = merge_team_features(matches, roll, window=10)

    dataset.to_parquet(OUT_DIR / "dataset.parquet", index=False)
    print("dataset:", dataset.shape)
    print(dataset.head())


if __name__ == "__main__":
    main()