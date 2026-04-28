# Dream11 AI Team Generator

This project uses machine learning to predict player performances and optimize Dream11 teams for IPL matches.

## Features

- Predicts player fantasy points using XGBoost and LSTM models
- Optimizes team selection with Integer Linear Programming
- Generates balanced teams with constraints (max 7 from one team, budget, etc.)

## Local Team Generation

To generate a Dream11 team for a specific match between two teams:

```bash
python generate_team.py "Team 1" "Team 2"
```

Example:
```bash
python generate_team.py "Chennai Super Kings" "Mumbai Indians"
```

This will output the best 11-player team with predicted points.

## Requirements

- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

## Project Structure

- `src/`: Core modules (preprocess, feature engineering, predict, optimize)
- `models/`: Trained ML models
- `datasets/`: IPL data files
- `generate_team.py`: Local team generation script

## Troubleshooting

- Ensure datasets are in the `../datasets/` folder
- Models should be in `models/` folder
- If no players found, check team names match exactly (case-sensitive)