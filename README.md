# Nix Community Survey 2024 analysis

A temporary repository to work on the analysis of the results of the Nix Community Survey 2024.

Requirements:
- Python 3.12+

Install dependencies.

```bash
pip install -r requirements.txt
```

Copy survey results CSV extract from LimeSurvey to `data/results-survey2024.csv`.

> [!NOTE]
> `data/results-survey2024-text_answers.json` was a JSON of manually analyzed and aggregated data from the amazing @fricklerhandwerk. <3

Run scripts.

```bash
python basic_charts.py
python advanced_charts.py
```

This will write aggregated data files and charts in `./output`.