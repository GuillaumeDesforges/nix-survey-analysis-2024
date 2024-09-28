# %%
import re

import polars as pl
import yaml

# %%
with open("data/survey.yaml") as f:
    survey = yaml.safe_load(f)

# %%
df = pl.read_csv("data/results-survey2024.csv").filter(
    pl.col("submitdate. Date submitted") != ""
)


# %%
def compute_stats(i_question: int, survey: dict, df: pl.DataFrame) -> pl.DataFrame:
    question = survey["questions"][i_question]
    type_ = question["type"]

    answers: pl.DataFrame
    if type_ == "single":
        choices = question["choices"]
        # empy allowed
        choices += [""]
        columns = [c for c in df.columns if c.startswith(f"q{i_question+1:02d}.")]
        assert len(columns) == 1
        column = columns[0]
        answers = df[column].value_counts()
        for v in answers[column].to_list():
            assert v in choices, f"{v} not in {choices}"
    elif type_ == "multiple":
        choices = question["choices"]
        columns = [c for c in df.columns if c.startswith(f"q{i_question+1:02d}[")]
        answers = (
            df[columns]
            .unpivot()
            .pivot("variable", values="value", index="value", aggregate_function="len")
        )
    elif type_ == "ranking":
        choices = question["choices"]
        columns = [c for c in df.columns if c.startswith(f"q{i_question+1:02d}[")]
        answers = (
            df[columns]
            .unpivot()
            .pivot("variable", values="value", index="value", aggregate_function="len")
            .fill_null(0)
            .rename({c: re.match(r".*(Rank (\d+))", c).group(1) for c in columns})
        )
    else:
        raise NotImplementedError

    return answers


compute_stats(0, survey, df).plot.barh()

# %%
