# %%
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Literal
import typing

import polars as pl
import altair as alt

with open("data/survey.json") as f:
    survey = json.load(f)

df = pl.read_csv("data/results-survey2024.csv").filter(
    pl.col("submitdate. Date submitted") != ""
)


# %%
NOT_ANSWERED = "Not answered"
OTHER = "Other"

if typing.TYPE_CHECKING:
    QuestionTypes = Literal["single"] | Literal["multiple"] | Literal["ranking"]


def compute_stats(
    i_question: int,
    question_type: "QuestionTypes",
    question_allow_other: bool,
    df: pl.DataFrame,
) -> pl.DataFrame:
    df = df.__copy__()
    question = deepcopy(survey["questions"][i_question])

    answers: pl.DataFrame
    match question_type:
        case "single":
            choice_columns = [
                c for c in df.columns if c.startswith(f"q{i_question+1:02d}.")
            ]
            assert len(choice_columns) == 1
            df = df.rename({choice_columns[0]: "choices"})
            s = df["choices"]
            s = s.set(s.str.len_chars() == 0, NOT_ANSWERED)
            answers = s.value_counts()
            answers = answers.with_columns(
                (pl.col("count") / pl.sum("count")).alias("percentage")
            )
            # check values are within designed choices
            choices = question["choices"]
            choices += [NOT_ANSWERED]
            if question_allow_other:
                choices += [OTHER]
            for v in answers["choices"].to_list():
                assert v in choices, f"'{v}' not in choices: {', '.join(choices)}"
        case "multiple":
            choices = question["choices"]
            choice_columns = [
                c for c in df.columns if c.startswith(f"q{i_question+1:02d}[")
            ]
            answers = (
                df[choice_columns]
                .unpivot()
                .pivot(
                    on="value",
                    values="value",
                    index="variable",
                    aggregate_function="len",
                )
                .with_columns(pl.col("variable").str.extract(r".*\[(.+)\]"))
                .rename({"variable": "choices"})
                .unpivot(index=["choices"])
                .rename({"value": "count"})
            )
        case "ranking":
            choices = question["choices"]
            choice_columns = [
                c for c in df.columns if c.startswith(f"q{i_question+1:02d}[")
            ]
            answers = (
                df[choice_columns]
                .unpivot()
                .pivot(
                    "variable", values="value", index="value", aggregate_function="len"
                )
                .fill_null(0)
                .rename(
                    {
                        c: m.group(1)
                        if (m := re.match(r".*(Rank (\d+))", c)) is not None
                        else ""
                        for c in choice_columns
                    }
                )
            )
        case _:
            raise NotImplementedError(question_type)

    return answers


def plot_answers(
    question_type: "QuestionTypes",
    question_prompt: str,
    answers: pl.DataFrame,
) -> alt.Chart | alt.LayerChart | alt.FacetChart:
    chart = alt.Chart(answers, height=alt.Step(40))

    bar_color: alt.Color
    row: alt.Row | None
    match question_type:
        case "single":
            chart = chart.encode(
                y=alt.Y("choices").sort("-x").title("Choices"),
                x=alt.X("count").title("Count"),
                text=alt.Text("percentage:Q", format=".1%"),
            )
            bar_color = alt.Color("choices", legend=None, sort="-x")
            row = None
        case "multiple":
            chart = (
                chart.transform_joinaggregate(total="sum(count)", groupby=["choices"])
                .transform_calculate(percent="datum.count / datum.total")
                .encode(
                    y=alt.Y("variable").sort("-x").title("Selected"),
                    x=alt.X("count").title("Count"),
                    text=alt.Text("percent:Q", format=".1%"),
                )
            )
            bar_color = alt.Color("variable", legend=None, sort="-x")
            row = alt.Row(
                "choices",
                header=alt.Header(
                    labelAngle=0,
                    labelAlign="left",  # https://github.com/vega/vega/issues/2233
                ),
            ).title("Choices")
        case "ranking":
            raise NotImplementedError

    chart = chart.encode(color=bar_color).mark_bar(size=25) + chart.mark_text(
        baseline="middle",
        align="left",
        dx=5,
        size=10,
    )
    if row:
        chart = chart.facet(row=row)
    chart = chart.properties(title=question_prompt).configure_title(anchor="middle")
    return chart


# i_question = 5
# question_type = survey["questions"][i_question]["type"]
# question_prompt = survey["questions"][i_question]["prompt"]
# question_allow_other = survey["questions"][i_question].get("allow_other", False)
# answers = compute_stats(
#     i_question=i_question,
#     question_type=question_type,
#     question_allow_other=question_allow_other,
#     df=df,
# )
# plot = plot_answers(
#     question_type=question_type,
#     question_prompt=question_prompt,
#     answers=answers,
# )
# plot


# %%
def strip_prompt(prompt: str):
    m = re.match(r"(.+\?)", prompt)
    if m is not None:
        return m.group(1)
    else:
        return prompt


def process_question(
    survey: dict,
    i_question: int,
    output_path: Path,
):
    question = survey["questions"][i_question]
    question_prompt: str = strip_prompt(question["prompt"])
    question_type = question["type"]
    if question_type == "text":
        print(f"Skip text question {i_question} ({question_prompt})")
        return
    question_allow_other = question.get("allow_other", False)
    answers = compute_stats(
        i_question=i_question,
        question_type=question_type,
        question_allow_other=question_allow_other,
        df=df,
    )
    chart = plot_answers(
        question_type=question_type,
        question_prompt=question_prompt,
        answers=answers,
    )

    output_path.mkdir(parents=True, exist_ok=True)

    answers_path = output_path / f"answers_{i_question:02}.json"
    with open(answers_path, "w") as f:
        json.dump(obj=answers.to_dicts(), fp=f)

    chart_plot_path = output_path / f"chart_plot_{i_question:02}.png"
    with open(chart_plot_path, "wb") as f:
        chart.save(fp=f, format="png", scale_factor=3)

    chart_json_path = output_path / f"chart_json_{i_question:02}.json"
    with open(chart_json_path, "w") as f:
        chart.save(fp=f, format="json")


OUTPUT_PATH = Path("output/")
for i_question in range(len(survey["questions"])):
    try:
        process_question(
            survey=survey,
            i_question=i_question,
            output_path=OUTPUT_PATH,
        )
    except Exception:
        print("Failed", i_question)

# %%
# process_question(
#     survey=survey,
#     i_question=22,
#     output_path=OUTPUT_PATH,
# )
