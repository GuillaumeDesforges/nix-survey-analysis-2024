# %%
import json
import re
import traceback
import typing
from copy import deepcopy
from pathlib import Path
from typing import Literal

import altair as alt
import polars as pl

with open("data/survey.json") as f:
    survey = json.load(f)

df = pl.read_csv("data/results-survey2024.csv").filter(
    pl.col("submitdate. Date submitted") != ""
)

OUTPUT_PATH = Path("output/")


# %%
NOT_ANSWERED = "Not answered"
OTHER = "Other"


def get_actual_choices(
    choices: list[str],
    allow_other: bool,
) -> list[str]:
    choices = deepcopy(choices)
    choices += [NOT_ANSWERED]
    if allow_other:
        choices += [OTHER]
    return choices


if typing.TYPE_CHECKING:
    QuestionTypes = (
        Literal["single"] | Literal["multiple"] | Literal["ranking"] | Literal["text"]
    )


def compute_stats(
    i_question: int,
    question: dict,
    df: pl.DataFrame,
) -> pl.DataFrame:
    df = df.__copy__()
    question = deepcopy(question)
    question_type: "QuestionTypes" = question["type"]
    question_allow_other = question.get("allow_other", False)
    question_choices = get_actual_choices(
        question.get("choices", []),
        allow_other=question_allow_other,
    )

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
            for v in answers["choices"].to_list():
                assert (
                    v in question_choices
                ), f"'{v}' not in choices: {', '.join(question_choices)}"
        case "multiple":
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
                .fill_null(0)
            )
        case "ranking":
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
            raise ValueError(f"Not implemented for question type {question_type}")

    return answers


def plot_answers(
    question: dict,
    answers: pl.DataFrame,
) -> alt.Chart | alt.LayerChart | alt.FacetChart:
    chart = alt.Chart(answers, height=alt.Step(40))
    question = deepcopy(question)
    question_type: "QuestionTypes" = question["type"]
    question_prompt = question["prompt"]
    question_allow_other = question.get("allow_other", False)
    question_keep_choice_order = question.get("keep_choice_order", False)
    question_choices = get_actual_choices(
        question.get("choices", []),
        allow_other=question_allow_other,
    )

    def y_sort(y):
        if question_keep_choice_order:
            return y.sort(question_choices)
        else:
            return y.sort("-x")

    bar_color: alt.Color
    row: alt.Row | None
    match question_type:
        case "single":
            chart = chart.encode(
                y=y_sort(alt.Y("choices")).title("Choices"),
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
                    y=alt.Y("variable").sort(["Yes", "No"]).title(None),
                    x=alt.X("count").title("Count"),
                    text=alt.Text("percent:Q", format=".1%"),
                )
            )
            bar_color = alt.Color("variable", legend=None, sort="-x")
            facet_choice_order: list[str]
            if question_keep_choice_order:
                facet_choice_order = question_choices
            else:
                facet_choice_order = (
                    answers.filter(pl.col("variable") == "Yes")
                    .sort(by="count", descending=True)["choices"]
                    .to_list()
                )
            row = alt.Row(
                "choices",
                sort=facet_choice_order,
                header=alt.Header(
                    labelAngle=0,
                    labelAlign="left",  # https://github.com/vega/vega/issues/2233
                ),
            ).title("Choices")
        case _:
            raise ValueError(f"Not implemented for question type {question_type}")

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
    question = deepcopy(survey["questions"][i_question])
    answers = compute_stats(
        i_question=i_question,
        question=question,
        df=df,
    )
    chart = plot_answers(
        question=question,
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


# helpful to debug
process_question(
    survey=survey,
    i_question=17,
    output_path=OUTPUT_PATH,
)

# %%
# RUN ALL
for i_question in range(len(survey["questions"])):
    try:
        process_question(
            survey=survey,
            i_question=i_question,
            output_path=OUTPUT_PATH,
        )
    except Exception:
        print("Failed", i_question)
        print(traceback.format_exc())

# %%
