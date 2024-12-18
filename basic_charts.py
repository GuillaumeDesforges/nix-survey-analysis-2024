# %%
import json
import re
import typing
from copy import deepcopy
from pathlib import Path
from textwrap import wrap
from typing import Literal

import altair as alt
import polars as pl

with open("data/survey.json") as f:
    survey = json.load(f)

df = pl.read_csv("data/results-survey2024.csv").filter(
    pl.col("submitdate. Date submitted") != ""
)

with open("data/results-survey2024-text_answers.json") as f:
    text_answers = {
        question_id: [
            {"choice": choice, "count": count} for choice, count in answers.items()
        ]
        for question_id, answers in json.load(f).items()
    }

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
    question: dict,
    df: pl.DataFrame,
    text_answers: dict,
) -> pl.DataFrame:
    df = df.__copy__()
    question = deepcopy(question)
    question_id = question["id"]
    question_type: "QuestionTypes" = question["type"]
    question_allow_other = question.get("allow_other", False)
    question_choices = get_actual_choices(
        question.get("choices", []),
        allow_other=question_allow_other,
    )

    answers: pl.DataFrame
    match question_type:
        case "single":
            choice_columns = [c for c in df.columns if c.startswith(f"{question_id}.")]
            assert len(choice_columns) == 1
            df = df.rename({choice_columns[0]: "choice"})
            s = df["choice"]
            s = s.set(s.str.len_chars() == 0, NOT_ANSWERED)
            answers = s.value_counts()
            answers = answers.with_columns(
                (pl.col("count") / pl.sum("count")).alias("percentage")
            )
            # check values are within designed choices
            for v in answers["choice"].to_list():
                assert (
                    v in question_choices
                ), f"'{v}' not in choices: {', '.join(question_choices)}"
        case "multiple":
            choice_columns = [c for c in df.columns if c.startswith(f"{question_id}[")]
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
                .rename({"variable": "choice"})
                .unpivot(index=["choice"])
                .rename({"value": "count"})
                .with_columns(pl.col("count").fill_null(0))
                .with_columns(
                    pl.col("variable")
                    .replace("Yes", "Selected")
                    .replace("No", "Not selected")
                )
            )
            answers = answers.join(
                answers.group_by("choice").agg(pl.sum("count").alias("total")),
                on=["choice"],
            )
            answers = answers.join(
                answers.filter(pl.col("variable") == "Selected").select(
                    "choice", pl.col("count").alias("by_choice_is_selected_count")
                ),
                on=["choice"],
            )
            answers = answers.with_columns(percentage=pl.col("count") / pl.col("total"))
        case "ranking":
            choice_columns = [c for c in df.columns if c.startswith(f"{question_id}[")]
            answers = (
                df[choice_columns]
                .unpivot()
                .pivot(
                    "variable",
                    values="value",
                    index="value",
                    aggregate_function="len",
                )
                .fill_null(0)
                .rename({"value": "choice"})
                .filter(pl.col("choice") != "")
                .rename(
                    {
                        c: f"Rank {int(m.group(2)):02}"
                        if (m := re.match(r".*(Rank (\d+))", c)) is not None
                        else ""
                        for c in choice_columns
                    }
                )
                .select("choice", *(f"Rank {i+1:02}" for i in range(5)))
                .unpivot(index="choice", value_name="count")
                .with_columns(
                    rank_in_rank=pl.col("count").rank(descending=True).over("variable")
                )
                .with_columns(
                    choice=pl.when(pl.col("rank_in_rank") <= 10)
                    .then(pl.col("choice"))
                    .otherwise(pl.lit("Other"))
                )
                .group_by([pl.col("choice"), pl.col("variable")])
                .agg(count=pl.sum("count"))
            )
        case "text":
            answers = pl.from_records(text_answers[question_id])
            count_median = answers.select(pl.median("count")).to_series()[0]
            limit = max(count_median, 10)
            answers = answers.sort("count", descending=True).limit(limit)
        case _:
            raise NotImplementedError(
                f"Not implemented for question type {question_type}"
            )

    return answers


def plot_answers(
    question: dict,
    answers: pl.DataFrame,
) -> alt.Chart | alt.LayerChart | alt.FacetChart:
    question = deepcopy(question)
    question_type: "QuestionTypes" = question["type"]
    question_prompt = strip_prompt(question["prompt"])
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

    chart = alt.Chart(answers, height=alt.Step(40))
    bar_color: alt.Color
    row: alt.Row | None
    bar_text_within = False
    chart_title = question_prompt
    match question_type:
        case "single":
            chart = chart.encode(
                y=y_sort(alt.Y("choice")).title("Choice"),
                x=alt.X("count").title("Count"),
                text=alt.Text("percentage:Q", format=".1%"),
            )
            bar_color = alt.Color("choice", legend=None, sort="-x")
            row = None
        case "multiple":
            chart = chart.transform_calculate(
                percent="datum.count / datum.total"
            ).encode(
                y=alt.Y("choice:N")
                .sort(
                    alt.SortField("by_choice_is_selected_count", order="descending"),
                    op="max",
                )
                .title(None),
                x=alt.X("sum(count):Q").stack("zero").title("Count"),
                detail=alt.Detail("variable:N"),
                order=alt.Order("variable", sort="descending"),
                text=alt.Text("sum(percentage):Q", format=".1%"),
            )
            bar_color = alt.Color("variable:N").title("")
            row = None
        case "ranking":
            chart = chart.encode(
                y=alt.Y("variable:N").title(None),
                x=alt.X("sum(count):Q").stack("zero").title("Count"),
                detail=alt.Detail("choice:N"),
                order=alt.Order("count:N", sort="descending"),
                text=alt.Text("choice[0]:N"),
            )

            bar_color = alt.Color("choice:N", sort="ascending").title("Choice")
            row = None
            bar_text_within = True
            chart_title += " (top 5 ranks, top 10 choices per rank)"
        case "text":
            # unfortunately, does not work for stacked charts
            chart.data = chart.data.with_columns(
                pl.col("choice").map_elements(
                    lambda s: "\n".join(wrap(s, width=30)), return_dtype=pl.String
                )
            )
            chart = chart.transform_calculate(choice="split(datum.choice, '\\n')")

            chart = chart.encode(
                y=y_sort(alt.Y("choice")).title("Choice"),
                x=alt.X("count").title("Count"),
                text=alt.Text("count:Q"),
            )
            bar_color = alt.Color("choice", legend=None, sort="-x")
            row = None
        case _:
            raise NotImplementedError(
                f"Not implemented for question type {question_type}: {question_prompt}"
            )

    bar_text_align, bar_text_dx, bat_text_color = (
        ("right", 0, "white") if bar_text_within else ("left", 5, "black")
    )
    chart = chart.encode(color=bar_color).mark_bar(size=25) + chart.mark_text(
        baseline="middle",
        align=bar_text_align,
        dx=bar_text_dx,
        size=10,
        color=bat_text_color,
    )
    if row:
        chart = chart.facet(row=row)
    chart = chart.properties(title=chart_title).configure_title(anchor="middle")
    return chart


def strip_prompt(prompt: str):
    m = re.match(r"(.+\?)", prompt)
    if m is not None:
        return m.group(1)
    else:
        return prompt


def process_question(
    question: dict,
    output_path: Path,
):
    question = deepcopy(question)
    question_id = question["id"]

    answers = compute_stats(
        question=question,
        df=df,
        text_answers=text_answers,
    )
    chart = plot_answers(
        question=question,
        answers=answers,
    )

    output_path.mkdir(parents=True, exist_ok=True)

    answers_path = output_path / f"answers_{question_id}.json"
    with open(answers_path, "w") as f:
        json.dump(obj=answers.to_dicts(), fp=f)

    chart_plot_path = output_path / f"chart_plot_{question_id}.png"
    with open(chart_plot_path, "wb") as f:
        chart.save(fp=f, format="png", scale_factor=3)

    chart_json_path = output_path / f"chart_json_{question_id}.json"
    with open(chart_json_path, "w") as f:
        chart.save(fp=f, format="json")


# helpful to debug
question = deepcopy(survey["questions"][26])
answers = compute_stats(
    question=question,
    df=df,
    text_answers=text_answers,
)
plot_answers(
    question=question,
    answers=answers,
)

# %%
# RUN ALL
for question in survey["questions"]:
    try:
        process_question(
            question=question,
            output_path=OUTPUT_PATH,
        )
    except NotImplementedError as e:
        print(f"{question["id"]} error={e}")
