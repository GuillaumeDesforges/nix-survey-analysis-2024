# %%
import json
import re
from pathlib import Path

import altair as alt
import polars as pl
from IPython.display import display

OUTPUT_PATH = Path("output/")


def _extract_question_choice_id(full_column_name: str) -> str | None:
    m = re.match(r"^(q[0-9]{2}(\[.*\])?)\..+", full_column_name)
    if m is None:
        return None
    return m.group(1)


assert _extract_question_choice_id("q01. Where do you live?") == "q01"
assert (
    _extract_question_choice_id(
        "q12[SQ001]. Which user types do you identify with? Select all that apply   \tType A: You love the idea behind Nix or NixOS.Maybe you spread the word among friends and coworkers.Maybe you are interested in or enthusiastic about any of the following: \t \t\tFree and open source software \t\tLinux \t\tHome automation \t\tOnline communities \t\tDistro-hopping \t \t \tType B. You’re here because you’re curious about Nix and how it works.Maybe you enjoy or want to learn functional programming, or you just want to learn new things.Maybe you identify yourself as: \t \t\tNix-curious developer \t\tStudent of a technical field \t\tEducator \t\tAcademic researcher \t \t \tType C: You use Nix or NixOS to get things done or boost your team’s productivity, you learn it to grow your career opportunities, or you have to use it on the job because someone said so.Maybe you identify yourself as: \t \t\tSystem administrator \t\tEmployee developer \t\tDevOps engineer \t\tNatural scientist \t\tOpen source software author \t\tEarly-career professional \t \t \tType D: You work or want to work on the Nix ecosystem rather than just with it.You are at least one of the following: \t \t\tAspiring contributor \t\tNovice contributor \t\tDrive-by contributor \t\tPackage maintainer \t\tCode owner \t\tCommunity team member \t\tSponsor \t \t \tType E: You make the strategic decisions for your team or company: which technologies to adopt, which skills to train your employees in, which projects to support or invest in, which services or products to offer.Maybe you’re a: \t \t\tEntrepreneur \t\tTeam lead \t\tSoftware architect \t\tCTO \t\tSales executive \t\tPublic service administrator \t \t   [A. I love the idea behind Nix]"
    )
    == "q12[SQ001]"
)
df: pl.DataFrame = pl.read_csv("./data/results-survey2024.csv")
original_columns = df.columns.copy()
df = df.rename(
    {
        full_column_name: column_name
        for full_column_name in df.columns
        if (column_name := _extract_question_choice_id(full_column_name)) is not None
    }
)
del _extract_question_choice_id
df: pl.DataFrame = df.with_columns(
    pl.when(pl.col(pl.String).str.len_chars() == 0)
    .then(pl.lit("Not answered"))
    .otherwise(pl.col(pl.String))
    .name.keep()
)
df.head(3)


with open("data/survey.json") as f:
    survey = json.load(f)


def get_question(column_name: str) -> dict | None:
    m = re.match(r"(q[0-9]{2})", column_name)
    if m is None:
        return None

    q_id = m.group(1)
    for question in survey["questions"]:
        if question["id"] == q_id:
            return question

    return None


def get_question_prompt(column_name: str) -> str | None:
    q = get_question(column_name)
    if not q:
        return None

    return q["prompt"]


assert get_question_prompt("q01") == "Where do you live?"


def get_choice_text(
    column_name: str,
) -> str | None:
    q = get_question(column_name)
    if not q:
        return None

    m = re.match(r"q[0-9]{2}\[SQ([0-9]{3})\]", column_name)
    if m is None:
        return None
    c_id = int(m.group(1)) - 1

    return q["choices"][c_id]


assert get_choice_text("q07[SQ003]") == "I use NixOS"


def save_output(
    table,
    chart,
    code,
    output_path: Path = OUTPUT_PATH,
):
    output_path.mkdir(parents=True, exist_ok=True)

    answers_path = output_path / f"answers_{code}.json"
    with open(answers_path, "w") as f:
        json.dump(obj=table.to_dicts(), fp=f)

    chart_plot_path = output_path / f"chart_plot_{code}.png"
    with open(chart_plot_path, "wb") as f:
        chart.save(fp=f, format="png", scale_factor=3)

    chart_json_path = output_path / f"chart_json_{code}.json"
    with open(chart_json_path, "w") as f:
        chart.save(fp=f, format="json")


# %%
q08_q07sq003 = df.group_by("q07[SQ003]", "q08").agg(pl.len().alias("count"))
display(q08_q07sq003)

"""
Now the question is: does the proportion of NixOS usage evolve through the years?
"""

chart_q08_q07sq003 = (
    alt.Chart(
        q08_q07sq003.filter(pl.col("q07[SQ003]") == "Yes")
        .select("q08", pl.col("count").alias("count_yes"))
        .join(
            q08_q07sq003.group_by("q08").agg(
                pl.col("count").sum().alias("count_total")
            ),
            on="q08",
        )
        .select(
            "q08",
            (pl.col("count_yes") / pl.col("count_total") * 100).alias("percent_yes"),
        )
    )
    .mark_bar()
    .encode(
        x=alt.X("q08", title=get_question_prompt("q08")).sort(
            (get_question("q08") or {}).get("choices")
        ),
        y=alt.Y("percent_yes", title="Use NixOS (%)"),
    )
)

display(chart_q08_q07sq003)

save_output(code="q08_q07sq003", table=q08_q07sq003, chart=chart_q08_q07sq003)


# %%
q08_q09 = df.group_by("q09", "q08").agg(pl.len().alias("count"))
display(q08_q09)

"""
Now the question is: does the self-evaluated skill level with Nix evolve through the years?
"""

chart_q08_q09 = (
    alt.Chart(
        q08_q09,
        title="Percentage of self-evaluated skill level with Nix, normalized by years of Nix experience",
    )
    .transform_joinaggregate(total="sum(count)", groupby=["q08"])
    .transform_calculate(frac=alt.datum.count / alt.datum.total * 100)
    .mark_rect()
    .encode(
        x=alt.X("q08", title=get_question_prompt("q08")).sort(
            (get_question("q08") or {}).get("choices")
        ),
        y=alt.Y("q09", title=get_question_prompt("q09")).sort(
            (get_question("q09") or {}).get("choices")
        ),
        color=alt.Color("frac:Q", scale=alt.Scale(domain=[0, 100]), title="Percentage"),
    )
)

display(chart_q08_q09)

save_output(code="q08_q09", table=q08_q09, chart=chart_q08_q09)

# %%
q08_q11 = df.group_by("q11", "q08").agg(pl.len().alias("count"))
display(q08_q11)

chart_q08_q11 = (
    alt.Chart(
        q08_q11,
        title="Percentage of medium from which people discovered Nix, normalized by years of Nix experience",
    )
    .transform_joinaggregate(total="sum(count)", groupby=["q08"])
    .transform_calculate(frac=alt.datum.count / alt.datum.total * 100)
    .mark_rect()
    .encode(
        x=alt.X("q08", title=get_question_prompt("q08")).sort(
            (get_question("q08") or {}).get("choices")
        ),
        y=alt.Y("q11", title=get_question_prompt("q11")).sort(
            (get_question("q11") or {}).get("choices")
        ),
        color=alt.Color("frac:Q"),
    )
)

display(chart_q08_q11)

save_output(code="q08_q11", table=q08_q11, chart=chart_q08_q11)


# %%
def reduce_join(dfs: list[pl.DataFrame], on: str | list[str]) -> pl.DataFrame:
    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, on=on, how="left")
    return result


q14_choices = (get_question("q14") or {})["choices"]
q14_choices_cols = [f"q14[SQ{i+1:03}]" for i in range(len(q14_choices))]
q08_q14 = df.group_by("q08", *q14_choices_cols).agg(pl.len().alias("count"))
q08_q14 = reduce_join(
    [
        q08_q14.filter(pl.col(col) == "Yes")
        .group_by("q08")
        .agg(pl.sum("count").alias(choice))
        for choice, col in zip(q14_choices, q14_choices_cols)
    ],
    on="q08",
).unpivot(index="q08", value_name="count")
display(q08_q14)

chart_q08_q14 = (
    alt.Chart(
        q08_q14,
        title="Count of usage of Nix installer normalized by years of Nix experience",
    )
    .transform_joinaggregate(total="sum(count)", groupby=["q08"])
    .transform_calculate(frac=alt.datum.count / alt.datum.total)
    .encode(
        x=alt.X("q08", title=get_question_prompt("q08")).sort(
            (get_question("q08") or {}).get("choices")
        ),
        y=alt.Y("variable", title=get_question_prompt("q14")),
        text=alt.Text("frac:Q", format=".0%"),
    )
)
chart_q08_q14 = chart_q08_q14.encode(
    color=alt.Color("frac:Q", title="Percentage", scale=alt.Scale(domain=[0, 1]))
).mark_rect() + chart_q08_q14.mark_text(size=5, color="black")

display(chart_q08_q14)

save_output(code="q08_q14", table=q08_q14, chart=chart_q08_q14)


# %%
q18_choices = (get_question("q18") or {})["choices"]
q18_choices_cols = [f"q18[SQ{i+1:03}]" for i in range(len(q18_choices))]
q08_q18 = df.group_by("q08", *q18_choices_cols).agg(pl.len().alias("count"))
q08_q18 = reduce_join(
    [
        q08_q18.filter(pl.col(col) == "Yes")
        .group_by("q08")
        .agg(pl.sum("count").alias(choice))
        for choice, col in zip(q18_choices, q18_choices_cols)
    ],
    on="q08",
).unpivot(index="q08", value_name="count")
display(q08_q18)

chart_q08_q18 = (
    alt.Chart(
        q08_q18,
        title="Count of usage of Nix installer normalized by years of Nix experience",
    )
    .transform_joinaggregate(total="sum(count)", groupby=["q08"])
    .transform_calculate(frac=alt.datum.count / alt.datum.total)
    .encode(
        x=alt.X("q08", title=get_question_prompt("q08")).sort(
            (get_question("q08") or {}).get("choices")
        ),
        y=alt.Y("variable", title=get_question_prompt("q18")),
        text=alt.Text("frac:Q", format=".0%"),
    )
)
chart_q08_q18 = chart_q08_q18.encode(
    color=alt.Color("frac:Q", title="Percentage", scale=alt.Scale(domain=[0, 1]))
).mark_rect() + chart_q08_q18.mark_text(size=5, color="black")

display(chart_q08_q18)

save_output(code="q08_q18", table=q08_q18, chart=chart_q08_q18)

# %%
multiple_choice_questions = [q for q in survey["questions"] if q["type"] == "multiple"]

for q_mc in multiple_choice_questions:
    q_mc_id = q_mc["id"]
    q_mc_prompt = q_mc["prompt"]
    q_mc_choices = q_mc["choices"]
    q_mc_choices_cols = [f"{q_mc_id}[SQ{i+1:03}]" for i in range(len(q_mc_choices))]

    q_mc_corrs = (
        (
            df.select(
                *(
                    pl.when(pl.col(c) == "Y")
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias(c)
                    for c in q_mc_choices_cols
                )
            )
            .corr()
            .with_columns(choice1=pl.Series(q_mc_choices))
        )
        .rename({c: t for t, c in zip(q_mc_choices, q_mc_choices_cols)})
        .unpivot(index="choice1", variable_name="choice2", value_name="corr")
    )

    chart_q_mc = (
        alt.Chart(
            q_mc_corrs,
            title=f"{q_mc_prompt} - Correlation",
        )
        .encode(
            x=alt.X("choice1", title="Choice"),
            y=alt.Y("choice2", title="Choice"),
            color=alt.Color("corr:Q", scale=alt.Scale(domain=[-1, 1])),
        )
        .mark_rect()
    )
    display(chart_q_mc)

    save_output(code=f"{q_mc_id}_corrs", table=q_mc_corrs, chart=chart_q_mc)


# %%
# WIP
multiple_choice_questions = [q for q in survey["questions"] if q["type"] == "multiple"]

for q_mc in multiple_choice_questions:
    q_mc_id = q_mc["id"]
    q_mc_choices = q_mc["choices"]
    q_mc_choices_cols = [f"{q_mc_id}[SQ{i+1:03}]" for i in range(len(q_mc_choices))]

    q_mc_corrs = (
        df.select(
            pl.concat_list(
                [
                    pl.when(pl.col(c) == "Y")
                    .then(pl.lit(t))
                    .otherwise(pl.lit(None))
                    .alias(c)
                    for t, c in zip(q_mc_choices, q_mc_choices_cols)
                ]
            )
            .list.drop_nulls()
            .list.eval('"' + pl.element() + '"')
            .list.join(",")
            .alias("selected_choices")
        )
        .group_by("selected_choices")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    q_mc_total = sum(q_mc_corrs["count"])
    q_mc_threshold = q_mc_total * 0.10
    q_mc_corrs = (
        q_mc_corrs.with_columns(
            selected_choices=pl.when(pl.col("count") >= 10)
            .then("selected_choices")
            .otherwise(pl.lit("other"))
        )
        .group_by("selected_choices")
        .agg(pl.sum("count").alias("count"))
    )

    display(q_mc_corrs)
