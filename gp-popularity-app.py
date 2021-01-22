import pathlib
from datetime import datetime
from typing import Dict, List

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import xarray as xr
from bokeh.layouts import gridplot
from bokeh.models import (
    Band,
    ColumnDataSource,
    DatetimeTickFormatter,
    HoverTool,
    LabelSet,
    NumeralTickFormatter,
    Span,
    Title,
)
from bokeh.palettes import cividis, inferno, viridis
from bokeh.plotting import figure
from scipy.special import expit as logistic

REPO_STEM = "/Users/alex_andorra/repos/pollsposition_models/popularity"

complete_data = pd.read_csv(
    f"{REPO_STEM}/plot_data/complete_popularity_data.csv", index_col=0, parse_dates=True
)

raw_polls = pd.read_csv(
    f"{REPO_STEM}/plot_data/raw_polls.csv", index_col=0, parse_dates=True
)
PREDICTION_COORDS = pd.read_csv(
    f"{REPO_STEM}/plot_data/prediction_coords.csv",
    index_col=0,
    parse_dates=["timesteps"],
)

trace_econ = az.from_netcdf(f"{REPO_STEM}/trace_raw_econ.nc")
pp_prop = xr.open_dataset(f"{REPO_STEM}/plot_data/post_pred_approval.nc")
pp_prop_5 = xr.open_dataset(f"{REPO_STEM}/plot_data/post_pred_approval_5.nc")
pp_prop_10 = xr.open_dataset(f"{REPO_STEM}/plot_data/post_pred_approval_10.nc")


def get_data_source(
    trace: az.InferenceData, post_pred_samples: xr.DataArray
) -> pd.DataFrame:
    source_df = (
        post_pred_samples.stack(sample=("chain", "draw"))
        .to_pandas()
        .droplevel(0, axis=1)
    )
    source_df.columns = source_df.columns.astype(str)

    source_df["baseline"] = logistic(trace.predictions["baseline"]).mean().data
    source_df["baseline_lower"] = (
        logistic(az.hdi(trace.predictions)["baseline"]).sel(hdi="lower").data
    )
    source_df["baseline_upper"] = (
        logistic(az.hdi(trace.predictions)["baseline"]).sel(hdi="higher").data
    )

    source_df["median_app"] = post_pred_samples.median(dim=("chain", "draw")).data
    source_df["median_low"] = np.squeeze(
        az.hdi(post_pred_samples, hdi_prob=0.75).sel(hdi="lower").to_array().data
    )
    source_df["median_high"] = np.squeeze(
        az.hdi(post_pred_samples, hdi_prob=0.75).sel(hdi="higher").to_array().data
    )

    return source_df


def get_last_update_date(reference_file: str) -> str:
    graph_file = pathlib.Path(f"{REPO_STEM}/{reference_file}")
    modified_time = datetime.fromtimestamp(graph_file.stat().st_mtime)
    return modified_time.strftime("%b %d, %Y")


def samples_subset(data_source: pd.DataFrame, frac: float = 0.1) -> Dict[str, List]:
    sub_source = data_source.filter(regex="\d", axis="columns").sample(
        frac=frac, replace=True, axis="columns"
    )

    dates = []
    draws = []
    for draw in sub_source.columns:
        dates.append(sub_source.index.values)
        draws.append(sub_source[draw].values)

    return {"dates": dates, "draws": draws}


source_annotations = ColumnDataSource(
    data=dict(
        dates=[
            pd.to_datetime("2002-05-14"),
            pd.to_datetime("2007-05-16"),
            pd.to_datetime("2012-05-11"),
            pd.to_datetime("2017-05-17"),
            pd.to_datetime("2020-03-17"),
            pd.to_datetime("2002-10-24"),
        ],
        ys=[0.95, 0.95, 0.95, 0.95, 0.95, 0.32],
        events=[
            "Switch to 5-year term",
            "Sarkozy elected",
            "Hollande elected",
            "Macron elected",
            "1st Covid Cases",
            "Historical approval mean",
        ],
    )
)


def make_plot(
    subtitle: str,
    palette,
    random_draws: Dict[str, List],
    data_source: pd.DataFrame,
    post_pred_samples: xr.Dataset,
):
    CDS = ColumnDataSource(data_source)

    p = figure(
        aspect_ratio=16 / 7.5,
        min_width=480,
        max_width=1600,
        x_axis_type="datetime",
        title="Evolution of French presidents' popularity over time",
        x_range=(
            pd.to_datetime("2012-01-01"),
            PREDICTION_COORDS["timesteps"].iloc[-1] + pd.DateOffset(months=3),
        ),
        y_range=(0, 1),
        toolbar_location="above",
        tools="xpan, box_zoom, xwheel_zoom, crosshair, reset, undo, save",
    )
    p.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y", days="%d/%m")
    p.yaxis[0].formatter = NumeralTickFormatter(format="00%")
    p.add_layout(
        Title(
            text=f"One quarter out-of-sample, if unemployment {subtitle}",
            align="center",
            text_font_style="italic",
            text_font_size="1.2em",
        ),
        "above",
    )
    p.title.text_font_size = "1.6em"
    p.title.align = "center"
    p.grid.grid_line_alpha = 0.5
    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = "% popularity"

    p.multi_line(
        xs=random_draws["dates"],
        ys=random_draws["draws"],
        color=palette[4],
        legend_label="Random samples",
    )
    p.patch(
        np.concatenate((data_source.index.values, data_source.index.values[::-1])),
        np.concatenate(
            (
                np.squeeze(az.hdi(post_pred_samples).sel(hdi="lower").to_array()),
                np.squeeze(az.hdi(post_pred_samples).sel(hdi="higher").to_array())[
                    ::-1
                ],
            )
        ),
        color=palette[3],
        line_alpha=0.4,
        fill_alpha=0.4,
        legend_label="94% HDI",
    )
    p.patch(
        np.concatenate((data_source.index.values, data_source.index.values[::-1])),
        np.concatenate(
            (
                np.squeeze(
                    az.hdi(post_pred_samples, hdi_prob=0.75).sel(hdi="lower").to_array()
                ),
                np.squeeze(
                    az.hdi(post_pred_samples, hdi_prob=0.75)
                    .sel(hdi="higher")
                    .to_array()
                )[::-1],
            )
        ),
        color=palette[2],
        line_alpha=0,
        fill_alpha=0.5,
        legend_label="75% HDI",
    )
    p.patch(
        np.concatenate((data_source.index.values, data_source.index.values[::-1])),
        np.concatenate(
            (
                np.squeeze(
                    az.hdi(post_pred_samples, hdi_prob=0.5).sel(hdi="lower").to_array()
                ),
                np.squeeze(
                    az.hdi(post_pred_samples, hdi_prob=0.5).sel(hdi="higher").to_array()
                )[::-1],
            )
        ),
        color=palette[1],
        line_alpha=0,
        fill_alpha=0.6,
        legend_label="50% HDI",
    )
    median_line = p.line(
        "timesteps",
        "median_app",
        color=palette[0],
        line_width=2,
        legend_label="Median",
        source=CDS,
    )
    p.scatter(
        raw_polls.index.values,
        raw_polls.p_approve.values,
        size=6,
        color=palette[5],
        legend_label="Observed polls",
        alpha=0.7,
    )

    labels = LabelSet(
        x="dates",
        y="ys",
        text="events",
        level="glyph",
        text_color="gray",
        text_font_style="italic",
        text_font_size="1em",
        text_align="center",
        source=source_annotations,
    )
    vline_0 = Span(
        location=source_annotations.data["dates"][0],
        dimension="height",
        line_color="gray",
        line_dash="dashed",
        line_width=1.5,
    )
    vline_1 = Span(
        location=source_annotations.data["dates"][1],
        dimension="height",
        line_color="gray",
        line_dash="dashed",
        line_width=1.5,
    )
    vline_2 = Span(
        location=source_annotations.data["dates"][2],
        dimension="height",
        line_color="gray",
        line_dash="dashed",
        line_width=1.5,
    )
    vline_3 = Span(
        location=source_annotations.data["dates"][3],
        dimension="height",
        line_color="gray",
        line_dash="dashed",
        line_width=1.5,
    )
    vline_4 = Span(
        location=source_annotations.data["dates"][4],
        dimension="height",
        line_color="gray",
        line_dash="dashed",
        line_width=1.5,
    )

    fifty_line = Span(
        location=0.5,
        dimension="width",
        line_color="gray",
        line_dash="dotted",
        line_width=1.5,
    )
    hist_band = Band(
        base="timesteps",
        lower="baseline_lower",
        upper="baseline_upper",
        source=CDS,
        fill_color="gray",
        fill_alpha=0.2,
    )
    hist_avg_line = Span(
        location=CDS.data["baseline"][0],
        dimension="width",
        line_color="gray",
        line_dash="dashdot",
        line_width=2,
    )

    p.renderers.extend(
        [
            labels,
            vline_0,
            vline_1,
            vline_2,
            vline_3,
            vline_4,
            fifty_line,
            hist_band,
            hist_avg_line,
        ]
    )

    p.legend.click_policy = "hide"
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"
    p.legend.background_fill_alpha = 0.6

    # Add the HoverTool to the figure
    TOOLTIPS = [
        ("Median app.", "@median_app{00%} in @timesteps{%b %Y}"),
        ("75% chance btw", "@median_low{00%} and @median_high{00%}"),
        ("Historic. avg. btw", "@baseline_lower{00%} and @baseline_upper{00%}"),
    ]
    p.add_tools(
        HoverTool(
            tooltips=TOOLTIPS,
            formatters={"@timesteps": "datetime"},
            mode="vline",
            renderers=[median_line],
        )
    )

    return p


def style_raw_polls(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = (
        raw_df.reset_index()
        .rename(
            columns={
                "index": "Date",
                "p_approve": "Approve",
                "p_disapprove": "Disapprove",
                "samplesize": "Sample",
                "method": "Method",
                "president": "President",
                "sondage": "Pollster",
            }
        )
        .sort_values(["Date", "Pollster"], ascending=[False, True])[
            [
                "Date",
                "Pollster",
                "Approve",
                "Disapprove",
                "Sample",
                "Method",
                "President",
            ]
        ]
    )
    return raw_df


source_df1 = get_data_source(trace_econ, pp_prop["post_pred_approval"])
source_df2 = get_data_source(trace_econ, pp_prop_5["post_pred_approval_5"])
source_df3 = get_data_source(trace_econ, pp_prop_10["post_pred_approval_10"])

random_draws1 = samples_subset(source_df1)
random_draws2 = samples_subset(source_df2)
random_draws3 = samples_subset(source_df3)

p1 = make_plot(
    subtitle=f"stays at {complete_data.unemployment.iloc[-1]}%",
    palette=viridis(6),
    random_draws=random_draws1,
    data_source=source_df1,
    post_pred_samples=pp_prop,
)

p2 = make_plot(
    subtitle="drops to 5%",
    palette=cividis(6),
    random_draws=random_draws2,
    data_source=source_df2,
    post_pred_samples=pp_prop_5,
)

p3 = make_plot(
    subtitle="increases to 10%",
    palette=inferno(6),
    random_draws=random_draws3,
    data_source=source_df3,
    post_pred_samples=pp_prop_10,
)

p2.title.text = None
p3.title.text = None
p2.x_range = p1.x_range
p3.x_range = p1.x_range

plot_layout = gridplot(
    children=[p1, p2, p3],
    ncols=1,
    sizing_mode="scale_both",
    toolbar_options=dict(logo="grey"),
)

# set up Streamlit dashboard:
st.set_page_config(
    page_title="PollsPosition",
    page_icon="https://alexandorra.github.io/pollsposition_blog/images/favicon.ico",
    layout="wide",
)
st.title("PollsPosition")
st.header("Forecasting French elections with Bayesian Statistics")
st.markdown(
    f"_By [Alexandre Andorra](https://twitter.com/alex_andorra), last updated "
    f"{get_last_update_date('gp-popularity.png')}_"
)

col1, col2 = st.beta_columns((2, 1))

with col1:
    st.subheader("How popular is the president?")
    st.bokeh_chart(plot_layout, use_container_width=True)

with col2:
    st.subheader("What is this?")
    st.markdown(
        """
        These plots try to answer a simple question: how does the popularity of French presidents 
        evolve with time? I compiled all the popularity opinion polls of French presidents since 
        the term limits switched to 5 years (in 2002), so we can also compare presidents and see
        that popularity is very periodic -- it seems natural that a president experiences a dip in 
        popularity in the middle of his term, so reacting to the latest release poll usually 
        amounts to reaction to noise. 
        
        Each line
        
        Counterfactual
        """
    )
    st.subheader("How does it work?")
    st.markdown(
        """
        Gaussian Process model
        https://alexandorra.github.io/pollsposition_blog/popularity/macron/gaussian%20processes
        /polls/2021/01/18/gp-popularity.html
        """
    )
    st.subheader("Latest polls:")
    st.dataframe(
        style_raw_polls(raw_polls)
        .style.background_gradient(
            cmap=sns.color_palette("viridis", as_cmap=True), low=0.4, subset=["Approve"]
        )
        .format({"Approve": "{:.0%}", "Disapprove": "{:.0%}"}),
        height=800,
    )

st.subheader("What to make of this?")
"bla bla bla"

st.subheader("About PollsPosition")
c1, c2 = st.beta_columns(2)
with c1:
    st.markdown(
        """
        The PollsPosition project is an [open-source](
        https://github.com/AlexAndorra/pollsposition_models) endeavor that stands on the 
        shoulders of 
        giants of the [Python](https://www.python.org/) data stack: [PyMC3](https://docs.pymc.io/) 
        for state-of-the-art MCMC algorithms, [ArviZ](https://arviz-devs.github.io/arviz/) and [
        Bokeh](
        https://docs.bokeh.org/en/latest/) for visualizations, and [Pandas](
        https://pandas.pydata.org/) for data cleaning.

        We warmly thank all the developers who give their time to develop these free, open-source 
        and 
        high quality scientific tools -- just like The Avengers, they really are true heroes.
    """
    )
with c2:
    st.markdown(
        """
        Sounds fun to you? And you're looking for a project to improve your Python and 
        Bayesian chops?! Well, feel free to [contribute pull requests](
        https://github.com/AlexAndorra/pollsposition_models) -- there is always something to do!

        If you want to learn more about PollsPosition and who I am, you can take a look at this [
        short summary](https://alexandorra.github.io/pollsposition_blog/about/). And feel free to 
        reach out on [Twitter](https://twitter.com/alex_andorra) if you want to talk about 
        statistical modeling under certainty, or how "polls are useless now because they missed 
        two elections in a row!" -- yeah, I'm a bit sarcastic.
        """
    )
