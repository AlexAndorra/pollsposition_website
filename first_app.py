from typing import Dict, List

import arviz as az
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from bokeh.layouts import column, layout, Spacer, gridplot
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

# https://blog.streamlit.io/introducing-new-layout-options-for-streamlit/
# https://docs.streamlit.io/en/stable/api.html?highlight=plotly#lay-out-your-app
# https://docs.streamlit.io/en/stable/main_concepts.html

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")  # set more options here
st.title("PollsPosition")
st.header("Forecasting French elections with Bayesian Statistics")

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


def standardize(series):
    """Standardize a pandas series"""
    return (series - series.mean()) / series.std()


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
        # sizing_mode="scale_both",
        aspect_ratio=16 / 7.5,
        # height_policy="fit",
        # width_policy="fit",
        # width=1200,
        #height=400,
        min_width=480,
        max_width=1600,
        # min_height=450,
        # max_height=450,
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

# p3.width_policy = "fit"
# p3.width = 1200
# p3.min_width = 550
# p3.max_width = 1350
# p3.height_policy = "fit"
# p3.height = 420
# p3.min_height = 300
# p3.max_height = 550
# plot_layout = layout(
#     children=[
#         [p1, p2],
#         [
#             Spacer(
#                 #width_policy="min",
#                 sizing_mode="scale_both",
#                 background="red",
#                 height_policy="fit",
#                 height=420,
#                 min_height=300,
#                 max_height=550,
#             ),
#             p3,
#             Spacer(
#                 #width_policy="min",
#                 sizing_mode="scale_both",
#                 background="red",
#                 height_policy="fit",
#                 height=420,
#                 min_height=300,
#                 max_height=550,
#             ),
#         ],
#     ],
#     sizing_mode="scale_both",
# )

#plot_layout = layout([p1, p2, p3], sizing_mode="scale_both")#, margin=(40, 170, 5, 160))
plot_layout = gridplot(
    children=[p1, p2, p3],
    ncols=1,
    sizing_mode="scale_both",
    toolbar_options = dict(logo='grey'),
)

col1, col2 = st.beta_columns((2, 1))

with col1:
    st.subheader("How popular is the president?")
    st.bokeh_chart(plot_layout, use_container_width=True)

with col2:
    st.subheader("Latest polls:")
    st.write(raw_polls.sort_index(ascending=False))
