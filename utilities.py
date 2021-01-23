import pathlib
from datetime import datetime
from typing import Dict, List, Tuple

import arviz as az
import numpy as np
import pandas as pd
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


# @st.cache
def extract_constant_data(repo_path: str):
    """
    Extract all the constant data needed to make the Bokeh plot and Streamlit app.
    """
    _complete_data = pd.read_csv(
        "https://raw.githubusercontent.com/AlexAndorra/pollsposition_models/master/popularity/plot_data/complete_popularity_data.csv",
        index_col=0,
        parse_dates=True,
    )
    _raw_polls = pd.read_csv(
        "https://raw.githubusercontent.com/AlexAndorra/pollsposition_models/master/popularity/plot_data/raw_polls.csv",
        index_col=0,
        parse_dates=True,
    )
    _prediction_coords = pd.read_csv(
        "https://raw.githubusercontent.com/AlexAndorra/pollsposition_models/master/popularity/plot_data/prediction_coords.csv",
        index_col=0,
        parse_dates=["timesteps"],
    )
    _trace_econ = az.from_netcdf(f"{repo_path}/trace_raw_econ.nc")
    _pp_prop = xr.open_dataset(f"{repo_path}/plot_data/post_pred_approval.nc")
    _pp_prop_5 = xr.open_dataset(f"{repo_path}/plot_data/post_pred_approval_5.nc")
    _pp_prop_10 = xr.open_dataset(f"{repo_path}/plot_data/post_pred_approval_10.nc")

    _source_annotations = ColumnDataSource(
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

    return (
        _complete_data,
        _raw_polls,
        _prediction_coords,
        _trace_econ,
        _pp_prop,
        _pp_prop_5,
        _pp_prop_10,
        _source_annotations,
    )


# setup data needed for this program
(
    complete_data,
    raw_polls,
    PREDICTION_COORDS,
    trace_econ,
    pp_prop,
    pp_prop_5,
    pp_prop_10,
    SOURCE_ANNOTATIONS,
) = extract_constant_data(
    repo_path="/Users/alex_andorra/repos/pollsposition_models/popularity"
)


def generate_app_input() -> Tuple[str, gridplot, pd.DataFrame]:
    """
    Generate objects needed by Streamlit to build the app
    """
    last_update = get_last_update_date(
        reference_file_path="/Users/alex_andorra/repos/pollsposition_models/popularity/gp"
        "-popularity.png"
    )

    source_df_list = []
    random_draws_list = []
    post_pred_approval_list = [
        pp_prop["post_pred_approval"],
        pp_prop_5["post_pred_approval_5"],
        pp_prop_10["post_pred_approval_10"],
    ]
    for post_pred_approval in post_pred_approval_list:
        source_df = generate_bokeh_data_source(trace_econ, post_pred_approval)
        random_draws = samples_subset(source_df)

        source_df_list.append(source_df)
        random_draws_list.append(random_draws)

    bokeh_plot_layout = generate_bokeh_layout(
        complete_data, random_draws_list, source_df_list, post_pred_approval_list
    )
    styled_raw_polls = style_raw_polls(raw_polls)

    return last_update, bokeh_plot_layout, styled_raw_polls


def get_last_update_date(reference_file_path: str) -> str:
    """
    Check last modified date of ``reference_file``, to display date in the app.
    """
    graph_file = pathlib.Path(f"{reference_file_path}")
    modified_time = datetime.fromtimestamp(graph_file.stat().st_mtime)
    return modified_time.strftime("%b %d, %Y")


def generate_bokeh_data_source(
    trace: az.InferenceData, post_pred_samples: xr.DataArray
) -> pd.DataFrame:
    """
    Turn posterior inference data into a dataframe for Bokeh.

    :param trace: inference data from the model.
    :param post_pred_samples: posterior predictive approval of president, under given unemployment.

    :return: posterior inference data as a dataframe.
    """
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
    """
    Take a random sample of ``frac`% of the posterior predictive samples, to display on plots.

    :param data_source: posterior predictive samples as a dataframe.
                        Result of ``func: generate_bokeh_data_source``.
    :param frac: fraction of subsamples to takes. Defaults to 10%.

    :return: the subsamples as a dictionary ``{"dates": dates, "draws": draws}``.
    """
    sub_source = data_source.filter(regex="\d", axis="columns").sample(
        frac=frac, replace=True, axis="columns"
    )

    dates = []
    draws = []
    for draw in sub_source.columns:
        dates.append(sub_source.index.values)
        draws.append(sub_source[draw].values)

    return {"dates": dates, "draws": draws}


def generate_bokeh_layout(
    complete_data: pd.DataFrame,
    random_draws_list: List[Dict[str, List]],
    source_df_list: List[pd.DataFrame],
    post_pred_approval_list: List[xr.DataArray],
) -> gridplot:
    """
    Generate the final Bokeh object that will be passed to Streamlit.

    :param complete_data: complete raw data, from models' repo.
    :param random_draws_list: list of random draws from the posterior predictive distribution.
                            Result of ``func: samples_subset``.
    :param source_df_list: complete posterior inference data, as a dataframe.
                        Result of ``func: generate_bokeh_data_source``.
    :param post_pred_approval_list: posterior predictive approval of president, under given
    unemployment.
    :return: Bokeh gridplot object.
    """
    p0 = make_bokeh_plot(
        subtitle=f"stays at {complete_data.unemployment.iloc[-1]}%",
        palette=viridis(6),
        random_draws=random_draws_list[0],
        data_source=source_df_list[0],
        post_pred_samples=post_pred_approval_list[0],
    )

    p1 = make_bokeh_plot(
        subtitle="drops to 5%",
        palette=cividis(6),
        random_draws=random_draws_list[1],
        data_source=source_df_list[1],
        post_pred_samples=post_pred_approval_list[1],
    )
    p1.add_layout(
        Title(
            text="'X% HDI' means that the true latent popularity has X% chance to be in the given "
            "interval",
            align="left",
            text_font_size="0.85rem",
            text_font_style="italic",
            text_color="gray",
        ),
        "above",
    )

    p2 = make_bokeh_plot(
        subtitle="increases to 10%",
        palette=inferno(6),
        random_draws=random_draws_list[2],
        data_source=source_df_list[2],
        post_pred_samples=post_pred_approval_list[2],
    )
    p2.add_layout(
        Title(
            text="'X% HDI' means that the true latent popularity has X% chance to be in the given "
            "interval",
            align="left",
            text_font_size="0.85rem",
            text_font_style="italic",
            text_color="gray",
        ),
        "above",
    )

    p1.title.text = None
    p2.title.text = None
    p1.x_range = p0.x_range
    p2.x_range = p0.x_range

    return gridplot(
        children=[p0, p1, p2],
        ncols=1,
        sizing_mode="scale_both",
        toolbar_options=dict(logo="grey"),
    )


def make_bokeh_plot(
    subtitle: str,
    palette,
    random_draws: Dict[str, List],
    data_source: pd.DataFrame,
    post_pred_samples: xr.Dataset,
) -> figure:
    """
    Make and return Bokeh plot of posterior predictive approval timeseries.

    :param subtitle: Subtitle to differentiate the unemployment scenario.
    :param palette: Bokeh palette
    :param random_draws: random draws from the posterior predictive distribution.
                        Result of ``func: samples_subset``.
    :param data_source: Complete posterior inference data, as a dataframe.
                        Result of ``func: generate_bokeh_data_source``.
    :param post_pred_samples: posterior predictive approval of president, under given unemployment.

    :return: Bokeh figure.
    """
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
        source=SOURCE_ANNOTATIONS,
    )
    vline_0 = Span(
        location=SOURCE_ANNOTATIONS.data["dates"][0],
        dimension="height",
        line_color="gray",
        line_dash="dashed",
        line_width=1.5,
    )
    vline_1 = Span(
        location=SOURCE_ANNOTATIONS.data["dates"][1],
        dimension="height",
        line_color="gray",
        line_dash="dashed",
        line_width=1.5,
    )
    vline_2 = Span(
        location=SOURCE_ANNOTATIONS.data["dates"][2],
        dimension="height",
        line_color="gray",
        line_dash="dashed",
        line_width=1.5,
    )
    vline_3 = Span(
        location=SOURCE_ANNOTATIONS.data["dates"][3],
        dimension="height",
        line_color="gray",
        line_dash="dashed",
        line_width=1.5,
    )
    vline_4 = Span(
        location=SOURCE_ANNOTATIONS.data["dates"][4],
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
    """
    Clean raw polls dataframe to later use Pandas styling module to get better display in app.

    :param raw_df: raw polls file, as a dataframe.

    :return: raw polls with human-readable column names, sorted values and sorted columns.
    """
    return (
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
