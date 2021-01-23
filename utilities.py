import pathlib
from datetime import datetime
from typing import Dict, List, Tuple

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit as logistic


# @st.cache
def extract_constant_data():
    """
    Extract all the constant data needed to make the Bokeh plot and Streamlit app.
    """
    BASE_URL = "https://raw.githubusercontent.com/AlexAndorra/pollsposition_models/master/popularity/plot_data"
    _complete_data = pd.read_csv(
        f"{BASE_URL}/complete_popularity_data.csv", index_col=0, parse_dates=True
    )
    _raw_polls = pd.read_csv(f"{BASE_URL}/raw_polls.csv", index_col=0, parse_dates=True)
    _prediction_coords = pd.read_csv(
        f"{BASE_URL}/prediction_coords.csv", index_col=0, parse_dates=["timesteps"]
    )
    _trace_predictions = az.from_netcdf("plot_data/trace_predictions.nc")
    _pp_prop = xr.open_dataset("plot_data/post_pred_approval.nc")
    _pp_prop_5 = xr.open_dataset("plot_data/post_pred_approval_5.nc")
    _pp_prop_10 = xr.open_dataset("plot_data/post_pred_approval_10.nc")

    return (
        _complete_data,
        _raw_polls,
        _prediction_coords,
        _trace_predictions,
        _pp_prop,
        _pp_prop_5,
        _pp_prop_10,
    )


# setup data needed for this program
(
    complete_data,
    raw_polls,
    prediction_coords,
    trace_predictions,
    pp_prop,
    pp_prop_5,
    pp_prop_10,
) = extract_constant_data()


def generate_app_input():  # -> Tuple[str, gridplot, pd.DataFrame]:
    """
    Generate objects needed by Streamlit to build the app
    """
    last_update = get_last_update_date(
        reference_file_path="plot_data/trace_predictions.nc"
    )

    source_df_list = []
    random_draws_list = []
    post_pred_approval_list = [
        pp_prop["post_pred_approval"],
        pp_prop_5["post_pred_approval_5"],
        pp_prop_10["post_pred_approval_10"],
    ]
    for post_pred_approval in post_pred_approval_list:
        source_df = generate_bokeh_data_source(trace_predictions, post_pred_approval)
        random_draws = samples_subset(source_df)

        source_df_list.append(source_df)
        random_draws_list.append(random_draws)

    styled_raw_polls = style_raw_polls(raw_polls)

    return (
        last_update,
        complete_data,
        raw_polls,
        prediction_coords,
        random_draws_list,
        source_df_list,
        post_pred_approval_list,
        styled_raw_polls,
    )


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

    :param trace: only the ``predictions`` group of the full inference data from the model.
    :param post_pred_samples: posterior predictive approval of president, under given unemployment.

    :return: posterior inference data as a dataframe.
    """
    source_df = (
        post_pred_samples.stack(sample=("chain", "draw"))
        .to_pandas()
        .droplevel(0, axis=1)
    )
    source_df.columns = source_df.columns.astype(str)

    source_df["baseline"] = logistic(trace.posterior["baseline"]).mean().data
    source_df["baseline_lower"] = (
        logistic(az.hdi(trace.posterior)["baseline"]).sel(hdi="lower").data
    )
    source_df["baseline_upper"] = (
        logistic(az.hdi(trace.posterior)["baseline"]).sel(hdi="higher").data
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
