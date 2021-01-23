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

p3 = make_plot(
    subtitle="increases to 10%",
    palette=inferno(6),
    random_draws=random_draws3,
    data_source=source_df3,
    post_pred_samples=pp_prop_10,
)
p3.add_layout(
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
    exp1 = st.beta_expander("What is this?", expanded=True)
    with exp1:
        st.markdown(
            """
            These plots try to answer a simple question: how does the popularity of French presidents 
            evolve with time? I compiled all the popularity opinion polls of French presidents since 
            the term limits switched to 5 years (in 2002), so we can also compare presidents and see
            that popularity is very periodic.
            
            Each line is a possible path for any president's approval, taking into account historical 
            variations, honeymoon effects (i.e when a president gets elected), the variation in the 
            unemployment rate, and temporal variations due to periodic effects withing a presidential 
            term.
            
            As each line is a possible scenario, the more concentrated the path, the more probable the
            associated scenario. In contrast, you can see that some paths are all alone (😢), which
            means that they can't be excluded, given our model and the observed data, but they have a
            low probability of happening.
            
            By the way, the plots are fully interactive, so fell free to play around with them -- you
            can zoom, pan, compare, reset, deactivate parts of the legend, etc. 
            Who said statistics weren't fun? 😜
            """
        )
    exp2 = st.beta_expander("Why three different plots?")
    with exp2:
        st.markdown(
            """
            Each plot makes out-of-sample predictions (i.e for which we have no data yet) for the next 
            three months. But each one of them simulates a different future universe, if you will: the 
            first one imagines that the unemployment rate stays more or less the same in the next 
            quarter, while the second one tries to guess what would happen to the president's popularity
            if unemployment were to drop to 5%. What about the third one? You guessed it: it simulates a
            world where unemployment jumps to 10%. 
    
            Of course, this is not very realistic, as unemployment rarely changes that drastically, 
            but this helps us to develop a probabilistic way of thinking, which is very useful in 
            uncertain situations -- like poker, medical diagnosis, or, you know, just life in general 🤷‍♂️
            
            It also helps further evaluate the model: if the different scenarios and their relative 
            differences are consistent with domain knowledge, this is a good point for our model.
            
            In a nutshe
            """
        )
    exp3 = st.beta_expander("How does it work?", expanded=True)
    with exp3:
        st.markdown(
            """
            Specifically, the model is a Gaussian Process regression, implemented in the Bayesian
            framework, thus making it possible to estimate uncertainty in the estimates (here, the true
            latent popularity of the president) as well as integrating domain knowledge when data become
            sparse and / or unreliable. 
            
            See [this tutorial](https://alexandorra.github.io/pollsposition_blog/popularity/macron/gaussian%20processes/polls/2021/01/18/gp-popularity.html) 
            for more details about the model.
            """
        )
    exp4 = st.beta_expander("Which polls are used?")
    with exp4:
        st.markdown(
            """
            All of them! At least to the best of my abilities: this has to be done by hand (there 
            aren't a lot of polls each months, so developing a little robot for each pollster to 
            automate that isn't really worth the trouble.
            
            All the data are open-sourced and free to access. So, if you see some polls missing or want 
            to contribute new ones, feel free to open pull requests on the 
            [GitHub repo](https://github.com/AlexAndorra/pollsposition_models/blob/master/data/raw_popularity_presidents.csv)! 
            """
        )
        st.dataframe(
            style_raw_polls(raw_polls)
            .style.background_gradient(
                cmap=sns.color_palette("viridis", as_cmap=True),
                low=0.4,
                subset=["Approve"],
            )
            .format({"Approve": "{:.0%}", "Disapprove": "{:.0%}"}),
            height=400,
        )

st.subheader("What to make of this?")
c1, c2 = st.beta_columns(2)
with c1:
    st.markdown(
        """
        There are many lessons to draw from these graphs, but the main ones seem to be:
    
        1. **French people don't seem to like their presidents that much**: their median approval 
        very rarely goes 
        above 60%. Instead, they spend most of their time around the 40% mark -- in fact, the historical
        average is estimated to be between 33% and 42%. This stands in 
        [sharp contrast to US presidents](https://projects.fivethirtyeight.com/trump-approval-ratings/)
        for instance, who are usually much more appreciated.
        
        2. However, **they can _really_ dislike their presidents**, François Hollande being the most 
        prominent
        example, as he the only one who spent more _below_ the historical average than above -- even
        dropping under 20% approval twice. Nicolas Sarkozy and Emmanuel Macron spent most of their time
        around the average, with only a small amount of variation. And Jacques Chirac (in his second 
        term)
        is the only one who was more popular than average during the majority of his term.
        
        3. **It is usual for a president to experience a dip in popularity after 1-2 years in 
        office**.
        It's just the honeymoon effect fading away, combined to 
        [reversion to the mean](https://en.wikipedia.org/wiki/Regression_toward_the_mean). Then 
        they 
        oscillate around this mean, which is itself a combination of the historical baseline 
        approval
        and a mean specific to the given president. Hollande's baseline for instance seemed to be 
        much
        lower than Macron's and Sarkozy's, with those two being basically at the historical 
        baseline.
        All that suggests that reacting to the most recent poll usually amounts to reacting to 
        noise.
        """
    )
with c2:
    st.markdown(
        """
        4. **Presidential approval's ceiling seems to slowly go down since the switch from a 7-year to a
        5-year term**. You can see that in the fact that the honeymoon effect is less powerful and fades
        away faster for more recent presidents. Don't blame the 5-year term so quickly though: we don't
        know which direction the effect goes: this shortening of the political cycle coincides with the
        appearance of 24/7 news media and social networks (and the ensuing shortening of the news 
        cycle).
        Maybe _that_ made French people more impatient with their heads of State? And maybe _that's_ why
        they approved the 5-year term proposal in the first place? Also, with the 5-year term, the
        possibility for [divided government](https://en.wikipedia.org/wiki/Cohabitation_(government)) 
        ("cohabitation") disappeared, and with it the option for the president to blame the opposition 
        for the country's troubles -- if divided governments were still possible, maybe presidents 
        would be more popular?
         
        5. **It's hard to argue for a "Covid effect"**. Macron gained 5 points between March and June 
        2020, going from 38% to 43*, but this "rally around the flag" faded away during the Summer.
        Most importantly, those numbers are in the historical average (33%-42%), which seems to be 
        Macron's cruising speed. As a result, things would have to go much worse (resp. much better) for
        him to start loosing (resp. gaining) popularity, as we can see in plot #3 (resp. #2).
        """
    )

st.subheader("About PollsPosition")
column1, column2 = st.beta_columns(2)
with column1:
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
        and high quality scientific tools -- just like The Avengers, they really are true heroes.
        """
    )
with column2:
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

"caching, Disqus"
