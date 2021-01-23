import seaborn as sns
import streamlit as st

from utilities import main

last_update, bokeh_plot_layout, styled_raw_polls = main()

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
    f"{last_update}_"
)

col1, col2 = st.beta_columns((2, 1))
with col1:
    st.subheader("How popular is the president?")
    st.bokeh_chart(bokeh_plot_layout, use_container_width=True)

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
            styled_raw_polls.style.background_gradient(
                cmap=sns.color_palette("viridis", as_cmap=True),
                low=0.4,
                subset=["Approve"],
            ).format({"Approve": "{:.0%}", "Disapprove": "{:.0%}"}),
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
