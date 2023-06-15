# Causal Inference Project

## Description
Most movies today have pre-premiere screening before releasing the movie to broad distribution. 
Among the people invited to those screenings, there are movie critics which publish their opinions about the movie in
different papers/magazines prior to the movies' broad distribution.<br>
In this work our goal is to estimate the causal effect of the movie's pre-release review score on its ROI

To estimate this effect we used "The Movies Dataset" combined with "Rotten Tomatoes Dataset" and some scraped data (see `data_preparation.py`).

## The code
1. `data_preparation.py` - webscraping + preprocessing of the data
2. `causal_analysis.py` - Causal analysis of the data using different models such as IPW, S-learner, T-learner and 1-NN matching.
