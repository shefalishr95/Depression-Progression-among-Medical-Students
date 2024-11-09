# Modeling Mental Health Progression in Medical Students Using Markov Chains

## Project Overview

This project aims to analyze mental health trends in medical students over time, focusing on the progression and fluctuations of depression states. The study uses data from the ETMED-L project, which tracks mental health indicators in medical students at the University of Lausanne through three waves (March 2021, November 2021, November 2022).

## Dataset

The dataset includes responses from 1,595 medical students and captures multiple aspects of mental health, such as:

- **Depression symptoms** (CES-D)
- **Anxiety symptoms** (STAI)
- **Burnout indicators** (MBI-SS)

In addition, the dataset includes biopsychosocial variables, including gender identity, social support, physical activity, satisfaction with health, and coping strategies.

## Objectives

1. **Model Mental Health Transitions**: Use a Markov chain to model the changes in depression states (e.g., low, moderate, severe) over time, with a goal of understanding equilibrium states and trends in mental health.
   
2. **Polynomial Regression**: Apply polynomial regression to study the non-linear trend in mental health, hypothesizing that mental health follows a parabolic shape during the course of medical school.

3. **Comparison of Models**: Compare the predictive capabilities of the Markov chain model and polynomial regression over time.

## Methods

- **Markov Chain Modeling**: Track transitions between different depression states and analyze the effects of covariates like social support and coping strategies on these transitions.
- **Polynomial Regression**: Model mental health trends to capture non-linear patterns as medical students progress through their education.


Source:
Carrard V, Berney S, Bourquin C, Ranjbar S, Castelao E, Schlegel K, et al. (2024) Mental health and burnout during medical school: Longitudinal evolution and covariates. PLoS ONE 19(4): e0295100. https://doi.org/10.1371/journal.pone.0295100