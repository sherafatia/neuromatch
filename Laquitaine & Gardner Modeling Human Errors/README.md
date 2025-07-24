# Laquitaine & Gardner Modeling Human Errors

A comprehensive analysis of human perceptual estimation using Bayesian observer models, based on the motion direction estimation dataset from Laquitaine & Gardner (2018).

## Overview

This project investigates how humans combine prior expectations and sensory evidence when estimating motion direction under uncertainty. Through a series of computational models, we test whether human behavior follows optimal Bayesian integration or relies on simpler heuristic strategies.

## Dataset

- **Source**: Laquitaine & Gardner, Neuron, 2018
- **Participants**: 12 human subjects
- **Trials**: ~83,000 motion direction estimation trials
- **Task**: Estimate the direction of moving dot stimuli with varying coherence levels
- **Manipulations**: 
  - Motion coherence (sensory uncertainty)
  - Prior distributions with different standard deviations (10°, 40°, 60°, 80°)

## Key Variables

- `motion_direction`: True stimulus motion direction
- `motion_coherence`: Stimulus reliability (0-1 scale)
- `estimate_x/y`: Human response coordinates
- `prior_std`: Standard deviation of experimental prior
- `prior_mean`: Mean of experimental prior (fixed at 225°)
- `reaction_time`: Response time

## Analysis Pipeline

### Hypothesis I: Linear Regression
- Tests whether circular distance between previous estimate and current stimulus predicts error
- **Result**: Very low predictive power (R² ≈ 0.01)

### Hypothesis II: Multivariate Prediction
- Combines multiple features: circular distance, coherence, prior strength, reaction time
- **Result**: Poor linear fit (R² ≈ 0.001), suggesting non-linear relationships

### Hypothesis III: Bayesian Linear Regression
- Adds uncertainty quantification to coefficient estimates
- **Result**: Confirms weak linear relationships with uncertainty bounds

### Hypothesis IV: Bayesian Observer Model
- Implements optimal Bayesian integration of prior and likelihood
- Models responses as von Mises distributions
- **Result**: Moderate fit (r = 0.373), captures coherence effects but shows systematic biases

### Hypothesis V: Switching Observer Model
- Tests discrete strategy switching between prior-based and sensory-based decisions
- Switching probability depends on motion coherence
- **Result**: Similar performance to Bayesian model, suggests heuristic rather than optimal integration

## Key Findings

1. **Human behavior deviates from optimal Bayesian integration**
2. **Motion coherence systematically affects estimation accuracy**
3. **Switching between strategies may better explain behavior than continuous integration**
4. **Large individual differences and unexplained variance remain**

## Requirements

```python
# Core packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# Machine learning
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Bayesian modeling
import pymc as pm
import arviz as az
import pytensor.tensor as pt
```

## Usage

1. **Data Loading**: The notebook includes utilities to load and preprocess the motion estimation data
2. **Model Fitting**: Each hypothesis section contains complete model implementation
3. **Visualization**: Comprehensive plots for model diagnostics and interpretation
4. **Model Comparison**: Compare Bayesian vs. switching observer models

## Model Implementations

### Bayesian Observer
```python
# Combines prior and likelihood using von Mises distributions
theta_posterior = integrate_prior_likelihood(prior, likelihood, coherence)
```

### Switching Observer
```python
# Probabilistic switching between strategies
p_prior = sigmoid(alpha * coherence + beta)
strategy = sample_strategy(p_prior)
```

## Results Summary

- **Bayesian Model**: Captures basic integration principles but shows range compression
- **Switching Model**: Similar predictive power, suggests discrete rather than continuous integration
- **Both models**: Explain ~37-44% of response correlation, indicating substantial unexplained variance

## Future Directions

1. Individual difference modeling
2. Non-linear integration models
3. Trial history effects
4. Confidence-based decision making
5. Neural network approaches

## Authors

- **Arefeh Sherafati** - Primary analysis and modeling
- **Data**: Courtesy of Neuromatch Academy Computational Neuroscience Track
- **Original Research**: Laquitaine & Gardner, Neuron, 2018

## Acknowledgments

Special thanks to Daphne Zhang, Jacob Boulrice, Shayla Schwartz, Yi Gao, and AI assistants Claude Sonnet 4 and GPT-4o for collaborative support.

## License

This analysis is provided for educational and research purposes. Original data courtesy of Laquitaine & Gardner (2018).
