# pyrlmala

The python implementation of Reinforcement Learning for Adaptive MALA.

# Requirement

- Platform
  - Ubuntu 22.04.4 LTS x86_64
- Language
  - Python 3.12.7
- Package
  - bridgestan == 2.5.0
  - coverage == 7.6.7
  - jaxtyping == 0.2.34
  - numpy == 2.1.3
  - posteriordb == 0.2.0
  - pytest-cov == 6.0.0
  - pytest == 8.3.3
  - scipy == 1.14.1
  - sphinx-rtd-theme == 3.0.2
  - sphinx == 8.1.3
  - stable-baselines3 == 2.3.2
  - tqdm == 4.66.6
  - wandb == 0.18.7
  - toolz == 1.0.0
  - pytest-mock == 3.14.0

```{bash}
uv sync
```

Please note that this project has not been tested in Windows system, consult the `Stan` and `Bridgestan`.

# Task (Update 09/01/2025)
- [ ] add an interactive slider for the learning rate
- [ ]  also explore RL-MALA
- [ ] think about whether we can have parallel environments (i.e. parallel MCMC, all driven by one agent)
- [ ] work towards a results table for one test problem (e.g. banana) whose columns are {ESJD , MMD}, and whose rows are e.g. {MALA ($\sigma$ = 0.1) , MALA ($\sigma$ = 0.5), MALA ($\sigma$ = 1), MALA ($\sigma$ = 2) , Barker ($\sigma$ = 0.1), Barker ($\sigma$ = 0.5), Barker ($\sigma$ = 1), Barker ($\sigma$ = 2), RL-MALA (trained on ESJD), RL-MALA (trained on Matthew's reward), RL-Barker (trained on ESJD), RL-Barker (trained on Matthew's reward) }  For the RL-algorithms, we would stop the policy optimisation and then use the (fixed) learned policy to generate the samples used for this assessment - let's not worry too much about equating the computational demands of the different methods for the moment.
