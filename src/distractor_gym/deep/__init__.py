"""Neural (torch) model-learning stack for the WP1 deep diagnostics.

Holds the value-aware dynamics models, the per-sample value-gradient (VJP)
machinery, and the offline training machinery over the continuous Distractor-Gym
suite. Torch is imported lazily inside the modules so the base package (numpy-only
tabular suite) imports without the optional ``deep`` extra.
"""
