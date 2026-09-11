"""Shared hyper-parameters, dataset sizes, and state/action ranges.

All knobs mirror Lambert et al. 2020 (Appendix tab:paramspets and tab:dataset)."""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PETS hyper-parameters (Lambert et al. 2020, Appendix tab:paramspets)
NET_WIDTH = 500
NET_DEPTH = 2
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.9
FULL_EPOCHS = 100

CEM_HORIZON = 25
CEM_SAMPLES = 400
CEM_ELITES = 40
CEM_ITERATIONS = 5
REWARD_TRIALS = 10

# Dataset sizes (Lambert et al. 2020, Appendix tab:dataset)
CP_GRID_SIZE = 16807
CP_EXPERT_SIZE = 2400
CP_ONPOLICY_SIZE = 3780
EXPERT_REWARD_THRESHOLD = 179.0

STATE_RANGES = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.5, 0.5], [-3.0, 3.0]])
ACTION_RANGE = np.array([-1.0, 1.0])

# MuJoCo cartpole model (MJCF), used by the MujocoCartPoleEnv fallback.
CP_XML = """
<mujoco model="cartpole">
  <compiler angle="radian"/>
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <body name="cart" pos="0 0 0.1">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2.5 2.5"/>
      <geom name="cart" type="box" size="0.1 0.1 0.04" pos="0 0 -0.04" mass="1.0"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0.1"/>
        <geom name="pole" type="capsule" fromto="0 0 0.1 0 0 0.6" size="0.02" mass="0.1"/>
        <inertial pos="0 0 0.35" mass="0.1" diaginertia="0.001 0.001 0.001"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="cart_motor" joint="slider" gear="10" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""