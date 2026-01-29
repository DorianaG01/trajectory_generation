# MPC Vehicle Controller for Path Tracking

This project implements a **Model Predictive Controller (MPC)** for the autonomous guidance of a vehicle using a **dynamic 6-state bicycle model**. The system solves a constrained optimization problem at each time step to track a reference trajectory while respecting physical limits and tire dynamics.

The simulation environment is built in **Python** using:
* **NumPy**: For high-performance numerical computations and matrix operations.
* **CVXPY**: For the convex formulation and solving of the optimization problem.
* **Matplotlib**: For real-time visualization and generating trajectory animations.

---

## Key Concepts

### Dynamic Vehicle Model
The vehicle is described by a non-linear 6-state model that accounts for lateral sliding and mass distribution:


* **States ($x$):**
    $$x = [X, Y, \phi, v_x, v_y, \omega]$$
    * $X, Y$: Global coordinates.
    * $\phi$: Yaw angle (heading).
    * $v_x, v_y$: Longitudinal and lateral velocities.
    * $\omega$: Yaw rate.

* **Inputs ($u$):**
    $$u = [d, \delta]$$
    * $d$: Duty cycle (motor traction/braking).
    * $\delta$: Steering angle.

Tire interactions are modeled using the **Pacejka Magic Formula**, which calculates lateral forces ($F_y$) based on slip angles ($\alpha$) and specific friction coefficients.

---

### Model Predictive Control (MPC) logic
The controller follows a **Receding Horizon** strategy:

1.  **Prediction**: Uses the non-linear model to estimate the vehicle's future states over a prediction horizon $N$.
2.  **Linearization**: Since the model is non-linear, it performs **Time-Varying Linearization (TV-MPC)** around a nominal trajectory at each step.
3.  **Optimization**: Solves a **Quadratic Program (QP)** to find the control sequence that minimizes a cost function (tracking error + control effort).
4.  **Feedback**: Only the first command of the sequence is applied; the process repeats at the next sampling time $T_s$ with updated sensor data.



---

## Project Structure

### `mpc_6stati.py` (Core Controller)
This module handles the physics and the math:
* **`tire_forces`**: Computes slip angles and non-linear tire reactions.
* **`f_cont`**: Defines the continuous-time ODEs for the vehicle dynamics.
* **`linearize_discretize`**: Performs central finite differences to obtain $A_d, B_d$, and $g$ matrices for the TV-QP formulation.
* **`mpc_step`**: Formulates the optimization problem with `cvxpy`, enforcing constraints on input magnitude (e.g., $|\delta| \leq 0.6$ rad) and input rates (to ensure smoothness).

### `main.py` (Simulation & Scenarios)
This script runs the actual experiment:
* **Velocity Profiles**: Defines target speeds (Ramp, Trapezoidal, or Sine wave).
* **Path Generation**: Creates geometric references such as a **Parabolic Path** ($y = 0.1x^2$) or **Sinusoidal Path**.
* **Simulation Loop**: Integrates the vehicle dynamics over time and stores the history for analysis.
* **Animation**: Uses `matplotlib.animation` to save the results as `mpc_tracking.gif`.

---

## Tested Scenarios

The reference path $(X^\*, Y^\*, \phi^\*)$ is generated using the following geometric functions:

### 1. Parabolic Tracking
Used to test the controller's ability to handle a continuously increasing curvature:
* **Position**: $Y^* = 0.1 \cdot (X^*)^2$
* **Heading**: $\phi^* = \arctan(0.2 \cdot X^*)$

### 2. Sinusoidal Tracking
Used to test S-curves and rapid changes in direction:
* **Position**: $Y^* = 0.5 \cdot \sin(0.5 \cdot X^*)$
* **Heading**: $\phi^* = \arctan(0.25 \cdot \cos(0.5 \cdot X^*))$
---

## Getting Started

### Prerequisites
Install the required libraries:
```bash
pip install numpy matplotlib cvxpy osqp
```

### Running the Project

Execute the main simulation script:

```bash
python main.py

```

Upon completion, the script will print control statistics (mean duty cycle, steering range) and generate a GIF animation of the tracking performance.

---
## Acknowledgments

This implementation is inspired by and based on the concepts found in the MPCC (Model Predictive Contouring Control) repository by Alexander Liniger.
https://github.com/alexliniger/MPCC.git

## Experimentation Guide

You can modify the controller's behavior by adjusting parameters in `main.py`:

* **Aggressiveness**: Increase `q_c` (lateral error) or `q_phi` (heading error) weights for tighter tracking.
* **Smoothing**: Increase `Rd` weights to reduce jerky steering movements.
* **Prediction**: Change `N` to adjust the look-ahead horizon.
* **Geometry**: Toggle the commented sections in `ref_window_from_x_with_vref` to switch between a Sine wave or a Parabola.


