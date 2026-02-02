
# Vehicle Dynamics Framework: Simulation, Estimation & Control

This repository hosts a comprehensive framework for autonomous vehicle research, focusing on the interplay between **data generation**, **state estimation**, and **optimal control**.

The project integrates physical modeling (Dynamic Bicycle Model) with Deep Learning (KalmanNet) and optimization-based control (MPC). Uniquely, **the MPC controller is used upstream to generate realistic datasets**, ensuring the training data represents physically feasible driving behaviors.

##  Project Structure

The repository is organized into three main modules:

```text
 ┣ 📂 generation_traj   # 1. Data Generation (Uses MPC logic)
 ┣ 📂 KalmanNet         # 2. State Estimation (AI + Filtering)
 ┗ 📂 MPC               # 3. Motion Control (Core Logic)

```

### 1. 🚗 Trajectory Generation (`/generation_traj`)

**Goal:** Create synthetic datasets for training and testing.
This module simulates a **Dynamic 6-State Bicycle Model**. Crucially, it utilizes the **MPC controllers** to drive the vehicle along various reference paths (splines, sinusoidal, parabolic).

* **Method:** The MPC computes optimal control inputs () to track a reference, generating realistic "closed-loop" trajectories.
* **Output:** Produces "Clean" (Ground Truth) and "Noisy" (Sensor) datasets (`combined_dataset_clean.csv`, etc.).

### 2. 🧠 State Estimation (`/KalmanNet`)

**Goal:** Estimate the true vehicle state from noisy observations.
This module implements **KalmanNet (KNet)**, a hybrid architecture that combines the domain knowledge of the Extended Kalman Filter (EKF) with the learnability of Recurrent Neural Networks (RNNs).

* **Input:** The noisy trajectories generated in step 1.
* **Strategies:** Supports Standard Sequential TBPTT, Gradient Accumulation, and Independent Chunking training.

### 3. 🎮 Model Predictive Control (`/MPC`)

**Goal:** Core implementation of the control logic.
This module contains the mathematical formulation of the **Model Predictive Controller** using `CVXPY`. It solves a constrained optimization problem at every time step to handle tire dynamics and actuator limits.

* **Usage:** This logic is imported by the generation module to create data, but can also be run standalone to test control performance.
* **Dynamics:** Accounts for lateral sliding (Pacejka formula) and mass distribution.

---

##  Workflow

The data flows through the system as follows:

1. **MPC-Driven Generation:**
Run `generation_traj/` scripts. The system instantiates the MPC controller to drive the simulated vehicle along complex paths. This ensures the dataset contains realistic kinematic and dynamic behaviors.
2. **Dataset Creation:**
The simulation saves the **Control Inputs** and **State Trajectories**, adding Gaussian noise to simulate GPS sensors.
3. **KalmanNet Training:**
Use `KalmanNet/` to train the model. The network learns to map the noisy sensor data back to the clean states produced by the MPC simulation.

---

##  Installation & Requirements

### Global Dependencies

```bash
pip install numpy pandas matplotlib scipy

```

### Module-Specific Dependencies

* **For KalmanNet:** `pip install torch wandb`
* **For MPC/Generation:** `pip install cvxpy osqp`

*Please refer to the `README.md` file inside each subfolder for detailed installation and usage instructions.*

---
