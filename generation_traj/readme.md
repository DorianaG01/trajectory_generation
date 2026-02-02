
#  Vehicle Trajectory Dataset Generation Pipeline 🚗

This repository contains a complete pipeline for generating synthetic vehicle trajectory datasets. By simulating a **dynamic 6-state bicycle model** with different control strategies, the pipeline produces high-quality data for training and evaluating autonomous driving models or state estimators.

##  Workflow Overview

To obtain the final dataset, simply run the generation scripts followed by the merge script. The entire process takes only a few minutes to produce thousands of trajectories.

1. **Generate Raw Data**: Run the two generation scripts to create diverse driving behaviors.
2. **Merge & Re-index**: Run the merge script to unify the data into a single consistent training set.
3. **Ready for Training**: The resulting combined CSV files are ready to be loaded into your Machine Learning pipeline.

---

##  Script Descriptions

### 1. `generation_type1.py` (Smooth/MPC-like) 🛺

Generates trajectories with fluid, continuous movements.

* **Method**: Uses Cubic Splines or MPC logic to ensure smooth transitions in steering and acceleration.
* **Outputs**: Creates `vehicle_mpc_clean.csv` and `vehicle_mpc_noisy_err.csv`.

### 2. `generation_type2.py` (Behavior-based/Piecewise) 🏎️

Generates more aggressive and varied maneuvers.

* **Method**: Uses a Finite State Machine to sample discrete behaviors like "Turn Left," "Accelerate," or "Cruise."
* **Outputs**: Creates `vehicle_piecewise_clean.csv` and `vehicle_piecewise_noisy_err.csv`.

### 3. `merge_datasets.py` (Data Aggregator) 🚔

This is the final step of the pipeline. It combines the data from both generation types.

* **ID Re-indexing**: It automatically calculates an ID offset (e.g., +5000) for the second dataset to ensure every trajectory in the final file has a **unique ID**.
* **Outputs**: Produces the final files:
* `combined_dataset_clean.csv` (**Ground Truth**)
* `combined_dataset_noisy.csv` (**Sensor Measurements**)



---

## Dataset Structure

The pipeline produces two specific types of data:

### **Clean Dataset (Ground Truth)** 🧹

* **Purpose**: Used as the "Target" or "Label" during training.
* **Content**: Contains the true physical states  without any distortion.

### **Noisy Dataset (Sensor Measurements)** 🎧

* **Purpose**: Used as the "Input" features for the model.
* **Content**: Simulates real-world sensor limitations by adding Gaussian noise to positions and velocities. It typically excludes orientation () to force the model to learn state estimation from movement.

---

## Setup and Execution

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib scipy tqdm

```

### 2. Generate and Merge

Run these commands in order:

```bash
python generation_type1.py
python generation_type2.py
python merge_datasets.py

```

---

## Summary of Dynamics

The simulation is based on a **Dynamic Bicycle Model** utilizing the **Pacejka Magic Formula**:

* **States**: 
* **Inputs**: 
* **Stability**: Includes safety clipping for yaw rate and longitudinal velocity to ensure physical realism.

---
