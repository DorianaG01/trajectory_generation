
# KalmanNet for Vehicle Trajectory Estimation

This repository contains the implementation of **KalmanNet (KNet)** applied to vehicle dynamics estimation. It combines the domain knowledge of the Extended Kalman Filter (EKF) with the learning capability of Recurrent Neural Networks (RNNs) to estimate the state of a vehicle observed through noisy GPS sensors.

## 📂 Repository Structure

The codebase is organized into core modules, training pipelines, and testing scripts:

| File Name | Description |
| --- | --- |
| **`kalman_net.py`** | Defines the KNet neural network architecture. |
| **`vehicle_model.py`** | Contains the physical system model (state transition , observation , and Jacobians). |
| **`data_loader.py`** | Utilities for loading, splitting, and normalizing the vehicle dataset. |
| **`pipeline.py`** | The main class handling the **Standard** and **Accumulation** training loops. |
| **`pipeline_indipendent_chunking.py`** | Specialized pipeline class for the **Chunked & Shuffled** training strategy. |
| **`pipeline_prediction.py`** | Pipeline class dedicated to the long-term prediction task. |
| **`training.py`** | Script to execute Standard or Gradient Accumulation training. |
| **`training_indipendent_chunking.py`** | Script to execute the Chunked/Shuffled training strategy. |
| **`training_prediction.py`** | Script to train the specific prediction model. |
| **`test_prediction.py`** | Script for inference, metric calculation (ADE/FDE), and plotting results. |
| **`test_vehicle.py`** | Unit tests to verify the vehicle model dynamics. |

---

## Training Strategies

This project implements three distinct training strategies to handle the complexity of long vehicle trajectories.

### 1. Standard TBPTT

* **Script:** `training.py` (with strategy set to `standard`)
* **Description:** Uses **Truncated Backpropagation Through Time**. The model processes trajectories sequentially and updates weights every  steps.
* **Use Case:** Best for learning temporal dependencies while managing memory usage.

### 2. Gradient Accumulation

* **Script:** `training.py` (with strategy set to `accumulation`)
* **Description:** Processes the trajectory in chunks but accumulates gradients over the entire sequence before performing a weight update.
* **Use Case:** Simulates full-batch training; useful when stability is prioritized over frequent updates.

### 3. Independent Chunking

* **Script:** `training_indipendent_chunking.py`
* **Description:** Long trajectories are sliced into smaller fixed-size segments (chunks). These chunks are **randomly shuffled** before being fed into the network.
* **Use Case:** Breaks temporal correlations to prevent overfitting to specific trajectory starting points and improves robustness.

---

##  Setup & Requirements

### 1. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies and avoid conflicts.

**On macOS and Linux:**

```bash
python3 -m venv .kalmanet
source .kalmanet/bin/activate

```

**On Windows:**

```bash
python -m venv .kalmanet
.\.kalmanet\Scripts\activate

```

### 2. Install Dependencies

Once the virtual environment is active, install the required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt

```
### Data Format

The system expects dataset files (e.g., `combined_dataset_clean.csv`) containing:

* **Time:** 
* **State:** 
* **Control Inputs:** Acceleration, Steering angle

---

## 🏃‍♂️ Usage

### 1. Training

To train the model, run the script corresponding to your desired strategy:

```bash
# For Standard or Accumulation training
python training.py

# For Independent Chunking training
python training_indipendent_chunking.py

```

*Note: Training logs (Loss in dB, Learning Rate) are automatically tracked via Weights & Biases (WandB) if you have an account.*

### 2. Evaluation & Testing

To evaluate the trained model on the test set and generate plots:

1. Ensure your best model weights (`.pt` file) are in the results folder.
2. Update the model path in `test_prediction.py`.
3. Run:

```bash
python test_prediction.py

```

### 3. Metrics

The testing script calculates the following metrics:

* **MSE (dB):** Mean Squared Error in Decibels.
* **ADE (Average Displacement Error):** Mean Euclidean distance between estimated and ground truth positions.
* **FDE (Final Displacement Error):** Error distance at the final time step.

---
