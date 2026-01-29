import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# -----------------------------
#  MODEL PARAMETERS
# -----------------------------
# Constants for motor torque, rolling resistance, tire friction, and dimensions
Params = {
    "Cm1": 0.287, "Cm2": 0.0545, "Cr0": 0.0518, "Cr2": 0.00035,
    "Br": 3.3852, "Cr": 1.2691, "Dr": 0.1737, "Bf": 2.579,
    "Cf": 1.2,    "Df": 0.192, "m": 0.041, "Iz": 27.8e-6,
    "lf": 0.029, "lr": 0.033, "g": 9.81, "maxAlpha": 0.6, "vx_zero": 0.3,
}

@dataclass
class ControlRules:
    """Soft rules with low noise to maintain yaw stability."""
    v_turn_max: float = 1.2             # Max speed during turns
    v_high: float = 4.0                 # High speed threshold
    d_range: Tuple[float, float] = (0.0, 0.33)  # Duty cycle range
    delta_turn_range: Tuple[float, float] = (0.015, 0.04) # Steering range for turns
    delta_straight_noise: float = 0.004  # Jitter during straight driving
    d_noise_std: float = 0.008          # Throttle noise
    delta_noise_std: float = 0.002      # Steering noise
    meas_noise_std: Dict[str, float] = None
    
    def __post_init__(self):
        if self.meas_noise_std is None:
            # Typical sensor measurement noise standard deviations
            self.meas_noise_std = {
                "X":    0.05,   # 5 mm
                "Y":    0.05,   # 5 mm
                "phi":  0.003,  # ≈ 0.17° 
                "vx":   0.010,  # 1 cm/s
                "vy":   0.003,  # 3 mm/s
                "omega":0.030,  # ≈ 1.7°/s
            }

# -----------------------------
#  VEHICLE DYNAMICS
# -----------------------------
def clamp(x, lo, hi):
    """Utility to constrain values within a range."""
    return np.minimum(np.maximum(x, lo), hi)

def tire_forces(x, u, p):
    """Calculates lateral and longitudinal tire forces."""
    _, _, _, vx, vy, omega = x; d, delta = u
    vx_eff = max(abs(vx), p["vx_zero"])
    
    # Calculate slip angles (alpha)
    alpha_f = -np.arctan2(omega * p["lf"] + vy, vx_eff) + delta
    alpha_r =  np.arctan2(omega * p["lr"] - vy, vx_eff)
    
    # Apply safety clamps to slip angles
    alpha_f = clamp(alpha_f, -p["maxAlpha"], p["maxAlpha"])
    alpha_r = clamp(alpha_r, -p["maxAlpha"], p["maxAlpha"])
    
    # Lateral forces (Magic Formula)
    Fy_f = p["Df"] * np.sin(p["Cf"] * np.arctan(p["Bf"] * alpha_f))
    Fy_r = p["Dr"] * np.sin(p["Cr"] * np.arctan(p["Br"] * alpha_r))
    
    # Longitudinal force (Motor torque minus rolling resistance)
    Frx = (p["Cm1"] - p["Cm2"] * vx_eff) * d - p["Cr0"] - p["Cr2"] * (vx_eff**2)
    return Fy_f, Fy_r, Frx

def f_cont(x, u, p):
    """Continuous-time dynamic bicycle model ODEs."""
    X, Y, phi, vx, vy, omega = x; d, delta = u
    m, Iz, lf, lr = p["m"], p["Iz"], p["lf"], p["lr"]
    Fy_f, Fy_r, Frx = tire_forces(x, u, p)
    
    # State derivatives
    Xdot = vx*np.cos(phi) - vy*np.sin(phi)
    Ydot = vx*np.sin(phi) + vy*np.cos(phi)
    phidot = omega
    vxdot = (Frx - Fy_f*np.sin(delta) + m*vy*omega) / m
    vydot = (Fy_r + Fy_f*np.cos(delta) - m*vx*omega) / m
    omegadot = (Fy_f*lf*np.cos(delta) - Fy_r*lr) / Iz
    return np.array([Xdot, Ydot, phidot, vxdot, vydot, omegadot])

def euler_step(x, u, p, Ts):
    """Numerical integration using Explicit Euler method."""
    return x + Ts * f_cont(x, u, p)

# -----------------------------
#  BEHAVIOR GENERATION
# -----------------------------
def sample_controls_piecewise(sim_steps, Ts, x0, p, rules: ControlRules, seed=42):
    """Generates a sequence of control inputs based on randomized driving modes."""
    prev_delta_cmd, delta_rate_max = 0.0, 0.30
    rng = np.random.default_rng(seed)
    U_sim, modes = np.zeros((sim_steps, 2)), [""] * sim_steps
    x_shadow = np.array(x0, dtype=float) # Internal state for local trajectory prediction
    v_floor, d_boost_min = 0.35, 0.15
    i = 0
    prev_mode = "init"
    
    while i < sim_steps:
        # Choose next driving mode based on logic transitions
        if prev_mode in ("turn_left", "turn_right"):
            # After a turn, force a straight segment (accelerate or cruise)
            mode = rng.choice(["accelerate", "cruise"], p=[0.5, 0.5])
        else:
            mode = rng.choice(["accelerate", "cruise", "turn_left", "turn_right"], 
                              p=[0.35, 0.35, 0.15, 0.15])
            
        seg_len = max(1, int(np.round(rng.uniform(0.4, 1.5) / Ts)))
        v = np.hypot(x_shadow[3], x_shadow[4])
        
        # Define behavior for chosen mode
        if mode == "accelerate":
            if v >= rules.v_high: mode = "cruise"
            d, delta = rng.uniform(0.3, rules.d_range[1]), rng.normal(0.0, rules.delta_straight_noise)
        elif mode == "cruise":
            d, delta = rng.uniform(-0.05, 0.2), rng.normal(0.0, rules.delta_straight_noise)
        elif mode in ("turn_left", "turn_right"):
            d = rng.uniform(0.0, 0.15) if v > rules.v_turn_max else rng.uniform(0.05, 0.25)
            mag = rng.uniform(*rules.delta_turn_range)
            # Reduce steering angle as speed increases to prevent spin-out
            scale = min(1.0, rules.v_turn_max / max(v, 1e-3))
            delta = (mag if mode == "turn_left" else -mag) * scale
        
        # Safety: check if car is stalling
        if v < 0.5:
            mode, d, delta = "accelerate", rng.uniform(0.5, 1.0), rng.normal(0.0, rules.delta_straight_noise)
            seg_len = max(seg_len, int(round(0.3 / Ts)))

        # Fill control buffers
        for _ in range(seg_len):
            if i >= sim_steps: break
            v = np.hypot(x_shadow[3], x_shadow[4])
            if v < v_floor: d = max(d, d_boost_min)
            
            d_k = float(np.clip(d, *rules.d_range))
            delta_k = float(np.clip(delta, -0.6, 0.6))
            
            # Rate limiting for steering (servo dynamics emulation)
            max_step = delta_rate_max * Ts
            delta_k = float(np.clip(delta_k, prev_delta_cmd - max_step, prev_delta_cmd + max_step))
            prev_delta_cmd = delta_k
            
            U_sim[i, 0], U_sim[i, 1], modes[i] = d_k, delta_k, mode
            
            # Update internal shadow state
            x_shadow = euler_step(x_shadow, U_sim[i], p, Ts)
            x_shadow[3] = max(x_shadow[3], 0.0) # Speed cannot be negative
            x_shadow[5] = float(np.clip(x_shadow[5], -6, 6)) # Yaw rate limit
            i += 1
        prev_mode = mode
    return U_sim, modes

# -----------------------------
#  DATASET GENERATION
# -----------------------------
def generate_dataset(num_traj, T, Ts, seed=2025):
    rules = ControlRules()
    rng = np.random.default_rng(seed)
    sim_steps_per_traj = int(np.round(T / Ts))
    all_trajectories_data: List[pd.DataFrame] = []

    print(f"Starting generation of {num_traj} trajectories...")
    for i in tqdm(range(num_traj), desc="Generating Trajectories"):
        # Randomized initial conditions
        x0 = np.array([
            rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-np.pi, np.pi),
            rng.uniform(0.2, 0.6), rng.uniform(-0.05, 0.05), rng.uniform(-1, 1)
        ], dtype=float)

        # 1. Create control sequence
        U_sim, modes = sample_controls_piecewise(sim_steps_per_traj, Ts, x0, Params, rules, seed=seed+i)

        # 2. Integrate physics model (Ground Truth)
        history_X_truth = np.empty((sim_steps_per_traj + 1, 6))
        history_X_truth[0, :] = x0
        for k in range(sim_steps_per_traj):
            x_dot = f_cont(history_X_truth[k, :], U_sim[k, :], Params)
            history_X_truth[k+1, :] = history_X_truth[k, :] + Ts*x_dot
            # Numerical stability clipping
            history_X_truth[k+1, 3] = max(history_X_truth[k+1, 3], 0.0)
            history_X_truth[k+1, 5] = float(np.clip(history_X_truth[k+1, 5], -6, 6))

        # 3. Add measurement noise
        N = sim_steps_per_traj + 1
        noise_rng = np.random.default_rng(12345 + i)
        noise = np.column_stack([
            noise_rng.normal(0, rules.meas_noise_std["X"], N),
            noise_rng.normal(0, rules.meas_noise_std["Y"], N),
            noise_rng.normal(0, rules.meas_noise_std["phi"], N),
            noise_rng.normal(0, rules.meas_noise_std["vx"], N),
            noise_rng.normal(0, rules.meas_noise_std["vy"], N),
            noise_rng.normal(0, rules.meas_noise_std["omega"], N),
        ])
        history_X_meas = history_X_truth + noise

        t_steps = np.arange(N) * Ts
        modes_full = modes + [""] # Match length for states (+1 compared to controls)

        # 4. Store Clean and Noisy versions
        for noise_type, state_history in [('clean', history_X_truth), ('noisy', history_X_meas)]:
            data = {
                't': t_steps,
                'X': state_history[:, 0], 'Y': state_history[:, 1], 'phi': state_history[:, 2],
                'vx': state_history[:, 3], 'vy': state_history[:, 4], 'omega': state_history[:, 5],
                'd': np.append(U_sim[:, 0], np.nan),
                'delta': np.append(U_sim[:, 1], np.nan),
                'trajectory_id': i, 'noise_type': noise_type, 'mode': modes_full
            }
            traj_df = pd.DataFrame(data)
            all_trajectories_data.append(traj_df)

    combined_dataset = pd.concat(all_trajectories_data, ignore_index=True, sort=False)
    print("Dataset generation completed.")
    return combined_dataset

# -----------------------------
#  ANIMATION UTILITY
# -----------------------------
def animate_trajectories(dataset, num_to_animate=10, interval=10, save_path=None):
    """Creates a GIF animation of the car trajectories."""
    fig, ax = plt.subplots(figsize=(10, 8))
    title = f"Animation of {num_to_animate} Trajectories"
    if save_path: title += f" ({'Clean' if 'clean' in save_path else 'Noisy'})"
    ax.set_title(title, fontsize=16)
    
    traj_ids = sorted(dataset['trajectory_id'].unique().tolist())
    if not traj_ids: raise ValueError("Empty dataset.")
    
    num_to_animate = min(num_to_animate, len(traj_ids))
    selected_ids = traj_ids[:num_to_animate]
    
    trajectories_data = []
    for tid in selected_ids:
        df = dataset[dataset['trajectory_id'] == tid].sort_values('t')
        if not df.empty:
            trajectories_data.append({'x': df['X'].values, 'y': df['Y'].values, 't': df['t'].values, 'id': tid})

    if not trajectories_data:
        print("No trajectories found for animation.")
        return

    # Set plot limits
    x_min, x_max = min(d['x'].min() for d in trajectories_data), max(d['x'].max() for d in trajectories_data)
    y_min, y_max = min(d['y'].min() for d in trajectories_data), max(d['y'].max() for d in trajectories_data)
    pad_x, pad_y = 0.05 * max(1e-6, (x_max - x_min)), 0.05 * max(1e-6, (y_max - y_min))
    
    max_len = max(len(d['t']) for d in trajectories_data)
    ax.set_xlabel('X Position [m]', fontweight='bold'); ax.set_ylabel('Y Position [m]', fontweight='bold')
    ax.grid(True); ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_min - pad_x, x_max + pad_x); ax.set_ylim(y_min - pad_y, y_max + pad_y)

    lines = [ax.plot([], [], lw=2, label=f'Traj {d["id"]}')[0] for d in trajectories_data]
    points = [ax.plot([], [], 'o', markersize=6, color=line.get_color())[0] for line in lines]
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
    ax.legend(loc='best')

    def init():
        for line, point in zip(lines, points): line.set_data([], []); point.set_data([], [])
        time_text.set_text('')
        return lines + points + [time_text]

    def animate(i):
        for j, d in enumerate(trajectories_data):
            idx = min(i, len(d['t']) - 1)
            lines[j].set_data(d['x'][:idx+1], d['y'][:idx+1])
            points[j].set_data([d['x'][idx]], [d['y'][idx]])
        t_show = trajectories_data[0]['t'][min(i, len(trajectories_data[0]['t']) - 1)]
        time_text.set_text(f'Time = {t_show:.2f} s')
        return lines + points + [time_text]

    ani = FuncAnimation(fig, animate, frames=max_len, init_func=init, interval=interval, blit=True)
    if save_path:
        try:
            writer = PillowWriter(fps=max(1, int(1000/interval)))
            ani.save(save_path, writer=writer, dpi=150)
            print(f"Animation saved to: {save_path}")
        except Exception as e: 
            print(f"Error while saving animation: {e}")
    plt.show()

# -----------------------------
#  MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    num_traj = 5000
    duration = 12.0
    time_step = 0.01

    output_csv_clean = "vehicle_piecewise_clean.csv"
    output_csv_noisy = "vehicle_piecewise_noisy.csv"
    output_gif_clean = "animation_piecewise_clean.gif"
    output_gif_noisy = "animation_piecewise_noisy.gif"

    # 1. Generate combined dataset
    combined_dataset = generate_dataset(
        num_traj=num_traj,
        T=duration,
        Ts=time_step,
        seed=42
    )

    # 2. Split and clean datasets for saving
    clean_dataset = combined_dataset[combined_dataset['noise_type'] == 'clean'].copy()
    noisy_dataset = combined_dataset[combined_dataset['noise_type'] == 'noisy'].copy()
    
    # Drop columns not needed for pure machine learning training files
    if 'noise_type' in clean_dataset.columns:
        clean_dataset = clean_dataset.drop(columns=['noise_type', 'mode'])
    if 'noise_type' in noisy_dataset.columns:
        # Note: Noisy dataset also drops ground truth 'phi' as it is often inferred or filtered
        noisy_dataset = noisy_dataset.drop(columns=['noise_type', 'mode', 'phi'])

    clean_dataset.to_csv(output_csv_clean, index=False)
    print(f"   -> Clean dataset saved as '{output_csv_clean}'")
    noisy_dataset.to_csv(output_csv_noisy, index=False)
    print(f"   -> Noisy dataset saved as '{output_csv_noisy}'")
    
    # 3. Visualization and Animation (First 10 trajectories)
    if not combined_dataset.empty:
        animate_trajectories(clean_dataset, num_to_animate=10, save_path=output_gif_clean)
        animate_trajectories(noisy_dataset, num_to_animate=10, save_path=output_gif_noisy)
