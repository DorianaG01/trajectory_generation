import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, Dict, List

# Vehicle and simulation parameters
Params = {
    "Cm1": 0.287, "Cm2": 0.0545, "Cr0": 0.0518, "Cr2": 0.00035,
    "Br": 3.3852, "Cr": 1.2691, "Dr": 0.1737, "Bf": 2.579,
    "Cf": 1.2,    "Df": 0.192, "m": 0.041, "Iz": 27.8e-6,
    "lf": 0.029, "lr": 0.033, "g": 9.81, "maxAlpha": 0.6, "vx_zero": 0.3,
}
np.random.seed(42)

@dataclass
class ControlRules:
    """Contains standard deviations for measurement noise."""
    meas_noise_std: Dict[str, float] = None
    def __post_init__(self):
        if self.meas_noise_std is None:
            self.meas_noise_std = {
                "X":    0.05,   # 5 mm
                "Y":    0.05,   # 5 mm
                "phi":  0.003,  # ≈ 0.17° 
                "vx":   0.010,  # 1 cm/s
                "vy":   0.003,  # 3 mm/s
                "omega":0.030,  # ≈ 1.7°/s
            }

def clamp(x, lo, hi):
    """Utility to clamp values within a range."""
    return np.minimum(np.maximum(x, lo), hi)

def tire_forces(x, u, p):
    """Calculates lateral and longitudinal tire forces based on the Pacejka Magic Formula."""
    _, _, _, vx, vy, omega = x; d, delta = u
    vx_eff = max(abs(vx), p["vx_zero"])
    
    # Slip angles calculation
    alpha_f = -np.arctan2(omega * p["lf"] + vy, vx_eff) + delta
    alpha_r =  np.arctan2(omega * p["lr"] - vy, vx_eff)
    alpha_f = clamp(alpha_f, -p["maxAlpha"], p["maxAlpha"])
    
    # Lateral forces
    Fy_f = p["Df"] * np.sin(p["Cf"] * np.arctan(p["Bf"] * alpha_f))
    Fy_r = p["Dr"] * np.sin(p["Cr"] * np.arctan(p["Br"] * alpha_r))
    
    # Longitudinal force (simplified motor/friction model)
    Frx = (p["Cm1"] - p["Cm2"] * vx_eff) * d - p["Cr0"] - p["Cr2"] * (vx_eff**2)
    return Fy_f, Fy_r, Frx

def f_cont(x, u, p):
    """Continuous-time ODEs for the dynamic bicycle model."""
    X, Y, phi, vx, vy, omega = x; d, delta = u
    m, Iz, lf, lr = p["m"], p["Iz"], p["lf"], p["lr"]
    Fy_f, Fy_r, Frx = tire_forces(x, u, p)
    
    Xdot = vx*np.cos(phi) - vy*np.sin(phi)
    Ydot = vx*np.sin(phi) + vy*np.cos(phi)
    phidot = omega
    vxdot = (Frx - Fy_f*np.sin(delta) + m*vy*omega) / m
    vydot = (Fy_r + Fy_f*np.cos(delta) - m*vx*omega) / m
    omegadot = (Fy_f*lf*np.cos(delta) - Fy_r*lr) / Iz
    return np.array([Xdot, Ydot, phidot, vxdot, vydot, omegadot])

def simulate_trajectory(x0, U_sim, time_step, p):
    """Simulates trajectory using Explicit Euler with stability clipping."""
    sim_steps = len(U_sim)
    history_X = np.empty((sim_steps + 1, 6))
    history_X[0, :] = x0
    
    for k in range(sim_steps):
        x_dot = f_cont(history_X[k, :], U_sim[k, :], p)
        history_X[k + 1, :] = history_X[k, :] + time_step * x_dot
        
        # Clipping for numerical and physical stability
        history_X[k + 1, 3] = max(history_X[k + 1, 3], 0.0)
        history_X[k + 1, 5] = float(np.clip(history_X[k + 1, 5], -6.0, 6.0))

    return history_X

def create_spline_signal(num_steps, Ts, d_mean, d_std, delta_mean, delta_std):
    """Generates smooth control signals using Cubic Splines."""
    if num_steps <= 1: return np.array([d_mean]), np.array([delta_mean])
    
    checkpoint_interval_s = np.random.uniform(3.0, 5.0)
    checkpoint_interval_steps = max(1, int(round(checkpoint_interval_s / Ts)))
    
    idx = np.arange(0, num_steps, checkpoint_interval_steps)
    if idx[-1] != num_steps - 1: idx = np.append(idx, num_steps - 1)
    
    d_chk, delta_chk = np.random.normal(d_mean, d_std, len(idx)), np.random.normal(delta_mean, delta_std, len(idx))
    s_d, s_delta = CubicSpline(idx, d_chk, bc_type='natural'), CubicSpline(idx, delta_chk, bc_type='natural')
    
    t = np.arange(num_steps)
    return s_d(t), s_delta(t)

def generate_smooth_profiles(total_steps, Ts, stats, mode='random'):
    """Creates control profiles (Duty cycle and Steering) for different driving modes."""
    if mode == 'random': mode = np.random.choice(['straight', 'sinusoid'], p=[0.5, 0.5])
    
    transient_duration_s = np.random.uniform(1.5, 3.0)
    transient_steps = min(int(transient_duration_s / Ts), total_steps)
    steady_state_steps = total_steps - transient_steps
    
    # Initial transient phase
    d_tr, delta_tr = create_spline_signal(num_steps=transient_steps, Ts=Ts, d_mean=stats['d_mean'], d_std=stats['d_std']*0.2, delta_mean=stats['delta_mean'], delta_std=stats['delta_std']*0.3)
    
    if steady_state_steps > 0:
        d_st = np.random.normal(stats['d_mean'], stats['d_std']*0.05, steady_state_steps)
        if mode == 'sinusoid':
            t_st = np.arange(steady_state_steps) * Ts
            period, amplitude = np.random.uniform(4.0, 8.0), np.random.uniform(stats['delta_std']*0.5, stats['delta_std']*1.5)
            phase, omega = np.random.uniform(0, 2*np.pi), 2*np.pi/period
            sinusoid, noise = amplitude*np.sin(omega*t_st + phase), np.random.normal(0, stats['delta_std']*0.1, steady_state_steps)
            delta_st = stats['delta_mean'] + sinusoid + noise
        else: 
            # Straight mode
            delta_st = np.random.normal(stats['delta_mean'], stats['delta_std']*0.01, steady_state_steps)
        
        d_profile, delta_profile = np.concatenate([d_tr, d_st]), np.concatenate([delta_tr, delta_st])
    else: 
        d_profile, delta_profile = d_tr, delta_tr
        
    return d_profile, delta_profile, mode

def apply_du_bounds(u, du_min, du_max):
    """Limits the rate of change (slew rate) of the control signal."""
    out = np.empty_like(u); out[0] = u[0]
    for k in range(1, len(u)):
        du = np.clip(u[k] - out[k-1], du_min, du_max)
        out[k] = out[k-1] + du
    return out

def create_trajectory_dataframe(state_history, U_sim, t_steps, traj_id, noise_type, mode):
    """Helper function to create a DataFrame. Includes 'phi' only for 'clean' data."""
    data = {
        't': t_steps, 
        'X': state_history[:, 0], 
        'Y': state_history[:, 1], 
        'vx': state_history[:, 3], 
        'vy': state_history[:, 4], 
        'omega': state_history[:, 5],
        'd': np.append(U_sim[:, 0], np.nan), 
        'delta': np.append(U_sim[:, 1], np.nan),
        'trajectory_id': traj_id, 
        'noise_type': noise_type, 
        'mode': mode
    }
    
    if noise_type == 'clean':
        data['phi'] = state_history[:, 2]
    
    return pd.DataFrame(data)

def plot_sample_controls(dataset, trajectory_id):
    """Plots Ground Truth controls for a specific trajectory."""
    traj_data = dataset[dataset['trajectory_id'] == trajectory_id]
    clean_traj = traj_data[traj_data['noise_type'] == 'clean'].copy().dropna(subset=['d', 'delta'])
    
    if clean_traj.empty:
        print(f"Missing control data for trajectory {trajectory_id}.")
        return
        
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    mode = clean_traj['mode'].iloc[0]
    fig.suptitle(f'Ground Truth Controls - Trajectory {trajectory_id} (Mode: {mode})', fontsize=16)
    
    axs[0].plot(clean_traj['t'], clean_traj['d'], lw=2, color='royalblue', label='Ground Truth (d)')
    axs[0].set_ylabel('d [-]'); axs[0].grid(True, linestyle=':'); axs[0].legend()
    
    axs[1].plot(clean_traj['t'], clean_traj['delta'], lw=2, color='royalblue', label='Ground Truth (delta)')
    axs[1].set_ylabel('delta [rad]'); axs[1].set_xlabel('Time [s]'); axs[1].grid(True, linestyle=':'); axs[1].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
    
def animate_trajectories(dataset, num_to_animate, noise_type_to_animate='noisy', save_path=None):
    """Animates trajectories for a given noise type ('clean' or 'noisy')."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal', adjustable='box')
    
    noise_type_str_capitalized = noise_type_to_animate.capitalize()
    title = f'Animation of {num_to_animate} Trajectories ({noise_type_str_capitalized})'
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X Position [m]'); ax.set_ylabel('Y Position [m]'); ax.grid(True, linestyle=':')
    
    anim_dataset = dataset[dataset['noise_type'] == noise_type_to_animate]
    traj_ids = anim_dataset['trajectory_id'].unique()
    
    trajectories_data = []
    for i in traj_ids[:num_to_animate]:
        traj = anim_dataset[anim_dataset['trajectory_id'] == i]
        trajectories_data.append({'x': traj['X'].values, 'y': traj['Y'].values, 't': traj['t'].values})
    
    if not trajectories_data:
        print(f"No '{noise_type_to_animate}' data found for animation.")
        plt.close(fig)
        return

    x_min = min(d['x'].min() for d in trajectories_data) - 1
    x_max = max(d['x'].max() for d in trajectories_data) + 1
    y_min = min(d['y'].min() for d in trajectories_data) - 1
    y_max = max(d['y'].max() for d in trajectories_data) + 1
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    
    lines = [ax.plot([], [], lw=2, label=f'Traj {traj_ids[i]}')[0] for i in range(len(trajectories_data))]
    points = [ax.plot([], [], 'o', markersize=8, color=line.get_color())[0] for line in lines]
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.legend(loc='best')

    max_len = max(len(d['t']) for d in trajectories_data)

    def init():
        for line, point in zip(lines, points): line.set_data([], []); point.set_data([], [])
        time_text.set_text('')
        return lines + points + [time_text]
    
    def animate(i):
        for j, data in enumerate(trajectories_data):
            idx = min(i, len(data['t']) - 1)
            lines[j].set_data(data['x'][:idx+1], data['y'][:idx+1])
            points[j].set_data([data['x'][idx]], [data['y'][idx]])
        
        t_now = trajectories_data[0]['t'][min(i, len(trajectories_data[0]['t']) - 1)]
        time_text.set_text(f'Time = {t_now:.2f} s')
        return lines + points + [time_text]

    ani = FuncAnimation(fig, animate, frames=max_len, init_func=init, interval=30, blit=True)
    if save_path:
        print(f"\nSaving animation to '{save_path}'...")
        writer = PillowWriter(fps=int(1000/30)); ani.save(save_path, writer=writer, dpi=120)
        print("Saving completed.")
    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    num_traj = 5000
    traj_duration = 12.0
    time_step = 0.01
    
    output_csv_clean = "vehicle_mpc_clean.csv" 
    output_csv_noisy = "vehicle_mpc_noisy_err005.csv"
    output_animation_noisy = "animation_smooth_noisy.gif"

    # Base statistics for control inputs
    mpc_stats = {'d_mean': 0.2161, 'd_std': 0.1314, 'delta_mean': 0.0035, 'delta_std': 0.0338}
    du_bounds = ((-0.1, 0.1), (-0.04, 0.04)) # Max rate of change per step
    
    rules = ControlRules() 
    noise_seed_base = 12345 

    all_trajectories_data = []
    sim_steps_per_traj = int(traj_duration / time_step)
    
    # Ranges for random initial states
    x_init_range = (-2.0, 2.0)
    y_init_range = (-2.0, 2.0)
    phi_init_range = (-np.pi, np.pi)
    vx_init_range = (0.4, 1.5)
    vy_init_range = (-0.05, 0.05)
    omega_init_range = (-1.0, 1.0)
    
    for i in tqdm(range(num_traj), desc="Generating Trajectories"):

        # Randomize initial state
        x0 = np.array([
            np.random.uniform(*x_init_range),
            np.random.uniform(*y_init_range),
            np.random.uniform(*phi_init_range),
            np.random.uniform(*vx_init_range),
            np.random.uniform(*vy_init_range),
            np.random.uniform(*omega_init_range)
        ])

        # Generate control profiles
        d_clean, delta_clean, mode = generate_smooth_profiles(sim_steps_per_traj, time_step, mpc_stats)
        
        # Add high-frequency noise to controls
        noise_d = np.random.normal(0, mpc_stats['d_std'] * 0.1, sim_steps_per_traj)
        noise_delta = np.random.normal(0, mpc_stats['delta_std'] * 0.1, sim_steps_per_traj)
        
        # Apply physical constraints (slew rate and limits)
        d_true = np.clip(apply_du_bounds(d_clean + noise_d, *du_bounds[0]), -1.0, 1.0)
        delta_true = np.clip(apply_du_bounds(delta_clean + noise_delta, *du_bounds[1]), -0.6, 0.6)
        U_sim_true = np.stack([d_true, delta_true], axis=1)

        # Physics simulation (Ground Truth)
        history_X_truth = simulate_trajectory(x0, U_sim_true, time_step, Params)

        # Generate noisy measurements
        N = sim_steps_per_traj + 1
        noise_rng = np.random.default_rng(noise_seed_base + i)
        meas_noise = np.column_stack([
            noise_rng.normal(0, rules.meas_noise_std["X"], N),     
            noise_rng.normal(0, rules.meas_noise_std["Y"], N),     
            noise_rng.normal(0, rules.meas_noise_std["phi"], N),   
            noise_rng.normal(0, rules.meas_noise_std["vx"], N),    
            noise_rng.normal(0, rules.meas_noise_std["vy"], N),    
            noise_rng.normal(0, rules.meas_noise_std["omega"], N), 
        ])
        
        history_X_meas = history_X_truth + meas_noise
        
        t_steps = np.arange(N) * time_step
        df_clean = create_trajectory_dataframe(history_X_truth, U_sim_true, t_steps, i, 'clean', mode)
        df_noisy = create_trajectory_dataframe(history_X_meas, U_sim_true, t_steps, i, 'noisy', mode)
        
        all_trajectories_data.extend([df_clean, df_noisy])
    
    # Merge all data
    final_dataset = pd.concat(all_trajectories_data, ignore_index=True, sort=False)
    
    # Save 'clean' dataset
    clean_dataset_only = final_dataset[final_dataset['noise_type'] == 'clean'].copy()
    # Remove columns not needed for the final file
    cols_to_drop_clean = [col for col in ['noise_type', 'mode'] if col in clean_dataset_only.columns]
    clean_dataset_only = clean_dataset_only.drop(columns=cols_to_drop_clean)
    
    # Reorder columns to put 'phi' in a logical position
    if 'phi' in clean_dataset_only.columns:
        cols_order = ['t', 'X', 'Y', 'phi', 'vx', 'vy', 'omega', 'd', 'delta', 'trajectory_id']
        final_cols = [col for col in cols_order if col in clean_dataset_only.columns]
        clean_dataset_only = clean_dataset_only[final_cols]
        
    clean_dataset_only.to_csv(output_csv_clean, index=False)
    print(f"   -> 'clean' dataset (with phi) saved as '{output_csv_clean}'")

    # Save 'noisy' dataset
    noisy_dataset_only = final_dataset[final_dataset['noise_type'] == 'noisy'].copy()
    # Remove metadata and 'phi' (often excluded in noisy observations depending on sensor suite)
    cols_to_drop_noisy = [col for col in ['noise_type', 'mode', 'phi'] if col in noisy_dataset_only.columns]
    noisy_dataset_only = noisy_dataset_only.drop(columns=cols_to_drop_noisy)
        
    noisy_dataset_only.to_csv(output_csv_noisy, index=False)
    print(f"   -> 'noisy' dataset (without phi) saved as '{output_csv_noisy}'")

    # Visualization
    if not final_dataset.empty:
        try:
            sinusoid_example_id = final_dataset[final_dataset['mode'] == 'sinusoid']['trajectory_id'].iloc[0]
            print(f"Plotting sample controls for trajectory {sinusoid_example_id}...")
            plot_sample_controls(final_dataset, sinusoid_example_id)
        except IndexError:
            print("No 'sinusoid' trajectory found for sample plot.")
        
        animate_trajectories(final_dataset, num_to_animate=10, 
                             noise_type_to_animate='noisy', 
                             save_path=output_animation_noisy)
