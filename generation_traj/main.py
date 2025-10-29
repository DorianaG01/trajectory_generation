import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from mpc_6stati import mpc_step, f_cont, Params  

p = Params

def d_steady_state(v):
    num = p["Cr0"] + p["Cr2"] * v**2 # somma delle resistenze (rotolamento + aerodinamica)
    den = p["Cm1"] - p["Cm2"] * v # spinta motore disponibile che diminuisce con la velocità 
    return num / den
# serve a trovare il comando del motore per mantenere la velocità desiderata
# input è la velocità e output è percentuale di acceleratore d necessaria
# è usato per impostare lo stato iniziale e dare all’MPC un punto di partenza realistico (non partire da 0)

v0_target = 1.0                        
u_prev = np.array([d_steady_state(v0_target), 0.0])
print(f"Stato iniziale: v0_target={v0_target:.2f} m/s, u_prev={u_prev}")


# Profili di velocità di riferimento vref(t) (N+1,)
# ogni funzione costruisve un array v di lunghezza N+1 , dove N è il numero di passi predittivi e Ts è il tempo di campionamento 
# l'array contiene i valori di velocità target vref[0], vref[1], ..., vref[N] per ogni passo predittivo

def vref_profile_ramp_cruise(N, Ts, v0=0.8, v_cruise=2.0, tramp=2.0):
    # Velocità che accelera linearmente da v0 a v_cruise in tramp secondi,
    t = np.arange(N + 1) * Ts
    v = v0 + (v_cruise - v0) * np.clip(t / tramp, 0.0, 1.0)
    return v

def vref_profile_trapezoid(N, Ts, v0=0.8, vmax=2.0, t_acc=2.0, t_flat=3.0, t_dec=2.0):
    # Velocità che accelera linearmente da v0 a vmax in t_acc secondi,
    # mantiene vmax per t_flat secondi, poi decelera a v0 in t_dec secondi.
    t = np.arange(N + 1) * Ts
    v = np.full(N + 1, v0)
    v = np.where(t <= t_acc, v0 + (vmax - v0) * (t / t_acc), vmax)
    v = np.where(t > t_acc + t_flat,
                 vmax - (vmax - v0) * ((t - (t_acc + t_flat)) / t_dec),
                 v)
    return np.clip(v, v0, vmax)

def vref_profile_sine(N, Ts, v_mean=1.5, v_amp=0.5, period_s=6.0):
    # Velocità che oscilla sinusoidalmente attorno a v_mean
    t = np.arange(N + 1) * Ts
    return v_mean + v_amp * np.sin(2 * np.pi * t / period_s)


# Genera il “percorso ideale” che il veicolo deve seguire, a partire da una posizione iniziale e da un profilo di velocità di riferimento (vref_seq)
def ref_window_from_x_with_vref(x_start, N, Ts, vref_seq):
    """
    vref_seq: array (N+1,) di velocità desiderata lungo l'orizzonte
    Ritorna path_ref shape (N+1, 3): [X*, Y*, phi*]
    # phi* è l'angolo di direzione del percorso in radianti 
    """
    vref_seq = np.asarray(vref_seq).reshape(N + 1)
    xs = np.zeros(N + 1)
    xs[0] = x_start
    for k in range(N):
        xs[k + 1] = xs[k] + vref_seq[k] * Ts  
    # calcola quanto il veicolo avanza lungo l’asse X ad ogni passo, assumendo che si muova a velocità vref_seq[k] per un intervallo Ts
   # Equazione di una sinusoide (es. y = 0.5 * sin(0.5 * x))
    """ys = 0.5 * np.sin(0.5 * xs)
    dydx = 0.25 * np.cos(0.5 * xs)           # derivata di y(x)
    phs = np.arctan(dydx)
    return np.stack([xs, ys, phs], axis=1)"""
    
 
    # Equazione di una parabola (es. y = 0.1 * x^2)
    ys = 0.1 * xs**2
    dydx = 0.2 * xs
    phs = np.arctan(dydx)
    return np.stack([xs, ys, phs], axis=1)

# Simulazione MPC
# Parametri simulazione
Ts = 0.02       # passo di simulazione 
sim_steps=600
N = 40           # orizzonte predittivo (numero di passi)
# Stato iniziale
x = np.array([0.0, 0.5, 0.0, v0_target, 0.0, 0.0])  # [X, Y, phi, v, delta, omega]
traj_X = []
traj_U = []
path_refs = []  # per salvare i riferimenti nel tempo


for t in range(sim_steps):
    vref_seq = vref_profile_ramp_cruise(N, Ts, v0=0.8, v_cruise=2.0, tramp=2.0)
    #vref_seq = vref_profile_trapezoid(N, Ts, v0=0.8, vmax=2.0, t_acc=2.0, t_flat=3.0, t_dec=2.0)
    #vref_seq = vref_profile_sine(N, Ts, v_mean=1.5, v_amp=0.5, period_s=6.0)
    path_ref = ref_window_from_x_with_vref(x_start=x[0], N=N, Ts=Ts, vref_seq=vref_seq)

    # salva il riferimento corrente
    path_refs.append(path_ref)

    u_cmd, status, info = mpc_step(x, u_prev, path_ref, Ts=Ts, N=N, params=Params, vref=vref_seq)
    x = x + Ts * f_cont(x, u_cmd, Params) # Eulero

    traj_X.append(x.copy())
    traj_U.append(u_cmd.copy())
    u_prev = u_cmd


traj_X = np.array(traj_X)
traj_U = np.array(traj_U)


t_u = np.arange(traj_U.shape[0]) * Ts
d_seq = traj_U[:, 0]
delta_seq = traj_U[:, 1]


# Statistiche per 'd' (acceleratore) usando la variabile 'd_seq'
d_mean = np.mean(d_seq)
d_std = np.std(d_seq)
d_min = np.min(d_seq)
d_max = np.max(d_seq)
print(f"\nAcceleratore 'd_seq':")
print(f"  - Media: {d_mean:.4f}")
print(f"  - Deviazione Standard: {d_std:.4f}")
print(f"  - Range: [{d_min:.4f}, {d_max:.4f}]")

# Statistiche per 'delta' (sterzo) usando la variabile 'delta_seq'
delta_mean = np.mean(delta_seq)
delta_std = np.std(delta_seq)
delta_min = np.min(delta_seq)
delta_max = np.max(delta_seq)
print(f"\nSterzo 'delta_seq':")
print(f"  - Media: {delta_mean:.4f}")
print(f"  - Deviazione Standard: {delta_std:.4f}")
print(f"  - Range: [{delta_min:.4f}, {delta_max:.4f}]")

print(max(omega_seq := traj_X[:, 5]))


fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle("Comandi MPC e Media Statistica")

# Plot d_seq
axs[0].plot(t_u, d_seq, linewidth=1.8, label='d_seq (segnale)')
axs[0].axhline(d_mean, color='r', linestyle='--', label=f'Media: {d_mean:.2f}')
axs[0].set_ylabel("d [-]")
axs[0].grid(True, linestyle=":")
axs[0].legend()

# Plot delta_seq
axs[1].plot(t_u, delta_seq, linewidth=1.8, label='delta_seq (segnale)')
axs[1].axhline(delta_mean, color='r', linestyle='--', label=f'Media: {delta_mean:.2f}')
axs[1].set_ylabel("δ [rad]")
axs[1].set_xlabel("Tempo [s]")
axs[1].grid(True, linestyle=":")
axs[1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("mpc_inputs_analysis.png", dpi=150)
plt.show()

# Animazione veicolo vs riferimento 

T = len(traj_X)
assert T > 1, "traj_X è vuoto!"

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(np.min(traj_X[:,0]) - 1, np.max(traj_X[:,0]) + 1)
ax.set_ylim(np.min(traj_X[:,1]) - 1, np.max(traj_X[:,1]) + 1)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title("MPC: veicolo vs riferimento")

# traiettoria percorsa (linea)
trail, = ax.plot([], [], 'b-', linewidth=2, label="veicolo")

# puntino veicolo (scatter, più robusto per un singolo punto)
pt = ax.scatter([], [], s=40, c='r', label="posizione")

# freccia heading (quiver)
arrow = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1, color='r')

# riferimento (tratteggio)
ref_ln, = ax.plot([], [], 'k--', linewidth=1.5, label="riferimento (finestra)")

ax.legend(loc="best")

def init():
    trail.set_data([], [])
    pt.set_offsets(np.empty((0, 2)))   # scatter array Nx2
    ref_ln.set_data([], [])
    
    return trail, pt, ref_ln, arrow

def update(i):
    # traiettoria percorsa fino a i
    trail.set_data(traj_X[:i+1, 0], traj_X[:i+1, 1])

    # punto corrente del veicolo
    pt.set_offsets(np.array([[traj_X[i, 0], traj_X[i, 1]]]))

    # freccia heading dalla posa corrente
    phi = traj_X[i, 2]
    dx, dy = np.cos(phi), np.sin(phi)
    arrow.set_offsets([traj_X[i, 0], traj_X[i, 1]])
    arrow.set_UVC(dx, dy)

    # finestra di riferimento corrente 
    if i < len(path_refs):
        pref = path_refs[i]
    else:
        pref = path_refs[-1]
    ref_ln.set_data(pref[:, 0], pref[:, 1])

    return trail, pt, ref_ln, arrow

ani = animation.FuncAnimation(
    fig, update, frames=T, init_func=init,
    blit=False, interval=40  
)

ani.save("mpc_ref_vs_vehicle_parab_X02_Y05.gif", writer=animation.PillowWriter(fps=25))
print("GIF salvata: mpc_ref_vs_vehicle.gif")
plt.show()