# trajectory_generation
# MPC Vehicle Controller for Path Tracking

Il codice qui presentato implementa un **Model Predictive Controller (MPC)** per la guida autonoma di un veicolo.

Il controllore utilizza un modello dinamico a **6 stati** e risolve un problema di ottimizzazione ad ogni passo per seguire una traiettoria di riferimento, gestendo anche vincoli fisici.

L'intera simulazione è implementata in **Python**, con:
- `numpy` per i calcoli numerici  
- `cvxpy` per la formulazione del problema di ottimizzazione  
- `matplotlib` per la visualizzazione  

---

##  Concetti Chiave

### Modello Dinamico del Veicolo
Il comportamento del veicolo è descritto da un modello **non lineare a 6 stati**:

- **Stati**:  

```math
x = [X, Y, \phi, v_x, v_y, \omega]
```

dove:  
- $X, Y$: posizione globale  
- $\phi$: angolo di imbardata  
- $v_x, v_y$: velocità longitudinali e laterali  
- $\omega$: velocità di imbardata  

---

- **Input**:  

```math
u = [d, \delta]
```

dove:  
- $d$: comando di trazione  
- $\delta$: angolo di sterzo  

Le forze delle gomme sono modellate usando una versione semplificata della **formula di Pacejka**.

---

### Model Predictive Control (MPC)

Il funzionamento segue un ciclo continuo:

1. **Prevede** → utilizza il modello dinamico per stimare il comportamento futuro del veicolo su un orizzonte finito \(N\).  
2. **Ottimizza** → calcola la sequenza ottimale di comandi \(u\) che minimizza una funzione di costo (errore di traiettoria, sforzo di controllo) rispettando i vincoli fisici.  
3. **Applica** → invia solo il primo comando della sequenza.  
4. **Ripete** → misura il nuovo stato e ricomincia.  

---

### Linearizzazione Time-Varying (TV-QP)
Poiché il modello è non lineare, non può essere risolto direttamente come QP.  
Ad ogni passo il modello viene **linearizzato** attorno a una traiettoria nominale, trasformando il problema in un **Quadratic Program (QP)** risolvibile in modo efficiente.

---

##  Note 

- **Assenza di Rumore**: simulazione deterministica (niente rumore di processo o di misura).  
- **Formulazione come Tracking**: l'errore è calcolato come distanza geometrica dal riferimento (trajectory tracking), invece che con coordinate di Frenet o errori angolari (path following).

---

##  Struttura dei File

### `mpc_6stati.py`
Contiene la logica del controllore e del modello fisico:
- `Params`: dizionario dei parametri fisici (massa, inerzia, coefficienti gomme, ecc.).  
- `tire_forces(x, u, p)`: calcola le forze generate dalle gomme.  
- `f_cont(x, u, p)`: implementa le equazioni del moto non lineari.  
- `linearize_discretize(x, u, Ts, p)`: linearizza e discretizza il modello.  
- `mpc_step(...)`: funzione principale dell’MPC che formula e risolve il QP con **cvxpy**.  

### `main.py`
Orchestra l’intera simulazione:
- **Profili di velocità** → `vref_profile_*`  
- **Generazione riferimenti** → `ref_window_from_x_with_vref(...)`  
- **Ciclo di simulazione** → aggiorna stato e comandi MPC ad ogni passo.  
- **Animazione** → genera `mpc_ref_vs_vehicle.gif` con `matplotlib.animation`.  

---

##  Scenari di Riferimento Testati

### 1. Percorso Sinusoidale
Utile per testare transizioni e curve costanti.

```python
ys = 0.5 * np.sin(0.5 * xs)
dydx = 0.25 * np.cos(0.5 * xs)
phs = np.arctan(dydx)
return np.stack([xs, ys, phs], axis=1)
```

### 2) Percorso Parabolico
Utile per testare la capacità del controllore di adattarsi a variazioni di curvatura.

```python
# Equazione di una parabola (es. y = 0.1 * x^2)
ys = 0.1 * xs**2
dydx = 0.2 * xs
phs = np.arctan(dydx)
return np.stack([xs, ys, phs], axis=1)
```
### Prerequisiti
```bash
pip install numpy matplotlib cvxpy osqp
```

### Come Eseguire il Progetto
```bash
python main.py
```

Lo script eseguirà la simulazione (potrebbe richiedere qualche minuto) e mostrerà un grafico.
Al termine, salverà l'animazione come mpc_ref_vs_vehicle.gif nella stessa cartella.

## Come Sperimentare

È possibile modificare la simulazione per testare diversi scenari:

- **Cambiare il Percorso**: attiva o disattiva i blocchi di codice commentati nella funzione `ref_window_from_x_with_vref` in `main.py`.

- **Modificare il Comportamento del Controllore**: all'interno della chiamata a `mpc_step` in `main.py`, si possono modificare i pesi della funzione di costo per alterare le priorità del controllore:
  - `q_c`, `q_phi`: aumenta per un inseguimento della traiettoria più aggressivo
  - `R`, `Rd`: aumenta per ottenere un controllo più fluido e meno dispendioso
  - `N`: modifica l'orizzonte di previsione

- **Cambiare le Condizioni Iniziali**: modifica la variabile `x` all'inizio del ciclo di simulazione in `main.py`.

