import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Параметры системы ===
N = 100
rcut = 2.5
dt = 0.005
n_steps_eq = 100
n_steps_avg = 50

# === Физические параметры для СИ ===
sigma = 3.405e-10  # м
epsilon = 1.65e-21  # Дж
kB = 1.38e-23  # Дж/К
m = 6.63e-26  # кг

def initialize_positions(L):
    n = int(np.ceil(np.sqrt(N)))
    x = np.linspace(0, L, n, endpoint=False)
    y = np.linspace(0, L, n, endpoint=False)
    xv, yv = np.meshgrid(x, y)
    pos = np.vstack([xv.ravel(), yv.ravel()]).T[:N]
    return pos

def initialize_velocities(T):
    v = np.random.randn(N, 2)
    v -= np.mean(v, axis=0)
    v *= np.sqrt(T / np.mean(np.sum(v**2, axis=1)))
    return v

def compute_forces(pos, L):
    forces = np.zeros_like(pos)
    potential_energy = 0.0
    virial = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rij = pos[i] - pos[j]
            rij -= L * np.round(rij / L)
            r2 = np.dot(rij, rij)
            if r2 < rcut**2:
                inv_r2 = 1 / r2
                inv_r6 = inv_r2 ** 3
                inv_r12 = inv_r6 ** 2
                f_scalar = 48 * inv_r2 * (inv_r12 - 0.5 * inv_r6)
                fij = f_scalar * rij
                forces[i] += fij
                forces[j] -= fij
                potential_energy += 4 * (inv_r12 - inv_r6)
                virial += np.dot(rij, fij)
    return forces, potential_energy, virial

def velocity_verlet(pos, vel, dt, L):
    forces, _, _ = compute_forces(pos, L)
    pos += vel * dt + 0.5 * forces * dt**2
    pos %= L
    f_new, potential_energy, virial = compute_forces(pos, L)
    vel += 0.5 * (forces + f_new) * dt
    return pos, vel, f_new, potential_energy, virial

def kinetic_energy(vel):
    return 0.5 * np.sum(vel**2)

def run_simulation(T, rho):
    L = np.sqrt(N / rho)
    pos = initialize_positions(L)
    vel = initialize_velocities(T)

    for step in tqdm(range(n_steps_eq)):
        pos, vel, forces, _, _ = velocity_verlet(pos, vel, dt, L)

    E_total, P_total = 0.0, 0.0
    for step in tqdm(range(n_steps_avg)):
        pos, vel, forces, potential, virial = velocity_verlet(pos, vel, dt, L)
        kin = kinetic_energy(vel)
        E_total += kin + potential

        V = L**2
        P = rho * T + virial / (2 * V)
        P_total += P

    E_avg = E_total / n_steps_avg / N
    P_avg = P_total / n_steps_avg
    return E_avg, P_avg

# === Задание 2: E(T) и Cv ===
rho_fixed = 0.4
T_list = np.linspace(0.1, 2.0, 20)
E_list = []

for T in T_list:
    print(f"[E(T)] Simulating T = {T:.2f}")
    E_avg, _ = run_simulation(T, rho_fixed)
    E_list.append(E_avg)

E_list = np.array(E_list)
dT = T_list[1] - T_list[0]
Cv_list = np.gradient(E_list, dT)

# === Перевод в СИ ===
E_SI = E_list * epsilon
Cv_SI = Cv_list * epsilon / kB
Cv_SI_mol = Cv_SI * 6.022e23

# === График E(T) и Cv ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(T_list, E_list, 'o-')
plt.xlabel('T (LJ)')
plt.ylabel('E/N (LJ)')
plt.title('E(T), ρ = 0.4')

plt.subplot(1, 2, 2)
plt.plot(T_list, Cv_SI_mol, 's-r')
plt.xlabel('T (LJ)')
plt.ylabel('Cv (Дж/моль·К)')
plt.title('Cv(T), ρ = 0.4')

plt.tight_layout()
plt.show()

# === Приблизительные температуры фазовых переходов ===
d2E = np.gradient(Cv_list, dT)
transitions = np.where(np.abs(d2E) > 5)[0]

print("\n=== Температуры фазовых переходов (приблизительно) ===")
for i in transitions:
    print(f"T ≈ {T_list[i]:.2f}")

# === Задание 3: Изотермы P(ρ) ===
T_iso = [0.3, 0.6, 1.0, 1.5]
rho_list = np.linspace(0.2, 0.8, 10)
pressures = {}

for T in T_iso:
    p_row = []
    for rho in rho_list:
        print(f"[P(ρ)] T = {T:.2f}, ρ = {rho:.3f}")
        _, P_avg = run_simulation(T, rho)
        p_row.append(P_avg)
    pressures[T] = p_row

# === График P(ρ) ===
plt.figure(figsize=(8, 6))
for T in T_iso:
    plt.plot(rho_list, pressures[T], '-o', label=f"T = {T:.2f}")
plt.xlabel("ρ (σ⁻²)")
plt.ylabel("P (в LJ)")
plt.title("Изотермы давления P(ρ)")
plt.legend()
plt.grid()
plt.show()
import matplotlib.animation as animation

def animate_simulation(T, rho, n_steps=500, interval=20):
    L = np.sqrt(N / rho)
    pos = initialize_positions(L)
    vel = initialize_velocities(T)

    fig, ax = plt.subplots(figsize=(5, 5))
    scat = ax.scatter(pos[:, 0], pos[:, 1], s=20, c='blue')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_title(f"Движение частиц при T={T}, ρ={rho}")
    ax.set_aspect('equal')

    def update(frame):
        nonlocal pos, vel
        pos, vel, _, _, _ = velocity_verlet(pos, vel, dt, L)
        scat.set_offsets(pos)
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=interval, blit=True)
    plt.show()
