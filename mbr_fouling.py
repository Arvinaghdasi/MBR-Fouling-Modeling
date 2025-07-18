
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import seaborn as sns

# Parameters for simulation
TMP0 = 5  # starting TMP in kPa
fouling_rate = 0.05
recovery = 0.02
colloid_growth = 0.1
smp_growth = 0.08

# Time setup
start = 0
end = 48
times = np.linspace(start, end, 500)

# The model (how TMP, colloid and SMP change over time)
def model(t, y, air_power):
    tmp = y[0]
    c = y[1]
    s = y[2]

    dt = fouling_rate * (c + s) - recovery * air_power
    dc = colloid_growth * (1 - 0.2 * air_power)
    ds = smp_growth * (1 - 0.3 * air_power)

    return [dt, dc, ds]

# Initial values
initial = [TMP0, 0, 0]

# Run simulation for 2 cases
output = {}
cases = [('Case 1: Constant Air (Wu)', 0.5), ('Case 2: Fine Air (Temmerman)', 1.0)]

for name, intensity in cases:
    sol = solve_ivp(model, (start, end), initial, args=(intensity,), t_eval=times)
    df = pd.DataFrame()
    df['Time'] = sol.t
    df['TMP'] = sol.y[0]
    df['Colloid'] = sol.y[1]
    df['SMP'] = sol.y[2]
    df['Case'] = name
    output[name] = df

# Merge both cases
df_all = pd.concat(output.values(), ignore_index=True)

# Plot 1: TMP
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_all, x='Time', y='TMP', hue='Case')
plt.title('TMP Change')
plt.xlabel('Time (h)')
plt.ylabel('TMP (kPa)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Colloidal Matter
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_all, x='Time', y='Colloid', hue='Case')
plt.title('Colloid Accumulation')
plt.xlabel('Time (h)')
plt.ylabel('Colloid (mg/L)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: SMP
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_all, x='Time', y='SMP', hue='Case')
plt.title('SMP Concentration')
plt.xlabel('Time (h)')
plt.ylabel('SMP (mg/L)')
plt.grid(True)
plt.tight_layout()
plt.show()
