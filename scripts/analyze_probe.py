"""
analyze_probe.py - turn a probe-controller log (mpc_log.npz) into an estimated
per-thruster effect in the BODY frame.

For each action channel it averages the body-frame linear and angular
acceleration while that channel is pulsed, minus the average while everything
is off (which removes buoyancy/gravity imbalance and drift). Each row is
roughly one column of the thruster allocation matrix, scaled by M^-1.

    python analyze_probe.py mpc_log.npz
"""

import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "mpc_log.npz"
d = np.load(path)
t, u = d["t"], d["action"]
vel_b, omega_b = d["vel_b"], d["omega_b"]
dt = np.median(np.diff(t))

# body-frame accelerations by finite difference
acc = np.gradient(vel_b, dt, axis=0)
alpha = np.gradient(omega_b, dt, axis=0)

active = np.abs(u) > 1e-6
idle = ~active.any(axis=1)
skip = int(0.3 / dt)  # ignore the first 0.3 s of each pulse (motor spin-up)

base_acc = acc[idle].mean(axis=0) if idle.any() else np.zeros(3)
base_alpha = alpha[idle].mean(axis=0) if idle.any() else np.zeros(3)

print(f"log: {path}   steps: {len(t)}   dt: {dt:.4f} s")
print(f"idle baseline  acc_b={np.round(base_acc, 3)}  alpha_b={np.round(base_alpha, 3)}\n")
print("ch | amp  |   du/dt    dv/dt    dw/dt  |   dp/dt    dq/dt    dr/dt   (body frame, minus idle)")
print("---+------+----------------------------+----------------------------")

for k in range(u.shape[1]):
    on = np.where(active[:, k])[0]
    if len(on) == 0:
        continue
    # drop the start of every contiguous pulse
    starts = on[np.insert(np.diff(on) > 1, 0, True)]
    mask = np.ones(len(on), bool)
    for s in starts:
        mask &= ~((on >= s) & (on < s + skip))
    idx = on[mask]
    if len(idx) == 0:
        continue
    amp = u[idx, k].mean()
    a = acc[idx].mean(axis=0) - base_acc
    w = alpha[idx].mean(axis=0) - base_alpha
    print(f"{k:2d} | {amp:+.2f} | {a[0]:+8.3f} {a[1]:+8.3f} {a[2]:+8.3f} | "
          f"{w[0]:+8.3f} {w[1]:+8.3f} {w[2]:+8.3f}")

print("\nThe dominant entries tell you each thruster's axis and sign "
      "(surge/sway/heave and roll/pitch/yaw).")