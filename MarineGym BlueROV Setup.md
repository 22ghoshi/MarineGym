---
title: MarineGym BlueROV Heavy — setup, fixes, teleop & MPC harness
date: 2026-11-03
tags: [marinegym, isaac-sim, bluerov, mpc, auv, simulation]
machine: Acer Swift SFX14-41G (RTX 3050 Ti 4 GB, driver 580)
---

# MarineGym BlueROV Heavy — setup, fixes, teleop & MPC harness

> [!summary]
> MarineGym (Isaac Sim 4.1) runs on the laptop with **one BlueROV Heavy, GPU physics with shrunk buffers, a display, and a hand-written control loop** instead of RL. Keyboard teleop works from its own window, and `mpc_run.py` is a template with an `MPCController` stub to fill in. This note records every fix needed to get there, plus everything learned about the vehicle model.

**Goal:** use the simulator to develop **MPC for a BlueROV**

---

## 1. Layout and environment

| Thing | Location |
|---|---|
| Isaac Sim 4.1.0 standalone | `/mnt/crucial/isaac410` |
| IsaacLab (Marine-RL fork) | `/mnt/crucial/IsaacLab` (`_isaac_sim` → `../isaac410`) |
| MarineGym | `/mnt/crucial/MarineGym` |
| My scripts | `/mnt/crucial/MarineGym/scripts/` |
| Conda env | `marinegym` (Python 3.10), in `/mnt/crucial/miniconda3/envs/marinegym` |

The root partition is nearly full, so **everything lives on `/mnt/crucial`**, including pip's temp and cache folders.

`~/.bashrc` must contain:
```bash
export ISAACSIM_PATH="/mnt/crucial/isaac410"    # the docs say $HOME/isaac410 — wrong for this machine
```

---

## 2. Installation (what actually worked)

Followed the MarineGym "Source Setup" guide, with these deviations:

```bash
# env + repos
conda create -n marinegym python=3.10 -y
conda activate marinegym
cd /mnt/crucial
git clone https://github.com/Marine-RL/MarineGym.git
git clone https://github.com/Marine-RL/IsaacLab.git
cd IsaacLab && ln -s ../isaac410 _isaac_sim && cd ..
cp -r MarineGym/conda_setup/etc "$CONDA_PREFIX"      # Isaac activation overlay
conda deactivate && conda activate marinegym

pip install --upgrade pip setuptools wheel
pip install usd-core==23.11 lxml==4.9.4 tqdm xxhash
cd IsaacLab && ./isaaclab.sh --install && cd ..
```

### 2.1 Fixes needed during install

| Problem | Cause | Fix |
|---|---|---|
| pip pulled **torch 2.14 / CUDA 13** and NCCL download failed | `tensordict==0.4.0`/`torchrl==0.4.0` need `torch>=2.3`; the env had no torch, so pip took the newest | Install a cu118 torch **first**: `pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118` |
| Packages leaking from `~/.local/lib/python3.10` | User site-packages on the path | `conda env config vars set PYTHONNOUSERSITE=1` |
| "No space left on device" during pip install | pip temp + cache live on the full root drive | `pip cache purge`, then `mkdir -p /mnt/crucial/tmp /mnt/crucial/pip-cache` and `conda env config vars set TMPDIR=/mnt/crucial/tmp PIP_CACHE_DIR=/mnt/crucial/pip-cache` |
| `Isaac Sim path:` empty / `/setup_conda_env.sh` not found | `ISAACSIM_PATH` not set | Set it in `~/.bashrc` (above) |
| `pydantic_core` errors from conda commands | The overlay puts Isaac's bundled pydantic on `PYTHONPATH` | Harmless. Optionally run conda from `(base)` with `-n marinegym` |
| torchvision / torchaudio mismatch | Old 2.2.2 builds left over | `pip install torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118` |
| gymnasium missing deps (if it shows up) | Undeclared | `pip install farama-notifications jax-jumpy` |

Then:
```bash
cd /mnt/crucial/MarineGym
pip install -e . --resume-retries 20
python -c "import marinegym; print('MarineGym installed successfully')"
```

### 2.2 Fixes needed to get `train.py` to run

| Problem | Fix |
|---|---|
| `train.py` not found | It's in `scripts/`, not the repo root |
| `\\` line continuations in the docs | The docs have a formatting bug: use a single `\`, or one line |
| `task=HoverRand` doesn't exist | Use `task=Hover` (only `Hover`, `Track`, `Landing` exist in `cfg/task/`) |
| `No module named 'setproctitle'` | `pip install setproctitle` (missing from `setup.py`) |
| `wandb.util.generate_id` missing | `pip install "wandb==0.17.9"` |
| `get_world_poses() got an unexpected keyword argument 'usd'` | Patch `marinegym/views/__init__.py` (below) |
| CUDA out of memory | Shrink the PhysX GPU buffers in `cfg/base/sim_base.yaml` (below) |
| Thousands of `Failed to create change watch ... No space left on device` | Not disk: the **inotify** watch limit. `echo "fs.inotify.max_user_watches=524288" \| sudo tee /etc/sysctl.d/90-inotify.conf && sudo sysctl --system` |

**`marinegym/views/__init__.py` patch.** Isaac 4.1's `omni.isaac.core` calls `get_world_poses(usd=...)` during init, before the physics view exists. Add a `usd` parameter and fall back to the parent class:
```python
def get_world_poses(self, env_indices=None, clone=True, usd=False):
    if usd or getattr(self, "_physics_view", None) is None:
        return super().get_world_poses(env_indices, clone=clone, usd=True)
    ...  # original body unchanged
```
Do the same for `get_local_poses` if it's overridden.

**`cfg/base/sim_base.yaml`: shrunk PhysX buffers** (originals are sized for thousands of envs). Keep a copy of the originals for bigger GPUs:
```yaml
gpu_max_rigid_contact_count: 65536
gpu_max_rigid_patch_count: 16384
gpu_found_lost_pairs_capacity: 262144
gpu_found_lost_aggregate_pairs_capacity: 1048576
gpu_total_aggregate_pairs_capacity: 262144
gpu_max_soft_body_contacts: 65536
gpu_max_particle_contacts: 65536
gpu_heap_capacity: 16777216
gpu_temp_buffer_capacity: 4194304
```

Smoke test that works (headless RL):
```bash
cd /mnt/crucial/MarineGym/scripts
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py task=Hover algo=ppo headless=true enable_livestream=false mode=train task.drone_model.name=iAUV task.env.num_envs=4 total_frames=5000 wandb.mode=offline
```

### 2.3 GPU-saving decisions (for the MPC use case)

- **1 environment** (`task.env.num_envs=1`).
- **GPU physics, not CPU.** CPU physics (`task.sim.device=cpu ...`) crashes with a device mismatch because MarineGym hardcodes `cuda` tensors in places. With 1 env and shrunk buffers, GPU physics costs little.
- **Lower resolution:** `"viewer.resolution=[1280,720]"` (the default is 2460×1440).
- **Render less often than physics:** render 1 of every 2 steps (`render_every=2`).
- **`lowgpu` mode** (optional) turns off RTX translucency, reflections, indirect diffuse, AO, sampled lighting, subsurface and caustics, and sets DLSS to performance. Translucency (water/glass) caused the original Vulkan OOM. With one env, full rendering (`lowgpu=false`) has fit so far.
- **Recommended:** `sudo prime-select on-demand` (then log out and in) so the desktop runs on the AMD iGPU and leaves the 4 GB free for Isaac.

---

## 3. Hydra config gotchas

- `scripts/train.yaml` has defaults `task: HoverRand`, `algo: ppo`. **Always pass `task=Hover`.**
- `algo/ppo` isn't a file. It's registered in Python by `import marinegym.learning` (ConfigStore). Any custom script must import that module, or Hydra fails with `Could not find 'algo/ppo'`.
- `sim: ${task.sim}` and `env: ${task.env}` are interpolations. **Override `task.sim.*` / `task.env.*`**, never `sim.*` (that fails with "Key 'device' is not in struct").
- New keys need a `+` prefix, e.g. `+mpc.controller=probe` and `+teleop.gain=0.5`.
- Physics: `dt = 0.016 s`, `substeps = 1`, gravity `[0, 0, -9.81]`.
- Default camera: `viewer.eye = [8, 0, 6]`, `lookat = [0, 0, 3]` (about 9 m from the robot).

---

## 4. Fixing the graphics

| Symptom | Cause | Fix |
|---|---|---|
| World and robot all white | USDs contain absolute paths from the authors' machines (`/home/david/...`, `~/Documents/...`, S3 URLs). Missing materials render white. | `fix_asset_paths.py --apply` rewrites them to the local copies shipped in the repo (`.bak` backups kept) |
| `PNG not supported: unknown PNG chunk type` / `broken data stream` (sand) | `Sand_BaseColor.png` in `Collected_EmptyMarine/` is truncated/corrupt | Download a fresh copy (below) and re-save the other PNGs with Pillow |
| "Sand above the robot" | Reflection of the seabed in the water surface (z = 10 m) seen from below | Cosmetic. Hide `Water` in the Stage panel if it bothers you |
| "No water", washed out | RTX has no underwater volume scattering | Distance fog (`WATER_FOG` in `mpc_run.py`) |
| Still white near the robot, even with fog | **The world's single light is far too bright**: a 100 × 100 m `RectLight`, intensity 15000, at z ≈ 7.5 m. The sand clips to white, and fog (blended before tone mapping) can't pull it back. | `dim_scene_lights()`: `+teleop.light=0.1` / `+mpc.light=0.1` (the default), which is about −3.3 EV. GUI test: Stage → `defaultGroundPlane/RectLight` → Exposure −3 |
| BlueROV stays plain-coloured | Its mesh has corrupted UVs (`corrupted data in primvar 'st_*'`) | Asset problem. Assign a plain coloured material if needed |
| Ghost second BlueROV | Hover's visual goal marker at `/World/envs/env_0/target` | `hide_target_marker()` hides it automatically. `+mpc.hide_target=false` brings it back |
| Pressing F turned the viewport black | Isaac's "frame selection" hotkey | Teleop now uses a separate key window, and heave-down moved to V |

**Sand texture repair** (already done):
```bash
cd /mnt/crucial/MarineGym/marinegym/robots/assets/usd/worlds/Collected_EmptyMarine/omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Natural/Sand
wget -O Sand_BaseColor.png https://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Natural/Sand/Sand_BaseColor.png
python -c "
from PIL import Image
for f in ['Sand_BaseColor.png','Sand_Normal.png','Sand_ORM.png']:
    im = Image.open(f); im.load(); print('ok', f, im.size, im.mode)
"
```
`Water_Normal.png`, `Sand_Normal.png` and `Sand_ORM.png` were re-saved with Pillow (originals kept as `.png.bak`).

**Unresolved asset paths that are fine:** `OmniPBR.mdl` (built into Isaac) and the three `Grid/...` textures (fetched from S3 when online).

### 4.1 Water fog settings

Render settings changed in the GUI **don't persist**, and MarineGym rebuilds the stage on every launch, so the scripts apply them at startup. All keys were verified to exist in Isaac 4.1 (`carb.settings.get_settings().get("/rtx/fog")`).

```python
WATER_FOG = {
    "/rtx/fog/enabled": True,
    "/rtx/fog/fogColor": [0.02, 0.22, 0.28],   # dark teal; I'd tried cyan [0.01, 0.56, 0.76] in the GUI
    "/rtx/fog/fogColorIntensity": 0.8,
    "/rtx/fog/fogStartDist": 0.5,
    "/rtx/fog/fogEndDist": 20.0,
    "/rtx/fog/fogDistanceDensity": 0.6,
    "/rtx/fog/fogZup/enabled": False,           # height fog off
}
```
- **Murkier water:** end 10–12 m, density 0.8. **Clearer water:** end 30–40 m, density 0.4.
- If the camera moves closer, scale the end distance down proportionally.
- Re-tune after dimming the light. With the sand no longer clipped, the fog reads much stronger.
- `print()` in Isaac's Script Editor goes to the **launching terminal**. Use `carb.log_warn(...)` to see output in the Console.

### 4.2 World geometry (`worlds/EmptyMarine.usd`, referenced at `/World/defaultGroundPlane`)

| Prim | Details |
|---|---|
| `Sand` | Plane at **z = −5 m**, 200 × 200 m, material `Sand_01` |
| `Water` | Surface plane at **z = +10 m**, material `Water_Blue_Ocean_Perlinwaves_01` |
| `RectLight` | 100 × 100 m, z ≈ 7.49 m, intensity 15000, exposure 0 (the only light) |
| Sky | `Unreal_Engine_4_Sky.usdz` (payload; path fixed by `fix_asset_paths.py`) |

Hover spawns the robot at (0, 0, 2) and randomizes the reset pose in x, y ∈ [−2.5, 2.5], z ∈ [1.5, 2.5], roll/pitch ±0.2π, yaw 0–2π. That's why the scripts **level the vehicle after every reset** (`upright`).

---

## 5. Commands

All commands are run from the scripts folder:
```bash
conda activate marinegym
cd /mnt/crucial/MarineGym/scripts
```

### 5.1 Teleop (keyboard)
```bash
python teleop.py task=Hover task.drone_model.name=BlueROVHeavy headless=false \
  task.env.num_envs=1 task.env.max_episode_length=1000000 \
  "viewer.resolution=[1280,720]" +teleop.lowgpu=false
```
A separate **"ROV teleop"** window opens. **Click it and keep it focused**, so keys never reach Isaac's shortcuts. The banner turns grey when it isn't focused, and losing focus releases all keys, so thrusters stop. Closing the window or pressing Ctrl+C quits and saves the log.

| Keys | Action (body frame, FLU) |
|---|---|
| W / S | surge forward / back |
| A / D | sway left / right |
| R / V | heave up / down |
| Q / E | yaw left / right |
| I / K | pitch nose down / up |
| J / L | roll left / right |
| = / − | gain ±0.1 (starts 0.5, range 0.1–1) |
| T | toggle auto-levelling (PD on roll/pitch: `−2·angle − 0.5·rate`) |
| SPACE (hold) | all thrusters off |

| Option | Default | Meaning |
|---|---|---|
| `+teleop.input=window\|isaac` | `window` | Key window, or Isaac viewport keys |
| `+teleop.gain=0.5` | 0.5 | Initial command gain |
| `+teleop.stabilize=true` | true | Auto-level roll/pitch when not commanded |
| `+teleop.lowgpu=true\|false` | true | Strip RTX features |
| `+teleop.fog=true\|false` | true | Water fog |
| `+teleop.light=0.1` | 0.1 | Scale world lights (1 = as shipped) |
| `+teleop.hide_target=true` | true | Hide the ghost goal marker |
| `+teleop.upright=true` | true | Level after reset |
| `+teleop.render_every=2` | 2 | Render 1 in N physics steps |
| `+teleop.realtime=true` | true | Pace to wall clock |
| `+teleop.log=PATH` | `teleop_log.npz` | Log file |

If the key window doesn't open, run `conda install tk` (tkinter missing). The script falls back to Isaac viewport keys.

### 5.2 MPC harness / probe
```bash
# probe: pulses each thruster channel in turn (one full cycle = 34 s)
python mpc_run.py task=Hover task.drone_model.name=BlueROVHeavy headless=false \
  task.env.num_envs=1 task.env.max_episode_length=1000000 \
  "viewer.resolution=[1280,720]" +mpc.controller=probe +mpc.duration=34 +mpc.lowgpu=false
python analyze_probe.py mpc_log.npz

# your MPC (once MPCController is filled in)
python mpc_run.py task=Hover task.drone_model.name=BlueROVHeavy headless=false \
  task.env.num_envs=1 task.env.max_episode_length=1000000 \
  "viewer.resolution=[1280,720]" +mpc.controller=mpc "+mpc.target=[0,0,2]" +mpc.lowgpu=false
```

| Option | Default | Meaning |
|---|---|---|
| `+mpc.controller=zero\|probe\|mpc` | probe | Which controller |
| `+mpc.render_every=N` | 2 | Render 1 in N steps |
| `+mpc.realtime=true\|false` | true | Pace to wall clock |
| `+mpc.duration=SECONDS` | until closed | Stop after this much sim time |
| `+mpc.target=[x,y,z]` | initial position | Position reference |
| `+mpc.log=PATH` | `mpc_log.npz` | Log file |
| `+mpc.upright=true\|false` | true | Level after reset |
| `+mpc.lowgpu=true\|false` | true | Strip RTX features |
| `+mpc.fog=true\|false` | true | Water fog |
| `+mpc.light=0.1` | 0.1 | Scale world lights |
| `+mpc.hide_target=true\|false` | true | Hide the ghost goal marker |

Stop with **one** Ctrl+C: a custom SIGINT handler saves the log. Isaac's own handler segfaulted before.

### 5.3 One-time maintenance
```bash
cd /mnt/crucial/MarineGym
python scripts/fix_asset_paths.py           # dry run
python scripts/fix_asset_paths.py --apply   # already applied
```

---

## 6. Files

### My scripts (`MarineGym/scripts/`)

| File | What it does |
|---|---|
| `mpc_run.py` | **Control-loop harness + shared helpers.** Hydra entry point on `train.yaml`, which imports `marinegym.learning`. Builds 1 env, applies render settings, hides the ghost, dims lights, levels the vehicle after reset, and runs a controller every step at about real time. Logs to `mpc_log.npz`. Exports helpers used by `teleop.py`: `apply_low_gpu_render_settings()`, `WATER_FOG` + `apply_water_fog()`, `dim_scene_lights()`, `hide_target_marker()`, `quat_rotate_inverse()`, `read_state(drone)`. Controllers: `ZeroController` (watch passive dynamics), `ProbeController` (amp 0.3, 2 s on / 2 s off per channel), `MPCController` (**stub to fill in**). |
| `teleop.py` | **Keyboard teleop.** Builds the 6×8 allocation matrix B from USD geometry, maps keys to a body wrench, then pseudo-inverse to 8 throttles, with optional auto-levelling. Reads back the thrust actually applied and the resulting wrench, sends a live status to the key window, and logs to `teleop_log.npz`. Saves B to `thruster_B.npy`. |
| `teleop_window.py` | **Tkinter key-capture window**, run as a subprocess by `teleop.py`. Protocol over pipes: it sends `READY` / `P <KEY>` / `R <KEY>` / `QUIT`, and receives `S <status>` / `QUIT`. Debounces X11 key auto-repeat (40 ms) and releases all keys on focus loss. Shows a key help, held keys, pose/velocity/attitude, a thruster table (location, push direction, throttle, thrust in N), and the body and world wrench. |
| `analyze_probe.py` | Reads `mpc_log.npz` from a probe run. For each channel it prints the mean **body-frame** linear and angular acceleration while that channel is on, minus the idle baseline (a rough column of M⁻¹B). Good for signs; poor for magnitudes. |
| `fix_asset_paths.py` | Scans MarineGym's USDs for broken absolute asset paths, finds the same file in the repo, and rewrites it as a relative path using `UsdUtils.ModifyAssetPaths`. Dry run by default, `--apply` to write, with `.bak` backups. |

### Outputs

| File | Contents |
|---|---|
| `mpc_log.npz` | Per step: time, state, action, solve time |
| `teleop_log.npz` | `t, pos_w, quat, vel_b, omega_b, wrench` (normalized command), `action` (throttles), `thrust_N` (applied, 8), `wrench_b`, `wrench_w` (applied, [N, N·m]) |
| `thruster_B.npy` | 6×8 allocation matrix, rows `[X, Y, Z, K, M, N]` in body FLU, **per 1 N of thrust** per column, about the `base_link` origin |

### Modified MarineGym files

- `cfg/base/sim_base.yaml`: shrunk PhysX GPU buffers.
- `marinegym/views/__init__.py`: `get_world_poses(..., usd=False)` fallback.
- USD files under `robots/assets/usd/**`: asset paths made relative (`.bak` kept).
- `worlds/Collected_EmptyMarine/.../Sand/*.png`, `Water_Normal.png`: repaired.

---

## 7. The BlueROV Heavy model in MarineGym

Sources: `marinegym/robots/assets/usd/BlueROVHeavy/BlueROVHeavy.yaml`, `marinegym/robots/drone/underwaterVehicle.py`, `marinegym/actuators/t200.py`, startup logs, and the probe run.

### 7.1 Frames and state

- **Isaac body frame is FLU:** x forward, y left, z up. World is z-up.
- Positive rotations are right-handed: +roll (about x) = **right side down**, +pitch (about y) = **nose down**, +yaw (about z) = **turn left**.
- **MarineGym's hydrodynamics are computed internally in FRD/NED-style signs.** It flips components [v, w, q, r] (indices 1, 2, 4, 5) before applying Fossen and flips the result back. If you write the MPC model in Fossen's FRD convention, convert in `read_state`. Mixing them gives sign errors in sway, heave, pitch and yaw.
- `read_state(drone)` returns `pos_w`, `quat` (**w, x, y, z**), `vel_w`, `omega_w`, `vel_b` (u, v, w), `omega_b` (p, q, r). That's Fossen's η and ν, in FLU.
- MarineGym's own `get_state()` = `[pos, rot, vel, heading, up, throttle*2−1]` (the obs is 34-dim for Hover). The harness doesn't depend on it.
- Action: `("agents", "action")`, shape `(1, 1, 8)`, **normalized throttle in [−1, 1]**, one per thruster `rotor_0` … `rotor_7`. Read the spec from `env.full_action_spec`.

### 7.2 Rigid body and hydrostatics

| Parameter | Value |
|---|---|
| Mass m | **11.5 kg** |
| Inertia diag (Ixx, Iyy, Izz) | **(0.21, 0.245, 0.245) kg·m²** |
| Volume | 0.0116499 m³ |
| Buoyancy | 997 · 9.8 · V ≈ **113.8 N** (weight ≈ 112.8 N, so **≈ +1 N net positive**) |
| CoB above CoM (`coBM`) | 0.01 m (restoring moment ≈ 1.14 N·m per rad, small angles) |

Buoyancy in MarineGym (FRD, from roll φ and pitch θ): `X = B sinθ`, `Y = −B sinφ cosθ`, `Z = −B cosφ cosθ`, `K = −d·B cosθ sinφ`, `M = −d·B sinθ`, with d = coBM. It's applied as an extra force at `base_link`. Gravity is applied by PhysX.

### 7.3 Hydrodynamic coefficients (Fossen, diagonal)

| DOF | Added mass | Linear damping | Quadratic damping |
|---|---|---|---|
| X (surge) | 5.5 | 4.03 | 18.18 |
| Y (sway) | 12.7 | 6.22 | 21.66 |
| Z (heave) | 14.57 | 5.18 | 36.99 |
| K (roll) | 0.12 | 0.07 | 1.55 |
| M (pitch) | 0.12 | 0.07 | 1.55 |
| N (yaw) | 0.12 | 0.07 | 1.55 |

(from `dsor-isr/farol_gazebo` per the yaml comment)

How the sim applies them, each step (`apply_hydrodynamic_forces`):
- Damping `D(ν)ν = (D_lin + D_quad·|ν|) ν`. It's effectively diagonal: the off-diagonal |v| couplings in the code get multiplied by a diagonal matrix and vanish.
- **Added mass is not a true mass-matrix term.** Force = −M_A · a, where a is a numerically differentiated, low-pass-filtered body acceleration (α = 0.3). Expect it to behave a little differently from the ideal M_A ν̇ in your model.
- Coriolis from added mass only: `C_A(ν)ν`, built from cross products of `M_A ν`.
- Optional flow (current) velocity is subtracted from ν first (zero by default).
- `drag_coef: 0.3` in the yaml isn't used by this path.

### 7.4 Thruster model (T200, `actuators/t200.py`)

For each thruster, every step (dt = 0.016 s):
1. `throttle ← throttle + 0.43·(clamp(cmd, −1, 1) − throttle)`: first-order lag, τ ≈ 28 ms.
2. Target RPM with a deadband of |throttle| ≤ 0.075:
   - throttle > 0.075: `rpm* = 3659.9·thr + 345.21`
   - throttle < −0.075: `rpm* = 3494.4·thr − 433.50`
3. `rpm ← α·rpm + (1−α)·rpm*`, with α = exp(−dt / 0.01), then clamp ±3900.
4. Thrust in N: `T = (KF / 4.4e-7) · 9.81 · p(rpm)`, KF = 0.8e-7, so the scale is 1.784:
   - rpm > 0: `p = 4.7368e-7·rpm² − 1.9275e-4·rpm + 0.084452`
   - rpm ≤ 0: `p = −3.8442e-7·rpm² − 1.6186e-4·rpm − 0.039139`
5. **Max thrust ≈ +11.7 N forward and ≈ −9.4 N reverse.** At zero RPM the reverse branch gives a small **≈ −0.07 N bias**.
6. Rotor reaction moments are **zero** (`moments = thrust · −dir · 0`).
7. **Thrust acts along each rotor prim's local +X axis**, applied at the rotor position (`apply_forces_and_torques_at_pos(is_global=False)`). It is *not* along local +Z. Assuming +Z was the cause of the early teleop axis mix-up.

The applied thrust per thruster can be read live as `drone.thrusts[..., i, 0]`; `teleop.py` does this.

### 7.5 Thruster layout and allocation

From the probe run (`analyze_probe.py`, amp 0.3, body-frame accelerations minus idle):

| ch | du/dt | dv/dt | dw/dt | dp/dt | dq/dt | dr/dt |
|---|---|---|---|---|---|---|
| 0 | +0.022 | +0.012 | −0.001 | +0.081 | −0.005 | +0.112 |
| 1 | +0.015 | −0.013 | +0.002 | +0.004 | −0.095 | −0.145 |
| 2 | −0.027 | +0.018 | +0.004 | −0.037 | −0.020 | −0.100 |
| 3 | −0.014 | −0.014 | +0.003 | +0.034 | +0.075 | +0.155 |
| 4 | +0.011 | −0.001 | +0.026 | −0.093 | −0.019 | −0.019 |
| 5 | +0.007 | +0.007 | +0.019 | −0.077 | −0.150 | −0.000 |
| 6 | −0.003 | −0.006 | +0.016 | +0.073 | −0.049 | +0.000 |
| 7 | −0.000 | +0.008 | +0.016 | −0.082 | +0.127 | +0.000 |

> [!warning] Magnitudes are not usable
> Only the signs are reliable. The vehicle reaches drag-limited speed in a fraction of a 2 s pulse, so average acceleration is far too small. The roll/pitch entries for the verticals are noisy.

**Thrust direction of positive throttle** (body FLU; `PROBE_DIRS` in `teleop.py`):

| ch | Type | +thrust pushes | Direction |
|---|---|---|---|
| 0 | horizontal, 45° vectored | forward + left | (+1, +1, 0)/√2 |
| 1 | horizontal, 45° vectored | forward + right | (+1, −1, 0)/√2 |
| 2 | horizontal, 45° vectored | back + left | (−1, +1, 0)/√2 |
| 3 | horizontal, 45° vectored | back + right | (−1, −1, 0)/√2 |
| 4–7 | vertical | up | (0, 0, +1) |

**Positions relative to `base_link`** (inferred from the first teleop B-matrix printout, probe yaw signs and the standard Heavy layout):

| ch | Position | Notes |
|---|---|---|
| 0 | front-right | x ≈ +0.14, y ≈ −0.10 |
| 1 | front-left | x ≈ +0.14, y ≈ +0.10 |
| 2 | back-right | x ≈ −0.15, y ≈ −0.10 |
| 3 | back-left | x ≈ −0.15, y ≈ +0.10 |
| 4 | right (likely front-right) | y ≈ −0.22, z ≈ −0.01 |
| 5 | left (likely front-left) | y ≈ +0.22, z ≈ −0.01 |
| 6 | right (likely back-right) | y ≈ −0.22, z ≈ 0 |
| 7 | left (likely back-left) | y ≈ +0.22, z ≈ 0 |

> [!todo] Confirm vertical-thruster x positions and all z heights
> `teleop.py` now prints the exact rotor positions from the USD, plus a front/back, left/right, top/bottom label per thruster, at startup and in the key window. Copy the exact numbers here. The probe pitch signs (ch5 −, ch7 +) support ch5 front and ch7 back.

Consistency check, yaw torque from horizontals: τz = x·dy − y·dx. For ch0 that's (0.14·0.707 − (−0.10)·0.707) ≈ **+0.17 N·m per N**, matching the probe's + sign. The signs for ch0–3 are + − − +, all matching the probe.

**Allocation used in teleop:**
- `B[:, i] = [d_i ; r_i × d_i]` per 1 N, in body FLU about the `base_link` origin.
- r_i comes from the USD rotor prims and d_i from the rotor's local +X.
- If an axis disagrees with `PROBE_DIRS` (dot product < 0.7), the probe direction is used instead, and the startup "allocation:" line says which.
- Six axes are controllable: surge, sway, heave, roll, pitch, yaw.
- The allocator in teleop is a row-normalized pseudo-inverse, scaled so the busiest thruster's throttle equals the largest requested axis command. **For the MPC**, map wrench → thrust in N with B, then invert the T200 curve to get throttle (or put the T200 steady-state curve in the model).

---

## 8. MPC: where things stand and next steps

- **Template:** `MPCController.__call__(state, ref, t) → u` (8 throttles in [−1, 1]) in `mpc_run.py`. `state` comes from `read_state` (FLU), and `ref["pos_w"]` from `+mpc.target`.
- **Plant model to use** (matches the sim): 6-DOF Fossen with M = M_RB + M_A, C_RB + C_A, D_lin + D_quad|ν|, the hydrostatics above, τ = B · T(throttle) with the T200 curve, and a first-order thruster lag of about 28 ms.
- **Watch the frame convention:** Isaac is FLU, while Fossen texts and MarineGym's internals use FRD.
- **System ID data:** `teleop_log.npz` has state, commanded throttles, applied thrust in N, and the applied wrench in body and world frames.
- **Ideas not done yet:**
  - Fill in the CasADi/acados MPC.
  - Add a `follow` option that moves the ghost marker to the reference pose.
  - Add a tonemap/exposure setting if dimming the lights isn't enough.
  - Prototype the MPC against a pure-Python Fossen model before running it in Isaac.

---

## 9. Known harmless messages

- `PhysX error: PxArticulationLink::setGlobalPose(): it is illegal ... eENABLE_DIRECT_GPU_API`: appears on reset. Harmless if the vehicle starts level.
- `Physics scenes stepping is not the same ...`: harmless.
- `Mesh ... has corrupted data in primvar 'st_*'`: the BlueROV mesh UVs (cosmetic).
- `functorch.vmap is deprecated`: harmless with torch 2.3.1.
- Long first launch (2+ min): shader compilation. Later launches are faster.
