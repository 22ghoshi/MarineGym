"""
teleop.py - drive a MarineGym vehicle (e.g. BlueROVHeavy) from the keyboard.

Put this next to mpc_run.py in MarineGym/scripts/ (it reuses its helpers) and run:

    python teleop.py task=Hover task.drone_model.name=BlueROVHeavy headless=false \
        task.env.num_envs=1 task.env.max_episode_length=1000000 \
        "viewer.resolution=[1280,720]" +teleop.lowgpu=false

A separate "ROV teleop" window opens (teleop_window.py, same folder). Keep THAT
window focused while driving - keys typed there never reach the Isaac GUI, so
none of its shortcuts (W/E/R tools, F frame, ...) interfere. Closing it quits.

Keys (body frame, FLU: x forward, y left, z up)
    W / S   surge forward / back         Q / E   yaw left / right
    A / D   sway left / right            I / K   pitch nose down / up
    R / V   heave up / down              J / L   roll left / right
    = / -   gain up / down               T       toggle auto-levelling
    SPACE   hold = all thrusters off     Ctrl+C  quit (log is saved)

Options (pass with a leading '+'):
    +teleop.input=window|isaac  separate key window, or Isaac viewport keys (default window)
    +teleop.gain=0.5            initial command gain, 0..1        (default 0.5)
    +teleop.stabilize=true      auto-level roll/pitch when not commanded (default true)
    +teleop.lowgpu=true|false   strip RTX features to save VRAM    (default true)
    +teleop.fog=true|false      underwater fog, see WATER_FOG in mpc_run.py (default true)
    +teleop.hide_target=true    hide Hover's ghost goal marker     (default true)
    +teleop.light=0.1           scale the world's lights, 1 = as shipped (default 0.1)
    +teleop.upright=true        level the vehicle after reset      (default true)
    +teleop.render_every=2      render 1 of every N physics steps  (default 2)
    +teleop.realtime=true       pace the loop to wall-clock        (default true)
    +teleop.log=PATH            .npz log of state/commands         (default teleop_log.npz)

The key window shows each thruster's location, throttle command (-1..1) and the thrust
actually applied [N] (after the motor lag; T200 model maxes out near +11.7 / -9.4 N), plus
the resulting wrench in body and world frames. The log has the same (thrust_N, wrench_b,
wrench_w) next to the commands, which is what you want for fitting the MPC model.
"""

import os
import signal
import subprocess
import sys
import threading
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from marinegym import init_simulation_app
import marinegym.learning  # noqa: F401  registers algo configs with Hydra

from mpc_run import (
    apply_low_gpu_render_settings,
    apply_water_fog,
    dim_scene_lights,
    hide_target_marker,
    read_state,
)

AXES = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# key name -> (axis index, sign).  FLU body frame, right-handed rotations.
KEYMAP = {
    "W": (0, +1), "S": (0, -1),   # surge
    "A": (1, +1), "D": (1, -1),   # sway  (+y is left)
    "R": (2, +1), "V": (2, -1),   # heave (+z is up). Not F: F is Isaac's "frame selection" hotkey
    "L": (3, +1), "J": (3, -1),   # roll  (+x rotation = right side down)
    "I": (4, +1), "K": (4, -1),   # pitch (+y rotation = nose down)
    "Q": (5, +1), "E": (5, -1),   # yaw   (+z rotation = turn left)
}

# Signs measured with the probe controller (analyze_probe.py), used to sanity-check
# / fix the per-thruster direction read from the USD, and as a fallback mixer.
PROBE_SIGNS = {  # channel -> {axis: sign}
    0: {0: +1, 1: +1}, 1: {0: +1, 1: -1}, 2: {0: -1, 1: +1}, 3: {0: -1, 1: -1},
    4: {2: +1}, 5: {2: +1}, 6: {2: +1}, 7: {2: +1},
}


# --------------------------------------------------------------------------
# Keyboard
# --------------------------------------------------------------------------
class Keyboard:
    def __init__(self):
        import carb.input
        import omni.appwindow

        self._carb_input = carb.input
        self._input = carb.input.acquire_input_interface()
        self._kb = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._kb, self._on_event)
        self.held = set()
        self.pressed = []  # one-shot presses, drained each step

    def _on_event(self, event, *args, **kwargs):
        name = str(event.input).split(".")[-1]
        t = self._carb_input.KeyboardEventType
        if event.type == t.KEY_PRESS:
            self.held.add(name)
            self.pressed.append(name)
        elif event.type == t.KEY_RELEASE:
            self.held.discard(name)
        return True

    closed = False

    def drain(self):
        p, self.pressed = self.pressed, []
        return p

    def held_keys(self):
        return set(self.held)

    def status(self, text):
        pass

    def close(self):
        try:
            self._input.unsubscribe_to_keyboard_events(self._kb, self._sub)
        except Exception:  # noqa: BLE001
            pass


class WindowKeyboard:
    """Keys from a separate Tk window (teleop_window.py) running in its own process."""

    def __init__(self, timeout=10.0):
        path = os.path.join(SCRIPT_DIR, "teleop_window.py")
        if not os.path.exists(path):
            raise RuntimeError(f"{path} not found")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self._lock = threading.Lock()
        self._held, self._pressed = set(), []
        self.closed = False
        ready = threading.Event()

        def reader():
            for line in self.proc.stdout:
                parts = line.split()
                if not parts:
                    continue
                kind = parts[0]
                if kind == "READY":
                    ready.set()
                elif kind == "P" and len(parts) > 1:
                    with self._lock:
                        self._held.add(parts[1])
                        self._pressed.append(parts[1])
                elif kind == "R" and len(parts) > 1:
                    with self._lock:
                        self._held.discard(parts[1])
                elif kind == "QUIT":
                    self.closed = True
            self.closed = True      # window process ended
            ready.set()

        threading.Thread(target=reader, daemon=True).start()
        if not ready.wait(timeout) or self.proc.poll() is not None:
            self.close()
            raise RuntimeError("key window did not start (is tkinter installed? `conda install tk`)")

    def drain(self):
        with self._lock:
            p, self._pressed = self._pressed, []
        return p

    def held_keys(self):
        with self._lock:
            return set(self._held)

    def status(self, text):
        try:
            self.proc.stdin.write("S " + text.replace("\n", "|") + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            self.closed = True

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            self.proc.terminate()


# --------------------------------------------------------------------------
# Thruster allocation
# --------------------------------------------------------------------------
# Thrust directions (body FLU) per channel, from the probe run. MarineGym applies each
# rotor's thrust along the rotor prim's local +X axis (underwaterVehicle.apply_action),
# so that axis is read from the USD and checked against these probe directions.
PROBE_DIRS = {
    0: (+1, +1, 0), 1: (+1, -1, 0), 2: (-1, +1, 0), 3: (-1, -1, 0),   # vectored, 45 deg
    4: (0, 0, 1), 5: (0, 0, 1), 6: (0, 0, 1), 7: (0, 0, 1),            # vertical
}


def thruster_matrix_from_usd(drone_name):
    """
    Build B (6 x n): column i = [d_i ; r_i x d_i] in the base_link frame, where r_i is
    the rotor position and d_i its thrust axis (local +X), both read from the USD.
    If an axis disagrees with PROBE_DIRS, the probe direction is used instead.
    Unit thrust per column (1 N) - so B @ thrust_N = body wrench [N, N m].
    Returns (B, positions, note).
    """
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = f"/World/envs/env_0/{drone_name}_0"
    base = stage.GetPrimAtPath(f"{root}/base_link")
    if not base.IsValid():
        return None, None, ""
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    base_inv = cache.GetLocalToWorldTransform(base).GetInverse()

    cols, positions, overridden, i = [], [], [], 0
    while True:
        rotor = stage.GetPrimAtPath(f"{root}/rotor_{i}")
        if not rotor.IsValid() or i not in PROBE_DIRS:
            break
        rel = cache.GetLocalToWorldTransform(rotor) * base_inv  # Gf: row-vector convention
        r = np.array(rel.ExtractTranslation(), dtype=float)
        probe = np.array(PROBE_DIRS[i], dtype=float)
        probe /= np.linalg.norm(probe)
        d = np.array(rel.TransformDir(Gf.Vec3d(1, 0, 0)), dtype=float)
        d /= max(np.linalg.norm(d), 1e-9)
        if d @ probe < 0.7:            # USD axis doesn't match measured behaviour
            d = probe
            overridden.append(i)
        cols.append(np.concatenate([d, np.cross(r, d)]))
        positions.append(r)
        i += 1
    if not cols:
        return None, None, ""
    note = "USD rotor +X axes" + (f" (probe dirs used for {overridden})" if overridden else "")
    return np.array(cols).T, np.array(positions), note


def check_against_probe(B):
    """Flip any column whose dominant direction disagrees with the probe measurements."""
    flipped = []
    for ch, expected in PROBE_SIGNS.items():
        if ch >= B.shape[1]:
            continue
        score = sum(sign * np.sign(B[axis, ch]) for axis, sign in expected.items())
        if score < 0:
            B[:, ch] *= -1
            flipped.append(ch)
    return B, flipped


def fallback_matrix(n):
    """Probe-sign mixer: surge/sway/yaw from the 4 vectored thrusters, heave from the rest."""
    B = np.zeros((6, n))
    yaw = {0: +1, 1: -1, 2: -1, 3: +1}
    for ch, expected in PROBE_SIGNS.items():
        if ch < n:
            for axis, sign in expected.items():
                B[axis, ch] = sign
            if ch in yaw:
                B[5, ch] = yaw[ch]
    return B


class Allocator:
    """Normalized wrench command (6,) in [-1,1] -> thruster commands (n,) in [-1,1]."""

    def __init__(self, B):
        self.B = B
        scale = np.abs(B).max(axis=1)
        self.controllable = scale > 1e-6
        scale[~self.controllable] = 1.0
        self.Bn = B / scale[:, None]          # each axis' full authority ~ 1
        self.pinv = np.linalg.pinv(self.Bn)

    def __call__(self, wrench):
        u = self.pinv @ wrench
        m = np.abs(u).max()
        if m < 1e-9:
            return u
        # Scale so the busiest thruster runs at the largest requested axis magnitude:
        # a single key at gain g drives the axis at fraction g of its full authority.
        return u * (np.abs(wrench).max() / m)


def roll_pitch(q_wxyz):
    roll, pitch, _ = rpy(q_wxyz)
    return roll, pitch


def rpy(q_wxyz):
    w, x, y, z = q_wxyz
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def qrot(q_wxyz, v):
    """Rotate vector(s) v (..., 3) by quaternion (w, x, y, z)."""
    w, qv = q_wxyz[0], np.asarray(q_wxyz[1:], dtype=float)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def qrot_inv(q_wxyz, v):
    q = np.asarray(q_wxyz, dtype=float).copy()
    q[1:] *= -1
    return qrot(q, v)


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, float)


def thruster_labels(positions, B):
    """Per thruster: 'front-left-top' style location + what positive thrust pushes."""
    n = B.shape[1]
    if positions is None or len(positions) != n:
        return ["?"] * n, ["?"] * n
    c = positions.mean(axis=0)            # labels relative to the thruster cluster centre
    words = [("front", "back"), ("left", "right"), ("top", "bottom")]
    locs = []
    for r in positions - c:
        parts = [(pos if v > 0 else neg) if abs(v) > 0.02 else "mid"
                 for v, (pos, neg) in zip(r, words)]
        locs.append("-".join(parts))
    pushes = []
    for d in B[:3].T:
        parts = [(pos if v > 0 else neg) for v, (pos, neg) in
                 zip(d, [("fwd", "back"), ("left", "right"), ("up", "down")]) if abs(v) > 0.3]
        pushes.append("+".join(parts) or "?")
    return locs, pushes


def applied_thrust(drone, n):
    """
    What the simulator actually applied this step (after the T200 motor lag):
    per-thruster thrust [N] and the resulting wrench about the base_link origin,
    in both the body (FLU) and world frames: (thrust_N, wrench_b, wrench_w) or None.
    """
    try:
        f = _np(drone.thrusts).reshape(-1, n, 3)[0, :, 0]
        rp, rq = drone.rotors_view.get_world_poses()
        rp, rq = _np(rp).reshape(-1, n, 3)[0], _np(rq).reshape(-1, n, 4)[0]
        bp, bq = drone.get_world_poses()
        bp, bq = _np(bp).reshape(-1, 3)[0], _np(bq).reshape(-1, 4)[0]
    except Exception:  # noqa: BLE001
        return None
    axes = np.stack([qrot(q, np.array([1.0, 0.0, 0.0])) for q in rq])  # thrust axis, world
    F = f[:, None] * axes
    force_w, torque_w = F.sum(0), np.cross(rp - bp, F).sum(0)
    wrench_w = np.concatenate([force_w, torque_w])
    wrench_b = np.concatenate([qrot_inv(bq, force_w), qrot_inv(bq, torque_w)])
    return f, wrench_b, wrench_w


def format_status(gain, stabilize, t, state, u, applied, locs, pushes):
    pos, vb = state["pos_w"], state["vel_b"]
    r, p, y = np.degrees(rpy(state["quat"]))
    lines = [
        f"gain {gain:.1f}   auto-level {'ON' if stabilize else 'OFF'}   t {t:7.1f} s",
        f"pos   x {pos[0]:+7.2f}   y {pos[1]:+7.2f}   z {pos[2]:+7.2f}  m   (world)",
        f"vel   u {vb[0]:+7.2f}   v {vb[1]:+7.2f}   w {vb[2]:+7.2f}  m/s (body)",
        f"att   roll {r:+6.1f}   pitch {p:+6.1f}   yaw {y:+6.1f}  deg",
        "",
        " #  location              +thrust pushes   cmd    thrust",
    ]
    f = applied[0] if applied is not None else [float("nan")] * len(u)
    for i in range(len(u)):
        lines.append(f" {i}  {locs[i]:<20}  {pushes[i]:<15} {u[i]:+5.2f}  {f[i]:+7.2f} N")
    lines += ["", "wrench       Fx      Fy      Fz   [N]      Tx      Ty      Tz   [N m]"]
    if applied is not None:
        for name, w in (("body ", applied[1]), ("world", applied[2])):
            lines.append(f"{name}  {w[0]:+7.2f} {w[1]:+7.2f} {w[2]:+7.2f}       "
                         f"{w[3]:+7.2f} {w[4]:+7.2f} {w[5]:+7.2f}")
        lines.append("body: x fwd y left z up (Tx roll, Ty pitch, Tz yaw);  world: Isaac x y z")
    else:
        lines.append("(applied thrust not readable from this vehicle)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    tcfg = cfg.get("teleop", {}) or {}
    gain = float(tcfg.get("gain", 0.5))
    stabilize = bool(tcfg.get("stabilize", True))
    render_every = max(1, int(tcfg.get("render_every", 2)))
    realtime = bool(tcfg.get("realtime", True))
    upright = bool(tcfg.get("upright", True))
    log_path = str(tcfg.get("log", "teleop_log.npz"))

    if cfg.headless:
        raise SystemExit("teleop needs a window: run with headless=false")

    simulation_app = init_simulation_app(cfg)
    if bool(tcfg.get("lowgpu", True)):
        apply_low_gpu_render_settings()
    if bool(tcfg.get("fog", True)):
        apply_water_fog(verbose=False)

    stop = {"flag": False}

    def _on_sigint(signum, frame):
        if stop["flag"]:
            raise KeyboardInterrupt
        print("\nCtrl+C received - saving log and shutting down...")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    from marinegym.envs.isaac_env import IsaacEnv
    from torchrl.envs.utils import step_mdp

    env = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=cfg.headless)
    env.eval()
    drone = env.drone
    if bool(tcfg.get("hide_target", True)):
        hide_target_marker()
    dim_scene_lights(float(tcfg.get("light", 0.1)))

    try:
        action_spec = env.full_action_spec[("agents", "action")]
    except (KeyError, AttributeError, IndexError):
        action_spec = env.action_spec
    act_shape = tuple(action_spec.shape)
    n_act = act_shape[-1]
    dt = float(cfg.sim.dt) * int(cfg.sim.substeps)

    def make_upright():
        try:
            pos, _ = drone.get_world_poses()
            q = torch.zeros(pos.shape[:-1] + (4,), device=pos.device)
            q[..., 0] = 1.0
            drone.set_world_poses(pos, q)
            drone.set_velocities(torch.zeros(pos.shape[:-1] + (6,), device=pos.device))
        except Exception as e:  # noqa: BLE001
            print(f"[teleop] could not reset to upright: {e}")

    td = env.reset()
    if upright:
        make_upright()

    # ---- thruster allocation from the vehicle's own geometry ----
    np.set_printoptions(precision=2, suppress=True)
    B, positions, note = thruster_matrix_from_usd(cfg.task.drone_model.name)
    if B is not None and B.shape[1] == n_act:
        source = note
        print("rotor positions in base frame (x fwd, y left, z up) [m]:")
        print(positions)
    else:
        B, source, positions = fallback_matrix(n_act), "probe-sign fallback (no roll/pitch)", None
    alloc = Allocator(B)
    locs, pushes = thruster_labels(positions, B)
    np.save("thruster_B.npy", B)

    print("\n========== teleop ==========")
    print(f"vehicle   : {cfg.task.drone_model.name}   thrusters: {n_act}   dt: {dt} s")
    print(f"allocation: {source}   (saved to thruster_B.npy)")
    print("B (rows surge,sway,heave,roll,pitch,yaw; unit thrust per column):")
    print(B)
    print("thrusters:")
    for i in range(n_act):
        print(f"  {i}: {locs[i]:<20} +thrust pushes {pushes[i]}")
    print("controllable axes:", [a for a, c in zip(AXES, alloc.controllable) if c])
    print("keys: WS surge  AD sway  RV heave  QE yaw  IK pitch  JL roll  "
          "=/- gain  T level  SPACE stop")
    print("============================\n")

    kb = None
    if str(tcfg.get("input", "window")) == "window":
        try:
            kb = WindowKeyboard()
            print("[teleop] key window open - click it and keep it focused while driving")
        except Exception as e:  # noqa: BLE001
            print(f"[teleop] {e}; falling back to Isaac viewport keys")
    if kb is None:
        kb = Keyboard()
        print("[teleop] using Isaac viewport keys - click the viewport to focus it")

    log = {k: [] for k in ["t", "pos_w", "quat", "vel_b", "omega_b", "wrench", "action",
                           "thrust_N", "wrench_b", "wrench_w"]}
    step, t = 0, 0.0
    status_every = max(1, int(round(0.25 / dt)))

    try:
        while simulation_app.is_running() and not stop["flag"] and not kb.closed:
            wall_start = time.perf_counter()

            for key in kb.drain():
                if key in ("EQUAL", "NUMPAD_ADD"):
                    gain = min(1.0, gain + 0.1)
                    print(f"[teleop] gain {gain:.1f}")
                elif key in ("MINUS", "NUMPAD_SUBTRACT"):
                    gain = max(0.1, gain - 0.1)
                    print(f"[teleop] gain {gain:.1f}")
                elif key == "T":
                    stabilize = not stabilize
                    print(f"[teleop] auto-levelling {'ON' if stabilize else 'OFF'}")

            state = read_state(drone)
            held = kb.held_keys()

            wrench = np.zeros(6)
            for key in held:
                if key in KEYMAP:
                    axis, sign = KEYMAP[key]
                    wrench[axis] += sign
            wrench = np.clip(wrench, -1, 1) * gain

            if stabilize and alloc.controllable[3] and alloc.controllable[4]:
                roll, pitch = roll_pitch(state["quat"])
                p, q = state["omega_b"][0], state["omega_b"][1]
                if wrench[3] == 0:
                    wrench[3] = np.clip(-2.0 * roll - 0.5 * p, -1, 1)
                if wrench[4] == 0:
                    wrench[4] = np.clip(-2.0 * pitch - 0.5 * q, -1, 1)

            u = np.zeros(n_act) if "SPACE" in held else alloc(wrench)

            td["agents", "action"] = torch.as_tensor(
                u, dtype=torch.float32, device=env.device).expand(act_shape).clone()
            if hasattr(env, "enable_render"):
                env.enable_render(step % render_every == 0)

            td = env.step(td)
            applied = applied_thrust(drone, n_act)   # read before a possible reset zeroes it
            if step % status_every == 0:
                kb.status(format_status(gain, stabilize, t, state, u, applied, locs, pushes))
            done = td["next", "done"].any().item()
            td = step_mdp(td)
            if done:
                td = env.reset()
                if upright:
                    make_upright()

            for k in ["pos_w", "quat", "vel_b", "omega_b"]:
                log[k].append(state[k])
            log["t"].append(t)
            log["wrench"].append(wrench)
            log["action"].append(u)
            nan = np.full(6, np.nan)
            log["thrust_N"].append(applied[0] if applied else np.full(n_act, np.nan))
            log["wrench_b"].append(applied[1] if applied else nan)
            log["wrench_w"].append(applied[2] if applied else nan)

            if step % int(round(1.0 / dt)) == 0:
                pos, vb = state["pos_w"], state["vel_b"]
                print(f"t={t:7.2f}s  pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})  "
                      f"vel_b=({vb[0]:+.2f},{vb[1]:+.2f},{vb[2]:+.2f})  gain={gain:.1f}  "
                      f"u={np.array2string(u, precision=2)}")

            step += 1
            t += dt
            if realtime:
                sleep = dt - (time.perf_counter() - wall_start)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        kb.close()
        if log["t"]:
            np.savez(log_path, **{k: np.asarray(v) for k, v in log.items()})
            print(f"Saved {len(log['t'])} steps to {log_path}")
        simulation_app.close()


if __name__ == "__main__":
    main()