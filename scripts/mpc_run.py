"""
mpc_run.py - run a single MarineGym vehicle (e.g. BlueROVHeavy) under a
hand-written controller (MPC, PID, probe, ...) instead of RL training.

Put this file in MarineGym/scripts/ (next to train.py / train.yaml) and run e.g.:

    python mpc_run.py task=Hover task.drone_model.name=BlueROVHeavy headless=false \
        task.env.num_envs=1 task.env.max_episode_length=1000000 \
        "viewer.resolution=[1280,720]" \
        +mpc.controller=probe +mpc.render_every=2 +mpc.realtime=true

Extra options (all optional, pass with a leading '+'):
    +mpc.controller=zero|probe|mpc   which controller to run       (default: probe)
    +mpc.render_every=N              render 1 of every N steps     (default: 2)
    +mpc.realtime=true|false         pace the loop to wall-clock   (default: true)
    +mpc.duration=SECONDS            stop after this much sim time (default: run until window closed)
    +mpc.target=[x,y,z]              position reference; default = initial position
    +mpc.log=PATH                    where to save the .npz log    (default: mpc_log.npz)
    +mpc.upright=true|false          level the vehicle after reset (default: true)
    +mpc.lowgpu=true|false           strip RTX features to save VRAM (default: true;
                                     false = full lighting/translucency, nicer but heavier)
    +mpc.fog=true|false              underwater distance fog (default: true; edit WATER_FOG to tune)
    +mpc.hide_target=true|false      hide Hover's ghost goal-marker vehicle (default: true)
    +mpc.light=0.1                   scale the world's lights; 1 = as shipped (default: 0.1,
                                     the shipped light over-exposes the sand to white)

Stop with Ctrl+C (once) or by closing the window; the log is saved either way.
"""

import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from marinegym import init_simulation_app
import marinegym.learning  # noqa: F401  registers the 'algo' configs (algo/ppo) with Hydra's ConfigStore


# --------------------------------------------------------------------------
# Low-GPU render settings (applied after the SimulationApp exists)
# --------------------------------------------------------------------------
def apply_low_gpu_render_settings():
    import carb

    s = carb.settings.get_settings()
    off = [
        "/rtx/translucency/enabled",          # this is what OOM'd before (water/glass)
        "/rtx/reflections/enabled",
        "/rtx/indirectDiffuse/enabled",
        "/rtx/ambientOcclusion/enabled",
        "/rtx/directLighting/sampledLighting/enabled",
        "/rtx/raytracing/subsurface/enabled",
        "/rtx/caustics/enabled",
    ]
    for key in off:
        s.set(key, False)
    s.set("/rtx/post/dlss/execMode", 0)        # DLSS "performance"
    s.set("/rtx/rendermode", "RaytracedLighting")  # real-time mode, never path tracing


# Underwater look: distance fog (values from tuning in Render Settings > Simple Fog).
WATER_FOG = {
    "/rtx/fog/enabled": True,
    "/rtx/fog/fogColor": [0.02, 0.22, 0.28],
    "/rtx/fog/fogColorIntensity": 0.8,
    "/rtx/fog/fogStartDist": 0.5,
    "/rtx/fog/fogEndDist": 20.0,
    "/rtx/fog/fogDistanceDensity": 0.6,
    "/rtx/fog/fogZup/enabled": False,       # height-based fog off
}


def apply_water_fog(verbose=True):
    """Apply WATER_FOG. Warns about any key this Kit version doesn't have."""
    import carb

    s = carb.settings.get_settings()
    for key, val in WATER_FOG.items():
        if s.get(key) is None:
            print(f"[mpc_run] fog setting not found in this Isaac version: {key}")
        s.set(key, val)
    if verbose:
        print(f"[mpc_run] fog settings now: {s.get('/rtx/fog')}")


def dim_scene_lights(scale=0.1, verbose=True):
    """
    The EmptyMarine world is lit by a 100 x 100 m RectLight at intensity 15000, 7.5 m
    above the origin. At Isaac's default camera exposure that is several stops too
    bright: the sand clips to white, and fog can't fix it because fog is blended in
    before tone mapping (fog-tinted white is still white). Scale every light outside
    /World/envs by `scale`, via its exposure (exposure += log2(scale); idempotent).
    """
    import math

    import omni.usd

    if scale is None or scale <= 0 or abs(scale - 1.0) < 1e-6:
        return
    stage = omni.usd.get_context().get_stage()
    ev = math.log2(scale)
    changed = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/envs/") or not prim.GetTypeName().endswith("Light"):
            continue
        attr = prim.GetAttribute("inputs:exposure")
        if not attr or not attr.IsValid():
            attr = prim.GetAttribute("exposure")
        if not attr or not attr.IsValid():
            continue
        base = _LIGHT_EXPOSURE_0.setdefault(path, float(attr.Get() or 0.0))
        attr.Set(base + ev)
        changed.append(path)
    if verbose:
        print(f"[mpc_run] light scale {scale:g} ({ev:+.1f} EV) on: {changed or 'no lights found'}")


_LIGHT_EXPOSURE_0 = {}


def hide_target_marker():
    """Hover spawns a second, visual-only vehicle at /World/envs/env_*/target as its goal marker. Hide it."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    hidden = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.GetName() == "target" and path.startswith("/World/envs/"):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
    print(f"[mpc_run] hid {hidden} target marker(s)")


# --------------------------------------------------------------------------
# State extraction - a fixed, documented layout for your MPC
# --------------------------------------------------------------------------
def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector v into the body frame of quaternion q (w, x, y, z)."""
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])
    return R.T @ v


def read_state(drone) -> dict:
    """
    Returns a dict for env 0 / agent 0:
        pos_w   (3,)  world position
        quat    (4,)  orientation, (w, x, y, z)  - Isaac convention
        vel_w   (3,)  world-frame linear velocity
        omega_w (3,)  world-frame angular velocity
        vel_b   (3,)  body-frame linear velocity   (u, v, w)  - Fossen nu_1
        omega_b (3,)  body-frame angular velocity  (p, q, r)  - Fossen nu_2
    """
    pos, quat = drone.get_world_poses()        # (N, 1, 3), (N, 1, 4)
    vel = drone.get_velocities()               # (N, 1, 6): lin (world), ang (world)
    pos = pos[0, 0].detach().cpu().numpy().astype(np.float64)
    quat = quat[0, 0].detach().cpu().numpy().astype(np.float64)
    vel = vel[0, 0].detach().cpu().numpy().astype(np.float64)
    vel_w, omega_w = vel[:3], vel[3:]
    return {
        "pos_w": pos,
        "quat": quat,
        "vel_w": vel_w,
        "omega_w": omega_w,
        "vel_b": quat_rotate_inverse(quat, vel_w),
        "omega_b": quat_rotate_inverse(quat, omega_w),
    }


# --------------------------------------------------------------------------
# Controllers - all return a numpy action of shape (n_act,) in the env's
# action units (MarineGym rotor/fin commands, typically normalized to [-1, 1])
# --------------------------------------------------------------------------
class ZeroController:
    """All commands zero - shows passive dynamics (buoyancy, damping)."""

    def __init__(self, n_act, dt, **_):
        self.n_act = n_act

    def __call__(self, state, ref, t):
        return np.zeros(self.n_act)


class ProbeController:
    """
    Pulses each action channel in turn (+amp for `on` s, then rest for `off` s).
    Watch the viewer and the log to learn which channel drives which thruster
    and in which direction - i.e. the thruster allocation your MPC needs.
    """

    def __init__(self, n_act, dt, amp=0.3, on=2.0, off=2.0, **_):
        self.n_act, self.amp, self.on, self.period = n_act, amp, on, on + off

    def __call__(self, state, ref, t):
        u = np.zeros(self.n_act)
        k = int(t // self.period) % self.n_act
        if (t % self.period) < self.on:
            u[k] = self.amp
        return u


class MPCController:
    """
    Put your MPC here. Suggested structure (CasADi / acados, CPU only):

      __init__: build the 6-DOF Fossen model from the SAME parameters the sim
                uses (marinegym/robots/assets/usd/BlueROVHeavy/BlueROVHeavy.yaml:
                mass, inertia, added mass, linear/quadratic damping, buoyancy,
                thruster positions/directions), plus the thrust map from
                marinegym/actuators/t200.py so the optimizer outputs thrusts and
                you convert them to normalized commands at the end.

      __call__: x = [pos_w, quat, vel_b, omega_b] from `state`, solve, return
                the first control move as normalized commands in [-1, 1].
    """

    def __init__(self, n_act, dt, **_):
        self.n_act, self.dt = n_act, dt
        raise NotImplementedError("Fill in MPCController before using +mpc.controller=mpc")

    def __call__(self, state, ref, t):
        raise NotImplementedError


CONTROLLERS = {"zero": ZeroController, "probe": ProbeController, "mpc": MPCController}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    mcfg = cfg.get("mpc", {}) or {}
    controller_name = str(mcfg.get("controller", "probe"))
    render_every = max(1, int(mcfg.get("render_every", 2)))
    realtime = bool(mcfg.get("realtime", True))
    duration = mcfg.get("duration", None)
    log_path = str(mcfg.get("log", "mpc_log.npz"))

    simulation_app = init_simulation_app(cfg)
    if bool(mcfg.get("lowgpu", True)):
        apply_low_gpu_render_settings()
    if bool(mcfg.get("fog", True)) and not cfg.headless:
        apply_water_fog()

    # Isaac's own SIGINT handler kills the process before `finally` runs, so the
    # log never gets saved. Replace it with one that just asks the loop to stop.
    import signal
    stop = {"flag": False}

    def _on_sigint(signum, frame):
        if stop["flag"]:          # second Ctrl+C: give up and exit hard
            raise KeyboardInterrupt
        print("\nCtrl+C received - finishing step, saving log, shutting down...")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    from marinegym.envs.isaac_env import IsaacEnv
    from torchrl.envs.utils import step_mdp

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    env = env_class(cfg, headless=cfg.headless)
    env.eval()
    drone = env.drone
    if bool(mcfg.get("hide_target", True)):
        hide_target_marker()
    if not cfg.headless:
        dim_scene_lights(float(mcfg.get("light", 0.1)))

    dt = float(cfg.sim.dt) * int(cfg.sim.substeps)
    # torchrl returns the leaf spec directly from .action_spec when there is a
    # single action key; the composite (keyed) version lives in .full_action_spec.
    try:
        action_spec = env.full_action_spec[("agents", "action")]
    except (KeyError, AttributeError, IndexError):
        action_spec = env.action_spec
    act_shape = tuple(action_spec.shape)          # (num_envs, n_agents, n_act)
    n_act = act_shape[-1]

    low = high = None
    space = getattr(action_spec, "space", None)
    if space is not None and hasattr(space, "low"):
        low = space.low.reshape(-1)[:n_act].cpu().numpy()
        high = space.high.reshape(-1)[:n_act].cpu().numpy()

    upright = bool(mcfg.get("upright", True))

    def make_upright():
        """Hover randomizes the start attitude; level the vehicle and zero its velocity."""
        try:
            pos, _ = drone.get_world_poses()
            q = torch.zeros(pos.shape[:-1] + (4,), device=pos.device)
            q[..., 0] = 1.0  # identity quaternion (w, x, y, z)
            drone.set_world_poses(pos, q)
            drone.set_velocities(torch.zeros(pos.shape[:-1] + (6,), device=pos.device))
        except Exception as e:  # noqa: BLE001
            print(f"[mpc_run] could not reset to upright: {e}")

    td = env.reset()
    if upright:
        make_upright()
    state = read_state(drone)

    target = mcfg.get("target", None)
    ref = {
        "pos_w": np.asarray(target, dtype=np.float64) if target is not None else state["pos_w"].copy(),
        "quat": state["quat"].copy(),
    }

    print("\n========== mpc_run ==========")
    print(f"vehicle        : {cfg.task.drone_model.name}")
    print(f"controller     : {controller_name}")
    print(f"dt             : {dt} s   (render every {render_every} steps, realtime={realtime})")
    print(f"action shape   : {act_shape}   bounds: {low} .. {high}")
    print(f"initial pos_w  : {state['pos_w']}")
    print(f"initial quat   : {state['quat']}  (w, x, y, z)")
    print(f"reference pos  : {ref['pos_w']}")
    print("=============================\n")

    controller = CONTROLLERS[controller_name](n_act=n_act, dt=dt)

    log = {k: [] for k in ["t", "pos_w", "quat", "vel_b", "omega_b", "action", "solve_ms"]}
    step, t = 0, 0.0

    try:
        while simulation_app.is_running() and not stop["flag"]:
            wall_start = time.perf_counter()
            state = read_state(drone)

            t0 = time.perf_counter()
            u = np.asarray(controller(state, ref, t), dtype=np.float64).reshape(n_act)
            solve_ms = (time.perf_counter() - t0) * 1e3
            if low is not None:
                u = np.clip(u, low, high)

            action = torch.as_tensor(u, dtype=torch.float32, device=env.device)
            td["agents", "action"] = action.expand(act_shape).clone()

            if hasattr(env, "enable_render"):
                env.enable_render((step % render_every == 0) and not cfg.headless)

            td = env.step(td)
            done = td["next", "done"].any().item()
            td = step_mdp(td)
            if done:
                print(f"[t={t:.2f}s] episode ended - resetting")
                td = env.reset()
                if upright:
                    make_upright()

            for k in ["pos_w", "quat", "vel_b", "omega_b"]:
                log[k].append(state[k])
            log["t"].append(t)
            log["action"].append(u)
            log["solve_ms"].append(solve_ms)

            if step % int(round(1.0 / dt)) == 0:
                p = state["pos_w"]
                print(f"t={t:7.2f}s  pos=({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})  "
                      f"u={np.array2string(u, precision=2)}  solve={solve_ms:.1f} ms")

            step += 1
            t += dt
            if duration is not None and t >= float(duration):
                break
            if realtime:
                sleep = dt - (time.perf_counter() - wall_start)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        if log["t"]:
            np.savez(log_path, **{k: np.asarray(v) for k, v in log.items()})
            print(f"Saved {len(log['t'])} steps to {log_path}")
        simulation_app.close()


if __name__ == "__main__":
    main()