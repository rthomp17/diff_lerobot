"""Plan a trajectory with ONLY the diffusion policy's alignment objective and visualize the MPC
optimization history in 3D end-effector space.

This is a pared-down sibling of ``init_pusht_planner.py`` for testing the policy alignment objective
in isolation. It deliberately does **not** touch the keypoint dynamics model or the T:

  * No SAM2 / red-blob / T-keypoint detection (the plan-time observation is just the dataset's start
    joint state + the first camera frame the policy conditions on).
  * No keypoint dynamics model is loaded or run. The two-stage MPC is given a no-op ``dummy_model``
    that echoes the joint actions and returns empty predicted states -- the same pattern
    ``difftree/dynaguide/calvin_planner.py`` uses. The policy alignment objective ignores the
    predicted states, so this is sufficient for a pure policy-directed plan.

The MPC's per-iteration sampled trajectories (joint-space) are forward-kinematics'd to end-effector
positions and drawn as a 3D plotly figure with a per-iteration slider, each trajectory colored by its
policy-objective cost -- the 3D-EE analog of ``init_pusht_planner.visualize_optimization_history``.

Run inside the `lerobot` conda env, e.g.:
    python scripts/plan_policy_objective_and_visualize.py \
        --repo-id Sorozco0612/pusht_200_20260706_130548 --episode 0 --n-actions 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import yaml

# Compatibility shim: the sibling `pusht_dynamics` repo has two copies of its code -- a stale
# top-level `/home/rthomp12/pusht_dynamics/utils.py` (namespace dir, no __init__) and the current
# packaged `pusht_dynamics_pkg/pusht_dynamics/`. `init_pusht_planner` puts `/home/rthomp12` on
# sys.path, which makes `import pusht_dynamics.utils` resolve to the *stale* file (missing
# `get_invariant_action_torch`), so importing `init_pusht_planner` / difftree fails. Pre-importing
# the packaged version here caches it in sys.modules before that chain runs, so the correct code is
# used. (Remove once the top-level pusht_dynamics leftovers are cleaned up / the package is
# installed.)
_PUSHT_DYNAMICS_PKG = "/home/rthomp12/pusht_dynamics/pusht_dynamics_pkg"
if _PUSHT_DYNAMICS_PKG not in sys.path:
    sys.path.insert(0, _PUSHT_DYNAMICS_PKG)
import pusht_dynamics.utils  # noqa: E402,F401  (import for its side effect: cache the packaged module)

# Importing init_pusht_planner runs its top-level sys.path setup (adds the sibling `diff_tree_search`
# / `pusht_dynamics` repos) so the `difftree` imports below resolve, and gives us the policy /
# planning helpers that don't touch detection or the dynamics model.
from init_pusht_planner import (
    DEFAULT_PLANNER_CONFIG,
    DEFAULT_POLICY_CONFIG,
    _iteration_costs_to_numpy,
    _policy_visual_key,
    build_policy,
    policy_objective,
)

import difftree.pusht.pusht_particle_model as pusht_particle_model
from difftree.common.viz_utils import get_color
from difftree.pusht.pusht_planner_utils import PushTPlanner, PushTPlannerConfig

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from visualize_arm_position import RobotState, plot_pose
from visualize_rollout import (
    DEFAULT_ROBOT_ID,
    DEFAULT_URDF,
    STATE_KEY,
    ee_pose_from_state,
    episode_frame_range,
    frame_to_uint8_hwc,
)

DEFAULT_OUT_HTML = str(Path(__file__).resolve().parent / "rollout_viz" / "planned_opt_history_3d.html")


def build_policy_only_planner(
    planner_config_path: str, horizon: int | None = None, action_dim: int | None = None
) -> PushTPlanner:
    """Construct a PushTPlanner whose two-stage MPC plans against a policy objective only.

    Avoids the keypoint dynamics model in two ways:

      1. ``PushTPlanner.__init__`` eagerly loads the keypoint dynamics weights via
         ``pusht_particle_model.load_model``. We monkeypatch that to a no-op returning ``(None, {})``
         for the duration of construction (mirroring ``build_policy``'s ``make_policy`` patch) so no
         weights are loaded.
      2. After construction we install a ``dummy_model`` (``lambda obs, actions: (np.empty(0),
         actions)``) on the planner and both MPC stages -- the calvin_planner pattern. The MPC
         tolerates empty predicted states and the policy objective ignores them.

    We also fix ``n_action_dim`` on both MPC stages to the joint-space dim (6). The MPCs default to
    2; without this, EvoSearch's second iteration adds ``(traj_len, 2)`` noise to ``(traj_len, 6)``
    seeds and crashes on the broadcast. ``action_dim`` is read from the YAML's ``action_dim`` key
    (the current ``PushTPlannerConfig`` dataclass drops that key, so we read the raw YAML), defaulting
    to 6.
    """
    configs = PushTPlannerConfig.from_yaml(planner_config_path)
    if not configs:
        raise ValueError(f"No planner configs found in {planner_config_path}")
    config = configs[0]
    if action_dim is None:
        with open(planner_config_path) as f:
            raw = yaml.safe_load(f)
        action_dim = int(raw[0].get("action_dim", 6)) if raw else 6
    if horizon is not None:
        config.horizon = horizon
        config.mpc_config = {**config.mpc_config, "traj_len": horizon}
        config.second_stage_mpc_config = {**config.second_stage_mpc_config, "traj_len": horizon}

    original_load_model = pusht_particle_model.load_model
    pusht_particle_model.load_model = lambda *args, **kwargs: (None, {})
    try:
        planner = PushTPlanner(config)
    finally:
        pusht_particle_model.load_model = original_load_model

    def dummy_model(obs, actions):
        # calvin_planner pattern: no dynamics -- return empty predicted states and echo the actions
        # the MPC is optimizing (the policy objective reads only the actions).
        return np.empty(0), actions

    planner.model = dummy_model
    planner.action_dim = action_dim
    for mpc in (planner.modal_mpc, planner.refinement_mpc):
        mpc.model = dummy_model
        mpc.n_action_dim = action_dim
        mpc.dim_samples = (mpc.traj_len, action_dim)
        mpc._reset_distribution()
    return planner


def build_plan_time_obs_no_detection(repo_id, episode=0, cam_key=None):
    """Build the minimal plan-time observation for a policy-only plan -- no SAM2, no T detection.

    Contains just what the policy objective + policy seeding + MPC touch on this path:
      * ``agent_pos`` / ``agent_state`` -- the episode's first-frame 6-DOF joint state. ``agent_state``
        is what ``Mpc.get_action`` integrates its delta trajectories from; ``agent_pos`` is what
        ``sample_policy_seed_sequences`` reads.
      * ``pixels`` -- ``{cam_short: image}`` (first camera frame) so ``preprocess_observation`` renames
        it to ``observation.images.<cam_short>`` for the diffusion policy's image conditioning.

    Returns ``(obs, debug)``.
    """
    dataset = LeRobotDataset(repo_id)
    start, _end = episode_frame_range(dataset, episode)
    first = dataset[start]
    joint_state = np.asarray(
        first[STATE_KEY].numpy() if hasattr(first[STATE_KEY], "numpy") else first[STATE_KEY],
        dtype=float,
    )

    cam = cam_key if cam_key is not None else list(dataset.meta.camera_keys)[0]
    image = frame_to_uint8_hwc(first[cam])  # (H, W, 3) uint8
    cam_short = cam.split(".")[-1]

    obs = {
        "agent_pos": joint_state,
        "agent_state": joint_state,
        "pixels": {cam_short: image},
    }
    debug = {"camera": cam, "joint_state": joint_state, "n_frames": _end - start}
    return obs, debug


def _iter_history_stages(history):
    """Normalize the two_stage_optimizer history into an ordered ``[(stage_label, hist)]`` list."""
    if isinstance(history, dict) and "delta_actions" in history:
        return [("", history)]
    if isinstance(history, dict):
        return [(str(k), v) for k, v in history.items()]
    return [(f"stage{i}", h) for i, h in enumerate(history)]


def _deltas_to_numpy(deltas):
    """A single iteration's stored delta actions may be a torch tensor; return an ndarray."""
    if hasattr(deltas, "detach"):
        deltas = deltas.detach().cpu().numpy()
    return np.asarray(deltas, dtype=float)


def _joint_seq_to_ee_positions(robot_state, joint_seq):
    """FK a (T, n_joints) absolute-joint sequence to EE positions (T, 3)."""
    return np.asarray([ee_pose_from_state(robot_state, js)[0] for js in joint_seq])


def visualize_optimization_history_3d_ee(
    history,
    obs,
    robot_state,
    max_trajectories_per_iter=40,
    policy_trajectories_ee=None,
    title="Planner optimization history (3D EE space)",
    out_html=None,
):
    """Build a plotly figure of the MPC optimization history in 3D end-effector space.

    3D-EE analog of ``init_pusht_planner.visualize_optimization_history`` (which plots the gym_pusht
    frame). One slider step per optimization iteration; within each step every sampled joint-space
    trajectory is turned into absolute joints (``agent_state + cumsum(delta)``, with the start state
    prepended so the line begins at the current EE), forward-kinematics'd to EE positions, and drawn
    as a 3D line colored by that trajectory's policy-objective cost (shared color scale across all
    iterations).

    Reads the current difftree MPC history keys ``delta_actions`` / ``costs`` (the older
    ``action_sequences`` keys that the 2D function reads are no longer emitted). ``history`` may be a
    single get_action history dict or the ``{"modal", "refinement"}`` dict from
    ``two_stage_optimizer(return_history=True)``.

    ``policy_trajectories_ee`` (optional ``(n, T, 3)`` EE positions from
    ``sample_policy_ee_trajectories``) is drawn as a static crimson reference overlay, visible across
    all slider steps, for comparison against the optimized trajectories. The scene axis ranges are
    locked over all plotted points so the box does not rescale as the slider changes iterations.
    """
    agent_state = np.asarray(obs["agent_state"], dtype=float)

    # Collect per-iteration (label, EE trajectories (n, T+1, 3), per-trajectory cost (n,)).
    iterations = []
    for stage_label, hist in _iter_history_stages(history):
        for it, (deltas_it, costs_it) in enumerate(zip(hist["delta_actions"], hist["costs"])):
            deltas = _deltas_to_numpy(deltas_it)
            if deltas.ndim == 2:
                deltas = deltas[np.newaxis]
            costs_arr = _iteration_costs_to_numpy(costs_it)
            n = min(deltas.shape[0], len(costs_arr))
            deltas, costs_arr = deltas[:n], costs_arr[:n]
            if max_trajectories_per_iter and n > max_trajectories_per_iter:
                keep = np.linspace(0, n - 1, max_trajectories_per_iter).astype(int)
                deltas, costs_arr = deltas[keep], costs_arr[keep]
            abs_joints = agent_state + np.cumsum(deltas, axis=1)  # (n, T, n_joints)
            # Prepend the start state so each EE line emanates from the current pose.
            abs_joints = np.concatenate(
                [np.broadcast_to(agent_state, (abs_joints.shape[0], 1, agent_state.shape[0])), abs_joints],
                axis=1,
            )
            ee = np.asarray([_joint_seq_to_ee_positions(robot_state, traj) for traj in abs_joints])
            label = f"{stage_label} it {it}" if stage_label else f"it {it}"
            iterations.append((label, ee, costs_arr))

    if not iterations:
        raise ValueError("Optimization history contained no iterations to plot")

    all_costs = np.concatenate([c for _, _, c in iterations])
    cmin, cmax = float(all_costs.min()), float(all_costs.max())
    cspan = (cmax - cmin) or 1.0

    # Static traces: a colorbar carrier + the base-link frame (always visible).
    colorbar_trace = go.Scatter3d(
        x=[None], y=[None], z=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(color=[cmin], colorscale="Viridis", cmin=cmin, cmax=cmax,
                    showscale=True, colorbar=dict(title="cost")),
    )
    base = robot_state.robot_urdf.link_fk(cfg=np.zeros(6))[robot_state.robot_urdf.link_map["base_link"]]
    base_pose = [*base[:3, 3], *robot_state.rot_matrix_to_quat(base[:3, :3])]
    base_fig = plot_pose(base_pose, axis_length=0.05, name="base_link")
    static_traces = [colorbar_trace, *list(base_fig.data)]

    # Reference overlay: trajectories sampled directly from the policy, drawn in a distinct fixed color
    # and kept visible across every slider step for comparison with the optimized (cost-colored) ones.
    if policy_trajectories_ee is not None:
        policy_ee = np.asarray(policy_trajectories_ee)
        for j in range(policy_ee.shape[0]):
            static_traces.append(go.Scatter3d(
                x=policy_ee[j, :, 0], y=policy_ee[j, :, 1], z=policy_ee[j, :, 2],
                mode="lines+markers",
                line=dict(color="crimson", width=4, dash="dash"),
                marker=dict(size=3, color="crimson"),
                name="policy sample" if j == 0 else f"policy sample {j}",
                legendgroup="policy", showlegend=(j == 0),
            ))
    n_static = len(static_traces)

    traces = list(static_traces)
    iteration_trace_ranges = []
    for it_idx, (_, ee, costs_arr) in enumerate(iterations):
        start = len(traces)
        norm = (costs_arr - cmin) / cspan
        colors = get_color("Viridis", norm)
        for j in range(ee.shape[0]):
            traces.append(go.Scatter3d(
                x=ee[j, :, 0], y=ee[j, :, 1], z=ee[j, :, 2], mode="lines+markers",
                line=dict(color=colors[j], width=3), marker=dict(size=2, color=colors[j]),
                name=f"traj {j} (cost {costs_arr[j]:.2f})", showlegend=False,
                visible=(it_idx == 0),
            ))
        iteration_trace_ranges.append((start, len(traces)))

    steps = []
    for it_idx, (label, _, _) in enumerate(iterations):
        visibility = [False] * len(traces)
        for k in range(n_static):
            visibility[k] = True
        s, e = iteration_trace_ranges[it_idx]
        for k in range(s, e):
            visibility[k] = True
        steps.append(dict(
            method="update",
            args=[{"visible": visibility}, {"title": f"{title} — {label}"}],
            label=str(it_idx),
        ))

    # Lock the scene axis ranges over every plotted point (all iterations + policy overlay + base) so
    # the box keeps a fixed size/shape as the slider reveals different iterations, instead of the axes
    # rescaling per step. aspectmode="data" then keeps true EE geometry (equal scale per axis), matching
    # visualize_rollout.build_ee_figure.
    pts = [ee.reshape(-1, 3) for _, ee, _ in iterations]
    pts.append(np.asarray(base[:3, 3]).reshape(1, 3))
    if policy_trajectories_ee is not None:
        pts.append(np.asarray(policy_trajectories_ee).reshape(-1, 3))
    all_pts = np.concatenate(pts, axis=0)
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    pad = np.maximum((maxs - mins) * 0.1, 0.02)
    xr, yr, zr = ([float(mins[i] - pad[i]), float(maxs[i] + pad[i])] for i in range(3))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title} — {iterations[0][0]}",
        sliders=[dict(active=0, steps=steps, currentvalue={"prefix": "iteration: "})],
        scene=dict(
            xaxis=dict(title="X", range=xr),
            yaxis=dict(title="Y", range=yr),
            zaxis=dict(title="Z", range=zr),
            aspectmode="data",
        ),
        uirevision="opt-history-3d",
        showlegend=True,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    if out_html is not None:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html)
        print(f"wrote optimization-history figure: {out_html}")
    return fig


def sample_full_horizon_seed_sequences(policy, obs, batch_size, traj_len):
    """Sample `batch_size` full-horizon joint-space seed trajectories (as MPC deltas).

    Mirrors ``PushTPolicy.infer_seed_trajectories`` (full-horizon diffusion samples via
    ``generate_full_actions``) rather than ``init_pusht_planner.sample_policy_seed_sequences``
    (which uses ``infer_target`` and returns only the ``n_action_steps`` slice -- too short here, the
    policy has horizon=16 but n_action_steps=8). Planner seeds MUST be full ``horizon`` length because
    the policy objective re-noises them through the diffusion U-Net, which is trained on full-horizon
    trajectories; a truncated seed crashes the U-Net.

    Reads the checkpoint's actual visual feature key (``observation.images.<cam>``) instead of the gym
    ``observation.image`` that difftree's ``infer_seed_trajectories`` hardcodes.

    Returns MPC delta seeds ``(batch_size, traj_len, action_dim)``:
        delta[0]   = target[0] - agent_pos ; delta[t>0] = target[t] - target[t-1].
    """
    samples = _sample_full_horizon_actions(policy, obs, batch_size)  # (B, horizon, action_dim) absolute
    if samples.shape[1] > traj_len:
        samples = samples[:, :traj_len]
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
    prev = np.broadcast_to(agent_pos[None, None, :], (samples.shape[0], 1, agent_pos.shape[0]))
    return np.diff(samples, axis=1, prepend=prev)


def _sample_full_horizon_actions(policy, obs, batch_size):
    """Sample ``batch_size`` full-horizon absolute joint-target trajectories from the diffusion policy.

    Full diffusion trajectories via ``generate_full_actions`` (shape ``(batch_size, horizon,
    action_dim)``), reading the checkpoint's real visual key (``observation.images.<cam>``) rather than
    the gym ``observation.image`` that difftree's ``infer_seed_trajectories`` hardcodes.
    """
    visual_key = _policy_visual_key(policy)
    n_obs_steps = policy.policy.config.n_obs_steps
    processed_obs = policy.preprocess_observation(obs)
    processed_obs = policy.preprocessor(processed_obs)
    batch_obs = {
        "observation.state": processed_obs["observation.state"]
        .unsqueeze(1).repeat_interleave(n_obs_steps, dim=1)
        .repeat_interleave(batch_size, dim=0).cuda().float(),
        "observation.images": processed_obs[visual_key]
        .unsqueeze(1).unsqueeze(1).repeat_interleave(n_obs_steps, dim=1)
        .repeat_interleave(batch_size, dim=0).cuda().float(),
    }
    with torch.inference_mode():
        actions = policy.policy.diffusion.generate_full_actions(batch_obs)  # (B, horizon, action_dim)
    actions = policy.postprocessor(actions)
    return np.asarray(actions.detach().cpu(), dtype=np.float32)


def sample_policy_ee_trajectories(policy, obs, robot_state, n_samples=5):
    """Sample ``n_samples`` full-horizon policy trajectories and FK them to EE positions.

    Returns an array ``(n_samples, horizon + 1, 3)`` of EE positions -- the start pose (FK of
    ``obs["agent_pos"]``) is prepended so each line emanates from the current EE, matching the
    optimized trajectories. Used as a static reference overlay in the 3D figure.
    """
    samples = _sample_full_horizon_actions(policy, obs, n_samples)  # (n, horizon, action_dim) absolute
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
    prev = np.broadcast_to(agent_pos[None, None, :], (samples.shape[0], 1, agent_pos.shape[0]))
    abs_joints = np.concatenate([prev, samples], axis=1)  # (n, horizon + 1, action_dim)
    return np.asarray([_joint_seq_to_ee_positions(robot_state, traj) for traj in abs_joints])


def _stage_active(mpc):
    """A CEM/EvoSearch stage does real work only if it samples trajectories and iterates.

    Some configs (e.g. difftree's ``pusht_planner_config.yaml``) zero out the modal stage
    (``num_sim_traj: 0``, ``opt_iter: 0``) to run refinement-only; sampling 0 seed trajectories would
    otherwise crash the diffusion sampler (batch dim 0 -> einops ZeroDivisionError)."""
    return mpc.num_sim_traj > 0 and mpc.opt_iter > 0


def run_policy_objective_plan(planner, policy, obs, policy_objective_weight=0.05):
    """Run the two-stage MPC using ONLY the diffusion policy's alignment objective as the cost.

    Drives the current difftree MPC API directly (``get_action(obs, objective, seed_delta_actions,
    return_history=True)``) rather than ``init_pusht_planner.two_stage_optimizer`` (which calls the
    older ``seed_sequences=`` signature). Reuses the still-current non-MPC helpers
    ``sample_full_horizon_seed_sequences`` and ``policy_objective``.

    Runs whichever of the two stages are active (see ``_stage_active``): refinement-only configs
    (modal zeroed) are supported by seeding the refinement stage with policy samples directly instead
    of with the (absent) modal elites.

    The MPC calls the objective as ``objective(world_frame_actions, predicted_states, obs)``;
    ``policy_objective(policy, states, actions, obs)`` ignores ``states`` and scores ``actions``.

    Returns ``(actions, history)`` where ``actions`` is the final best absolute-joint trajectory
    ``(traj_len, action_dim)`` and ``history`` holds the active stages' get_action histories.
    """
    def objective(actions, states, obs_):
        # The MPC hands `actions` (world-frame joint waypoints) back as a CUDA tensor; policy_objective
        # expects a numpy array (it does np.asarray then re-uploads to CUDA itself).
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        return policy_objective_weight * policy_objective(policy, states, actions, obs_)

    modal, refine = planner.modal_mpc, planner.refinement_mpc
    if not _stage_active(modal) and not _stage_active(refine):
        raise ValueError(
            "No active MPC stage: both modal and refinement have num_sim_traj/opt_iter == 0. "
            "Check the planner config's mpc_config / second_stage_mpc_config."
        )

    history = {}
    actions = None
    modal_hist = None
    if _stage_active(modal):
        seeds = sample_full_horizon_seed_sequences(policy, obs, modal.num_sim_traj, modal.traj_len)
        actions, _traj, modal_hist = modal.get_action(obs, objective, seeds, return_history=True)
        history["modal"] = modal_hist

    if _stage_active(refine):
        # Prefer the modal search's lowest-cost elite deltas; if modal didn't run (refinement-only
        # config), seed refinement with fresh full-horizon policy samples instead.
        elites = modal_hist.get("lowest_cost_elites") if modal_hist is not None else None
        if elites is not None and len(elites) > 0:
            refine_seeds = elites
        else:
            refine_seeds = sample_full_horizon_seed_sequences(policy, obs, refine.num_sim_traj, refine.traj_len)
        actions, _traj2, refinement_hist = refine.get_action(
            obs, objective, seed_delta_actions=refine_seeds, return_history=True
        )
        history["refinement"] = refinement_hist

    if hasattr(actions, "detach"):
        actions = actions.detach().cpu().numpy()
    return np.asarray(actions), history


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True, help="Dataset to sample the start state from.")
    parser.add_argument("--episode", type=int, default=0, help="Episode to sample the start state from.")
    parser.add_argument("--planner-config", default=DEFAULT_PLANNER_CONFIG,
                        help="Path to the PushTPlannerConfig YAML file.")
    parser.add_argument("--policy-config", default=DEFAULT_POLICY_CONFIG,
                        help="Path to the PushTPolicyConfig YAML file (loads the diffusion checkpoint).")
    parser.add_argument("--n-actions", type=int, default=16,
                        help="Number of actions the planner optimizes over (MPC traj_len).")
    parser.add_argument("--policy-objective-weight", type=float, default=0.05,
                        help="Weight of the policy alignment objective in the (sole) planning cost.")
    parser.add_argument("--cam-key", default=None,
                        help="Camera key for the policy image conditioning (default: first camera).")
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="URDF path for forward kinematics.")
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID, help="Robot id (calibration label).")
    parser.add_argument("--max-trajectories-per-iter", type=int, default=40,
                        help="Subsample this many trajectories per iteration when plotting (FK cost).")
    parser.add_argument("--n-policy-samples", type=int, default=5,
                        help="Draw this many trajectories sampled directly from the policy as a static "
                             "reference overlay (0 to disable).")
    parser.add_argument("--out-html", default=DEFAULT_OUT_HTML, help="Output HTML path for the figure.")
    parser.add_argument("--no-show", action="store_true", help="Only write the HTML; don't fig.show().")
    args = parser.parse_args()

    planner = build_policy_only_planner(args.planner_config, horizon=args.n_actions)
    print(f"Initialized policy-only PushT planner from {args.planner_config}")
    print(f"  mpc_type:   {planner.mpc_type}")
    print(f"  horizon:    {planner.horizon}")
    print(f"  action_dim: {planner.action_dim}")
    print(f"  modal traj_len={planner.modal_mpc.traj_len}, n_action_dim={planner.modal_mpc.n_action_dim}")
    print(f"  dynamics model: dummy (no keypoint model loaded)")

    policy = build_policy(args.policy_config)
    print(f"Initialized PushT policy from {args.policy_config}")
    print(f"  policy_path: {policy.policy_path}")
    print(f"  horizon:     {policy.horizon}")

    obs, debug = build_plan_time_obs_no_detection(args.repo_id, episode=args.episode, cam_key=args.cam_key)
    print(f"\nPlan-time observation from {args.repo_id} episode {args.episode} (no detection):")
    print(f"  camera:                  {debug['camera']}")
    print(f"  agent_pos (6-DOF joints): {np.round(obs['agent_pos'], 3)}")

    print(f"\nPlanning over {args.n_actions} actions with the policy alignment objective only "
          f"(weight={args.policy_objective_weight})...")
    actions, history = run_policy_objective_plan(
        planner, policy, obs, policy_objective_weight=args.policy_objective_weight
    )
    print(f"  best action trajectory (joint space) shape: {actions.shape}")
    print(f"  first planned joint waypoint: {np.round(actions[0], 3)}")
    print(f"  last planned joint waypoint:  {np.round(actions[-1], 3)}")

    robot_state = RobotState(urdf_path=args.urdf, id=args.robot_id, load_meshes=False)

    policy_ee = None
    if args.n_policy_samples > 0:
        print(f"\nSampling {args.n_policy_samples} reference trajectories from the policy...")
        policy_ee = sample_policy_ee_trajectories(policy, obs, robot_state, n_samples=args.n_policy_samples)

    fig = visualize_optimization_history_3d_ee(
        history, obs, robot_state,
        max_trajectories_per_iter=args.max_trajectories_per_iter,
        policy_trajectories_ee=policy_ee,
        out_html=args.out_html,
    )
    if not args.no_show:
        fig.show()
    return planner, obs, actions, history


if __name__ == "__main__":
    main()
