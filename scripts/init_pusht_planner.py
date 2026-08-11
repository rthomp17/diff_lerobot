"""Initialize a PushT keypoint planner.

Loads a PushTPlannerConfig from a YAML file (modeled off
../diff_tree_search/configs/pusht_planner_config.yaml) and constructs the
PushTPlanner defined in ../diff_tree_search/difftree/pusht/pusht_planner_utils.py.

The `difftree` package lives in the sibling `diff_tree_search` repo and isn't
installed in this env, so we add it to sys.path before importing.

Example:
    python scripts/init_pusht_planner.py \
        --planner-config scripts/pusht_planner_config.yaml
"""

import argparse
import gc
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image

# Make the sibling repos importable:
#   - `diff_tree_search` provides `difftree`
#   - the keypoint dynamics model imports `pusht_dynamics`, a top-level namespace
#     package living directly under the home dir (parent of these repos).
SCRIPT_DIR = Path(__file__).resolve().parent
HOME_ROOT = SCRIPT_DIR.parent.parent
DIFF_TREE_SEARCH_ROOT = HOME_ROOT / "diff_tree_search"
# `pusht_dynamics` uses flat, top-level imports internally (e.g. `import data_generator`),
# so its own directory has to be on the path as well as the home dir above it.
PUSHT_DYNAMICS_ROOT = HOME_ROOT / "pusht_dynamics"
# SCRIPT_DIR is on the path so we can reuse the FK / frame-transform helpers from the
# sibling scripts (visualize_rollout.py, visualize_arm_position.py).
for _path in (str(DIFF_TREE_SEARCH_ROOT), str(HOME_ROOT), str(PUSHT_DYNAMICS_ROOT), str(SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from difftree.common.utils import radial_trajectories, to_json_serializable, without_keys
from difftree.common.viz_utils import get_color
from difftree.pusht.pusht_env_utils import plot_n_actions
import difftree.pusht.pusht_policy_utils as pusht_policy_utils
from difftree.pusht.pusht_planner_utils import PushTPlanner, PushTPlannerConfig, PushTPlanningConstraint
from difftree.pusht.pusht_policy_utils import PushTPolicy, PushTPolicyConfig

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.utils import OBS_IMAGE

# FK + frame-transform + perception helpers reused from the sibling visualization scripts.
from visualize_arm_position import RobotState
from visualize_rollout import (
    DEFAULT_ROBOT_ID,
    DEFAULT_SAM_WEIGHTS,
    DEFAULT_URDF,
    STATE_KEY,
    align_canonical_t_to_mask,
    ee_pose_from_state,
    episode_frame_range,
    find_red_centroid,
    frame_to_uint8_hwc,
    load_sam2,
    pixel_to_workspace_xy,
    resolve_sam_device,
    segment_at_point,
    workspace_xy_to_gym_pusht,
)

DEFAULT_PLANNER_CONFIG = "/home/rthomp12/diff_tree_search/configs/real_pusht_planner_config.yaml"
DEFAULT_POLICY_CONFIG = str(Path(__file__).resolve().parent / "pusht_policy_config.yaml")


def _visualize_action_history(best_action_history, env):
    """Visualize the history of best action sequences from MPC as a plotly figure.

    Ported from PushTCoupledPolicy._visualize_action_history.
    """
    fig = env.create_figure()
    actions_array = np.array(best_action_history)
    num_iterations = len(best_action_history)
    colors = get_color("viridis", np.linspace(0, 1, num_iterations))
    fig = plot_n_actions(fig, actions_array, colors)
    return fig


def two_stage_optimizer(
    planner,
    obs,
    costs,
    env,
    policy=None,
    visualize_history=False,
    seed_planner=True,
    seed_sequences=None,
    n_seed_trajectories=40,
    seed_trajectory_length=8,
    seed_trajectory_radius=10,
    optimization_history_folder=None,
    step=None,
    return_history=False,
):
    """Use MPC to optimize a trajectory that minimizes cost from the policy objective
    along with inference-time constraints.

    Independent-helper port of PushTCoupledPolicy.two_stage_optimizer: the former
    `self.planner`/`self.policy` are now the explicit `planner` and `policy` arguments.

    Seed trajectories for the modal MPC come from one of:
      - explicit `seed_sequences` (shape ``(n_seeds, traj_len, action_dim)``), used as-is;
      - `seed_planner=True` + a `policy` (the original policy-directed radial seeding); or
      - neither, giving no seeds (only valid for the CEM MPC, which samples its own).
    EvoSearchMpc requires non-empty seeds, so pass `seed_sequences` when using it without a
    policy.
    """
    # Seeding initial MPC trajectories
    if seed_sequences is not None:
        seed_sequences = np.asarray(seed_sequences)
    elif seed_planner:
        if policy is None:
            raise ValueError("seed_planner=True requires a `policy` to generate seed sequences")
        # Inlined from PushTCoupledPolicy.pusht_get_seed_sequences.
        seed_sequences = policy.infer_target(obs, 2, None, visualize=False, env=env)
        radius = np.linalg.norm(seed_sequences[0] - seed_sequences[-1])
        seed_sequences = np.array(
            radial_trajectories(obs["agent_pos"][0], 40, radius, n_steps=planner.modal_mpc.traj_len)
        )
        seed_sequences = np.array([np.diff(s, prepend=[s[0]], axis=0) for s in seed_sequences])
    else:
        seed_sequences = []

    actions, paths, modal_optimization_history = planner.modal_mpc.get_action(
        obs,
        lambda a, s, o: sum([p.weight * p.cost(s, a, o) for p in costs]),
        env=env,
        seed_sequences=seed_sequences,
        return_history=True,
    )

    if visualize_history and env is not None:
        agent_trajectories = modal_optimization_history["best_action_sequences"]
        fig = _visualize_action_history(agent_trajectories, env)
        fig.update_layout(title_text="MODAL Best Action Per Iteration (blue = older)")
        fig.show()
        input("MODAL cont?")

    # Seed trajectories for refinement are the lowest cost elites over all stages of the modal optimization.
    second_stage_seed_trajectories = modal_optimization_history["lowest_cost_elites"]

    actions, paths, refinement_optimization_history = planner.refinement_mpc.get_action(
        obs,
        lambda a, s, o: sum([p.weight * p.cost(s, a, o) for p in costs]),
        seed_sequences=second_stage_seed_trajectories,
        env=env,
        return_history=True,
    )
    if visualize_history and env is not None:
        agent_trajectories = (
            np.cumsum(np.array(refinement_optimization_history["best_action_sequences"]), axis=1) + obs["agent_pos"]
        )
        fig = _visualize_action_history(agent_trajectories, env)
        fig.update_layout(title_text="REF Best Action Per Iteration (blue = older)")
        fig.show()
        input("Ref  cont?")

    # Save optimization data if desired
    if not visualize_history and optimization_history_folder is not None:
        filepath = f"{optimization_history_folder}/{step}_optimization.json"
        Path(optimization_history_folder).mkdir(parents=True, exist_ok=True)
        print("saving opt history")
        optimization_data = to_json_serializable(
            {
                "observation": without_keys(obs, ["pixels"]),
                "actions": actions,
                # "paths": paths, # Removed when dealing with DINO-WM predictions
                "action_history": modal_optimization_history["action_sequences"]
                + refinement_optimization_history["action_sequences"],
                "best_action_history": modal_optimization_history["best_action_sequences"]
                + refinement_optimization_history["best_action_sequences"],
                "costs": modal_optimization_history["costs"] + refinement_optimization_history["costs"],
            }
        )
        with open(filepath, "w") as f:
            json.dump(optimization_data, f)

    if return_history:
        return actions, paths, {"modal": modal_optimization_history, "refinement": refinement_optimization_history}
    return actions, paths


class JointSpaceKeypointModel:
    """Wrap the keypoint dynamics model so the planner can plan in joint space.

    The MPC calls the model as ``model(obs, actions)``. Here ``actions`` are absolute
    joint-space action sequences of shape ``(B, T, n_joints)`` (or ``(T, n_joints)`` for a
    single trajectory) -- the MPC has already turned its per-step deltas into absolute
    joint waypoints (``obs["agent_pos"] + np.cumsum(...)``). Each waypoint is transformed,
    per timestep, before being handed to the underlying keypoint model:

      1. absolute joints -> end-effector ``(x, y)`` in workspace meters, via the same
         forward-kinematics pipeline used in visualize_rollout.py (``ee_pose_from_state``);
         the Z value and orientation are dropped for now.
      2. workspace ``(x, y)`` -> gym_pusht frame, via ``workspace_xy_to_gym_pusht``. The T's
         bar width (meters) needed for this frame conversion is read from
         ``obs["t_bar_width_meters"]``.

    The resulting gym-frame XY sequences (shape ``(B, T, 2)``) are passed to the keypoint
    model, which consumes them as absolute XY targets, unchanged from how it exists today.
    """

    def __init__(self, keypoint_model, robot_state, n_joints=6):
        self.keypoint_model = keypoint_model
        self.robot_state = robot_state
        self.n_joints = n_joints

    def joint_actions_to_gym_xy(self, joint_actions, t_bar_width_meters):
        """Transform absolute joint-space actions into gym_pusht XY actions.

        Input shape ``(B, T, n_joints)`` or ``(T, n_joints)``; output matches with the last
        dimension replaced by 2 (gym-frame XY).
        """
        actions = np.asarray(joint_actions, dtype=float)
        squeeze = actions.ndim == 2
        if squeeze:
            actions = actions[np.newaxis]

        batch, horizon, _ = actions.shape

        # (1)+(2) FK each joint vector -> workspace (x, y) -> gym_pusht frame
        gym_xy = np.zeros((batch, horizon, 2))
        for b in range(batch):
            for t in range(horizon):
                trans, _quat = ee_pose_from_state(self.robot_state, actions[b, t])
                gym_xy[b, t] = workspace_xy_to_gym_pusht(
                    float(trans[0]), float(trans[1]), t_bar_width_meters
                )

        if squeeze:
            gym_xy = gym_xy[0]
        return gym_xy

    def __call__(self, obs, actions):
        gym_xy = self.joint_actions_to_gym_xy(actions, obs["t_bar_width_meters"])
        simulated_paths, _echoed_gym_actions = self.keypoint_model(obs, gym_xy)
        # Return the joint-space actions (not the wrapped model's gym-XY echo) so the MPC's
        # planned action trajectory stays in the joint space it is optimizing over.
        return simulated_paths, actions


def build_joint_space_model(
    keypoint_model,
    urdf_path=DEFAULT_URDF,
    robot_id=DEFAULT_ROBOT_ID,
    n_joints=6,
) -> JointSpaceKeypointModel:
    """Construct a JointSpaceKeypointModel, loading a FK-only RobotState from the URDF."""
    robot_state = RobotState(urdf_path=urdf_path, id=robot_id, load_meshes=False)
    return JointSpaceKeypointModel(keypoint_model, robot_state, n_joints)


def install_joint_space_model(planner, **kwargs) -> JointSpaceKeypointModel:
    """Wrap ``planner.model`` for joint-space planning and install it on the planner + MPCs.

    The MPC objects captured a reference to the original ``planner.model`` at construction
    time, so they must be repointed as well.
    """
    wrapped = build_joint_space_model(planner.model, **kwargs)
    planner.model = wrapped
    planner.modal_mpc.model = wrapped
    planner.refinement_mpc.model = wrapped
    return wrapped


# Canonical T keypoint names (from visualize_rollout.build_canonical_t) ordered to match
# gym_pusht's PushTEnv.get_keypoints() vertex order, so the constructed ``particles`` line up
# with what the keypoint dynamics model was trained on:
#     0───────────1        0=bar_tl 1=bar_tr
#     │           │
#     3───4───5───2        3=bar_bl 4=stem_tl 5=stem_tr 2=bar_br
#         │   │
#         7───6            7=stem_bl 6=stem_br
GYM_KEYPOINT_ORDER = (
    "bar_tl", "bar_tr", "bar_br_out", "bar_bl_out", "stem_tl", "stem_tr", "stem_br", "stem_bl",
)

# gym_pusht renders in a 512x512 window; its center is the placeholder push target.
GYM_PUSHT_WINDOW = 512.0
GYM_PUSHT_WINDOW_CENTER = GYM_PUSHT_WINDOW / 2.0


def build_plan_time_obs(
    repo_id,
    episode=0,
    urdf_path=DEFAULT_URDF,
    robot_id=DEFAULT_ROBOT_ID,
    sam_weights=DEFAULT_SAM_WEIGHTS,
    sam_device="cuda",
    cam_key=None,
):
    """Build a plan-time observation from the first frame of a sampled dataset episode.

    Reuses the pipelines from visualize_rollout.py to fill the observation the joint-space
    planner needs:

      * **agent position** -- forward-kinematics the first frame's 6-DOF joint state to an
        end-effector ``(x, y)`` (Z/orientation dropped), treated as workspace meters, then
        mapped into the gym_pusht frame. This is the initial end-effector the keypoint model
        integrates from (``obs["info"]["pos_agent"]``).
      * **T keypoints** -- red-blob seed -> SAM2 mask -> canonical-T alignment on the first
        frame gives 8 pixel-space corners; each is mapped pixel -> workspace meters ->
        gym_pusht frame, ordered to match ``get_keypoints`` (``obs["particles"]``).

    ``obs["agent_pos"]`` is the raw 6-DOF joint vector (NOT the XY position): the MPC plans in
    joint space and forms its action trajectories as ``obs["agent_pos"] + np.cumsum(deltas)``,
    which the JointSpaceKeypointModel wrapper then FKs back to gym XY.

    Returns ``(obs, debug)`` where ``debug`` carries the intermediate perception/FK products.

    NOTE: following visualize_rollout.py, the pixel->workspace and FK->workspace mappings are
    the acknowledged "HACK" (no camera extrinsics yet), so both the agent XY and the T
    keypoints are only approximately in a shared metric frame before the gym conversion.
    """
    dataset = LeRobotDataset(repo_id)
    start, _end = episode_frame_range(dataset, episode)
    first = dataset[start]
    joint_state = np.asarray(
        first[STATE_KEY].numpy() if hasattr(first[STATE_KEY], "numpy") else first[STATE_KEY],
        dtype=float,
    )

    # --- agent position: FK -> workspace (x, y) ---
    robot_state = RobotState(urdf_path=urdf_path, id=robot_id, load_meshes=False)
    trans, _quat = ee_pose_from_state(robot_state, joint_state)
    agent_ws = (float(trans[0]), float(trans[1]))

    # --- T keypoints: red-blob -> SAM2 -> canonical-T alignment on the first frame ---
    resolved_device = resolve_sam_device(sam_device)
    sam = load_sam2(sam_weights, device=resolved_device)
    cam_keys = [cam_key] if cam_key is not None else list(dataset.meta.camera_keys)

    align = None
    used_cam = None
    pil_used = None
    for cam in cam_keys:
        pil_img = Image.fromarray(frame_to_uint8_hwc(first[cam]))
        seed = find_red_centroid(pil_img)
        if seed is None:
            continue
        det = segment_at_point(sam, pil_img, seed, device=resolved_device)
        if det is None:
            continue
        # Single frame: bootstrap scale jointly with pose (fixed_scale=None).
        candidate = align_canonical_t_to_mask(det["mask"], fixed_scale=None, device=resolved_device)
        if candidate is not None:
            align, used_cam, pil_used = candidate, cam, pil_img
            break
    if align is None:
        raise RuntimeError(
            f"Could not detect/align the T in the first frame of episode {episode} "
            f"across cameras {cam_keys}"
        )

    # SAM2 is only needed for the one-shot T segmentation above; the mask/alignment products it
    # produced already live on the CPU (numpy), so drop the model and hand its GPU blocks back to
    # the driver before the policy/diffusion forward passes run — otherwise it sits resident for
    # the whole planning call and competes with the `num_sim_traj`-wide diffusion batch (CUDA OOM).
    del sam, det
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # keypoints: pixel -> workspace meters
    kp_ws = {
        name: pixel_to_workspace_xy(px, py, pil_used.size)
        for name, (px, py) in align["keypoints_px"].items()
    }
    # T bar width (meters) = distance between the two top bar corners, used for the gym scale.
    t_bar_width_meters = float(np.linalg.norm(np.subtract(kp_ws["bar_tl"], kp_ws["bar_tr"])))

    # --- translate agent + keypoints into gym_pusht space ---
    agent_gym = np.asarray(workspace_xy_to_gym_pusht(*agent_ws, t_bar_width_meters), dtype=float)
    particles_gym = np.asarray(
        [workspace_xy_to_gym_pusht(*kp_ws[name], t_bar_width_meters) for name in GYM_KEYPOINT_ORDER],
        dtype=float,
    )  # (8, 2), ordered to match get_keypoints

    # Raw first-frame image kept alongside the plan-time obs so `sample_policy_seed_sequences`
    # can build a policy-compatible observation (the SO101 diffusion policy conditions on
    # `observation.state` + `observation.images.<cam>` — joint state + camera image).
    first_frame_image = np.asarray(pil_used)  # (H, W, 3) uint8
    cam_short = used_cam.split(".")[-1] if used_cam else "front"

    obs = {
        # 6-DOF joint config: the MPC's joint-space anchor (agent_pos + cumsum(joint deltas)).
        "agent_pos": joint_state,
        "agent_state": joint_state,
        "particles": particles_gym,
        "t_bar_width_meters": t_bar_width_meters,
        "info": {
            # gym-frame agent XY the keypoint model integrates the end-effector from.
            "pos_agent": agent_gym,
            "vel_agent": np.zeros(2),
        },
        "obstacles": [],
        "memory": None,
        # Policy-side inputs stashed for `sample_policy_seed_sequences`: a `pixels` dict
        # `{cam_short: image}` so `preprocess_observation` renames it to
        # `observation.images.<cam_short>` (the policy's expected image feature key).
        "pixels": {cam_short: first_frame_image},
    }
    debug = {
        "camera": used_cam,
        "joint_state": joint_state,
        "agent_ws": agent_ws,
        "agent_gym": agent_gym,
        "keypoints_ws": kp_ws,
        "particles_gym": particles_gym,
        "alignment": align,
    }
    return obs, debug


def _policy_image_rename_map(policy_path: str) -> dict[str, str]:
    """Map the gym-PushT image key onto the checkpoint's expected image feature key.

    ``PushTPolicy.load_policy`` derives the features it validates against from the built-in
    gym ``PushtEnv`` (a schema stand-in only — no gym image is ever fed to the policy), whose
    single camera is the generic ``observation.image`` (``OBS_IMAGE``). Our diffusion
    checkpoint, however, was trained on an SO101 dataset and expects a named camera feature
    (e.g. ``observation.images.front``). Reading the checkpoint config tells us that expected
    key so we can build ``{OBS_IMAGE: <expected>}``. Returns an empty dict if the checkpoint
    has no (or more than one) visual input feature, in which case no rename is inferred.
    """
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    visual_keys = [k for k, ft in cfg.input_features.items() if ft.type == FeatureType.VISUAL]
    if len(visual_keys) != 1:
        return {}
    return {OBS_IMAGE: visual_keys[0]}


def build_policy(policy_config_path: str) -> PushTPolicy:
    """Load a PushTPolicyConfig YAML and return an initialized PushTPolicy.

    Mirrors `build_planner`: reads the first config entry from the YAML file, constructs the
    policy (which loads the referenced pretrained diffusion checkpoint), and returns it.

    ``PushTPolicy.load_policy`` calls ``make_policy(env_cfg=PushtEnv())`` without a
    ``rename_map``, so ``make_policy`` validates the checkpoint's expected image feature
    (``observation.images.<cam>``) against the gym env's schema key (``observation.image``).
    That check is spurious for this offline planning flow: we never step the gym env, and the
    policy is built entirely from the checkpoint config — the planner and policy just share the
    dataset-sampled plan-time observation. We temporarily wrap the ``make_policy`` symbol that
    ``pusht_policy_utils`` calls to inject the image ``rename_map``, which aligns the keys and
    makes ``make_policy`` skip that offline validation. At inference,
    ``sample_policy_seed_sequences`` still feeds a ``pixels={"<cam>": img}`` observation that
    ``preprocess_observation`` maps to the expected ``observation.images.<cam>`` key.
    """
    configs = PushTPolicyConfig.from_yaml(policy_config_path)
    if not configs:
        raise ValueError(f"No policy configs found in {policy_config_path}")

    rename_map = _policy_image_rename_map(configs[0].policy_path)
    original_make_policy = pusht_policy_utils.make_policy

    def _make_policy_with_rename(*args, **kwargs):
        kwargs.setdefault("rename_map", rename_map)
        return original_make_policy(*args, **kwargs)

    pusht_policy_utils.make_policy = _make_policy_with_rename
    try:
        return PushTPolicy(configs[0])
    finally:
        pusht_policy_utils.make_policy = original_make_policy


def build_planner(planner_config_path: str, horizon=None) -> PushTPlanner:
    """Load the planner config YAML and return an initialized PushTPlanner.

    When the config plans in joint space (``action_dim != 2``), the keypoint model is
    wrapped with a JointSpaceKeypointModel so the MPC's joint-space waypoints are mapped
    through FK -> gym_pusht XY before the dynamics model sees them.

    `horizon`, when given, overrides the planned trajectory length: it sets ``config.horizon``
    and the ``traj_len`` of both MPC stages, so the planner optimizes a `horizon`-action
    trajectory instead of the config default.
    """
    configs = PushTPlannerConfig.from_yaml(planner_config_path)
    if not configs:
        raise ValueError(f"No planner configs found in {planner_config_path}")
    config = configs[0]
    if config.model_type != "keypoint":
        raise ValueError(
            f"Expected a keypoint planner config, got model_type={config.model_type!r}"
        )
    if horizon is not None:
        config.horizon = horizon
        config.mpc_config = {**config.mpc_config, "traj_len": horizon}
        config.second_stage_mpc_config = {**config.second_stage_mpc_config, "traj_len": horizon}
    planner = PushTPlanner(config)
    if config.action_dim != 2:
        install_joint_space_model(planner, n_joints=config.action_dim)
    return planner


def demo_center_cost(states, actions, obs):
    """Placeholder planning cost: distance of the predicted final T-keypoint centroid to the
    gym_pusht window center, per trajectory.

    Called by the MPC as ``cost(simulated_paths, action_trajectories, obs)`` where
    ``simulated_paths`` has shape ``(n_traj, traj_len + 1, n_particles, 2)``. Operates only on
    the predicted T paths, so it is independent of the (joint-space) action dimension and
    returns the ``(n_traj,)`` array the MPC expects. Stands in for the real, joint-space-aware
    task/collision costs that still need to be written.
    """
    states = np.asarray(states)
    final_keypoints = states[:, -1]  # (n_traj, n_particles, 2)
    centroid = final_keypoints.mean(axis=1)  # (n_traj, 2)
    target = np.array([GYM_PUSHT_WINDOW_CENTER, GYM_PUSHT_WINDOW_CENTER])
    return np.linalg.norm(centroid - target, axis=-1)


def _iter_history_stages(history):
    """Normalize the various history shapes into an ordered list of ``(stage_label, hist)``.

    Accepts a single get_action history dict, the ``{"modal": ..., "refinement": ...}`` dict
    returned by ``two_stage_optimizer(return_history=True)``, or a list of history dicts.
    """
    if isinstance(history, dict) and "action_sequences" in history:
        return [("", history)]
    if isinstance(history, dict):
        # e.g. {"modal": ..., "refinement": ...} -- keep insertion order.
        return [(str(k), v) for k, v in history.items()]
    return [(f"stage{i}", h) for i, h in enumerate(history)]


def _iteration_costs_to_numpy(costs_it):
    """A single iteration's costs may be a torch tensor or ndarray; return a 1-D ndarray."""
    if hasattr(costs_it, "detach"):
        costs_it = costs_it.detach().cpu().numpy()
    return np.ravel(np.asarray(costs_it, dtype=float))


def visualize_optimization_history(
    history,
    planner,
    obs,
    max_trajectories_per_iter=40,
    title="Planner optimization history (gym_pusht frame)",
    out_html=None,
):
    """Build a plotly figure of the planner's optimization history in the gym_pusht frame.

    One slider step per optimization iteration; within each step every sampled trajectory's
    action sequence is drawn as a line in gym_pusht coordinates, colored by that trajectory's
    cost (shared color scale across all iterations).

    The stored action sequences are *delta joint* trajectories, so each is turned into absolute
    joints (``obs["agent_pos"] + cumsum``) and then FK-mapped into the gym frame via the
    planner's JointSpaceKeypointModel wrapper -- the same transform the model itself applies.

    `history` may be a single get_action history dict, the ``{"modal", "refinement"}`` dict
    from ``two_stage_optimizer(return_history=True)``, or a list of history dicts. Returns the
    figure (and writes it to `out_html` when given).
    """
    wrapper = planner.model
    if not isinstance(wrapper, JointSpaceKeypointModel):
        raise TypeError(
            "visualize_optimization_history expects a joint-space planner "
            "(planner.model must be a JointSpaceKeypointModel)"
        )
    agent_pos = np.asarray(obs["agent_pos"], dtype=float)
    t_bar_width_meters = obs["t_bar_width_meters"]

    # Collect per-iteration (label, gym trajectories (n, T, 2), per-trajectory cost (n,)).
    iterations = []
    for stage_label, hist in _iter_history_stages(history):
        for it, (seqs, costs_it) in enumerate(zip(hist["action_sequences"], hist["costs"])):
            deltas = np.asarray(seqs, dtype=float)
            if deltas.ndim == 2:
                deltas = deltas[np.newaxis]
            costs_arr = _iteration_costs_to_numpy(costs_it)
            n = deltas.shape[0]
            # EvoSearchMpc stores elite actions (fewer) but all costs; elites are the lowest
            # costs and are stored in ascending-cost order, so take the n smallest to align.
            if len(costs_arr) == n:
                traj_costs = costs_arr
            elif len(costs_arr) > n:
                traj_costs = np.sort(costs_arr)[:n]
            else:
                traj_costs = np.resize(costs_arr, n)
            if max_trajectories_per_iter and n > max_trajectories_per_iter:
                keep = np.linspace(0, n - 1, max_trajectories_per_iter).astype(int)
                deltas, traj_costs = deltas[keep], traj_costs[keep]
            abs_joints = agent_pos + np.cumsum(deltas, axis=1)
            gym = wrapper.joint_actions_to_gym_xy(abs_joints, t_bar_width_meters)
            label = f"{stage_label} it {it}" if stage_label else f"it {it}"
            iterations.append((label, gym, traj_costs))

    if not iterations:
        raise ValueError("Optimization history contained no iterations to plot")

    all_costs = np.concatenate([c for _, _, c in iterations])
    cmin, cmax = float(all_costs.min()), float(all_costs.max())
    cspan = (cmax - cmin) or 1.0

    traces = []
    # Trace 0: an invisible marker carrying the shared colorbar.
    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(color=[cmin], colorscale="Viridis", cmin=cmin, cmax=cmax,
                    showscale=True, colorbar=dict(title="cost")),
    ))
    # Trace 1: the detected T keypoints (always visible, for reference).
    particles = np.asarray(obs["particles"], dtype=float).reshape(-1, 2)
    t_outline = np.vstack([particles, particles[:1]])
    traces.append(go.Scatter(
        x=t_outline[:, 0], y=t_outline[:, 1], mode="lines+markers",
        line=dict(color="black", width=2), marker=dict(size=5, color="black"),
        name="T keypoints", showlegend=True, hoverinfo="skip",
    ))
    n_static = len(traces)

    iteration_trace_ranges = []
    for it_idx, (_, gym, traj_costs) in enumerate(iterations):
        start = len(traces)
        norm = (traj_costs - cmin) / cspan
        colors = get_color("Viridis", norm)
        for j in range(gym.shape[0]):
            traces.append(go.Scatter(
                x=gym[j, :, 0], y=gym[j, :, 1], mode="lines+markers",
                line=dict(color=colors[j], width=1.5), marker=dict(size=4, color=colors[j]),
                name=f"traj {j} (cost {traj_costs[j]:.2f})", showlegend=False,
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

    # Axis ranges cover both the trajectories and the gym window, with equal aspect.
    all_xy = np.concatenate(
        [np.concatenate([g.reshape(-1, 2) for _, g, _ in iterations], axis=0), particles,
         np.array([[0, 0], [GYM_PUSHT_WINDOW, GYM_PUSHT_WINDOW]])], axis=0)
    pad = 20.0
    x_range = [float(all_xy[:, 0].min()) - pad, float(all_xy[:, 0].max()) + pad]
    y_range = [float(all_xy[:, 1].min()) - pad, float(all_xy[:, 1].max()) + pad]

    fig = go.Figure(data=traces)
    # gym_pusht window outline for reference.
    fig.add_shape(type="rect", x0=0, y0=0, x1=GYM_PUSHT_WINDOW, y1=GYM_PUSHT_WINDOW,
                  line=dict(color="#bbb", width=1, dash="dash"))
    fig.update_layout(
        title=f"{title} — {iterations[0][0]}",
        sliders=[dict(active=0, steps=steps, currentvalue={"prefix": "iteration: "})],
        xaxis=dict(title="gym x", range=x_range, constrain="domain"),
        yaxis=dict(title="gym y", range=y_range, scaleanchor="x", scaleratio=1.0),
        showlegend=True,
        margin=dict(l=40, r=10, t=50, b=10),
    )
    if out_html is not None:
        fig.write_html(out_html)
        print(f"wrote optimization-history figure: {out_html}")
    return fig


def save_optimization_history_video(fig, out_path="optimization_history.mp4", fps=2,
                                    width=900, height=800, scale=1):
    """Render each slider step of an optimization-history figure to a frame and write an mp4.

    Reuses the figure's slider steps (one per optimization iteration): for each step the
    corresponding trace-visibility and title are applied to a slider-less copy of the figure,
    which is rasterized (requires the ``kaleido`` plotly image backend) and stacked into a
    video written with imageio/ffmpeg.
    """
    import imageio.v3 as iio

    sliders = fig.layout.sliders
    if not sliders or not sliders[0].steps:
        raise ValueError("figure has no slider steps to render into a video")
    steps = sliders[0].steps

    # Work on a copy so the interactive figure is untouched; drop the slider UI from frames.
    frame_fig = go.Figure(fig)
    frame_fig.update_layout(sliders=[])

    frames = []
    for step in steps:
        visibility = step.args[0]["visible"]
        for trace, visible in zip(frame_fig.data, visibility):
            trace.visible = visible
        if len(step.args) > 1 and "title" in step.args[1]:
            frame_fig.update_layout(title=step.args[1]["title"])
        png = frame_fig.to_image(format="png", width=width, height=height, scale=scale)
        frames.append(iio.imread(png))

    iio.imwrite(out_path, np.stack(frames), fps=fps, codec="libx264")
    print(f"wrote optimization-history video: {out_path} ({len(frames)} frames @ {fps} fps)")
    return out_path


def sample_policy_seed_sequences(policy, obs, batch_size, traj_len):
    """Sample `batch_size` joint-space seed trajectories from the diffusion policy.

    Reads the current joint state from ``obs["agent_pos"]`` and the camera image from
    ``obs["pixels"]`` (stashed by ``build_plan_time_obs``). Calls ``policy.infer_target``
    ``batch_size`` times — the underlying diffusion model draws fresh noise on each call, so
    each call yields an independent action-sequence sample. The returned actions are absolute
    6-DOF joint targets of length ``horizon``; we truncate to ``traj_len`` and convert into
    the MPC's delta convention:

        delta[0]   = target[0] - agent_pos
        delta[t>0] = target[t] - target[t-1]

    matching how ``run_one_planning_step`` seeds random deltas today (the MPC integrates the
    delta trajectory as ``agent_pos + cumsum(delta)`` to recover absolute joint waypoints).

    Returns a numpy array of shape ``(batch_size, traj_len, action_dim)``.
    """
    agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)  # (n_joints,)
    samples = []
    for _ in range(batch_size):
        policy_obs = {"agent_pos": agent_pos, "pixels": obs["pixels"]}
        actions = policy.infer_target(policy_obs)
        actions = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
        # `predict_action_sequence` transposes to (horizon, batch, action_dim); with batch=1
        # after `preprocess_observation` auto-batches the obs, we get (horizon, 1, action_dim).
        actions = np.squeeze(actions, axis=1) if actions.ndim == 3 and actions.shape[1] == 1 else actions
        samples.append(actions)
    samples = np.stack(samples, axis=0)  # (batch_size, horizon, action_dim)
    if samples.shape[1] > traj_len:
        samples = samples[:, :traj_len]
    prev = np.broadcast_to(agent_pos[None, None, :], (samples.shape[0], 1, agent_pos.shape[0]))
    return np.diff(samples, axis=1, prepend=prev)


def run_one_planning_step(
    planner, obs, costs=None, policy=None, seed_scale=1.0, rng_seed=0, return_history=False
):
    """Run a single two-stage MPC optimization from `obs`.

    Returns ``(actions, paths)``, or ``(actions, paths, history)`` when ``return_history`` is
    set (``history`` is ``{"modal": ..., "refinement": ...}``, each a get_action history dict).

    Seeding: when `policy` is provided, seed trajectories come from
    ``sample_policy_seed_sequences`` (fresh diffusion samples from the trained pushT policy,
    converted to joint-space deltas). Otherwise falls back to small Gaussian joint-delta noise
    of shape ``(num_sim_traj, traj_len, action_dim)`` — EvoSearchMpc needs non-empty seeds and
    the difftree policy's built-in 2-D radial seeding doesn't apply to joint-space planning.

    `costs` defaults to a single placeholder ``demo_center_cost`` on the predicted T paths.
    The planner's packaged ``collision_avoidance_constraint`` assumes 2-D XY actions and breaks
    on 6-DOF joint action trajectories, so the real task/collision costs still need a
    joint-space rewrite; the placeholder operates only on the predicted T keypoints, so it is
    action-dim-agnostic and gives the MPC a real per-trajectory signal while exercising the
    full seed -> FK-wrapper -> keypoint-dynamics pipeline.
    """
    if costs is None:
        costs = [PushTPlanningConstraint(demo_center_cost, weight=1.0)]
    mpc = planner.modal_mpc
    if policy is not None:
        seeds = sample_policy_seed_sequences(policy, obs, mpc.num_sim_traj, mpc.traj_len)
    else:
        rng = np.random.default_rng(rng_seed)
        seeds = rng.normal(scale=seed_scale, size=(mpc.num_sim_traj, mpc.traj_len, planner.config.action_dim))
    return two_stage_optimizer(
        planner,
        obs,
        costs,
        env=None,
        seed_planner=False,
        seed_sequences=seeds,
        return_history=return_history,
    )


def _policy_visual_key(policy) -> str:
    """Return the diffusion policy's single visual input-feature key (e.g. ``observation.images.front``).

    ``policy_objective`` builds the diffusion model's ``observation.images`` batch tensor from
    the camera image ``preprocess_observation`` produced. The difftree original hardcodes the
    gym key ``observation.image``; here we read the actual key off the checkpoint config so it
    also works for a dataset-trained ``observation.images.<cam>`` policy.
    """
    visual_keys = [k for k, ft in policy.policy.config.input_features.items() if ft.type == FeatureType.VISUAL]
    if len(visual_keys) != 1:
        raise ValueError(f"Expected exactly one visual input feature, got {visual_keys}")
    return visual_keys[0]


def _policy_denoise_loss(policy, batch_obs, timestep, actions_raw):
    """One diffusion denoising pass -> per-trajectory mean L2 distance to ``actions_raw``.

    Runs ``compute_loss(..., return_prediction=True)`` at the given `timestep`, unnormalizes the
    denoised prediction with the policy postprocessor, and returns ``(loss, prediction)`` where
    ``loss`` is the ``(bs,)`` mean-over-horizon L2 distance between prediction and the raw
    (unnormalized) actions. ``prediction`` is returned for the ``return_prediction`` path.
    """
    _, prediction = policy.policy.diffusion.compute_loss(
        batch_obs, timestep, over_batch=True, return_prediction=True
    )
    prediction = policy.postprocessor(prediction)
    actions_t = torch.as_tensor(actions_raw, dtype=prediction.dtype, device=prediction.device)
    loss = torch.linalg.norm(prediction.detach() - actions_t, dim=-1).cpu().numpy()
    return np.mean(loss, axis=1), prediction


def policy_objective(policy, states, actions, obs, diff_timestep=1, return_prediction=False, chunk_size=8):
    """Diffusion-policy planning cost: how far the MPC's joint action trajectories sit from what
    the trained policy would denoise them toward, per trajectory.

    Standalone port of ``PushTPolicy.policy_objective`` reconciled with this script's setup: the
    former ``self`` is the explicit `policy` argument, and the camera image is read from the
    policy's actual visual feature key (``observation.images.<cam>``) via ``_policy_visual_key``
    instead of the difftree original's hardcoded gym key ``observation.image``.

    Called by the MPC (through ``PushTPlanningConstraint.cost``) as
    ``cost(simulated_paths, action_trajectories, obs)``:
      * ``states``  -- predicted T-keypoint paths; unused, kept for the cost signature.
      * ``actions`` -- absolute joint waypoints ``(n_traj, traj_len, action_dim)`` the MPC formed
        as ``obs["agent_state"] + cumsum(deltas)``. The SO101 policy's action space is these same
        6-DOF joint targets, so the diffusion prediction and ``actions`` live in one space and the
        per-step L2 distance below is meaningful.

    The diffusion forward is evaluated over the ``n_traj`` trajectories in mini-batches of
    ``chunk_size`` (each conditioned on the same single observation) rather than one giant batch:
    encoding ``n_traj x n_obs_steps`` images at full resolution through the vision backbone twice
    (normal + modal) otherwise peaks well past GPU memory (CUDA OOM). Everything runs under
    ``no_grad`` since the costs only rank trajectories for the gradient-free EvoSearch MPC.

    Returns a ``(n_traj,)`` cost array (mean over the horizon of the L2 distance between the
    policy's denoised prediction and ``actions``, plus a 0.1-weighted higher-noise "modal" term),
    or ``(cost, prediction)`` when ``return_prediction`` is set.
    """
    visual_key = _policy_visual_key(policy)
    actions = np.asarray(actions)
    n_traj = actions.shape[0]

    # Normalize the (single) observation + all actions once; the per-chunk batches below just
    # slice/repeat these. n_obs_steps == 2 for this policy (repeat_interleave(2, dim=1)).
    processed_obs = policy.preprocess_observation(obs)
    processed_obs["action"] = torch.from_numpy(actions).cuda().float()
    processed_obs = policy.preprocessor(processed_obs)
    obs_state = processed_obs["observation.state"].unsqueeze(1).repeat_interleave(2, dim=1).cuda().float()
    obs_image = (
        processed_obs[visual_key].unsqueeze(1).unsqueeze(1).repeat_interleave(2, dim=1).cuda().float()
    )
    norm_actions = processed_obs["action"]

    losses, modal_losses, predictions = [], [], []
    with torch.no_grad():
        for i in range(0, n_traj, chunk_size):
            sl = slice(i, i + chunk_size)
            bs = norm_actions[sl].shape[0]
            batch_obs = {
                "observation.state": obs_state.repeat_interleave(bs, dim=0),
                "observation.images": obs_image.repeat_interleave(bs, dim=0),
                "action": norm_actions[sl],
                "action_is_pad": torch.zeros(actions[sl].shape[:2], dtype=torch.bool),
            }
            timestep = torch.ones((bs,), dtype=torch.int64).cuda() * diff_timestep
            loss, prediction = _policy_denoise_loss(policy, batch_obs, timestep, actions[sl])
            modal_loss, _ = _policy_denoise_loss(policy, batch_obs, timestep * 10, actions[sl])
            losses.append(loss)
            modal_losses.append(modal_loss)
            if return_prediction:
                predictions.append(prediction.detach().cpu())

    loss = np.concatenate(losses) + 0.1 * np.concatenate(modal_losses)
    if return_prediction:
        return loss, torch.cat(predictions, dim=0)
    return loss


def sample_policy_directed_planner(
    planner,
    policy,
    obs,
    base_costs=None,
    policy_objective_weight=0.05,
    seed_scale=1.0,
    rng_seed=0,
    return_history=False,
):
    """Plan with a set of base planning costs AND the diffusion policy objective.

    Independent-helper port of ``PushTCoupledPolicy.sample_policy_directed_planner``: the former
    ``self.planner``/``self.policy`` are the explicit `planner`/`policy` arguments. It recreates
    the difftree coupling

        coupled_costs = base_costs + [PushTPlanningConstraint(policy_objective, 0.05)]

    -- the base planning costs plus the trained diffusion policy's objective as an added soft cost
    (default weight ``0.05``) -- then runs the two-stage MPC over those costs. The policy objective
    is the local, joint-space-reconciled ``policy_objective`` (bound to `policy`), not
    ``PushTPolicy.policy_objective`` (which hardcodes the gym image key).

    `base_costs` defaults to ``planner.planning_constraints`` (matching the difftree original).
    NOTE: for the joint-space planner those packaged constraints (e.g. ``collision_avoidance_constraint``)
    assume 2-D XY actions and crash on 6-DOF joint trajectories, so a joint-space-compatible cost
    (e.g. the placeholder ``demo_center_cost``) must be passed explicitly until they are rewritten.

    Seeding differs from the difftree original: rather than the 2-D radial ``seed_planner=True``
    path (which doesn't apply to joint-space planning), it seeds through this script's
    ``run_one_planning_step`` convention (fresh diffusion samples -> joint-space deltas).

    Returns ``(actions, paths)``, or ``(actions, paths, history)`` when ``return_history`` is set.
    """
    if base_costs is None:
        base_costs = planner.planning_constraints
    coupled_costs = list(base_costs * 0) + [
        PushTPlanningConstraint(partial(policy_objective, policy), policy_objective_weight)
    ]
    return run_one_planning_step(
        planner,
        obs,
        costs=coupled_costs,
        policy=policy,
        seed_scale=seed_scale,
        rng_seed=rng_seed,
        return_history=return_history,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner-config",
        default=DEFAULT_PLANNER_CONFIG,
        help="Path to the PushTPlannerConfig YAML file.",
    )
    parser.add_argument(
        "--policy-config",
        default=DEFAULT_POLICY_CONFIG,
        help="Path to the PushTPolicyConfig YAML file (loads the diffusion policy checkpoint).",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="If set, build a plan-time observation from the first frame of this dataset.",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode to sample the start state from.")
    parser.add_argument("--cam-key", default=None, help="Camera key to use for T detection (default: try all).")
    parser.add_argument("--sam-weights", default=DEFAULT_SAM_WEIGHTS, help="SAM2 weights path.")
    parser.add_argument("--sam-device", default="cuda", help="Device for SAM2 (cuda/cpu).")
    parser.add_argument(
        "--n-actions",
        type=int,
        default=16,
        help="Number of actions the planner optimizes over in the single planning call.",
    )
    parser.add_argument(
        "--policy-directed",
        action="store_true",
        help="Couple the diffusion policy objective into the planning cost "
        "(sample_policy_directed_planner) instead of the placeholder demo_center_cost.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Also save the optimization-history slider frames as an mp4 in the cwd.",
    )
    parser.add_argument("--video-fps", type=float, default=2.0, help="Frames-per-second for --save-video.")
    args = parser.parse_args()

    # Plan over `--n-actions` actions by overriding the trajectory length of both MPC stages.
    planner = build_planner(args.planner_config, horizon=args.n_actions)
    print(f"Initialized PushT keypoint planner from {args.planner_config}")
    print(f"  mpc_type: {planner.mpc_type}")
    print(f"  horizon:  {planner.horizon}")
    print(f"  model_type: {planner.config.model_type}")
    print(f"  action_dim: {planner.config.action_dim}")
    print(f"  joint_space_wrapped: {isinstance(planner.model, JointSpaceKeypointModel)}")
    print(f"  planning_constraints: {len(planner.planning_constraints)}")

    policy = build_policy(args.policy_config)
    print(f"Initialized PushT policy from {args.policy_config}")
    print(f"  policy_path: {policy.policy_path}")
    print(f"  horizon:     {policy.horizon}")

    if args.repo_id is not None:
        obs, debug = build_plan_time_obs(
            args.repo_id,
            episode=args.episode,
            sam_weights=args.sam_weights,
            sam_device=args.sam_device,
            cam_key=args.cam_key,
        )
        print(f"\nPlan-time observation from {args.repo_id} episode {args.episode}:")
        print(f"  T detected on camera: {debug['camera']}")
        print(f"  agent_pos (6-DOF joints): {np.round(obs['agent_pos'], 3)}")
        print(f"  info.pos_agent (gym XY):  {np.round(obs['info']['pos_agent'], 2)}")
        print(f"  t_bar_width_meters:       {obs['t_bar_width_meters']:.4f}")
        print(f"  particles (gym, {obs['particles'].shape}):")
        for name, pt in zip(GYM_KEYPOINT_ORDER, obs["particles"]):
            print(f"    {name:>10}: {np.round(pt, 1)}")

        # Call the planner over `--n-actions` actions from the sampled start state, seeding
        # the two-stage MPC with fresh samples from the loaded diffusion policy. With
        # `--policy-directed`, the diffusion policy objective is coupled into the planning cost
        # (planner constraints + policy_objective); otherwise the placeholder demo_center_cost
        # is used.
        cost_mode = "policy-directed (planner constraints + policy objective)" if args.policy_directed else "placeholder demo_center_cost"
        print(f"\nPlanning over {args.n_actions} actions from the start state [{cost_mode}]...")
        if args.policy_directed:
            # The planner's packaged constraints are 2-D-only and crash on 6-DOF joint actions,
            # so couple the policy objective with the joint-space-compatible placeholder cost
            # instead of `planner.planning_constraints` (see sample_policy_directed_planner NOTE).
            actions, paths, history = sample_policy_directed_planner(
                planner,
                obs=obs,
                policy=policy,
                base_costs=[PushTPlanningConstraint(demo_center_cost, weight=1.0)],
                return_history=True,
            )
        else:
            actions, paths, history = run_one_planning_step(
                planner, obs, policy=policy, return_history=True
            )
        actions = np.asarray(actions)
        paths = np.asarray(paths)
        print(f"  best action trajectory (joint space) shape: {actions.shape}")
        print(f"  predicted T-keypoint path shape:            {paths.shape}")
        print(f"  first planned joint waypoint:  {np.round(actions[0], 3)}")
        print(f"  last planned joint waypoint:   {np.round(actions[-1], 3)}")

        # Visualize the optimization history of that planning call.
        fig = visualize_optimization_history(history, planner, obs)
        fig.show()
        if args.save_video:
            save_optimization_history_video(fig, fps=args.video_fps)
        return planner, obs, (actions, paths), history

    return planner


if __name__ == "__main__":
    main()
