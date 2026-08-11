from functools import partial

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_PREFIX
from lerobot.utils.random_utils import set_seed
from visualize_arm_position import RobotState, plot_pose
import numpy as np

CKPT = "/home/rthomp12/diff_lerobot/outputs/train/diffusion_so101_adjusted_32/checkpoints/last/pretrained_model"

"""
validation action MSE (100 samples, Sorozco0612/new-test-data): 128.536481
training   action MSE (100 samples, Sorozco0612/record-test): 16.745604

validation action MSE (100 samples, Sorozco0612/new-test-data): 36.945628
training action MSE (100 samples, Sorozco0612/record-test): 2.222926

"""

CKPT = "/home/rthomp12/diff_lerobot/outputs/train/diffusion_so101_record_test_64/checkpoints/last/pretrained_model"
DEVICE = "cuda"
SAMPLE_IDX = 100
SEED = 1000
EVAL_REPO_ID = "Sorozco0612/new-test-data"
# Number of action steps the diffusion policy predicts. None -> use the model's trained
# config.horizon. Must be a multiple of the U-Net downsampling factor (2**(len(down_dims)-1)).
DIFFUSION_HORIZON = 32


def load_eval_dataset(cfg, repo_id, horizon=DIFFUSION_HORIZON, root=None, episodes=None):
    """Build a dataset from `repo_id` while keeping the temporal stacking the policy expects.

    The policy and its pre/post processors are still created from the original training dataset
    (so normalization stats and feature shapes match the checkpoint). Only the evaluation samples
    come from `repo_id`. The delta_timestamps are resolved from the policy config against this
    dataset's metadata, so each sample gets the same observation history as in training.

    `horizon` controls how many ground-truth action steps each sample carries. When None it follows
    the policy's trained config.horizon; otherwise the ACTION delta_timestamps are overridden to
    span exactly `horizon` steps (so the retrieved ground truth matches the diffusion prediction
    horizon, even when it exceeds the trained horizon).
    """
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root, revision=cfg.dataset.revision)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    if horizon is not None:
        # Mirror DiffusionConfig.action_delta_indices but with `horizon` steps instead of the
        # trained config.horizon, so the ground-truth action length equals the prediction horizon.
        n_obs = cfg.policy.n_obs_steps
        action_indices = range(1 - n_obs, 1 - n_obs + horizon)
        delta_timestamps = dict(delta_timestamps or {})
        delta_timestamps[ACTION] = [i / ds_meta.fps for i in action_indices]

    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
    )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def diffusion_horizon_factor(policy):
    """Sequence length the diffusion U-Net requires the action horizon to be a multiple of.

    The conv U-Net downsamples once per `down_dims` transition (i.e. len(down_dims) - 1 times),
    halving the sequence length each time, so the horizon must be divisible by 2**(that count) for
    the skip connections to line up."""
    return 2 ** (len(policy.config.down_dims) - 1)


def predict_actions(policy, preprocessor, postprocessor, sample, horizon=None):
    """Run the full diffusion sampling loop on one sample, returning (pred_actions, gt_action) in
    dataset units, both shaped (n_steps, action_dim) on CPU.

    `horizon` sets the number of action steps the diffusion policy predicts. When None, the model's
    trained `config.horizon` is used; otherwise we seed `conditional_sample` with noise of that
    length (it must be a multiple of `diffusion_horizon_factor(policy)`). The ground truth is
    truncated to the same number of steps so the two stay index-aligned."""
    gt_action = sample[ACTION].clone()
    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in sample.items()}
    batch = preprocessor(batch)

    if policy.config.image_features:
        batch[OBS_IMAGES] = torch.stack([batch[k] for k in policy.config.image_features], dim=-4)

    noise = None
    if horizon is not None:
        factor = diffusion_horizon_factor(policy)
        if horizon % factor != 0:
            raise ValueError(f"horizon={horizon} must be a multiple of {factor} for this U-Net.")
        action_dim = policy.config.action_feature.shape[0]
        noise = torch.randn(
            1, horizon, action_dim, device=DEVICE, dtype=next(policy.diffusion.parameters()).dtype
        )

    with torch.no_grad():
        global_cond = policy.diffusion._prepare_global_conditioning(batch)
        norm_actions = policy.diffusion.conditional_sample(batch_size=1, global_cond=global_cond, noise=noise)

    pred_actions = postprocessor(norm_actions).squeeze(0).to("cpu")
    gt_action = gt_action.to("cpu")
    n = min(len(pred_actions), len(gt_action))  # align lengths if horizon != dataset horizon
    return pred_actions[:n], gt_action[:n]


def velocity_mse(actions_a, actions_b):
    """MSE between the per-step velocities of two action sequences.

    Each input is shaped (n_steps, action_dim). The velocity at step t is the L2 distance between
    consecutive actions (||a[t+1] - a[t]||), giving a length-(n_steps - 1) velocity profile per
    sequence. We compare the two profiles elementwise and return the mean squared error (a Python
    float). The sequences are truncated to a common length first, so they may differ in horizon."""
    n = min(len(actions_a), len(actions_b))
    actions_a, actions_b = actions_a[:n], actions_b[:n]

    vel_a = (actions_a[1:] - actions_a[:-1]).norm(dim=-1)  # (n - 1,) per-step speed
    vel_b = (actions_b[1:] - actions_b[:-1]).norm(dim=-1)
    return (vel_a - vel_b).pow(2).mean().item()


def evaluate_action_mse(
    policy, preprocessor, postprocessor, dataset, n_samples=100, seed=SEED, predict_fn=predict_actions
):
    """Mean action-prediction MSE over `n_samples` random samples of `dataset`.

    For each sample we sample an action prediction via `predict_fn` (use `predict_actions` for a
    diffusion policy or `predict_actions_act` for an ACT policy) and compare it against the
    ground-truth action (in dataset units). Returns the mean MSE across samples (a Python float)."""
    n_samples = min(n_samples, len(dataset))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)

    #set_seed(seed)  # fix any sampling noise (e.g. diffusion prior) so the metric is reproducible
    total_mse = 0.0
    for i, idx in enumerate(indices):
        print(f"evaluating sample {i+1}/{n_samples} ({idx})")
        pred_actions, gt_action = predict_fn(policy, preprocessor, postprocessor, dataset[int(idx)])
        total_mse += (pred_actions - gt_action).pow(2).mean().item()

    return total_mse / n_samples


def evaluate_velocity_mse(
    policy,
    preprocessor,
    postprocessor,
    dataset,
    n_samples=100,
    seed=SEED,
    predict_fn=predict_actions,
    mse_threshold=1.0,
    urdf_path="/home/rthomp12/diff_lerobot/scripts/so101_new_calib.urdf",
    robot_id="bender_follower_arm",
):
    """Mean velocity MSE over `n_samples` random samples, then plot each sample's EE pose.

    For each sample we predict an action via `predict_fn` and compare its per-step velocity profile
    against the ground-truth action's via `velocity_mse`. We also stash each sample's
    `observation.state` (its current frame) alongside the resulting MSE.

    At the end we render the end-effector pose for every saved state with the same RobotState /
    link_fk / plot_pose pipeline used later in `main`. States whose velocity MSE exceeds
    `mse_threshold` are drawn in a distinct color so outliers stand out. Returns the mean velocity
    MSE across samples (a Python float)."""
    n_samples = min(n_samples, len(dataset))

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)

    total_mse = 0.0
    saved = []  # (observation.state current frame, velocity mse) per sample
    for i, idx in enumerate(indices):
        print(f"evaluating sample {i+1}/{n_samples} ({idx})")
        sample = dataset[int(idx)]
        pred_actions, gt_action = predict_fn(policy, preprocessor, postprocessor, sample)
        mse = velocity_mse(pred_actions, gt_action)
        total_mse += mse

        state = sample["observation.state"]
        if state.ndim > 1:
            state = state[-1]  # current frame of the n_obs_steps stack
        saved.append((state.to("cpu"), mse))

    import plotly.graph_objects as go

    fig = go.Figure()
    robot_state = RobotState(urdf_path=urdf_path, id=robot_id)

    base_pose = robot_state.robot_urdf.link_fk(cfg=np.zeros(6))[robot_state.robot_urdf.link_map["base_link"]]
    trans = base_pose[:3, 3]
    quat = robot_state.rot_matrix_to_quat(base_pose[:3, :3])
    combo = [trans[0], trans[1], trans[2], quat[0], quat[1], quat[2], quat[3]]
    fig = plot_pose(combo, axis_length=0.05, fig=fig, name="base_link")

    for t, (state, mse) in enumerate(saved):
        joint_positions = robot_state.convert_lerobot_action_to_radians(state)
        ee_pose = robot_state.robot_urdf.link_fk(cfg=joint_positions)[
            robot_state.robot_urdf.link_map["gripper_frame_link"]
        ]
        trans = ee_pose[:3, 3]
        quat = robot_state.rot_matrix_to_quat(ee_pose[:3, :3])
        combo = [trans[0], trans[1], trans[2], quat[0], quat[1], quat[2], quat[3]]

        above = mse > mse_threshold
        colors = ["red", "salmon", "lightcoral"] if above else ["pink", "lightgreen", "lightblue"]
        name = f"state_{t}_mse_{mse:.4f}" + ("_above" if above else "")
        fig = plot_pose(combo, axis_length=0.05, fig=fig, name=name, colors=colors)

    fig.show()

    return total_mse / n_samples


ACT_CKPT = "/home/rthomp12/diff_lerobot/outputs/pretrained_model"


def load_act_policy(ds_meta, policy_path=ACT_CKPT):
    """Load the ACT policy (and its pre/post processors) at `policy_path`.

    `load_policy` is generic (it dispatches on the checkpoint's config type via make_policy), so
    this is a thin, explicitly-named wrapper for the ACT checkpoint to keep its loading/sampling
    path separate from the diffusion one."""
    return load_policy(policy_path, ds_meta)


def predict_actions_act(policy, preprocessor, postprocessor, sample, current_idx=0):
    """Sample an action chunk from an ACT policy on one sample.

    ACT is not a diffusion policy: there's no iterative sampling loop. We just run the
    transformer once via `predict_action_chunk`, which also handles image stacking internally
    (so, unlike the diffusion path, we must NOT pre-stack OBS_IMAGES ourselves).

    The dataset samples carry the *diffusion* policy's temporal windowing, which differs from what
    ACT expects, so we adapt them here:
      - Observations come stacked over `n_obs_steps` frames; ACT wants a single current observation,
        so we collapse each observation key to its last (current) frame.
      - The ground-truth action stack starts `current_idx` steps before the current timestep (from
        the diffusion action_delta_indices) and only spans the diffusion horizon, while ACT predicts
        `chunk_size` steps from the current timestep onward. We align both to the current timestep
        and compare the overlapping window.

    Returns (pred_actions, gt_action) in dataset units, both (n_overlap, action_dim) on CPU."""
    gt_action = sample[ACTION].clone()

    sample = dict(sample)
    for key in list(sample):
        if key.startswith(OBS_PREFIX) and isinstance(sample[key], torch.Tensor):
            sample[key] = sample[key][-1]  # current frame of the n_obs_steps stack

    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in sample.items()}
    batch = preprocessor(batch)

    with torch.no_grad():
        norm_actions = policy.predict_action_chunk(batch)

    pred_actions = postprocessor(norm_actions).squeeze(0).to("cpu")
    gt_action = gt_action[current_idx:].to("cpu")  # drop pre-current steps so gt starts at t
    n_overlap = min(len(pred_actions), len(gt_action))
    return pred_actions[:n_overlap], gt_action[:n_overlap]


def load_policy(policy_path, ds_meta):
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    # Ensure the pretrained weights are loaded by make_policy.
    policy_config.pretrained_path = policy_path

    policy = make_policy(
        cfg=policy_config,
        ds_meta=ds_meta,
    )

    preprocessor_overrides = {
        "device_processor": {"device": DEVICE},
        "rename_observations_processor": {"rename_map": {}},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=policy_config.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    return policy, preprocessor, postprocessor


def main():
    #set_seed(SEED)

    # 1. Recover the exact training config (dataset + policy) from the checkpoint.
    cfg = TrainPipelineConfig.from_pretrained(CKPT)

    # 2. Rebuild the original training dataset. Its meta builds the policy and pre/post processors
    #    (so normalization stats and feature shapes match the checkpoint), and it also serves as a
    #    sampling source. Built horizon-aware so its ground-truth action length matches the
    #    diffusion prediction horizon.
    train_dataset = load_eval_dataset(
        cfg, cfg.dataset.repo_id, root=cfg.dataset.root, episodes=cfg.dataset.episodes
    )

    # 3. Load the policy and its pre/post processors from the *original* dataset meta.
    policy, preprocessor, postprocessor = load_policy(CKPT, train_dataset.meta)
    policy.eval()

    # 3b. Pull the evaluation samples from a *different* repo, keeping the temporal stacking
    #     the policy expects.
    dataset = load_eval_dataset(cfg, EVAL_REPO_ID)

    # 3c-i. Load the ACT policy (a different kind of policy) for comparison, with its own
    #       pre/post processors but the same dataset meta.
    act_policy, act_preprocessor, act_postprocessor = load_act_policy(train_dataset.meta)
    act_policy.eval()

    # 3c. Compare generalization: mean action-prediction MSE over 100 validation samples vs the
    #     same metric over the training dataset, for both the diffusion and ACT policies. A large
    #     gap (validation >> training) indicates overfitting. The dataset uses the diffusion
    #     policy's temporal windowing, so ACT aligns its prediction to the current timestep, which
    #     sits `n_obs_steps - 1` steps into the ground-truth action stack.
    act_predict_fn = partial(predict_actions_act, current_idx=cfg.policy.n_obs_steps - 1)
    diff_predict_fn = partial(predict_actions, horizon=DIFFUSION_HORIZON)

#     mean_vel_mse = evaluate_velocity_mse(
#     policy, preprocessor, postprocessor, dataset,
#     n_samples=50, predict_fn=diff_predict_fn, mse_threshold=3.0,
# )
    
#     exit(0)
    # val_mse = evaluate_action_mse(policy, preprocessor, postprocessor, dataset, n_samples=20, predict_fn=diff_predict_fn)
    # train_mse = evaluate_action_mse(policy, preprocessor, postprocessor, train_dataset, n_samples=20, predict_fn=diff_predict_fn)
    # act_val_mse = evaluate_action_mse(
    #     act_policy, act_preprocessor, act_postprocessor, dataset, n_samples=20, predict_fn=act_predict_fn
    # )
    # act_train_mse = evaluate_action_mse(
    #     act_policy, act_preprocessor, act_postprocessor, train_dataset, n_samples=20, predict_fn=act_predict_fn
    # )
    # print(f"[diffusion] validation action MSE (100 samples, {EVAL_REPO_ID}): {val_mse:.6f}")
    # print(f"[diffusion] training   action MSE (100 samples, {cfg.dataset.repo_id}): {train_mse:.6f}")
    # print(f"[diffusion] generalization gap (val - train): {val_mse - train_mse:.6f}")
    # print(f"[act]       validation action MSE (100 samples, {EVAL_REPO_ID}): {act_val_mse:.6f}")
    # print(f"[act]       training   action MSE (100 samples, {cfg.dataset.repo_id}): {act_train_mse:.6f}")
    # print(f"[act]       generalization gap (val - train): {act_val_mse - act_train_mse:.6f}")

    # 4-6. Grab one data point and predict its action horizon with the diffusion policy.
    #      predict_actions seeds conditional_sample with noise of length DIFFUSION_HORIZON (or the
    #      model's trained horizon when None) and returns the prediction + index-aligned ground
    #      truth, both in dataset units on CPU.
    SAMPLE_IDX = np.random.randint(0, 500)
    sample = dataset[SAMPLE_IDX]
    pred_actions, gt_action = diff_predict_fn(policy, preprocessor, postprocessor, sample)

    # 6b. Predict the same sample with the ACT policy for side-by-side visualization.
    act_pred_actions, _ = predict_actions_act(
        act_policy, act_preprocessor, act_postprocessor, sample, current_idx=cfg.policy.n_obs_steps - 1
    )

    import plotly.graph_objects as go
    fig = go.Figure()

    robot_state = RobotState(
        urdf_path="/home/rthomp12/diff_lerobot/scripts/so101_new_calib.urdf",
        id="bender_follower_arm",
    )


    base_pose = robot_state.robot_urdf.link_fk(cfg=np.zeros(6))[robot_state.robot_urdf.link_map["base_link"]]
    trans = base_pose[:3, 3]
    quat = robot_state.rot_matrix_to_quat(base_pose[:3, :3])
    combo = [trans[0], trans[1], trans[2], quat[0], quat[1], quat[2], quat[3]]
    fig = plot_pose(combo, axis_length=0.05, fig=fig, name=f"base_link")

    for t, initial_action in enumerate(gt_action):
        joint_positions = robot_state.convert_lerobot_action_to_radians(initial_action)
        ee_pose = robot_state.robot_urdf.link_fk(cfg=joint_positions)[robot_state.robot_urdf.link_map["gripper_frame_link"]]

        trans = ee_pose[:3, 3]
        quat = robot_state.rot_matrix_to_quat(ee_pose[:3, :3])
        combo = [trans[0], trans[1], trans[2], quat[0], quat[1], quat[2], quat[3]]
        fig = plot_pose(combo, axis_length=0.05, fig=fig, name=f"ground_truth_{t}")

    for t, initial_action in enumerate(pred_actions):
        joint_positions = robot_state.convert_lerobot_action_to_radians(initial_action)
        ee_pose = robot_state.robot_urdf.link_fk(cfg=joint_positions)[robot_state.robot_urdf.link_map["gripper_frame_link"]]

        trans = ee_pose[:3, 3]
        quat = robot_state.rot_matrix_to_quat(ee_pose[:3, :3])
        combo = [trans[0], trans[1], trans[2], quat[0], quat[1], quat[2], quat[3]]
        fig = plot_pose(combo, axis_length=0.05, fig=fig, name=f"pred_action_{t}", colors=["pink", "lightgreen", "lightblue"])

    for t, initial_action in enumerate(act_pred_actions):
        joint_positions = robot_state.convert_lerobot_action_to_radians(initial_action)
        ee_pose = robot_state.robot_urdf.link_fk(cfg=joint_positions)[robot_state.robot_urdf.link_map["gripper_frame_link"]]

        trans = ee_pose[:3, 3]
        quat = robot_state.rot_matrix_to_quat(ee_pose[:3, :3])
        combo = [trans[0], trans[1], trans[2], quat[0], quat[1], quat[2], quat[3]]
        fig = plot_pose(combo, axis_length=0.05, fig=fig, name=f"act_pred_action_{t}", colors=["orange", "yellow", "purple"])
    fig.show()



    # 7. Compare the full horizon, index-aligned (both indexed by action_delta_indices).
    err = pred_actions - gt_action
    per_step_l2 = err.norm(dim=-1)
    mse = err.pow(2).mean()
    mae = err.abs().mean()

    torch.set_printoptions(precision=4, sci_mode=False)
    print(f"eval dataset: {EVAL_REPO_ID} (policy/processors from {cfg.dataset.repo_id})")
    print(f"sample index: {SAMPLE_IDX}")
    print(f"horizon (action steps compared): {gt_action.shape[0]}")
    print(f"\nground-truth action (dataset units):\n{gt_action}")
    print(f"\npredicted action (inference, dataset units):\n{pred_actions}")
    print(f"\nper-step L2 error:\n{per_step_l2}")
    print(f"\noverall MSE: {mse.item():.6f}")
    print(f"overall MAE: {mae.item():.6f}")


if __name__ == "__main__":
    main()
