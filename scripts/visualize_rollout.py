"""Visualize rollouts in a LeRobot dataset (e.g. Sorozco0612/eval_so101_diffusion_batch_64).

For a chosen episode this produces two artifacts in `--out-dir`:

  1. An mp4 per camera built from the image observations (the raw rollout video).
  2. An interactive plotly figure plotting the robot's end-effector pose at each timestep,
     for both the ground-truth `observation.state` and the predicted `action` (both 6-DOF joint
     vectors run through forward kinematics).

Run inside the `lerobot` conda env, e.g.:
    python scripts/visualize_rollout.py \
        --repo-id Sorozco0612/eval_so101_diffusion_batch_64 --episode 0

The EE forward-kinematics path reuses the same RobotState / link_fk / plot_pose pipeline as
run_one_example.py, so the poses are in the same frame/units.
"""

import argparse
import base64
import io
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, no_update
from PIL import Image, ImageDraw, ImageFont

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from visualize_arm_position import RobotState, plot_pose

# Defaults mirror the URDF/robot used in run_one_example.py so the EE frame matches.
DEFAULT_URDF = "/home/rthomp12/diff_lerobot/scripts/so101_new_calib.urdf"
DEFAULT_ROBOT_ID = "bender_follower_arm"
EE_LINK = "gripper_frame_link"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
DEFAULT_SAM_WEIGHTS = "sam2_b.pt"
# HSV red is split across the hue wraparound (0 and 180), so we threshold both bands.
RED_HSV_RANGES = (((0, 150, 150), (12, 255, 255)),
                  ((170, 150, 150), (180, 255, 255)))
RED_MIN_AREA = 50  # px^2 — reject specks/noise as candidate centroids

# Canonical T mask + keypoints, in a fixed-size canvas. Dimensions follow the classic Push-T
# proportions (bar 4× wider than tall, stem 1×3 extending down from the bar's center). All
# keypoints are the 8 outer corners of the T outline, expressed in this canonical pixel frame.
# The align step fits (x, y, rotation, scale) to map these into the camera image.
CANONICAL_T_CANVAS = 200
CANONICAL_T_BAR = (120, 30)   # (width, height) of the horizontal bar
CANONICAL_T_STEM = (30, 90)   # (width, height) of the vertical stem

# HACK: linear pixel→XY workspace mapping used to overlay image-space detections onto the same
# XY plane as the EE trajectory. This is only correct if the camera happens to be a pure ortho
# top-down view exactly covering this rectangle — which it isn't. Replace with a proper
# unprojection (camera intrinsics + extrinsics + table plane) once calibration is available.
WORKSPACE_XY_BOUNDS = (-0.5, 0.0, 0.5, 0.75)  # (xmin, ymin, xmax, ymax)


def episode_frame_range(dataset, ep_idx):
    """Return the (from, to) global frame indices for episode `ep_idx`.

    LeRobot stores per-episode frame spans in the episode metadata as `dataset_from_index` /
    `dataset_to_index` (half-open), so the episode's frames are dataset[from:to]."""
    ep = dataset.meta.episodes[int(ep_idx)]
    return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])


def frame_to_uint8_hwc(img):
    """Convert a camera observation to an HxWx3 uint8 array for video writing.

    LeRobot camera keys are returned as CHW float tensors in [0, 1]; some datasets store raw uint8
    HWC instead. Handle both so the writer always sees uint8 HWC."""
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def make_episode_videos(dataset, ep_idx, out_dir, fps=None):
    """Write one mp4 per camera from an episode's image observations. Returns the written paths."""
    start, end = episode_frame_range(dataset, ep_idx)
    camera_keys = dataset.meta.camera_keys
    fps = fps or int(round(dataset.fps))

    written = []
    for cam in camera_keys:
        frames = [frame_to_uint8_hwc(dataset[i][cam]) for i in range(start, end)]
        safe_cam = cam.replace(".", "_").replace("/", "_")
        out_path = out_dir / f"episode_{int(ep_idx):03d}_{safe_cam}.mp4"
        iio.imwrite(out_path, np.stack(frames), fps=fps, codec="libx264")
        print(f"wrote video: {out_path} ({len(frames)} frames @ {fps} fps)")
        written.append(out_path)
    return written


def ee_pose_from_state(robot_state, state):
    """Forward-kinematics the EE pose for one `observation.state` (6 joint values).

    Returns (translation (3,), quaternion xyzw (4,)) in the URDF base frame."""
    state = state.numpy() if hasattr(state, "numpy") else np.asarray(state)
    joint_positions = robot_state.convert_lerobot_action_to_radians(state)
    ee = robot_state.robot_urdf.link_fk(cfg=joint_positions)[robot_state.robot_urdf.link_map[EE_LINK]]
    trans = ee[:3, 3]
    quat = robot_state.rot_matrix_to_quat(ee[:3, :3])
    return trans, quat


def joint_seq_ee_poses(robot_state, joint_seq):
    """FK a sequence of 6-DOF joint vectors to EE poses. Returns (positions (N,3), quats (N,4))."""
    positions, quats = [], []
    for js in joint_seq:
        trans, quat = ee_pose_from_state(robot_state, js)
        positions.append(trans)
        quats.append(quat)
    return np.asarray(positions), np.asarray(quats)


def add_ee_trajectory(fig, positions, quats, name, colorscale, axes_colors, pose_axes_stride,
                      axis_length=0.03):
    """Add a time-colored EE trajectory (line+markers, one point per timestep) plus subsampled
    orientation axes to `fig`."""
    steps = np.arange(len(positions))
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode="lines+markers",
            line=dict(color=steps, colorscale=colorscale, width=4),
            marker=dict(size=3, color=steps, colorscale=colorscale,
                        colorbar=dict(title=f"{name} timestep")),
            name=name,
        )
    )
    if pose_axes_stride:
        for t in range(0, len(positions), pose_axes_stride):
            fig = plot_pose([*positions[t], *quats[t]], axis_length=axis_length, fig=fig,
                            name=f"{name}_t{t}", colors=axes_colors)
    return fig


def frame_to_data_uri(img, fmt="JPEG", quality=85):
    """Encode an image observation as a base64 data URI so it can be shown in an html.Img."""
    arr = frame_to_uint8_hwc(img)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, quality=quality)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def pil_to_data_uri(pil_img, fmt="JPEG", quality=85):
    """Encode a PIL.Image as a base64 data URI."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def resolve_sam_device(device):
    """Resolve the requested device, falling back to CPU when CUDA isn't available."""
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("SAM: CUDA requested but not available — falling back to CPU.")
        return "cpu"
    return device


def find_red_centroid(pil_img, min_area=RED_MIN_AREA):
    """Return the (x, y) centroid of the largest red blob in `pil_img`, or None if none qualify.

    Red spans the HSV hue wraparound, so we OR two ranges. `min_area` rejects specks that are
    almost certainly camera noise rather than the T block."""
    import cv2
    rgb = np.asarray(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in RED_HSV_RANGES:
        mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if stats[idx, cv2.CC_STAT_AREA] < min_area:
        return None
    cx, cy = centroids[idx]
    return float(cx), float(cy)


def load_sam2(weights=DEFAULT_SAM_WEIGHTS, device="cuda"):
    """Lazy-load SAM2 so `ultralytics` import cost is only paid when segmentation is requested."""
    from ultralytics import SAM
    model = SAM(weights)
    model.to(device)
    return model


def segment_at_point(sam, pil_img, point, device="cuda"):
    """Run SAM2 with a single positive point prompt at `point` (x, y in pixels).

    Returns {"mask": (H, W) bool array, "xyxy": [x1, y1, x2, y2] from mask, "conf": float}
    or None if SAM produced nothing usable."""
    arr = np.asarray(pil_img.convert("RGB"))
    results = sam(source=arr, points=[[float(point[0]), float(point[1])]], labels=[1],
                  device=device, verbose=False)
    if not results or results[0].masks is None or len(results[0].masks.data) == 0:
        return None
    r = results[0]
    mask = r.masks.data[0].cpu().numpy().astype(bool)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x1, y1 = float(xs.min()), float(ys.min())
    x2, y2 = float(xs.max()), float(ys.max())
    conf = float(r.boxes.conf[0]) if r.boxes is not None and len(r.boxes) else 0.0
    return {"mask": mask, "xyxy": [x1, y1, x2, y2], "conf": conf}


def build_canonical_t():
    """Build the canonical T mask and its 8 outer-corner keypoints in a canonical pixel frame.

    Returns:
        (mask, keypoints) where mask is a uint8 (H, W) 0/255 array with the T at the canvas
        center, and keypoints is a dict {name: (x, y)} in the same coordinate frame."""
    H = W = CANONICAL_T_CANVAS
    bw, bh = CANONICAL_T_BAR
    sw, sh = CANONICAL_T_STEM
    mask = np.zeros((H, W), dtype=np.uint8)
    # Center the whole T (bar + stem) vertically on the canvas.
    total_h = bh + sh
    top = (H - total_h) // 2
    cx = W // 2
    bar_top, bar_bot = top, top + bh
    stem_top, stem_bot = bar_bot, bar_bot + sh
    mask[bar_top:bar_bot, cx - bw // 2:cx + bw // 2] = 255
    mask[stem_top:stem_bot, cx - sw // 2:cx + sw // 2] = 255
    keypoints = {
        "bar_tl": (cx - bw // 2, bar_top),
        "bar_tr": (cx + bw // 2, bar_top),
        "bar_br_out": (cx + bw // 2, bar_bot),
        "stem_tr": (cx + sw // 2, stem_top),
        "stem_br": (cx + sw // 2, stem_bot),
        "stem_bl": (cx - sw // 2, stem_bot),
        "stem_tl": (cx - sw // 2, stem_top),
        "bar_bl_out": (cx - bw // 2, bar_bot),
    }
    return mask, keypoints


def _mask_contour_points(mask_uint8):
    """Return an (N, 2) float32 array of external-contour pixel coordinates for `mask_uint8` (0/255).

    Concatenates all disjoint external contours so a fragmented mask (e.g. an occluded T split
    into two pieces) still contributes every boundary pixel."""
    import cv2
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate([c.reshape(-1, 2) for c in contours], axis=0).astype(np.float32)


def _uniform_subsample(points, n):
    """Uniformly-spaced subsample of `points` down to at most `n` rows."""
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, n).astype(int)
    return points[idx]


def align_canonical_t_to_mask(observed_mask_bool, fixed_scale=None,
                              n_init=12, n_steps=150, lr=0.1,
                              n_canon_points=200, n_obs_points=400, device="cpu"):
    """Continuous optimization of a 2D similarity transform (tx, ty, θ, [s]) that maps the
    canonical T contour onto the observed-mask contour.

    Two modes chosen by `fixed_scale`:
      - `None` (bootstrap mode): scale is optimized jointly with pose, and cost is *symmetric*
        chamfer (canon→obs + obs→canon). The backward term prevents the one-sided-chamfer
        degeneracy where the canonical shrinks or grows to zero cost. Use this once per camera
        (typically on frame 0) to estimate a good scale.
      - `<float>` (fixed-scale mode): scale is held constant and cost is *one-sided* chamfer
        obs→canon (for each observed contour point, distance to the nearest warped canonical
        point). This is tolerant to heavy occlusion — missing canonical regions cost nothing —
        but relies on the fixed scale to prevent the canonical growing to swallow every observed
        point. Standard mode for per-frame alignment once scale is known.

    Follows the batched-initial-poses + Adam-on-chamfer pattern from shape_warping's
    `shape_reconstruction` (originally 3D point clouds); chamfer is highly non-convex in rotation,
    so we seed `n_init` batches with evenly-spaced initial angles and let Adam refine each in
    parallel, then pick the batch element with lowest final cost.

    Returns None on a degenerate mask, otherwise:
        {angle_rad, scale, iou, chamfer, affine (2x3 canonical→image), keypoints_px}."""
    import cv2
    import torch
    if int(observed_mask_bool.sum()) < 10:
        return None
    canon_mask, canon_kp = build_canonical_t()
    canon_pts_full = _mask_contour_points(canon_mask)
    obs_pts_full = _mask_contour_points((observed_mask_bool.astype(np.uint8)) * 255)
    if len(canon_pts_full) < 8 or len(obs_pts_full) < 8:
        return None

    canon_pts = _uniform_subsample(canon_pts_full, n_canon_points)
    obs_pts = _uniform_subsample(obs_pts_full, n_obs_points)

    canon_centroid = canon_pts_full.mean(axis=0)
    canon_centered = canon_pts - canon_centroid

    obs_centroid = obs_pts.mean(axis=0)
    obs_area = float(observed_mask_bool.sum())
    canon_area = float((canon_mask > 0).sum())
    optimize_scale = fixed_scale is None
    init_scale = (float(fixed_scale) if not optimize_scale
                  else float(np.sqrt(obs_area / max(canon_area, 1.0))))

    dev = torch.device(device)
    canon_t = torch.from_numpy(canon_centered).to(dev)
    obs_t = torch.from_numpy(obs_pts).to(dev)

    thetas0 = np.linspace(0.0, 2.0 * np.pi, n_init, endpoint=False).astype(np.float32)
    tx_p = torch.full((n_init,), float(obs_centroid[0]), device=dev, requires_grad=True)
    ty_p = torch.full((n_init,), float(obs_centroid[1]), device=dev, requires_grad=True)
    theta_p = torch.tensor(thetas0, device=dev, requires_grad=True)
    log_s_p = torch.full((n_init,), float(np.log(init_scale)), device=dev,
                        requires_grad=optimize_scale)

    trainable = [tx_p, ty_p, theta_p]
    if optimize_scale:
        trainable.append(log_s_p)
    opt = torch.optim.Adam(trainable, lr=lr)

    def _cost():
        s = torch.exp(log_s_p)
        cos, sin = torch.cos(theta_p), torch.sin(theta_p)
        R = torch.stack(
            [torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)],
            dim=-2,
        ) * s[:, None, None]
        warped = torch.einsum("bij,nj->bni", R, canon_t) + torch.stack([tx_p, ty_p], dim=-1)[:, None, :]
        d2 = ((warped.unsqueeze(2) - obs_t[None, None, :, :]) ** 2).sum(-1)  # (B, Nc, No)
        d = torch.sqrt(d2.clamp_min(1e-12))
        if optimize_scale:
            # Symmetric chamfer keeps scale bounded during bootstrap.
            return d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1)
        # One-sided obs → canon: tolerant to canonical regions missing from the observed mask.
        return d.min(dim=1).values.mean(dim=1)

    for _ in range(n_steps):
        opt.zero_grad()
        cost = _cost()
        cost.sum().backward()
        opt.step()

    with torch.no_grad():
        final_cost = _cost().cpu().numpy()
        best_b = int(np.argmin(final_cost))
        chamfer = float(final_cost[best_b])
        best_theta = float(theta_p[best_b].item())
        best_tx = float(tx_p[best_b].item())
        best_ty = float(ty_p[best_b].item())
        best_s = float(torch.exp(log_s_p[best_b]).item())

    cos, sin = np.cos(best_theta), np.sin(best_theta)
    R2 = np.array([[cos, -sin], [sin, cos]]) * best_s
    t2 = np.array([best_tx, best_ty]) - R2 @ canon_centroid
    A = np.hstack([R2, t2.reshape(-1, 1)]).astype(np.float32)

    H, W = observed_mask_bool.shape
    warped_mask = cv2.warpAffine(canon_mask, A, (W, H), flags=cv2.INTER_NEAREST) > 127
    inter = int(np.logical_and(warped_mask, observed_mask_bool).sum())
    union = int(np.logical_or(warped_mask, observed_mask_bool).sum())
    iou = inter / max(union, 1)

    keypoints_px = {}
    for name, (x, y) in canon_kp.items():
        p = A @ np.array([x, y, 1.0], dtype=np.float32)
        keypoints_px[name] = (float(p[0]), float(p[1]))

    return {
        "angle_rad": (best_theta + np.pi) % (2.0 * np.pi) - np.pi,
        "scale": best_s,
        "iou": float(iou),
        "chamfer": chamfer,
        "affine": A,
        "keypoints_px": keypoints_px,
    }


def draw_detection(pil_img, det, seed_point=None, keypoints_px=None, alignment_info=None,
                   mask_color=(0, 255, 127), mask_alpha=0.4,
                   box_color="#00FF7F", seed_color="#FFD400", kp_color="#FF3B30"):
    """Overlay SAM2's mask + bounding box on the image, plus the red-blob seed point and any
    aligned canonical T keypoints when provided.

    `det` may be None (nothing detected); in that case the seed point is still drawn if present
    so it's visible where the red-blob step landed even when SAM2 didn't produce a mask."""
    out = pil_img.copy().convert("RGB")
    if det is not None:
        arr = np.asarray(out).copy()
        overlay = arr.copy()
        overlay[det["mask"]] = np.array(mask_color, dtype=np.uint8)
        arr = (arr * (1 - mask_alpha) + overlay * mask_alpha).astype(np.uint8)
        out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    if det is not None:
        x1, y1, x2, y2 = det["xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        label = f"SAM2 {det['conf']:.2f}"
        if alignment_info is not None:
            label += f"  T-fit {alignment_info['chamfer']:.2f}px"
        tx, ty = x1, max(0, y1 - 16)
        tb = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(tb, fill=box_color)
        draw.text((tx, ty), label, fill="black", font=font)
    if seed_point is not None:
        sx, sy = seed_point
        r = 6
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], outline=seed_color, width=3)
    if keypoints_px:
        r = 4
        for name, (kx, ky) in keypoints_px.items():
            draw.ellipse([kx - r, ky - r, kx + r, ky + r], fill=kp_color, outline="black")
            draw.text((kx + r + 2, ky - r - 2), name, fill=kp_color, font=font)
    return out


def build_ee_figure(gt_pos, gt_quat, pred_pos, pred_quat, base_pose, t, pose_axes_stride,
                    scene_ranges, title):
    """Build the 3D EE-trajectory figure showing GT and predicted trajectories up to timestep `t`."""
    upto = int(t) + 1
    fig = go.Figure()
    fig = plot_pose(base_pose, axis_length=0.05, fig=fig, name="base_link")
    fig = add_ee_trajectory(fig, gt_pos[:upto], gt_quat[:upto], name="ground_truth (state)",
                            colorscale="Viridis", axes_colors=["red", "green", "blue"],
                            pose_axes_stride=pose_axes_stride)
    fig = add_ee_trajectory(fig, pred_pos[:upto], pred_quat[:upto], name="predicted (action)",
                            colorscale="Hot", axes_colors=["orange", "yellow", "purple"],
                            pose_axes_stride=pose_axes_stride)
    # Highlight the current-timestep poses so the slider position is obvious.
    fig = plot_pose([*gt_pos[t], *gt_quat[t]], axis_length=0.05, fig=fig, name=f"gt@t={t}")
    fig = plot_pose([*pred_pos[t], *pred_quat[t]], axis_length=0.05, fig=fig, name=f"pred@t={t}",
                    colors=["orange", "yellow", "purple"])
    (xr, yr, zr) = scene_ranges
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X", range=xr),
            yaxis=dict(title="Y", range=yr),
            zaxis=dict(title="Z", range=zr),
            aspectmode="data",
        ),
        # Keep the user's camera view stable across slider updates.
        uirevision="ee-view",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def pixel_to_workspace_xy(px, py, image_size, ws=WORKSPACE_XY_BOUNDS):
    """HACK: linearly map an image pixel (`px`, `py`) into workspace XY meters.

    Uses a **uniform** (letterbox) scale — the same meters-per-pixel factor in both axes — so a
    T that's e.g. 4:1 in the image stays 4:1 in workspace. If the image aspect ratio doesn't
    match the workspace aspect ratio (as is the case for a 848×480 camera into a 0.5×0.5 m
    workspace here), the mapped image occupies a centered sub-rectangle of the workspace rather
    than filling it — pixel corners no longer hit workspace corners exactly, but the mapped T
    keeps its true shape.

    Replace with a proper intrinsics + extrinsics + table-plane unprojection once camera
    calibration is available."""
    W, H = image_size
    xmin, ymin, xmax, ymax = ws
    ws_w, ws_h = xmax - xmin, ymax - ymin
    scale = min(ws_w / W, ws_h / H)  # meters per pixel, uniform
    # Center the mapped image sub-rectangle within the workspace bounds.
    x_off = xmin + (ws_w - W * scale) / 2.0
    y_off = ymin + (ws_h - H * scale) / 2.0
    return float(x_off + px * scale), float(y_off + py * scale)


# gym_pusht default frame: 512×512 units; T built with `add_tee(scale=30)` → bar width
# `length*scale = 120` units.
GYM_PUSHT_WINDOW = 512
GYM_PUSHT_T_BAR_WIDTH = 120


def workspace_xy_to_gym_pusht(x, y, t_bar_width_meters,
                              workspace_bounds=WORKSPACE_XY_BOUNDS,
                              gym_window=GYM_PUSHT_WINDOW,
                              gym_t_bar_width=GYM_PUSHT_T_BAR_WIDTH):
    """Translate an XY_ee-display point (meters) into gym_pusht coordinates (0..gym_window).

    Assumptions the user requested: the T has the same physical scale in both frames — the T's
    bar width is `t_bar_width_meters` in workspace and `gym_t_bar_width` gym-units in the env —
    and the two coordinate frames are centered on the same point (the center of
    `workspace_bounds` and of the gym window)."""
    xmin, ymin, xmax, ymax = workspace_bounds
    ws_cx, ws_cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    gym_c = gym_window / 2.0
    gym_units_per_meter = gym_t_bar_width / t_bar_width_meters
    gx = (x - ws_cx) * gym_units_per_meter + gym_c
    gy = (y - ws_cy) * gym_units_per_meter + gym_c
    return float(gx), float(gy)


def gym_pusht_to_workspace_xy(gx, gy, t_bar_width_meters,
                              workspace_bounds=WORKSPACE_XY_BOUNDS,
                              gym_window=GYM_PUSHT_WINDOW,
                              gym_t_bar_width=GYM_PUSHT_T_BAR_WIDTH):
    """Inverse of `workspace_xy_to_gym_pusht` — gym_pusht coordinates → workspace-XY meters."""
    xmin, ymin, xmax, ymax = workspace_bounds
    ws_cx, ws_cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    gym_c = gym_window / 2.0
    meters_per_gym_unit = t_bar_width_meters / gym_t_bar_width
    x = (gx - gym_c) * meters_per_gym_unit + ws_cx
    y = (gy - gym_c) * meters_per_gym_unit + ws_cy
    return float(x), float(y)


# Distinct-hue colors so per-camera keypoints are visually separable when overlaid.
_CAM_PALETTE = ["#FF3B30", "#34C759", "#007AFF", "#FF9500", "#AF52DE", "#5AC8FA"]


def build_ee_xy_figure(gt_pos, pred_pos, t, keypoints_by_cam_ws=None,
                       ws=WORKSPACE_XY_BOUNDS, title=""):
    """Build a top-down 2D XY view of EE trajectories up to timestep `t`, overlaid with the T
    keypoints projected into the (hack) workspace-XY frame."""
   
    xmin, ymin, xmax, ymax = ws
    upto = int(t) + 1
    fig = go.Figure()
    # Display convention: horizontal axis shows workspace-Y, vertical shows workspace-X, and the
    # horizontal (Y) axis is reversed. Every data source below feeds `x=<workspace Y>` and
    # `y=<workspace X>` so this transform is fully contained in the plot builder — the underlying
    # data (gt_pos, keypoints_by_cam_ws) stays in native (X, Y) order.
    fig.add_shape(type="rect", x0=ymin, y0=xmin, x1=ymax, y1=xmax,
                  line=dict(color="#888", width=1, dash="dash"), fillcolor="rgba(0,0,0,0)")
    fig.add_trace(go.Scatter(
        x=gt_pos[:upto, 1], y=gt_pos[:upto, 0], mode="lines+markers",
        line=dict(color="rgba(50,100,200,0.5)", width=2),
        marker=dict(size=4, color=list(range(upto)), colorscale="Viridis"),
        name="gt EE",
    ))
    fig.add_trace(go.Scatter(
        x=pred_pos[:upto, 1], y=pred_pos[:upto, 0], mode="lines+markers",
        line=dict(color="rgba(200,100,50,0.5)", width=2),
        marker=dict(size=4, color=list(range(upto)), colorscale="Hot"),
        name="pred EE",
    ))
    fig.add_trace(go.Scatter(
        x=[gt_pos[t, 1]], y=[gt_pos[t, 0]], mode="markers",
        marker=dict(size=12, color="blue", symbol="circle-open", line=dict(width=3)),
        name=f"gt@t={t}",
    ))
    fig.add_trace(go.Scatter(
        x=[pred_pos[t, 1]], y=[pred_pos[t, 0]], mode="markers",
        marker=dict(size=12, color="red", symbol="circle-open", line=dict(width=3)),
        name=f"pred@t={t}",
    ))
    if keypoints_by_cam_ws:
        for i, (cam, kps) in enumerate(keypoints_by_cam_ws.items()):
            color = _CAM_PALETTE[i % len(_CAM_PALETTE)]
            # kps values are (workspace_X, workspace_Y); apply the same axis swap the EE traces
            # use — workspace-Y on plot-x, workspace-X on plot-y.
            ys = [kps[n][1] for n in kps]
            xs = [kps[n][0] for n in kps]
            fig.add_trace(go.Scatter(
                x=xs + [xs[0]], y=ys + [ys[0]], mode="lines+markers",
                line=dict(color=color, width=1, dash="dot"),
                marker=dict(size=6, color=color),
                name=f"T@{cam}",
            ))
    # Plot-axis bounds pull from the *swapped* data: plot-x from workspace-Y, plot-y from workspace-X.
    plot_x = np.concatenate([gt_pos[:, 1], pred_pos[:, 1], [ymin, ymax]])
    plot_y = np.concatenate([gt_pos[:, 0], pred_pos[:, 0], [xmin, xmax]])
    pad = 0.03
    fig.update_layout(
        title=title,
        # Horizontal axis shows workspace-Y and is reversed (larger Y on the left).
        xaxis=dict(title="Y (m)",
                   range=[float(plot_x.min()) - pad, float(plot_x.max()) + pad],
                   scaleanchor="y", scaleratio=1.0),
        yaxis=dict(title="X (m)",
                   range=[float(plot_y.max()) + pad, float(plot_y.min()) - pad]),
        showlegend=False,
        margin=dict(l=40, r=10, t=30, b=30),
        uirevision="xy-view",
    )
    return fig


def run_dash_app(dataset, ep_idx, urdf_path=DEFAULT_URDF, robot_id=DEFAULT_ROBOT_ID,
                 pose_axes_stride=5, host="127.0.0.1", port=8050, debug=False,
                 sam_weights=DEFAULT_SAM_WEIGHTS, sam_device="cuda"):
    """Launch a Dash app that shows the EE trajectory up to a slider-controlled timestep alongside
    the camera observation(s) at that timestep, with an on-demand Detect button.

    Detect finds the largest red blob in each camera frame (HSV threshold + largest connected
    component), then uses that centroid as a positive point prompt for SAM2 to produce a
    segmentation mask + bounding box. The mask is drawn as a semi-transparent overlay and the box
    outlines it; the seed centroid is drawn as a yellow ring so the red-blob step is visible even
    when SAM2 doesn't return a mask.

    Trajectories are precomputed once and the slider only re-slices them, so scrubbing stays
    snappy. Camera frames are kept as PIL images so detections can be re-drawn without decoding
    the JPEG each time. SAM2 is loaded lazily on the first Detect click, and results are cached
    per (timestep, camera) so scrubbing back to a previously-detected frame reuses them."""
    start, end = episode_frame_range(dataset, ep_idx)
    n_steps = end - start
    if n_steps == 0:
        raise SystemExit(f"episode {ep_idx} has no frames")
    robot_state = RobotState(urdf_path=urdf_path, id=robot_id, load_meshes=False)

    cam_keys = list(dataset.meta.camera_keys)
    states, actions = [], []
    pil_by_cam = {cam: [] for cam in cam_keys}
    for i in range(start, end):
        item = dataset[i]
        states.append(item[STATE_KEY])
        actions.append(item[ACTION_KEY])
        for cam in cam_keys:
            pil_by_cam[cam].append(Image.fromarray(frame_to_uint8_hwc(item[cam])))

    gt_pos, gt_quat = joint_seq_ee_poses(robot_state, states)
    pred_pos, pred_quat = joint_seq_ee_poses(robot_state, actions)

    base = robot_state.robot_urdf.link_fk(cfg=np.zeros(6))[robot_state.robot_urdf.link_map["base_link"]]
    base_pose = [*base[:3, 3], *robot_state.rot_matrix_to_quat(base[:3, :3])]

    # Lock scene ranges so the plot doesn't rescale as new points are revealed.
    all_pos = np.concatenate([gt_pos, pred_pos, np.asarray([base[:3, 3]])], axis=0)
    mins, maxs = all_pos.min(axis=0), all_pos.max(axis=0)
    pad = np.maximum((maxs - mins) * 0.1, 0.05)
    scene_ranges = tuple([float(mins[i] - pad[i]), float(maxs[i] + pad[i])] for i in range(3))

    slider_step = max(1, n_steps // 20)
    marks = {i: str(i) for i in range(0, n_steps, slider_step)}
    marks[n_steps - 1] = str(n_steps - 1)

    # Detection state persists across callbacks for the lifetime of the app. The model is loaded
    # lazily on the first click so `ultralytics` (heavy import) isn't touched otherwise. The cache
    # stores {(t, cam): (seed, det, align)} so scrubbing back reuses results. `scale_per_cam`
    # holds the per-camera T-scale bootstrapped from frame 0 on the first successful detection;
    # all subsequent per-frame alignments run with fixed scale + one-sided obs→canon chamfer.
    sam_state = {"model": None, "cache": {}, "scale_per_cam": {}}
    resolved_sam_device = resolve_sam_device(sam_device)

    def _bootstrap_scale(cam):
        """One-time per-camera scale estimate from frame 0 via scale-free symmetric-chamfer fit.
        Falls back to no-scale (returns None) if frame 0 has no usable mask; caller may still
        run per-frame alignment without a fixed scale in that case."""
        if cam in sam_state["scale_per_cam"]:
            return sam_state["scale_per_cam"][cam]
        pil0 = pil_by_cam[cam][0]
        seed0 = find_red_centroid(pil0)
        if seed0 is None:
            return None
        det0 = segment_at_point(sam_state["model"], pil0, seed0, device=resolved_sam_device)
        if det0 is None:
            return None
        align0 = align_canonical_t_to_mask(det0["mask"], fixed_scale=None,
                                           device=resolved_sam_device)
        if align0 is None:
            return None
        s = float(align0["scale"])
        sam_state["scale_per_cam"][cam] = s
        print(f"bootstrapped T scale for {cam} from frame 0: {s:.3f}")
        return s

    app = Dash(__name__)
    app.title = f"Rollout viz — ep {ep_idx}"
    app.layout = html.Div(
        [
            html.H3(f"Episode {int(ep_idx)} — {n_steps} steps"),
            html.Div(
                [
                    html.Button("Detect (red blob → SAM2)", id="detect-btn", n_clicks=0,
                                style={"padding": "6px 14px"}),
                    html.Span(id="detect-status",
                              children="No detection run yet.",
                              style={"marginLeft": "12px", "fontSize": "13px",
                                     "color": "#555"}),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "center",
                       "marginBottom": "10px"},
            ),
            html.Div(
                [
                    dcc.Graph(id="ee-plot", style={"height": "708px", "flex": "1 1 60%"}),
                    html.Div(
                        [
                            html.Div(id="cam-column",
                                     style={"display": "flex", "flexDirection": "column",
                                            "gap": "12px", "overflowY": "auto",
                                            "maxHeight": "440px"}),
                            dcc.Graph(id="ee-xy-plot", style={"height": "260px"}),
                        ],
                        style={"flex": "1 1 40%", "display": "flex",
                               "flexDirection": "column", "gap": "8px"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "alignItems": "stretch"},
            ),
            html.Div(
                dcc.Slider(id="timestep", min=0, max=n_steps - 1, step=1, value=0, marks=marks,
                           tooltip={"placement": "bottom", "always_visible": True}),
                style={"marginTop": "16px", "padding": "0 12px"},
            ),
        ],
        style={"fontFamily": "sans-serif", "padding": "12px"},
    )

    @app.callback(
        Output("ee-plot", "figure"),
        Output("ee-xy-plot", "figure"),
        Output("cam-column", "children"),
        Output("detect-status", "children"),
        Input("timestep", "value"),
        Input("detect-btn", "n_clicks"),
    )
    def _update(t, n_clicks):
        from dash import ctx
        t = int(t or 0)
        status = no_update

        if ctx.triggered_id == "detect-btn" and n_clicks:
            if sam_state["model"] is None:
                print(f"loading SAM2 weights: {sam_weights} (device={resolved_sam_device})")
                sam_state["model"] = load_sam2(sam_weights, device=resolved_sam_device)
            per_cam = []
            for cam in cam_keys:
                # Ensure per-cam scale is established from frame 0 before the first fit.
                cam_scale = _bootstrap_scale(cam)
                pil_img = pil_by_cam[cam][t]
                seed = find_red_centroid(pil_img)
                det = None
                align = None
                if seed is not None:
                    det = segment_at_point(sam_state["model"], pil_img, seed,
                                           device=resolved_sam_device)
                if det is not None:
                    align = align_canonical_t_to_mask(det["mask"], fixed_scale=cam_scale,
                                                      device=resolved_sam_device)
                sam_state["cache"][(t, cam)] = (seed, det, align)
                if seed is None:
                    per_cam.append(f"{cam}: no red blob")
                elif det is None:
                    per_cam.append(f"{cam}: seed only")
                elif align is None:
                    per_cam.append(f"{cam}: mask ok, T-fit failed")
                else:
                    per_cam.append(f"{cam}: T-fit chamfer={align['chamfer']:.2f}px")
            status = f"t={t} — " + "; ".join(per_cam)

        title = f"EE trajectory — episode {int(ep_idx)} — t={t}/{n_steps - 1}"
        fig = build_ee_figure(gt_pos, gt_quat, pred_pos, pred_quat, base_pose, t,
                              pose_axes_stride, scene_ranges, title)
        cam_children = []
        keypoints_by_cam_ws = {}
        for cam in cam_keys:
            pil_img = pil_by_cam[cam][t]
            seed, det, align = sam_state["cache"].get((t, cam), (None, None, None))
            if seed is not None or det is not None:
                rendered = draw_detection(
                    pil_img, det, seed_point=seed,
                    keypoints_px=align["keypoints_px"] if align is not None else None,
                    alignment_info=align,
                )
            else:
                rendered = pil_img
            if align is not None:
                # Project pixel-space keypoints into the (hack) workspace XY frame for the XY plot.
                keypoints_by_cam_ws[cam] = {
                    name: pixel_to_workspace_xy(px, py, pil_img.size)
                    for name, (px, py) in align["keypoints_px"].items()
                }

                caption = (f"{cam} — SAM2 {det['conf']:.2f}, "
                           f"T-fit chamfer {align['chamfer']:.2f}px, "
                           f"θ={np.degrees(align['angle_rad']):+.1f}°")
            elif det is not None:
                caption = f"{cam} — SAM2 mask ({det['conf']:.2f}), T-fit failed"
            elif seed is not None:
                caption = f"{cam} — seed only (SAM2 empty)"
            else:
                caption = cam
            cam_children.append(html.Div([
                html.Div(caption, style={"fontWeight": "bold", "marginBottom": "4px"}),
                html.Img(src=pil_to_data_uri(rendered),
                         style={"width": "100%", "border": "1px solid #ccc",
                                "borderRadius": "4px"}),
            ]))
        xy_title = (f"EE + T (XY) — t={t}/{n_steps - 1} — "
                    "HACK pixel→XY (needs extrinsics)")
        xy_fig = build_ee_xy_figure(gt_pos, pred_pos, t,
                                    keypoints_by_cam_ws=keypoints_by_cam_ws or None,
                                    title=xy_title)
        return fig, xy_fig, cam_children, status

    print(f"launching Dash app at http://{host}:{port} (episode {ep_idx}, {n_steps} steps)")
    app.run(host=host, port=port, debug=debug)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="Sorozco0612/pusht_200_20260706_130548",
                        help="HuggingFace dataset repo id to visualize.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize.")
    parser.add_argument("--out-dir", default="/home/rthomp12/diff_lerobot/scripts/rollout_viz",
                        help="Directory for the output mp4(s) and html.")
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="URDF path for forward kinematics.")
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID, help="Robot id (calibration label).")
    parser.add_argument("--pose-axes-stride", type=int, default=5,
                        help="Draw EE orientation axes every N timesteps (0 = trajectory only).")
    parser.add_argument("--fps", type=int, default=None, help="Video fps (defaults to dataset fps).")
    parser.add_argument("--write-videos", action="store_true",
                        help="Also write one mp4 per camera to --out-dir (off by default now that the "
                             "Dash app shows the same frames interactively).")
    parser.add_argument("--host", default="127.0.0.1", help="Dash app host.")
    parser.add_argument("--port", type=int, default=8050, help="Dash app port.")
    parser.add_argument("--debug", action="store_true", help="Run the Dash app in debug mode.")
    parser.add_argument("--sam-weights", default=DEFAULT_SAM_WEIGHTS,
                        help="Ultralytics SAM2 weights for the Detect button (auto-downloaded on "
                             "first use).")
    parser.add_argument("--sam-device", default="cuda",
                        help="Device for SAM2 inference. Defaults to CUDA (falls back to CPU if "
                             "unavailable).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(args.repo_id)
    n_ep = dataset.meta.total_episodes
    if not 0 <= args.episode < n_ep:
        raise SystemExit(f"episode {args.episode} out of range (dataset has {n_ep} episodes)")

    if args.write_videos:
        make_episode_videos(dataset, args.episode, out_dir, fps=args.fps)
    run_dash_app(dataset, args.episode, urdf_path=args.urdf, robot_id=args.robot_id,
                 pose_axes_stride=args.pose_axes_stride, host=args.host, port=args.port,
                 debug=args.debug, sam_weights=args.sam_weights, sam_device=args.sam_device)


if __name__ == "__main__":
    main()
