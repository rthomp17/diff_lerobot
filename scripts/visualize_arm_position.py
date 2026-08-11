

#Function for mapping to pose
#Plotly function for mapping poses

import numpy as np
import plotly.graph_objects as go


import os
import json
import numpy as np
import open3d as o3d
from urchin import URDF
from foxglove.schemas import (
    FrameTransform,
    Vector3,
    Quaternion
)

# joint limit written in USD (degree)
SO101_FOLLOWER_USD_JOINT_LIMLITS = {
    "shoulder_pan.pos": (-110.0, 110.0),
    "shoulder_lift.pos": (-100.0, 100.0),
    "elbow_flex.pos": (-100.0, 90.0),
    "wrist_flex.pos": (-95.0, 95.0),
    "wrist_roll.pos": (-160.0, 160.0),
    "gripper.pos": (-10, 100.0),
}

# motor limit written in real device (normalized to related range)
SO101_FOLLOWER_MOTOR_LIMITS = {
    "shoulder_pan.pos": (-100.0, 100.0),
    "shoulder_lift.pos": (-100.0, 100.0),
    "elbow_flex.pos": (-100.0, 100.0),
    "wrist_flex.pos": (-100.0, 100.0),
    "wrist_roll.pos": (-100.0, 100.0),
    "gripper.pos": (0.0, 100.0),
}

class RobotState:

    def __init__(self, urdf_path, id, load_meshes=True):
        self.robot_urdf = URDF.load(urdf_path)

        # robot_calibration_path = f"/home/sorozco0612/.cache/huggingface/lerobot/calibration/robots/so101_follower/{id}.json"

        # with open(robot_calibration_path, "r") as f:
        #     calib = json.load(f)

        # self.PHYS_RANGES = self.compute_phys_ranges(calib)

        # Cache URDF meshes. Skip when only forward kinematics (link_fk) is needed, since FK relies
        # on the URDF joint structure, not the visual meshes.
        self.robot_meshes_o3d = {}
        if load_meshes:
            self.load_robot_meshes()

    def ticks_to_radians(self, raw, homing_offset):
        TICKS_PER_REV = 4096
        return (raw + homing_offset) * (2 * np.pi / TICKS_PER_REV)
        #return (raw) * (2 * np.pi / TICKS_PER_REV)

    def compute_phys_ranges(self, calib_dict):
        phys_ranges = {}

        for joint, data in calib_dict.items():
            range_min = data["range_min"]
            range_max = data["range_max"]
            offset    = data["homing_offset"]

            lo = self.ticks_to_radians(range_min, offset)
            hi = self.ticks_to_radians(range_max, offset)

            phys_ranges[joint] = [float(lo), float(hi)]

        return phys_ranges

    def rot_matrix_to_quat(self, R):
        """
        Convert a 3x3 rotation matrix to quaternion [x, y, z, w].
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return np.array([x, y, z, w], dtype=np.float64)

    def load_robot_meshes(self):
        """Load and sample all robot meshes upfront."""
        for link in self.robot_urdf.links:
            for visual in link.visuals:
                if not hasattr(visual.geometry, "mesh"):
                    continue

                mesh_path = os.path.join("real_world/robot", visual.geometry.mesh.filename)
                mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)

                if mesh_o3d.is_empty():
                    print(f"[WARN] Empty mesh: {mesh_path}")
                    continue

                self.robot_meshes_o3d[(link.name, mesh_path)] = mesh_o3d

    def convert_lerobot_action_to_radians(self, joint_state):
        """
        Convert the action from Lerobot to LeIsaac. Just convert value, not include the format.
        """

        processed_action = np.zeros(6)
        joint_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS
        motor_limits = SO101_FOLLOWER_MOTOR_LIMITS

        for idx, joint_name in enumerate(joint_limits):
            motor_limit_range = motor_limits[joint_name]
            joint_limit_range = joint_limits[joint_name]
            motor_range = motor_limit_range[1] - motor_limit_range[0]
            joint_range = joint_limit_range[1] - joint_limit_range[0]
            motor_degree = joint_state[idx] - motor_limit_range[0] #joint_name
            processed_degree = motor_degree / motor_range * joint_range + joint_limit_range[0]
            processed_radius = processed_degree / 180.0 * np.pi  # convert to radian
            processed_action[idx] = processed_radius

        return processed_action

    def sample_robot_points(self, obs, tuned_joint_offsets):
        """Return sampled + transformed robot points for both arms."""
        robot_pts = []

        joint_positions = self.convert_lerobot_action_to_radians(obs)

        for link in self.robot_urdf.links:
            visuals = link.visuals
            if len(visuals) == 0:
                continue

            for visual in visuals:
                if not hasattr(visual.geometry, "mesh"):
                    continue

                key = (link.name, os.path.join("real_world/robot", visual.geometry.mesh.filename))
                mesh_o3d = self.robot_meshes_o3d[key]

                # Sample raw mesh points
                pts_mesh = np.asarray(mesh_o3d.sample_points_uniformly(1000).points)

                # Apply visual origin
                T_vis = visual.origin
                R_vis = T_vis[:3, :3]
                t_vis = T_vis[:3, 3]
                pts_visual = (R_vis @ pts_mesh.T).T + t_vis

                # FK for robot
                T1 = self.robot_urdf.link_fk(cfg=joint_positions)[self.robot_urdf.link_map[link.name]]
                R1, t1 = T1[:3, :3], T1[:3, 3]
                robot_pts.append((R1 @ pts_visual.T).T + t1)

        return np.array(robot_pts)

    def get_eef_pos(self, obs, tuned_joint_offsets):

        joint_positions = self.get_joint_positions(obs, tuned_joint_offsets)

        return self.robot_urdf.link_fk(cfg=joint_positions)[self.robot_urdf.link_map["gripper_frame_link"]]

    def get_transforms(self, obs, tuned_joint_offsets):

        transforms = []

        joint_positions = self.convert_lerobot_action_to_radians(obs)

        # Compute forward kinematics with updated joint positions
        fk_poses = self.robot_urdf.link_fk(cfg=joint_positions)

        # World -> Base
        transforms.append(
            FrameTransform(
                parent_frame_id="world",
                child_frame_id="base",
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            )
        )

        for joint in self.robot_urdf.joints:
            parent_link = joint.parent
            child_link = joint.child
            T_parent = fk_poses[self.robot_urdf.link_map[parent_link]]
            T_child = fk_poses[self.robot_urdf.link_map[child_link]]

            # Local transform from parent->child
            T_local = np.linalg.inv(T_parent) @ T_child
            trans = T_local[:3, 3]
            quat = self.rot_matrix_to_quat(T_local[:3, :3])
            transforms.append(
                FrameTransform(
                    parent_frame_id=parent_link,
                    child_frame_id=child_link,
                    translation=Vector3(x=float(trans[0]), y=float(trans[1]), z=float(trans[2])),
                    rotation=Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
                )
            )

        return transforms, self.sample_robot_points(obs, tuned_joint_offsets)
    

def quaternion_to_rotation_matrix(quat):
    """Convert a quaternion into a 3x3 rotation matrix.

    Args:
        quat: Array-like of length 4 in (x, y, z, w) order.

    Returns:
        A (3, 3) numpy array representing the rotation.
    """
    x, y, z, w = quat
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("Cannot convert a zero-norm quaternion to a rotation matrix.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def plot_pose(pose, axis_length=0.1, fig=None, name="pose", colors = ["red", "green", "blue"]):
    """Plot a pose (position + quaternion) as 3D coordinate axes using plotly.

    The pose is drawn as three arrows originating at the position, oriented by
    the quaternion: X (red), Y (green), Z (blue).

    Args:
        pose: Array-like of length 7 as [px, py, pz, qx, qy, qz, qw].
        axis_length: Length of each drawn axis arrow.
        fig: An existing plotly Figure to add the pose to. If None, a new one
            is created.
        name: Label prefix for the traces (useful when plotting several poses).

    Returns:
        The plotly Figure containing the pose.
    """
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,):
        raise ValueError(f"Expected a length-7 pose vector, got shape {pose.shape}.")

    position = pose[:3]
    rotation = quaternion_to_rotation_matrix(pose[3:])

    if fig is None:
        fig = go.Figure()

    
    labels = ["x", "y", "z"]
    for i in range(3):
        end = position + axis_length * rotation[:, i]
        fig.add_trace(
            go.Scatter3d(
                x=[position[0], end[0]],
                y=[position[1], end[1]],
                z=[position[2], end[2]],
                mode="lines+markers",
                line=dict(color=colors[i], width=6),
                marker=dict(size=[4, 6], color=colors[i]),
                name=f"{name}-{labels[i]}",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        )
    )
    return fig


if __name__ == "__main__":
    # Identity orientation at the origin.
    example_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    plot_pose(example_pose).show()
