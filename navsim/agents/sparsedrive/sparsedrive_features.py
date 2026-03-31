from enum import IntEnum
from typing import Any, Dict, List, Tuple
from omegaconf import DictConfig
from pathlib import Path
import pickle
import copy

from PIL import Image
import cv2
import random
import numpy as np
import torch
from pyquaternion import Quaternion

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

from .sparsedrive_config import SparseDriveConfig


class SparseDriveFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for TransFuser."""

    def __init__(self, config: SparseDriveConfig):
        """
        Initializes feature builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "sparsedrive_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        features["camera_feature"] = self._get_camera_feature(agent_input)
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )

        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        camera_keys = ["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"]
        camera_features = []
        for cameras in agent_input.cameras:
            camera_feature = {key: {} for key in camera_keys}
            for key in camera_keys:
                camera_data = getattr(cameras, key)
                camera_feature[key]["image_path"] = camera_data.image_path
                camera_feature[key]["sensor2lidar_rotation"] = camera_data.sensor2lidar_rotation
                camera_feature[key]["sensor2lidar_translation"] = camera_data.sensor2lidar_translation
                camera_feature[key]["intrinsics"] = camera_data.intrinsics
                camera_feature[key]["distortion"] = camera_data.distortion
            camera_features.append(camera_feature)

        return camera_features

    def pipeline(self, features, targets, token, test_mode, vis=False):
        camera_features = features["camera_feature"]

        ## only last frame used
        frame_info = camera_features[-1]
        results = self.get_camera_params(frame_info)
        results = self.load_images(results)
        results = self.resize_crop_flip_img(results, test_mode)
        results, targets = self.ego_rotation(results, targets, test_mode)
        results = self.photo_metric_distortion(results, test_mode)
        results = self.normalize_img(results)
        results = self.data_adapter(results)

        features["camera_feature"] = results
        return features, targets, token

    def get_camera_params(self, frame_info):
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsic = []
        cam2lidar_rts = []
        distortions = []
        for cam in self._config.cams:
            cam_info = frame_info[cam]
            ## image path
            image_path = cam_info["image_path"]
            image_paths.append(cam_info["image_path"])
            ## distortion
            distortions.append(cam_info["distortion"])
            ## cam2lidar
            cam2lidar_rt = np.eye(4)
            cam2lidar_rt[:3, :3] = cam_info["sensor2lidar_rotation"]
            cam2lidar_rt[:3, 3] = cam_info["sensor2lidar_translation"]
            cam2lidar_rts.append(cam2lidar_rt)
            ## lidar2cam
            lidar2cam_rt = np.eye(4)
            lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
            lidar2cam_t = (
                cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
            )
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            ## intrinsic
            intrinsic = copy.deepcopy(cam_info["intrinsics"])
            cam_intrinsic.append(intrinsic)
            ## lidar2img
            viewpad = np.eye(4)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            lidar2img_rt = viewpad @ lidar2cam_rt.T
            lidar2img_rts.append(lidar2img_rt)
            lidar2cam_rts.append(lidar2cam_rt.T)

        results = dict(
            image_paths=image_paths,
            distortions=distortions,
            lidar2img=lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            cam2lidar=cam2lidar_rts,
            cam_intrinsic=cam_intrinsic,
        )

        return results

    def load_images(self, results):
        image_paths = results["image_paths"]
        imgs = [np.array(Image.open(str(image_path))) for image_path in image_paths]
        if self._config.to_bgr:
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
        results["imgs"] = imgs
        results["img_shape"] = [x.shape[:2] for x in imgs]
        return results

    def resize_crop_flip_img(self, results, test_mode):
        H, W = self._config.H, self._config.W
        fH, fW = self._config.final_dim
        if not test_mode:
            resize = np.random.uniform(*self._config.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self._config.bot_pct_lim))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self._config.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self._config.rot_lim)
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self._config.bot_pct_lim)) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
        }

        imgs = results["imgs"]
        lidar2img = results["lidar2img"]
        cam_intrinsic = results["cam_intrinsic"]
        N = len(imgs)
        new_imgs = []
        for i in range(N):
            img, mat = self._img_transform(
                imgs[i], aug_config,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            lidar2img[i] = mat @ lidar2img[i]
            cam_intrinsic[i][:3, :3] = mat[:3, :3] @ cam_intrinsic[i][:3, :3]

        results["imgs"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        results["lidar2img"] = lidar2img
        results["cam_intrinsic"] = cam_intrinsic

        return results
    
    def _img_transform(self, img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array(
                [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix

    def ego_rotation(self, results, targets, test_mode):
        if not test_mode:
            angle = np.random.uniform(*self._config.rot3d_range)
        else:
            angle = 0
        
        if angle == 0:
            return results, targets

        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_T_2d = rot_mat[:2, :2].T
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["imgs"])
        lidar2img = results["lidar2img"]
        lidar2cam = results["lidar2cam"]
        for view in range(num_view):
            lidar2img[view] = lidar2img[view] @ rot_mat_inv
            lidar2cam[view] = lidar2cam[view] @ rot_mat_inv
        results["lidar2img"] = lidar2img
        results["lidar2cam"] = lidar2cam

        def wrap_to_pi_half_open(angle):
            """Map angle (rad) to [-pi, pi)."""
            return (angle + np.pi) % (2 * np.pi) - np.pi

        path = targets["path"].numpy()
        path[:, :2] = (path[:, :2] @ rot_mat_T_2d)
        path[:, 2] += angle
        path[:, 2] = wrap_to_pi_half_open(path[:, 2])
        targets["path"] = torch.tensor(path)

        trajectory = targets["trajectory"].numpy()
        trajectory[:, :2] = (trajectory[:, :2] @ rot_mat_T_2d)
        trajectory[:, 2] += angle
        trajectory[:, 2] = wrap_to_pi_half_open(trajectory[:, 2])
        targets["trajectory"] = torch.tensor(trajectory)

        return results, targets
            
    def photo_metric_distortion(self, results, test_mode):
        if test_mode or not self._config.photo_metric_distortion:
            return results

        brightness_delta = 32
        contrast_range = (0.5, 1.5)
        saturation_range = (0.5, 1.5)
        hue_delta = 18
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

        imgs = results["imgs"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if np.random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            if self._config.to_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # random saturation
            if np.random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if np.random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            if self._config.to_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

            # random contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if np.random.randint(2):
                img = img[..., np.random.permutation(3)]
            new_imgs.append(img)
        results["imgs"] = new_imgs
        return results

    def normalize_img(self, results):
        mean = np.array(self._config.img_mean, dtype=np.float32)
        std = np.array(self._config.img_std, dtype=np.float32)

        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))

        imgs = results["imgs"]
        for i in range(len(imgs)):
            img = imgs[i].copy().astype(np.float32)
            cv2.subtract(img, mean, img)  # inplace
            cv2.multiply(img, stdinv, img)  # inplace
            imgs[i] = img

        results["imgs"] = imgs

        return results

    def data_adapter(self, results):
        results.pop("image_paths")
        for key in ['distortions', 'lidar2img', 'lidar2cam', 'cam2lidar', 'cam_intrinsic']:
            results[key] = torch.tensor(np.stack(results[key]))

        imgs = [img.transpose(2, 0, 1) for img in results["imgs"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        imgs = torch.tensor(imgs)
        results["imgs"] = imgs

        results["projection_mat"] = results["lidar2img"].float()
        results["image_wh"] = np.ascontiguousarray(
            np.array(results["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )

        return results


class SparseDriveTargetBuilder(AbstractTargetBuilder):
    """Output target builder for TransFuser."""

    def __init__(self, config: SparseDriveConfig):
        """
        Initializes target builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "sparsedrive_target"

    def compute_targets(self, scene: Scene, cfg: DictConfig) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        trajectory = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._config.trajectory_sampling.num_poses).poses
        )

        ## lateral path & londitudinal velocity
        data_path = Path(cfg.navsim_log_path)
        log_name = scene.scene_metadata.log_name
        initial_token = scene.scene_metadata.initial_token
        log_pickle_path = data_path / f"{log_name}.pkl"
        scene_dict_list = pickle.load(open(log_pickle_path, "rb"))
        for idx, scene_dict in enumerate(scene_dict_list):
            token = scene_dict["token"]
            if token != initial_token:
                continue
            path, path_mask = self._get_future_path(idx, scene_dict_list)
            pad_trajectory = torch.cat([torch.zeros(1, 2), trajectory[:, :2]], dim=0)
            velocity = torch.norm(pad_trajectory[1:] - pad_trajectory[:-1], dim=-1) / self._config.vel_time_interval
            break

        return {
            "trajectory": trajectory,
            "path": path,
            "path_mask": path_mask,
            "velocity": velocity,
        }

    def _get_future_path(self, idx, scene_dict_list):
        num_pts = self._config.len_path
        interval = self._config.path_interval
        global_ego_poses = []
        distances = [0.0]
        accumulated_distance = 0.0
        max_dis = num_pts * interval
        for frame_idx in range(idx, len(scene_dict_list)):
            scene_frame = scene_dict_list[frame_idx]
            ego_translation = scene_frame["ego2global_translation"]
            ego_quaternion = Quaternion(*scene_frame["ego2global_rotation"])
            ego_pose = np.array(
                [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
                dtype=np.float64,
            )

            if global_ego_poses:
                prev_pose = global_ego_poses[-1]
                distance = np.linalg.norm(ego_pose[:2] - prev_pose[:2])
                distances.append(distance)
                accumulated_distance += distance

            global_ego_poses.append(ego_pose)

            if accumulated_distance > max_dis:
                break

        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[0]), np.array(global_ego_poses, dtype=np.float64)
        )

        distances = np.cumsum(distances)
        target_distance = np.arange(1, (num_pts + 1), 1) * interval
        path = np.array(
            [
                np.interp(target_distance, distances, local_ego_poses[:, 0]),
                np.interp(target_distance, distances, local_ego_poses[:, 1]),
                np.interp(target_distance, distances, local_ego_poses[:, 2]),
            ]
        ).T

        # limit yaw of path to [-pi, pi)
        path[:, 2] = (path[:, 2] + np.pi) % (2 * np.pi) - np.pi

        path_mask = np.ones(num_pts, dtype=np.float32)
        valid_points = min(num_pts, int(np.floor(accumulated_distance / interval)))
        path_mask[valid_points:] = 0
        return torch.tensor(path), torch.tensor(path_mask)