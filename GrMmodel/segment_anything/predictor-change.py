# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import numpy as np
import torch
from segment_anything.modeling.sam import Sam
from typing import Optional, Tuple
from .utils.transforms import ResizeLongestSide
import torch
from typing import List, Tuple

import torch


class ResizeLongestSide:
    def __init__(self, size):
        self.size = size

    def apply_image(self, image):
        h, w = image.shape[:2]
        scale = self.size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        return resized_image

    def apply_coords(self, coords, orig_size):
        h, w = orig_size
        scale = self.size / max(h, w)
        return coords * scale


class NormalizeCoords:
    def __init__(self, orig_size):
        self.orig_size = orig_size

    def apply_coords(self, coords):
        h, w = self.orig_size
        return coords * [w, h]


class SamPredictor:
    def __init__(
            self,
            sam_model: Sam,
            device: str = "cuda"
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
          device (str): The device to use, can be "cuda" or "cpu".
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()
        self.original_size = None  # 添加 self.original_size 属性，初始为 None
        self._device = device

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        # 在 setter 方法中防止设备属性被修改
        raise AttributeError("Can't set attribute 'device'. Use the __init__ argument 'device' instead.")

    def set_image(
            self,
            image: np.ndarray,
            image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        if input_image.size == 0:
            raise ValueError("Invalid image size after transformation. Please select another image.")

        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        # 创建 NormalizeCoords 转换
        self.normalize_coords = NormalizeCoords(image.shape[:2])
        self.set_torch_image(input_image_torch, input_image.shape[:2])
        # 更新self.original_size
        self.original_size = input_image.shape[:2]

    # ... 其他方法不变 ...

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:

            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]


        if mask_input is not None:

            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]
        else:
            mask_input_tensor = None


        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            # mask_input_tensor,

            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    # def predict(self, prompt: str, point_coords: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     使用给定的提示和点坐标预测掩膜。
    #
    #     参数：
    #       prompt (str)：用于掩膜预测的提示。
    #       point_coords (List[Tuple[int, int]])：包含点坐标的列表，每个点坐标由 (x, y) 组成。
    #
    #     返回：
    #       Tuple[np.ndarray, np.ndarray, np.ndarray]：包含：
    #         - masks (np.ndarray)：预测的掩膜。形状为 (num_points, height, width) 的 3D 二进制 numpy 数组。
    #         - scores (np.ndarray)：预测得分。形状为 (num_points, num_classes) 的 2D numpy 数组，
    #           包含每个类别的得分。
    #         - logits (np.ndarray)：原始 logits。形状为 (num_points, num_classes) 的 2D numpy 数组，
    #           包含模型输出的未归一化值。
    #     """
    #     # 将 prompt 转换为张量并获取图像嵌入
    #     with torch.no_grad():
    #         prompt = self.model.encode_prompt(prompt, self.device)
    #         image_embeddings = self.model.encode_image(self.image_torch, self.device)
    #
    #     # 将 point_coords 转换为 torch 张量
    #     coords_torch = torch.tensor(point_coords, dtype=torch.float32, device=self.device)
    #
    #     # 重复 prompt 和 image embeddings，以匹配每个点
    #     num_points = len(point_coords)
    #     prompt_tiled = prompt.expand(num_points, -1)
    #     image_embeddings_tiled = image_embeddings.expand(num_points, -1)
    #
    #     # 预测掩膜和 logits
    #     with torch.no_grad():
    #         masks_torch, logits_torch = self.model.predict_mask_and_logits(
    #             prompt_tiled, coords_torch, image_embeddings_tiled
    #         )
    #
    #     # 将掩膜和 logits 转换为 numpy 数组
    #     masks = masks_torch.cpu().numpy()
    #     logits = logits_torch.cpu().numpy()
    #
    #     # 从 logits 计算得分
    #     scores = torch.softmax(logits_torch, dim=1).cpu().numpy()
    #
    #     return masks, scores, logits

    # def predict(self, point_coords: List[Tuple[int, int]], point_labels: List[int], box: Tuple[int, int, int, int],
    #             multimask_output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #
    #     """
    #     使用给定的点坐标和标签预测掩膜。
    #
    #     参数：
    #       point_coords (List[Tuple[int, int]])：包含点坐标的列表，每个点坐标由 (x, y) 组成。
    #       point_labels (List[int]): 包含与点坐标对应的标签的列表。0 表示负标签，1 表示正标签。
    #       box (Tuple[int, int, int, int]): 表示包围框的元组，由 (x_min, y_min, x_max, y_max) 组成。
    #
    #     返回：
    #       Tuple[np.ndarray, np.ndarray, np.ndarray]：包含：
    #         - masks (np.ndarray)：预测的掩膜。形状为 (num_points, height, width) 的 3D 二进制 numpy 数组。
    #         - scores (np.ndarray)：预测得分。形状为 (num_points, num_classes) 的 2D numpy 数组，
    #           包含每个类别的得分。
    #         - logits (np.ndarray)：原始 logits。形状为 (num_points, num_classes) 的 2D numpy 数组，
    #           包含模型输出的未归一化值。
    #     """
    #     # 将 point_coords 和 point_labels 转换为 torch 张量
    #     coords_torch = torch.tensor(point_coords, dtype=torch.float32, device=self.device)
    #     labels_torch = torch.tensor(point_labels, dtype=torch.long, device=self.device)
    #
    #     # 重复 box 和 image embeddings，以匹配每个点
    #     num_points = len(point_coords)
    #
    #     # 确保 box 不为空
    #     if box is not None and box[0] is not None:
    #         box = torch.tensor([box], dtype=torch.float32, device=self.device)
    #         # 使用 self.transform.resize_boxes() 处理 box 参数
    #         box_torch = self.transform.resize_boxes(box, self.original_size, self.image_size)
    #     else:
    #         raise ValueError("box 参数不能为空。")
    #
    #     # 之前的代码保持不变
    #
    #     # 确保 point_coords 和 point_labels 不为空
    #     if not point_coords or not point_labels:
    #         raise ValueError("point_coords 和 point_labels 参数不能为空。")
    #
    #     # 确保 point_coords 和 point_labels 的长度相等
    #     if len(point_coords) != len(point_labels):
    #         raise ValueError("point_coords 和 point_labels 的长度应相等。")
    #
    #
    #     # 预测掩膜和 logits
    #     with torch.no_grad():
    #         masks_torch, logits_torch = self.predict_torch(
    #             coords_torch,
    #             labels_torch,
    #             boxes=box_torch,
    #             multimask_output=False,
    #         )
    #
    #     # 将掩膜和 logits 转换为 numpy 数组
    #     masks = masks_torch.cpu().numpy()
    #     logits = logits_torch.cpu().numpy()
    #
    #     # 从 logits 计算得分
    #     scores = torch.softmax(logits_torch, dim=1).cpu().numpy()
    #
    #     return masks, scores, logits





    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
