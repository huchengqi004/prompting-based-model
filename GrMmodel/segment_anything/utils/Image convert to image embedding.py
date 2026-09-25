import numpy as np
import os
from glob import glob
import argparse
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate npy files (image embeddings and/or ground truth) from images, masks, or npz files."
    )

    parser.add_argument("--img_path", type=str, default=None,
                        help="Path to the folder containing original images (optional).")
    parser.add_argument("--gt_path", type=str,

                        # default=r'A:\B_diskexpand\Code\GrMflakes\data\SEM\modelresults\testimage\SEMAP-smallscale\pipeline_temp\forecoarse',
                        # default=r'A:\B_diskexpand\Code\GrMflakes\data\OM-Grflakes\Test\images\BGRwithGraycontrastotsu\BGR',
                        # default=r'A:\B_diskexpand\Code\GrMflakes\data\OM-Grflakes\Test\images\BGRwithGraycontrastotsu\otsu',
                        # default=r'A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\OM\coarse-fore100back150\fore',
                        default=r'A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\AFM\fore',
                        help="Path to the folder containing masks/ground truth (optional).")
    parser.add_argument("--npz_path", type=str, default=None,
                        help="Path to an existing npz file (optional). If provided, skip image/mask processing.")

    parser.add_argument("-o", "--save_path", type=str,
                        # default=r'A:\B_diskexpand\Code\GrMflakes\data\SEM\modelresults\testimage\SEMAP-smallscale\pipeline_temp\forecoarse_npy',
                        # default=r'A:\B_diskexpand\Code\GrMflakes\data\SEM\modelresults\testimage\SEMAP-largescale\pipeline_temp\forecoarse_npy',
                        # default=r'A:\B_diskexpand\Code\GrMflakes\data\OM-Grflakes\Test\images\GrSiO2otsu_npy',
                        # default=r'A:\B_diskexpand\Code\GrMflakes\data\OM-Grflakes\Test\images\GrSiO2BGR_npy',
                        # default=r'A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\OM\coarse-fore100back150\fore',
                        default=r'A:\B_diskexpand\software-patent\SAM-Graphene\PPTdata\AFM\Fore_npy',
                        help="Directory to save npy files (and optionally npz).")
    parser.add_argument("--data_name", type=str, default="Grflakes",
                        help="Dataset name used for the intermediate npz file.")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--img_name_suffix", type=str, default=".jpg")
    parser.add_argument("--label_id", type=int, default=255)

    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


def prepare_mask(gt_data, label_id, image_size):
    if len(gt_data.shape) == 3:
        gt_data = gt_data[:, :, 0]
    assert len(gt_data.shape) == 2, "Mask must be 2D"

    gt_data = (gt_data == label_id)
    gt_data = transform.resize(
        gt_data, (image_size, image_size), order=0,
        preserve_range=True, mode="constant"
    )
    gt_data = np.uint8(gt_data)
    return gt_data


def prepare_image(image_data, image_size):
    if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
        image_data = image_data[:, :, :3]
    if len(image_data.shape) == 2:
        image_data = np.repeat(image_data[:, :, None], 3, axis=-1)

    lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - np.min(image_data_pre))
        / (np.max(image_data_pre) - np.min(image_data_pre))
        * 255.0
    )
    image_data_pre[image_data == 0] = 0

    image_data_pre = transform.resize(
        image_data_pre, (image_size, image_size), order=3,
        preserve_range=True, mode="constant", anti_aliasing=True
    )
    image_data_pre = np.uint8(image_data_pre)
    return image_data_pre


def process_folders(img_path, gt_path, image_size, img_name_suffix, label_id, sam_model=None, device=None):
    """
    处理成对的原图和 mask，返回：
        imgs: 预处理后的图像列表
        gts: 二值 mask 列表
        img_embeddings: 图像嵌入列表
        gt_filenames: mask 文件名（不含扩展名）
        img_filenames: 原图文件名（不含扩展名，仅对应有嵌入的样本）
    """
    imgs, gts, img_embeddings = [], [], []
    gt_filenames, img_filenames = [], []

    mask_files = sorted(os.listdir(gt_path)) if gt_path else []
    image_dict = {}
    if img_path:
        for f in sorted(os.listdir(img_path)):
            image_dict[f.split('.')[0]] = f  # 键为不带扩展名的文件名

    for gt_name in tqdm(mask_files, desc="Processing paired masks"):
        gt_path_full = os.path.join(gt_path, gt_name)
        gt_data = io.imread(gt_path_full)
        gt_resized = prepare_mask(gt_data, label_id, image_size)

        if np.sum(gt_resized) <= 100:
            continue

        gts.append(gt_resized)
        gt_filenames.append(os.path.splitext(gt_name)[0])  # 记录 mask 文件名

        gt_base = os.path.splitext(gt_name)[0]
        if gt_base in image_dict:
            img_name = image_dict[gt_base]
            image_data = io.imread(os.path.join(img_path, img_name))
            img_pre = prepare_image(image_data, image_size)
            imgs.append(img_pre)
            img_filenames.append(os.path.splitext(img_name)[0])  # 记录原图文件名

            if sam_model is not None:
                sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                resize_img = sam_transform.apply_image(img_pre)
                resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
                input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])
                with torch.no_grad():
                    embedding = sam_model.image_encoder(input_image)
                    img_embeddings.append(embedding.cpu().numpy()[0])

    return imgs, gts, img_embeddings, gt_filenames, img_filenames


def process_only_masks(gt_path, image_size, label_id):
    gts, gt_filenames = [], []
    mask_files = sorted(os.listdir(gt_path))
    for gt_name in tqdm(mask_files, desc="Processing masks only"):
        gt_data = io.imread(os.path.join(gt_path, gt_name))
        gt_resized = prepare_mask(gt_data, label_id, image_size)
        if np.sum(gt_resized) > 100:
            gts.append(gt_resized)
            gt_filenames.append(os.path.splitext(gt_name)[0])
    return gts, gt_filenames


def process_only_images(img_path, image_size, sam_model, device):
    imgs, img_embeddings, img_filenames = [], [], []
    image_files = sorted(os.listdir(img_path))
    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    for img_name in tqdm(image_files, desc="Processing images only"):
        image_data = io.imread(os.path.join(img_path, img_name))
        img_pre = prepare_image(image_data, image_size)
        imgs.append(img_pre)
        img_filenames.append(os.path.splitext(img_name)[0])

        resize_img = sam_transform.apply_image(img_pre)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
        input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])
        with torch.no_grad():
            embedding = sam_model.image_encoder(input_image)
            img_embeddings.append(embedding.cpu().numpy()[0])
    return imgs, img_embeddings, img_filenames


def process_npz(npz_file, save_path, sam_model=None, device=None):
    data = np.load(npz_file)
    print("npz 文件包含的键:", data.files)

    if 'gts' not in data:
        raise ValueError("npz 文件中没有 'gts' 键。")

    gts = data['gts']
    img_embeddings = None

    if 'img_embeddings' in data:
        img_embeddings = data['img_embeddings']
    elif 'imgs' in data and sam_model is not None:
        print("npz 中没有 img_embeddings，正在用 imgs 重新计算...")
        imgs = data['imgs']
        img_embeddings = []
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        for img in tqdm(imgs, desc="Computing embeddings"):
            resize_img = sam_transform.apply_image(img)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])
            with torch.no_grad():
                embedding = sam_model.image_encoder(input_image)
                img_embeddings.append(embedding.cpu().numpy()[0])
        img_embeddings = np.stack(img_embeddings, axis=0)
    else:
        raise ValueError("无法获取 img_embeddings")

    save_emb_path = os.path.join(save_path, "npy_embs")
    save_gt_path = os.path.join(save_path, "npy_gts")
    os.makedirs(save_emb_path, exist_ok=True)
    os.makedirs(save_gt_path, exist_ok=True)

    # npz 中不保留文件名，因此使用索引保存
    for i in range(len(gts)):
        np.save(os.path.join(save_gt_path, f"{i}.npy"), gts[i])
    if img_embeddings is not None:
        for i in range(len(img_embeddings)):
            np.save(os.path.join(save_emb_path, f"{i}.npy"), img_embeddings[i])
    print(f"已保存 {len(gts)} 个 gt 和 {len(img_embeddings) if img_embeddings is not None else 0} 个嵌入。")


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # 情况1：从 npz 生成 npy
    if args.npz_path is not None:
        print("模式：从 npz 文件生成 npy")
        need_sam = False
        data = np.load(args.npz_path)
        if 'img_embeddings' not in data:
            need_sam = True
        if need_sam:
            if args.checkpoint is None:
                raise ValueError("npz 中没有 img_embeddings，需要提供 SAM checkpoint。")
            sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
            process_npz(args.npz_path, args.save_path, sam_model, args.device)
        else:
            process_npz(args.npz_path, args.save_path)
        return

    # 情况2：同时提供 img_path 和 gt_path
    if args.img_path is not None and args.gt_path is not None:
        if args.checkpoint is None:
            raise ValueError("需要计算图像嵌入，请提供 SAM checkpoint。")
        sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)

        imgs, gts, img_embeddings, gt_filenames, img_filenames = process_folders(
            args.img_path, args.gt_path, args.image_size,
            args.img_name_suffix, args.label_id, sam_model, args.device
        )

        save_emb_path = os.path.join(args.save_path, "npy_embs")
        save_gt_path = os.path.join(args.save_path, "npy_gts")
        os.makedirs(save_emb_path, exist_ok=True)
        os.makedirs(save_gt_path, exist_ok=True)

        # 保存 gt，文件名与 mask 一致
        for gt, name in zip(gts, gt_filenames):
            np.save(os.path.join(save_gt_path, f"{name}.npy"), gt)
        # 保存嵌入，文件名与原图一致
        for emb, name in zip(img_embeddings, img_filenames):
            np.save(os.path.join(save_emb_path, f"{name}.npy"), emb)

        print(f"已保存 {len(gt_filenames)} 个 gt 和 {len(img_filenames)} 个嵌入。")
        return

    # 情况3：仅处理 mask
    if args.gt_path is not None:
        gts, gt_filenames = process_only_masks(args.gt_path, args.image_size, args.label_id)
        save_gt_path = os.path.join(args.save_path, "npy_gts")
        os.makedirs(save_gt_path, exist_ok=True)
        for gt, name in zip(gts, gt_filenames):
            np.save(os.path.join(save_gt_path, f"{name}.npy"), gt)
        print(f"已保存 {len(gt_filenames)} 个 gt。")
        return

    # 情况4：仅处理原图
    if args.img_path is not None:
        if args.checkpoint is None:
            raise ValueError("需要计算图像嵌入，请提供 SAM checkpoint。")
        sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
        imgs, img_embeddings, img_filenames = process_only_images(
            args.img_path, args.image_size, sam_model, args.device
        )
        save_emb_path = os.path.join(args.save_path, "npy_embs")
        os.makedirs(save_emb_path, exist_ok=True)
        for emb, name in zip(img_embeddings, img_filenames):
            np.save(os.path.join(save_emb_path, f"{name}.npy"), emb)
        print(f"已保存 {len(img_filenames)} 个嵌入。")
        return

    print("未提供有效输入，请至少指定 --img_path、--gt_path 或 --npz_path 之一。")


if __name__ == "__main__":
    main()

# import numpy as np
# import os
# from glob import glob
# import argparse
# from skimage import transform, io, segmentation
# from tqdm import tqdm
# import torch
# from segment_anything import sam_model_registry
# from segment_anything.utils.transforms import ResizeLongestSide
# from PIL import Image
#
# # 获取当前文件目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Generate npy files (image embeddings and/or ground truth) from images, masks, or npz files.")
#
#     # 输入选项
#     parser.add_argument(
#         "--img_path", type=str, default=None,
#         help="Path to the folder containing original images (optional)."
#     )
#     parser.add_argument(
#         "--gt_path", type=str,
#         default=r'A:\B_diskexpand\Code\GrMflakes\data\SEM\modelresults\testimage\SEMAP\pipeline_temp\contrast_fore',
#         help="Path to the folder containing masks/ground truth (optional)."
#     )
#     parser.add_argument(
#         "--npz_path", type=str, default = None,
#         help="Path to an existing npz file (optional). If provided, skip image/mask processing."
#     )
#
#     # 输出
#     parser.add_argument(
#         "-o", "--save_path", type=str, default=r'A:\B_diskexpand\Code\GrMflakes\data\SEM\modelresults\testimage\SEMAP\pipeline_temp\forecoarse_npy',
#         help="Directory to save npy files (and optionally npz)."
#     )
#     parser.add_argument(
#         "--data_name", type=str, default="Grflakes",
#         help="Dataset name used for the intermediate npz file."
#     )
#
#     # 处理参数
#     parser.add_argument("--image_size", type=int, default=256, help="Image size for resizing masks and images.")
#     parser.add_argument("--img_name_suffix", type=str, default=".jpg", help="Suffix of original image files.")
#     parser.add_argument("--label_id", type=int, default=255, help="Pixel value for foreground in masks.")
#
#     # SAM 模型参数（仅当需要计算嵌入时使用）
#     parser.add_argument("--model_type", type=str, default="vit_b", help="SAM model type.")
#     parser.add_argument(
#         "--checkpoint", type=str, default=None,
#         help="Path to SAM checkpoint. Required if computing embeddings."
#     )
#     parser.add_argument("--device", type=str, default="cuda:0", help="Device for SAM model.")
#
#
#     return parser.parse_args()
#
#
# def prepare_mask(gt_data, label_id, image_size):
#     """将 mask 转换为二值图像并缩放到 image_size。"""
#     if len(gt_data.shape) == 3:
#         gt_data = gt_data[:, :, 0]
#     assert len(gt_data.shape) == 2, "Mask must be 2D"
#
#     gt_data = (gt_data == label_id)
#     gt_data = transform.resize(
#         gt_data,
#         (image_size, image_size),
#         order=0,
#         preserve_range=True,
#         mode="constant",
#     )
#     gt_data = np.uint8(gt_data)
#     return gt_data
#
#
# def prepare_image(image_data, image_size):
#     """预处理原始图像：转为 RGB、归一化并缩放。"""
#     if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
#         image_data = image_data[:, :, :3]
#     if len(image_data.shape) == 2:
#         image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
#
#     lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
#     image_data_pre = np.clip(image_data, lower_bound, upper_bound)
#     image_data_pre = (
#             (image_data_pre - np.min(image_data_pre))
#             / (np.max(image_data_pre) - np.min(image_data_pre))
#             * 255.0
#     )
#     image_data_pre[image_data == 0] = 0
#
#     image_data_pre = transform.resize(
#         image_data_pre,
#         (image_size, image_size),
#         order=3,
#         preserve_range=True,
#         mode="constant",
#         anti_aliasing=True,
#     )
#     image_data_pre = np.uint8(image_data_pre)
#     return image_data_pre
#
#
# def process_folders(img_path, gt_path, image_size, img_name_suffix, label_id, sam_model=None, device=None):
#     """
#     从原始图像文件夹和 mask 文件夹生成 imgs, gts, img_embeddings 列表。
#     如果未提供 img_path，则 imgs 和 img_embeddings 为空；如果未提供 gt_path，则 gts 为空。
#     """
#     imgs, gts, img_embeddings = [], [], []
#
#     # 读取 mask 文件列表（如果提供了 gt_path）
#     if gt_path is not None:
#         mask_files = sorted(os.listdir(gt_path))
#     else:
#         mask_files = []
#
#     # 读取图像文件列表（如果提供了 img_path）
#     if img_path is not None:
#         image_files = sorted(os.listdir(img_path))
#         # 按文件名前缀匹配 mask 和 image（这里简单使用相同前缀，忽略后缀）
#         image_dict = {f.split('.')[0]: f for f in image_files}
#     else:
#         image_dict = {}
#
#     # 遍历 mask 文件（如果存在）
#     for gt_name in tqdm(mask_files, desc="Processing masks"):
#         gt_path_full = os.path.join(gt_path, gt_name)
#         gt_data = io.imread(gt_path_full)
#         gt_resized = prepare_mask(gt_data, label_id, image_size)
#
#         if np.sum(gt_resized) <= 100:
#             continue  # 过滤小目标
#
#         gts.append(gt_resized)
#
#         # 查找对应的原图
#         gt_base = gt_name.split('.')[0]
#         if gt_base in image_dict:
#             img_name = image_dict[gt_base]
#             image_data = io.imread(os.path.join(img_path, img_name))
#             img_pre = prepare_image(image_data, image_size)
#             imgs.append(img_pre)
#
#             # 计算嵌入
#             if sam_model is not None:
#                 sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
#                 resize_img = sam_transform.apply_image(img_pre)
#                 resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
#                 input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])
#                 with torch.no_grad():
#                     embedding = sam_model.image_encoder(input_image)
#                     img_embeddings.append(embedding.cpu().numpy()[0])
#         else:
#             # 未提供 img_path 或没有找到对应原图，仍保留 gt，但 imgs 和 embeddings 不添加
#             # 为了保持列表长度一致，我们可以选择跳过没有原图的样本？这里决定保留所有 mask，即使没有原图
#             # 但 imgs 和 img_embeddings 长度会少于 gts，后续需要处理
#             # 此处选择：只保留同时有 mask 和原图的样本，如果用户要求只处理 mask 则单独处理
#             # 为了灵活性，如果 img_path 未提供，则 gts 单独保存，imgs 和 embeddings 为空
#             pass
#
#     return imgs, gts, img_embeddings
#
#
# def process_only_masks(gt_path, image_size, label_id):
#     """仅处理 mask 文件夹，返回 gts 列表。"""
#     gts = []
#     mask_files = sorted(os.listdir(gt_path))
#     for gt_name in tqdm(mask_files, desc="Processing masks only"):
#         gt_data = io.imread(os.path.join(gt_path, gt_name))
#         gt_resized = prepare_mask(gt_data, label_id, image_size)
#         if np.sum(gt_resized) > 100:
#             gts.append(gt_resized)
#     return gts
#
#
# def process_only_images(img_path, image_size, sam_model, device):
#     """仅处理原图文件夹，返回 imgs 和 img_embeddings 列表。"""
#     imgs, img_embeddings = [], []
#     image_files = sorted(os.listdir(img_path))
#     sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
#     for img_name in tqdm(image_files, desc="Processing images only"):
#         image_data = io.imread(os.path.join(img_path, img_name))
#         img_pre = prepare_image(image_data, image_size)
#         imgs.append(img_pre)
#
#         resize_img = sam_transform.apply_image(img_pre)
#         resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
#         input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])
#         with torch.no_grad():
#             embedding = sam_model.image_encoder(input_image)
#             img_embeddings.append(embedding.cpu().numpy()[0])
#     return imgs, img_embeddings
#
#
# def process_npz(npz_file, save_path, sam_model=None, device=None):
#     """从已有 npz 文件生成 npy（嵌入和 gt）。如果 npz 中没有 img_embeddings，则用 imgs 重新计算。"""
#     data = np.load(npz_file)
#     print("npz 文件包含的键:", data.files)
#
#     if 'gts' in data:
#         gts = data['gts']
#     else:
#         raise ValueError("npz 文件中没有 'gts' 键。")
#
#     if 'img_embeddings' in data:
#         img_embeddings = data['img_embeddings']
#     elif 'imgs' in data and sam_model is not None:
#         print("npz 中没有 img_embeddings，正在用 imgs 重新计算...")
#         imgs = data['imgs']
#         img_embeddings = []
#         sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
#         for img in tqdm(imgs, desc="Computing embeddings"):
#             resize_img = sam_transform.apply_image(img)
#             resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
#             input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])
#             with torch.no_grad():
#                 embedding = sam_model.image_encoder(input_image)
#                 img_embeddings.append(embedding.cpu().numpy()[0])
#         img_embeddings = np.stack(img_embeddings, axis=0)
#     else:
#         raise ValueError("无法获取 img_embeddings：npz 中没有 'img_embeddings'，也没有 'imgs' 或未提供 SAM 模型。")
#
#     # 保存 npy
#     save_emb_path = os.path.join(save_path, "npy_embs")
#     save_gt_path = os.path.join(save_path, "npy_gts")
#     os.makedirs(save_emb_path, exist_ok=True)
#     os.makedirs(save_gt_path, exist_ok=True)
#
#     num_samples = min(len(gts), len(img_embeddings))
#     for i in range(num_samples):
#         np.save(os.path.join(save_emb_path, f"{i}.npy"), img_embeddings[i])
#         np.save(os.path.join(save_gt_path, f"{i}.npy"), gts[i])
#     print(f"已保存 {num_samples} 个样本的 npy 文件。")
#
#
# def main():
#     args = parse_args()
#
#     # 创建输出目录
#     os.makedirs(args.save_path, exist_ok=True)
#
#     # 如果提供了 npz_path，直接处理 npz
#     if args.npz_path is not None:
#         print("模式：从 npz 文件生成 npy")
#         # 如果需要计算嵌入，且 npz 中没有 img_embeddings，则需要 SAM 模型
#         need_sam = False
#         data = np.load(args.npz_path)
#         if 'img_embeddings' not in data:
#             need_sam = True
#         if need_sam:
#             if args.checkpoint is None:
#                 raise ValueError("npz 中没有 img_embeddings，需要提供 SAM checkpoint 以重新计算。")
#             sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
#             process_npz(args.npz_path, args.save_path, sam_model, args.device)
#         else:
#             process_npz(args.npz_path, args.save_path)
#         return
#
#     # 否则根据 img_path 和 gt_path 的组合处理
#     imgs, gts, img_embeddings = [], [], []
#
#     # 初始化 SAM 模型（如果需要计算嵌入）
#     sam_model = None
#     need_embedding = args.img_path is not None
#     if need_embedding:
#         if args.checkpoint is None:
#             raise ValueError("需要计算图像嵌入，请提供 SAM checkpoint。")
#         sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
#
#     # 处理 mask（如果提供）
#     if args.gt_path is not None:
#         print("处理 mask 文件夹...")
#         gts = process_only_masks(args.gt_path, args.image_size, args.label_id)
#     else:
#         print("未提供 mask 文件夹，将不生成 gt。")
#
#     # 处理图像（如果提供）
#     if args.img_path is not None:
#         print("处理图像文件夹并计算嵌入...")
#         imgs, img_embeddings = process_only_images(args.img_path, args.image_size, sam_model, args.device)
#     else:
#         print("未提供图像文件夹，将不生成图像嵌入。")
#
#     # 保存 npz（如果同时有 imgs 和 gts，且用户可能想要保留）
#     if len(imgs) > 0 and len(gts) > 0:
#         # 注意：此时 imgs 和 gts 的长度可能不同，因为 process_only_masks 和 process_only_images 独立处理
#         # 为了保持一致性，这里需要匹配：只保留同时有 imgs 和 gts 的样本
#         # 但当前独立处理无法匹配。因此更合理的是使用 process_folders 同时处理。我们改为使用 process_folders 函数。
#         pass
#
#     # 由于上述简单独立处理无法保证 img 和 mask 配对，我们推荐使用 process_folders 函数
#     # 为了保持脚本简洁，这里重建为使用 process_folders
#     print("重新使用配对处理流程...")
#     imgs, gts, img_embeddings = process_folders(
#         args.img_path, args.gt_path, args.image_size, args.img_name_suffix,
#         args.label_id, sam_model, args.device
#     )
#
#     # 保存 npz（可选）
#     if len(imgs) > 0 and len(gts) > 0:
#         npz_path = os.path.join(args.save_path, args.data_name + ".npz")
#         np.savez_compressed(
#             npz_path,
#             imgs=np.stack(imgs, axis=0) if len(imgs) > 0 else None,
#             gts=np.stack(gts, axis=0),
#             img_embeddings=np.stack(img_embeddings, axis=0) if len(img_embeddings) > 0 else None,
#         )
#         print(f"已保存 npz 文件: {npz_path}")
#
#     # 保存 npy 文件
#     save_emb_path = os.path.join(args.save_path, "npy_embs")
#     save_gt_path = os.path.join(args.save_path, "npy_gts")
#     os.makedirs(save_emb_path, exist_ok=True)
#     os.makedirs(save_gt_path, exist_ok=True)
#
#     num_samples = min(len(gts), len(img_embeddings)) if len(img_embeddings) > 0 else len(gts)
#     for i in range(num_samples):
#         if len(img_embeddings) > 0:
#             np.save(os.path.join(save_emb_path, f"{i}.npy"), img_embeddings[i])
#         np.save(os.path.join(save_gt_path, f"{i}.npy"), gts[i])
#
#     print(f"已保存 {num_samples} 个 gt 和 {len(img_embeddings)} 个嵌入 npy 文件。")
#
#
# if __name__ == "__main__":
#     main()