#!/usr/bin/env python3
"""Adapter 训练数据集。

从 data/atomaction_codebook_index.json 读取标签与元数据，训练时在线读取原始 PNG。

约定：
	1. 图像保持**原始 uint8**，全部预处理（→128→224→ImageNet 归一化）由
	   AdapterImageTokenizer 在 forward 内完成，保证训练与部署走同一条管线。
	2. 指令是 short 原文（含句尾句号），去句号由 AdapterTextTokenizer.normalize_instruction 负责。
	3. 相机顺序与 config/model.yaml::Adapter.image_tokenizer.camera_names 必须一致。
	4. adapter_collate_fn 的输出直接对齐 Adapter.forward(camera_images, instructions) 的签名。

用法（自检）：
	python data/adapter_dataset.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INDEX = REPO_ROOT / "data" / "atomaction_codebook_index.json"
# 与 data/import_codebook_index.py::ADAPTER_CAMERAS 及 Adapter 的 camera_names 保持一致。
ADAPTER_CAMERAS = ("front", "wrist")
CAMERA_IMAGE_FIELDS = {"front": "img_front", "wrist": "img_wrist"}


"""Adapter 训练数据集。

输入：
	__init__:
		index_path: 标签 JSON 路径，需已由 build_adapter_split.py 写入 split 字段。
		split: `train` 或 `val`。
		dataset_root: AtomAction_Dataset 根目录，默认取 JSON meta 中记录的路径。
	__getitem__:
		index: 样本下标。

输出：
	__getitem__:
		front / wrist: torch.uint8，[3, H, W]，原始 PNG 未做任何预处理。
		instruction: str，short 指令原文。
		k_global: torch.long，标量。
		k_detail: torch.long，[num_detail]。
		phase_id / action / task: str，供分组指标使用。
"""
class AdapterDataset(Dataset):

	def __init__(
		self,
		index_path: Path = DEFAULT_INDEX,
		split: str = "train",
		dataset_root: Optional[str] = None,
	) -> None:
		self.index_path = Path(index_path)
		self.split = str(split)

		with self.index_path.open("r", encoding="utf-8") as file:
			payload = json.load(file)

		self.meta: Dict[str, Any] = payload["meta"]
		if "split_strategy" not in self.meta:
			raise ValueError(
				f"{self.index_path} does not carry a split. Run data/build_adapter_split.py first."
			)

		self.records: List[Dict[str, Any]] = [
			record for record in payload["records"] if record.get("split") == self.split
		]
		if not self.records:
			available = sorted({record.get("split") for record in payload["records"]})
			raise ValueError(f"No records for split={self.split}. Available splits: {available}")

		self.dataset_root = Path(dataset_root or self.meta["dataset_root"])
		if not self.dataset_root.is_dir():
			raise FileNotFoundError(f"dataset_root does not exist: {self.dataset_root}")
		self.num_detail = int(self.meta["num_detail"])

	def __len__(self) -> int:
		return len(self.records)

	"""读取单张 RGB 图像为 uint8 张量。

	维度变化：
		PNG -> [H, W, 3] -> [3, H, W]
	"""
	@staticmethod
	def load_image(image_path: Path) -> torch.Tensor:
		with Image.open(image_path) as image:
			image_array = np.array(image.convert("RGB"))
		return torch.from_numpy(image_array).permute(2, 0, 1).contiguous()

	def __getitem__(self, index: int) -> Dict[str, Any]:
		record = self.records[index]
		phase_dir = self.dataset_root / record["phase_id"]

		sample: Dict[str, Any] = {
			"instruction": record["instruction"],
			"k_global": torch.tensor(int(record["k_global"]), dtype=torch.long),
			"k_detail": torch.tensor([int(value) for value in record["k_detail"]], dtype=torch.long),
			"phase_id": record["phase_id"],
			"action": record["action"],
			"task": record["task"],
		}
		for camera in ADAPTER_CAMERAS:
			sample[camera] = self.load_image(phase_dir / record[CAMERA_IMAGE_FIELDS[camera]])
		return sample


"""将 AdapterDataset 的样本列表整理成批数据。

输入：
	batch: List[Dict[str, Any]]，每个元素对应 AdapterDataset.__getitem__ 的返回值。

输出：
	camera_images: Dict[str, torch.Tensor]，每路相机 [B, 3, H, W]，uint8。
	instructions: List[str]，长度 B。
	k_global: torch.LongTensor，[B]。
	k_detail: torch.LongTensor，[B, num_detail]。
	meta: Dict[str, List[str]]，phase_id / action / task。
"""
def adapter_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
	if not isinstance(batch, Sequence) or len(batch) == 0:
		raise ValueError("batch must be a non-empty sequence")

	return {
		"camera_images": {
			camera: torch.stack([sample[camera] for sample in batch], dim=0)
			for camera in ADAPTER_CAMERAS
		},
		"instructions": [sample["instruction"] for sample in batch],
		"k_global": torch.stack([sample["k_global"] for sample in batch], dim=0),
		"k_detail": torch.stack([sample["k_detail"] for sample in batch], dim=0),
		"meta": {
			"phase_id": [sample["phase_id"] for sample in batch],
			"action": [sample["action"] for sample in batch],
			"task": [sample["task"] for sample in batch],
		},
	}


if __name__ == "__main__":
	for split_name in ("train", "val"):
		dataset = AdapterDataset(split=split_name)
		print(f"{split_name}: {len(dataset)} samples | dataset_root={dataset.dataset_root}")

	dataset = AdapterDataset(split="val")
	batch = adapter_collate_fn([dataset[index] for index in range(4)])
	for camera, images in batch["camera_images"].items():
		print(f"  {camera}: {tuple(images.shape)} {images.dtype}")
	print(f"  instructions: {batch['instructions']}")
	print(f"  k_global: {batch['k_global'].tolist()}")
	print(f"  k_detail[0]: {batch['k_detail'][0].tolist()}")
	print(f"  meta.action: {batch['meta']['action']}")
