#!/usr/bin/env python3
"""AtomAction 双码本索引导出脚本。

用冻结的 VQAP Stage 1 编码器为每个 phase 求出 (k_global, k_detail[9])，
连同图像路径 / 指令 / 帧号写入 JSON，作为 Adapter 的静态监督标签。

约定：
	1. 模型配置取自 checkpoint 内嵌的 model_args，并与当前 config/model.yaml 交叉校验，不一致直接报错。
	2. 只构建 AtomAction_NSVQ 子模块，按 `atomaction_nsvq.` 前缀过滤 checkpoint 权重，strict 加载。
	3. 指令按 source_phase_stats.phase_index 回查，而不是 phase 目录名（后者是样本序号，会静默查错）。
	4. 图像路径优先取 phase_metadata.json 的 view_selection_cache，缺失时回落到 {start_frame}.png。
	5. 全程 float32 + eval 模式，硬量化取 argmin，避免 bf16 在近邻边界上翻码。

用法：
	export COPPELIASIM_ROOT=/home/weizeming/weizeming/CoppeliaSim
	export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
	python data/import_codebook_index.py \
		--checkpoint checkpoints/vqap_pretrain/stage1/latest.pth \
		--output data/atomaction_codebook_index.json
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from data.dataset import AtomActionDataset, load_global_config
from data.utils import AtomActionDataset_collate_fn
from model.module.atomaction_nsvq import AtomAction_NSVQ
from model.module.nsvq import compute_perplexity

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "vqap_pretrain" / "stage1" / "latest.pth"
DEFAULT_CONFIG = REPO_ROOT / "config" / "model.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "atomaction_codebook_index.json"

MODULE_PREFIX = "atomaction_nsvq."
ADAPTER_CAMERAS = ("front", "wrist")
INSTRUCTION_STYLE = "short"

# 训练时的 per-rank batch size，困惑度对账必须用同一口径（ckpt 记录的是批内困惑度按 epoch 平均）。
PERPLEXITY_GROUP_SIZE = 64
REFERENCE_PERPLEXITY_GLOBAL = 17.78
REFERENCE_PERPLEXITY_DETAIL = 51.21
# 唯一 (task, variation, source_phase_index) 语义组合数。
# 设计文档记录的 586 是含 pose-adjust 的 18 动作口径；train_actions 白名单（17 动作）内为 572，
# 且与 variation_metadata 中的 phase_descriptions 条目一一对应（无孤儿描述、无缺描述样本）。
REFERENCE_SEMANTIC_COMBINATIONS = 572


"""流式计算文件 sha256。"""
def compute_file_sha256(file_path: Path, chunk_size: int = 1 << 20) -> str:
	digest = hashlib.sha256()
	with file_path.open("rb") as file:
		for chunk in iter(lambda: file.read(chunk_size), b""):
			digest.update(chunk)
	return digest.hexdigest()


"""携带样本下标的薄包装，不改动 AtomActionDataset 本身。"""
class IndexedDataset(Dataset):

	def __init__(self, dataset: AtomActionDataset) -> None:
		self.dataset = dataset

	def __len__(self) -> int:
		return len(self.dataset)

	def __getitem__(self, index: int) -> Dict[str, Any]:
		sample = self.dataset[index]
		sample["sample_index"] = index
		return sample


"""剥离样本下标后调用训练用 collate，保证训练管线收到的输入结构不变。"""
def indexed_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
	sample_indices = [sample.pop("sample_index") for sample in batch]
	collated_batch = AtomActionDataset_collate_fn(batch)
	collated_batch["sample_index"] = torch.tensor(sample_indices, dtype=torch.long)
	return collated_batch


"""比对 checkpoint 内嵌的模型配置与当前配置文件，不一致直接报错。"""
def validate_model_args(checkpoint_model_args: Dict[str, Any], config_path: Path) -> None:
	with config_path.open("r", encoding="utf-8") as file:
		current_config = yaml.safe_load(file)

	checkpoint_section = checkpoint_model_args.get("AtomAction_NSVQ")
	current_section = current_config.get("AtomAction_NSVQ")
	if checkpoint_section != current_section:
		raise ValueError(
			"checkpoint model_args['AtomAction_NSVQ'] does not match the current config. "
			f"config={config_path}. Export must use the exact training-time configuration."
		)


"""构建 AtomAction_NSVQ 并按前缀过滤加载 checkpoint 权重。"""
def load_atomaction_nsvq(
	checkpoint: Dict[str, Any],
	config_path: Path,
	device: torch.device,
) -> AtomAction_NSVQ:
	validate_model_args(checkpoint["model_args"], config_path)

	model = AtomAction_NSVQ(model_args=checkpoint["model_args"])
	filtered_state_dict = {
		key[len(MODULE_PREFIX):]: value
		for key, value in checkpoint["model"].items()
		if key.startswith(MODULE_PREFIX)
	}

	expected_keys = set(model.state_dict().keys())
	if set(filtered_state_dict.keys()) != expected_keys:
		missing_keys = sorted(expected_keys - set(filtered_state_dict.keys()))
		unexpected_keys = sorted(set(filtered_state_dict.keys()) - expected_keys)
		raise ValueError(
			"Filtered checkpoint keys do not match AtomAction_NSVQ state_dict. "
			f"missing={missing_keys[:5]} unexpected={unexpected_keys[:5]}"
		)

	model.load_state_dict(filtered_state_dict, strict=True)
	model.eval()
	model.to(device=device)
	return model


"""读取单个 phase 的元数据。"""
def load_phase_metadata(phase_path: Path) -> Dict[str, Any]:
	with (phase_path / "phase_metadata.json").open("r", encoding="utf-8") as file:
		return json.load(file)


"""读取 variation 元数据，同一 variation 下上百个 phase 共用，做缓存。"""
@lru_cache(maxsize=None)
def load_variation_metadata(variation_dir: str) -> Dict[str, Any]:
	with (Path(variation_dir) / "variation_metadata.json").open("r", encoding="utf-8") as file:
		return json.load(file)


"""按 source phase_index 回查 short 指令。

⚠️ 不能用 phase 目录名（= output_phase_index，样本序号）查，
   实测 push/close_drawer/variation1/phase_001 会查到语义相反的指令且不报错。
"""
def resolve_instruction(variation_dir: Path, source_phase_index: int) -> Optional[str]:
	variation_metadata = load_variation_metadata(str(variation_dir))
	for phase_description in variation_metadata.get("phase_descriptions", []):
		if int(phase_description.get("phase_index", -1)) != int(source_phase_index):
			continue
		for description in phase_description.get("descriptions", []):
			if str(description.get("style", "")).strip().lower() == INSTRUCTION_STYLE:
				return str(description.get("text", "")).strip()
	return None


"""解析某个相机的起始帧图像相对路径。

优先取 view_selection_cache 中记录的 start_path，缺失时回落到 {start_frame}.png。
"""
def resolve_image_path(phase_path: Path, phase_metadata: Dict[str, Any], camera: str) -> Optional[str]:
	start_frame = int(phase_metadata["source_phase_stats"]["start_frame"])
	cached_views = phase_metadata.get("view_selection_cache", {}).get("views", {})
	cached_view = cached_views.get(camera) if isinstance(cached_views, dict) else None

	if isinstance(cached_view, dict) and cached_view.get("start_path"):
		image_path = str(cached_view["start_path"])
	else:
		image_path = f"{camera}_rgb/{start_frame}.png"

	if not (phase_path / image_path).is_file():
		return None
	return image_path


"""为一个批次的索引结果组装 JSON 记录。"""
def build_records(
	dataset: AtomActionDataset,
	sample_indices: Sequence[int],
	global_indices: Sequence[int],
	detail_indices: Sequence[Sequence[int]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
	records: List[Dict[str, Any]] = []
	issues: List[str] = []

	for sample_index, global_index, detail_index in zip(sample_indices, global_indices, detail_indices):
		sample = dataset.samples[int(sample_index)]
		phase_path = Path(sample["phase_path"])
		variation_dir = phase_path.parent
		phase_metadata = load_phase_metadata(phase_path)
		source_phase_stats = phase_metadata["source_phase_stats"]
		source_phase_index = int(source_phase_stats["phase_index"])

		instruction = resolve_instruction(variation_dir, source_phase_index)
		phase_id = str(phase_path.relative_to(dataset.dataset_root))
		if instruction is None:
			issues.append(f"missing instruction: {phase_id} (source_phase_index={source_phase_index})")

		image_paths: Dict[str, Optional[str]] = {}
		for camera in ADAPTER_CAMERAS:
			image_path = resolve_image_path(phase_path, phase_metadata, camera)
			if image_path is None:
				issues.append(f"missing {camera} start image: {phase_id}")
			image_paths[camera] = image_path

		records.append(
			{
				"phase_id": phase_id,
				"action": sample["action"],
				"task": sample["task"],
				"variation": sample["variation"],
				"phase": phase_path.name,
				"source_phase_index": source_phase_index,
				"trajectory_length": int(source_phase_stats["length"]),
				"k_global": int(global_index),
				"k_detail": [int(value) for value in detail_index],
				"start_frame": int(source_phase_stats["start_frame"]),
				"end_frame": int(source_phase_stats["end_frame"]),
				"keyframe_index": int(source_phase_stats["keyframe_index"]),
				"img_front": image_paths["front"],
				"img_wrist": image_paths["wrist"],
				"instruction": instruction,
			}
		)

	return records, issues


"""按固定组大小随机分组计算困惑度并取平均，与训练时的批内口径对齐。"""
def compute_grouped_perplexity(
	indices: torch.Tensor,
	codebook_size: int,
	group_size: int,
	seed: int = 0,
) -> float:
	generator = torch.Generator().manual_seed(seed)
	permutation = torch.randperm(indices.shape[0], generator=generator)
	shuffled_indices = indices[permutation]

	perplexities = []
	for start in range(0, shuffled_indices.shape[0] - group_size + 1, group_size):
		group = shuffled_indices[start:start + group_size].reshape(-1)
		perplexities.append(float(compute_perplexity(group, codebook_size)))

	if not perplexities:
		return float(compute_perplexity(shuffled_indices.reshape(-1), codebook_size))
	return sum(perplexities) / len(perplexities)


"""导出后的验收检查。硬性项不通过直接返回 False。"""
def run_assertions(records: List[Dict[str, Any]], meta: Dict[str, Any], expected_count: int) -> bool:
	passed = True

	def report(name: str, condition: bool, extra: str = "", hard: bool = True) -> None:
		nonlocal passed
		status = "PASS" if condition else ("FAIL" if hard else "WARN")
		print(f"[{status}] {name} {extra}")
		if hard and not condition:
			passed = False

	report("A1 记录数", len(records) == expected_count, f"{len(records)} / {expected_count}")

	phase_ids = {record["phase_id"] for record in records}
	report("A2 phase_id 唯一", len(phase_ids) == len(records), f"{len(phase_ids)}")

	global_ok = all(0 <= record["k_global"] < meta["global_codebook_size"] for record in records)
	detail_ok = all(
		len(record["k_detail"]) == meta["num_detail"]
		and all(0 <= value < meta["detail_codebook_size"] for value in record["k_detail"])
		for record in records
	)
	report("A3 索引值域", global_ok and detail_ok)

	missing_instruction = [record["phase_id"] for record in records if not record["instruction"]]
	report("A4 指令覆盖率 100%", not missing_instruction, f"missing={len(missing_instruction)}")

	missing_image = [
		record["phase_id"] for record in records if not record["img_front"] or not record["img_wrist"]
	]
	report("A5 双相机起始帧存在", not missing_image, f"missing={len(missing_image)}")

	frame_mismatch = [
		record["phase_id"]
		for record in records
		if Path(record["img_front"] or "").name != f"{record['start_frame']}.png"
		or Path(record["img_wrist"] or "").name != f"{record['start_frame']}.png"
	]
	report("A6 图像帧号与 start_frame 一致", not frame_mismatch, f"mismatch={len(frame_mismatch)}")

	combinations = {(record["task"], record["variation"], record["source_phase_index"]) for record in records}
	report(
		"A7 唯一 (task, variation, source_phase_index) 组合数",
		len(combinations) == REFERENCE_SEMANTIC_COMBINATIONS,
		f"{len(combinations)} (设计文档记录 {REFERENCE_SEMANTIC_COMBINATIONS})",
		hard=False,
	)

	global_tensor = torch.tensor([record["k_global"] for record in records], dtype=torch.long)
	detail_tensor = torch.tensor([record["k_detail"] for record in records], dtype=torch.long)
	grouped_global = compute_grouped_perplexity(global_tensor, meta["global_codebook_size"], PERPLEXITY_GROUP_SIZE)
	grouped_detail = compute_grouped_perplexity(detail_tensor, meta["detail_codebook_size"], PERPLEXITY_GROUP_SIZE)
	report(
		"A8 批内困惑度对账（组大小 64）",
		abs(grouped_global - REFERENCE_PERPLEXITY_GLOBAL) < 2.0
		and abs(grouped_detail - REFERENCE_PERPLEXITY_DETAIL) < 6.0,
		f"ppl_g={grouped_global:.2f} (训练 {REFERENCE_PERPLEXITY_GLOBAL}) | "
		f"ppl_d={grouped_detail:.2f} (训练 {REFERENCE_PERPLEXITY_DETAIL})",
		hard=False,
	)

	dataset_global = float(compute_perplexity(global_tensor, meta["global_codebook_size"]))
	dataset_detail = float(compute_perplexity(detail_tensor.reshape(-1), meta["detail_codebook_size"]))
	print(
		f"[INFO] 数据集级困惑度: ppl_g={dataset_global:.2f} / {meta['global_codebook_size']} | "
		f"ppl_d={dataset_detail:.2f} / {meta['detail_codebook_size']}"
	)
	print(
		f"[INFO] 码本覆盖: 全局 {len({record['k_global'] for record in records})} / {meta['global_codebook_size']} | "
		f"细节 {len({value for record in records for value in record['k_detail']})} / {meta['detail_codebook_size']}"
	)

	slot_unique_counts = [len(set(record["k_detail"])) for record in records]
	print(
		f"[INFO] 样本内细节槽位唯一索引数: 平均 {sum(slot_unique_counts) / len(slot_unique_counts):.2f} / {meta['num_detail']}"
	)

	return passed


def main() -> int:
	parser = argparse.ArgumentParser(description="Export AtomAction dual-codebook indices for Adapter training")
	parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	parser.add_argument("--dataset-root", type=str, default=None)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--num-workers", type=int, default=8)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--limit", type=int, default=None, help="只导出前 N 个样本，用于试跑")
	args = parser.parse_args()

	global_config = load_global_config()
	dataset_config = global_config.get("atomactiondataset", {})
	dataset_root = args.dataset_root or dataset_config.get("dataset_root", "AtomAction_Dataset")
	if not Path(dataset_root).is_absolute():
		dataset_root = str(REPO_ROOT / dataset_root)

	device = torch.device(args.device)
	print(f"checkpoint: {args.checkpoint}")
	checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
	print(f"  epoch={checkpoint['epoch']} global_step={checkpoint['global_step']} stage={checkpoint['stage']}")
	checkpoint_sha256 = compute_file_sha256(args.checkpoint)
	print(f"  sha256={checkpoint_sha256}")

	model = load_atomaction_nsvq(checkpoint=checkpoint, config_path=args.config, device=device)
	detail_module = model.detail_codebook_module
	global_module = model.global_codebook_module
	meta = {
		"checkpoint": str(args.checkpoint),
		"checkpoint_sha256": checkpoint_sha256,
		"checkpoint_epoch": int(checkpoint["epoch"]),
		"checkpoint_stage": int(checkpoint["stage"]),
		"dataset_root": dataset_root,
		"train_actions": list(global_config.get("train_actions", [])),
		"instruction_style": INSTRUCTION_STYLE,
		"global_codebook_size": int(global_module.quantizer.codebook_size),
		"detail_codebook_size": int(detail_module.quantizer.codebook_size),
		"num_detail": int(detail_module.num_queries),
		"exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
	}

	dataset = AtomActionDataset(
		dataset_root=dataset_root,
		top_k=int(dataset_config.get("top_k", 1)),
		view_selector_kwargs=dataset_config.get("view_selector"),
	)
	print(f"dataset: {len(dataset)} samples | device={device}")

	indexed_dataset: Dataset = IndexedDataset(dataset)
	if args.limit is not None:
		indexed_dataset = torch.utils.data.Subset(indexed_dataset, list(range(min(args.limit, len(dataset)))))

	data_loader = DataLoader(
		indexed_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=indexed_collate_fn,
		pin_memory=(device.type == "cuda"),
		drop_last=False,
	)

	all_sample_indices: List[int] = []
	all_global_indices: List[int] = []
	all_detail_indices: List[List[int]] = []
	total_batches = len(data_loader)
	for batch_index, batch in enumerate(data_loader):
		trajectory_data = {key: value.to(device=device) for key, value in batch["trajectory_data"].items()}
		trajectory_mask = batch["trajectory_mask"].to(device=device)
		outputs = model.encode_codebook_indices(trajectory_data, trajectory_mask)

		all_sample_indices.extend(batch["sample_index"].tolist())
		all_global_indices.extend(outputs["global_codeindex"].cpu().tolist())
		all_detail_indices.extend(outputs["detail_codeindices"].cpu().tolist())

		if (batch_index + 1) % 50 == 0 or batch_index + 1 == total_batches:
			print(f"  encoded {batch_index + 1}/{total_batches} batches", flush=True)

	print("assembling records ...", flush=True)
	records, issues = build_records(
		dataset=dataset,
		sample_indices=all_sample_indices,
		global_indices=all_global_indices,
		detail_indices=all_detail_indices,
	)
	for issue in issues[:20]:
		print(f"  [issue] {issue}")
	if len(issues) > 20:
		print(f"  [issue] ... and {len(issues) - 20} more")

	meta["num_records"] = len(records)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	with args.output.open("w", encoding="utf-8") as file:
		json.dump({"meta": meta, "records": records}, file, ensure_ascii=False)
	print(f"written: {args.output} ({args.output.stat().st_size / 2 ** 20:.1f} MB)")

	print("\n=== 验收检查 ===")
	expected_count = len(indexed_dataset)
	passed = run_assertions(records=records, meta=meta, expected_count=expected_count)
	print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
	return 0 if passed else 1


if __name__ == "__main__":
	raise SystemExit(main())
