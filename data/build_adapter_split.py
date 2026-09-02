#!/usr/bin/env python3
"""Adapter 训练集划分脚本。

在 data/atomaction_codebook_index.json 上原地写入每条记录的 `split` 字段（train / val），
不产生额外的 manifest 文件。

划分策略（strategy = "phase_id_per_task_variation"）：
	1. 划分单位是 `(task, variation)` 下的 `phase_XXX` 编号，**所有 action 目录共用同一份选择**。
	   同一 `(task, variation)` 下 `phase_XXX` 在各 action 目录中指向同一个 demo episode
	   （136 组中 124 组的各 action 目录 phase 集合完全相同），因此这样切不会把同一 demo 的
	   兄弟 phase 分到 train / val 两侧。
	2. 每个 `(task, variation)` 内独立抽 `max(1, round(N * val_ratio))` 个 phase 编号进 val，
	   保证每个 `action/task/variation` 目录都有 val 样本。
	3. 每组的随机种子由 `sha256(seed | task | variation)` 派生，与遍历顺序无关，
	   同 seed 重跑得到逐条相同的划分。

⚠️ 重跑 data/import_codebook_index.py 会重写 index JSON 并覆盖 `split` 字段。
   划分完全由 (seed, val_ratio, 记录内容) 决定，重跑导出后再执行本脚本即可恢复逐条相同的划分。

用法：
	python data/build_adapter_split.py --val-ratio 0.05 --seed 0
"""

import argparse
import collections
import hashlib
import json
import os
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INDEX = REPO_ROOT / "data" / "atomaction_codebook_index.json"
SPLIT_STRATEGY = "phase_id_per_task_variation"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"


"""由全局种子与分组键派生该组的随机种子，与遍历顺序无关。"""
def derive_group_seed(base_seed: int, task: str, variation: str) -> int:
	digest = hashlib.sha256(f"{base_seed}|{task}|{variation}".encode("utf-8")).digest()
	return int.from_bytes(digest[:8], "big")


"""为每个 (task, variation) 选出进入 val 的 phase 编号集合。"""
def select_validation_phases(
	records: List[Dict[str, Any]],
	val_ratio: float,
	seed: int,
) -> Dict[Tuple[str, str], Set[str]]:
	phases_by_group: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
	for record in records:
		phases_by_group[(record["task"], record["variation"])].add(record["phase"])

	validation_phases: Dict[Tuple[str, str], Set[str]] = {}
	for group_key, phases in phases_by_group.items():
		sorted_phases = sorted(phases)
		num_validation = max(1, round(len(sorted_phases) * val_ratio))
		generator = random.Random(derive_group_seed(seed, group_key[0], group_key[1]))
		validation_phases[group_key] = set(generator.sample(sorted_phases, num_validation))
	return validation_phases


"""统计并打印划分结果，返回硬性检查是否全部通过。"""
def report_split(records: List[Dict[str, Any]], val_ratio: float) -> bool:
	passed = True

	def report(name: str, condition: bool, extra: str = "", hard: bool = True) -> None:
		nonlocal passed
		status = "PASS" if condition else ("FAIL" if hard else "WARN")
		print(f"[{status}] {name} {extra}")
		if hard and not condition:
			passed = False

	split_counts = collections.Counter(record["split"] for record in records)
	total = len(records)
	report(
		"S1 每条记录都有 split 且总数守恒",
		split_counts[TRAIN_SPLIT] + split_counts[VAL_SPLIT] == total,
		f"train={split_counts[TRAIN_SPLIT]} val={split_counts[VAL_SPLIT]} total={total}",
	)

	actual_ratio = split_counts[VAL_SPLIT] / total
	report(
		"S2 val 总占比在目标 ±0.5% 内",
		abs(actual_ratio - val_ratio) <= 0.005,
		f"{actual_ratio * 100:.2f}% (目标 {val_ratio * 100:.2f}%)",
	)

	# 每个 action/task/variation 目录的 val 比例。
	directory_counts: Dict[Tuple[str, str, str], List[int]] = collections.defaultdict(lambda: [0, 0])
	for record in records:
		entry = directory_counts[(record["action"], record["task"], record["variation"])]
		entry[0] += 1
		entry[1] += int(record["split"] == VAL_SPLIT)
	ratios = [validation / total_count for total_count, validation in directory_counts.values()]
	empty_directories = sum(1 for ratio in ratios if ratio == 0.0)
	report(
		"S3 每个 action/task/variation 目录都含 val 样本",
		empty_directories == 0,
		f"目录数={len(directory_counts)} 无 val 的目录={empty_directories} | "
		f"比例 min={min(ratios):.3f} 中位={statistics.median(ratios):.3f} max={max(ratios):.3f}",
	)

	# episode 完整性：同一 (task, variation, phase) 不得跨 split。
	episode_splits: Dict[Tuple[str, str, str], Set[str]] = collections.defaultdict(set)
	for record in records:
		episode_splits[(record["task"], record["variation"], record["phase"])].add(record["split"])
	crossing_episodes = [key for key, splits in episode_splits.items() if len(splits) > 1]
	report(
		"S4 无 (task, variation, phase) 跨 split",
		not crossing_episodes,
		f"episode 数={len(episode_splits)} 跨 split={len(crossing_episodes)}",
	)

	train_records = [record for record in records if record["split"] == TRAIN_SPLIT]
	val_records = [record for record in records if record["split"] == VAL_SPLIT]
	for name, field in (("任务", "task"), ("原子", "action")):
		train_values = {record[field] for record in train_records}
		all_values = {record[field] for record in records}
		report(f"S5 train 覆盖全部{name}", train_values == all_values, f"{len(train_values)}/{len(all_values)}")

	train_codes = {record["k_global"] for record in train_records}
	val_codes = {record["k_global"] for record in val_records}
	report("S6 train 覆盖全部 36 个全局码", len(train_codes) == 36, f"train={len(train_codes)} val={len(val_codes)}")

	# 指令-only 基线：语义组内众数。
	# val 侧必须用 train 拟合的众数来评估 —— val 每组只有约 5 个样本，用 val 自拟合的众数会被
	# 小样本偏差显著抬高（实测 66.5% vs 诚实值 58.0%），不能作为 Adapter 的对照线。
	for field, label in (("k_global", "全局码"), ("k_detail", "细节码槽0")):
		group_counters: Dict[Tuple[str, str, int], collections.Counter] = collections.defaultdict(collections.Counter)
		for record in train_records:
			value = record[field] if field == "k_global" else record[field][0]
			group_counters[(record["task"], record["variation"], record["source_phase_index"])][value] += 1
		train_modes = {key: counter.most_common(1)[0][0] for key, counter in group_counters.items()}

		train_hits = sum(counter.most_common(1)[0][1] for counter in group_counters.values())
		val_hits = sum(
			1
			for record in val_records
			if train_modes.get((record["task"], record["variation"], record["source_phase_index"]))
			== (record[field] if field == "k_global" else record[field][0])
		)
		print(
			f"[INFO] 指令-only 基线（{label}）: train 自身 {100 * train_hits / len(train_records):.1f}% | "
			f"train 众数 -> val {100 * val_hits / len(val_records):.1f}%"
		)

	val_tasks = {record["task"] for record in val_records}
	val_actions = {record["action"] for record in val_records}
	print(f"[INFO] val 覆盖任务 {len(val_tasks)}/69 | 原子 {len(val_actions)}/17")
	return passed


"""原子化写回 JSON，避免中途失败损坏原文件。"""
def write_index_json(index_path: Path, payload: Dict[str, Any]) -> None:
	temp_path = index_path.with_suffix(index_path.suffix + ".tmp")
	with temp_path.open("w", encoding="utf-8") as file:
		json.dump(payload, file, ensure_ascii=False)
	os.replace(temp_path, index_path)


def main() -> int:
	parser = argparse.ArgumentParser(description="Assign train/val split to the AtomAction codebook index")
	parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
	parser.add_argument("--val-ratio", type=float, default=0.05)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--dry-run", action="store_true", help="只统计不写回")
	args = parser.parse_args()

	if not 0.0 < args.val_ratio < 1.0:
		raise ValueError("val_ratio must be in (0, 1)")

	with args.index.open("r", encoding="utf-8") as file:
		payload = json.load(file)
	records = payload["records"]
	print(f"index: {args.index} | {len(records)} records")

	validation_phases = select_validation_phases(records=records, val_ratio=args.val_ratio, seed=args.seed)
	for record in records:
		group_key = (record["task"], record["variation"])
		record["split"] = VAL_SPLIT if record["phase"] in validation_phases[group_key] else TRAIN_SPLIT

	split_counts = collections.Counter(record["split"] for record in records)
	payload["meta"].update(
		{
			"split_strategy": SPLIT_STRATEGY,
			"val_ratio": float(args.val_ratio),
			"split_seed": int(args.seed),
			"split_counts": {TRAIN_SPLIT: split_counts[TRAIN_SPLIT], VAL_SPLIT: split_counts[VAL_SPLIT]},
			"split_created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
		}
	)

	print("\n=== 划分检查 ===")
	passed = report_split(records=records, val_ratio=args.val_ratio)

	if args.dry_run:
		print("\n[dry-run] 未写回")
	else:
		write_index_json(args.index, payload)
		print(f"\nwritten: {args.index} ({args.index.stat().st_size / 2 ** 20:.1f} MB)")

	print(f"{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
	return 0 if passed else 1


if __name__ == "__main__":
	raise SystemExit(main())
