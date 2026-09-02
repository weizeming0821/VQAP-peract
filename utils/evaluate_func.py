"""Adapter 评估指标模块。

指标分三层（Adapter_Design.md §4.7）：
	第 1 层 单索引准确率：k_global 的 micro / macro / top-5 / 按类频分层 top-1。
	第 2 层 联合准确率：k_global 正确且 9 个 k_detail 全部正确 —— 一次子任务注入的整组码向量是否完全正确。
	第 3 层 码向量误差：Stage 3 消费的是码向量而非索引，用 ‖z_pred − z_true‖ / 平均码间距衡量"错得有多远"。

主指标（best checkpoint 判据）：val 的 k_global micro top-1。
"""

import collections
from typing import Any, Dict, Optional, Sequence

import torch


"""按类别频次把类划分为 head / mid / tail 三层。

输入：
	labels: [N]，真值索引。
	codebook_size: int，类别总数 K。

输出：
	tier_to_classes: Dict[str, List[int]]，每层包含的类别索引。
"""
def build_frequency_tiers(labels: torch.Tensor, codebook_size: int) -> Dict[str, list]:
	counts = torch.bincount(labels.reshape(-1), minlength=int(codebook_size))
	sorted_classes = torch.argsort(counts, descending=True).tolist()
	tier_size = max(1, len(sorted_classes) // 3)
	return {
		"head": sorted_classes[:tier_size],
		"mid": sorted_classes[tier_size: 2 * tier_size],
		"tail": sorted_classes[2 * tier_size:],
	}


"""计算 micro top-k 准确率。

输入：
	logits: [N, K]
	labels: [N]
	topk: int

输出：
	准确率标量（float）。
"""
def compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, topk: int = 1) -> float:
	if labels.numel() == 0:
		return 0.0
	k = min(int(topk), logits.shape[-1])
	predictions = logits.topk(k, dim=-1).indices	# [N, K] -> [N, k]
	correct = (predictions == labels.unsqueeze(-1)).any(dim=-1)	# [N, k] -> [N]
	return float(correct.float().mean())


"""计算 macro top-1 准确率：先对每个出现过的类别算准确率，再取平均。

输入：
	predictions: [N]，top-1 预测。
	labels: [N]，真值。

输出：
	(macro_accuracy, per_class_accuracy)
"""
def compute_macro_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> tuple:
	per_class_accuracy: Dict[int, float] = {}
	for class_index in sorted(set(labels.tolist())):
		class_mask = labels == class_index
		per_class_accuracy[int(class_index)] = float((predictions[class_mask] == class_index).float().mean())
	if not per_class_accuracy:
		return 0.0, per_class_accuracy
	return sum(per_class_accuracy.values()) / len(per_class_accuracy), per_class_accuracy


"""计算按分组（原子动作 / 任务）的 top-1 准确率。

输入：
	predictions: [N]
	labels: [N]
	group_names: 长度 N 的分组名列表。

输出：
	Dict[str, float]，每组的 top-1 准确率。
"""
def compute_group_accuracy(
	predictions: torch.Tensor,
	labels: torch.Tensor,
	group_names: Sequence[str],
) -> Dict[str, float]:
	correct_counts: Dict[str, list] = collections.defaultdict(lambda: [0, 0])
	is_correct = (predictions == labels).tolist()
	for group_name, correct in zip(group_names, is_correct):
		entry = correct_counts[str(group_name)]
		entry[0] += int(correct)
		entry[1] += 1
	return {name: entry[0] / entry[1] for name, entry in sorted(correct_counts.items())}


"""计算码向量的归一化误差：‖z_pred − z_true‖ / 平均码间距。

输入：
	predictions: [N] 或 [N, S]，预测索引。
	labels: 同形状的真值索引。
	codebook: [K, D]，冻结码本。

输出：
	归一化误差（float）。0 表示全对；接近 1 表示与随机猜同级。
"""
def compute_codeword_error(
	predictions: torch.Tensor,
	labels: torch.Tensor,
	codebook: torch.Tensor,
) -> float:
	if predictions.numel() == 0:
		return 0.0

	flat_predictions = predictions.reshape(-1).to(device=codebook.device)
	flat_labels = labels.reshape(-1).to(device=codebook.device)
	predicted_codewords = codebook.index_select(0, flat_predictions)	# [N] -> [N, D]
	target_codewords = codebook.index_select(0, flat_labels)	# [N] -> [N, D]
	distances = (predicted_codewords - target_codewords).norm(dim=-1)	# [N, D] -> [N]

	num_codes = codebook.shape[0]
	pairwise_distances = torch.cdist(codebook, codebook)	# [K, D] -> [K, K]
	off_diagonal = pairwise_distances[~torch.eye(num_codes, dtype=torch.bool, device=codebook.device)]
	return float(distances.mean() / off_diagonal.mean().clamp_min(1e-6))


"""计算 Adapter 的三层评估指标。

输入：
	global_logits: [N, K_g]
	detail_logits: [N, N_detail, K_d]
	k_global: [N]
	k_detail: [N, N_detail]
	actions / tasks: 长度 N 的分组名列表，可为 None。
	global_codebook / detail_codebook: [K, D]，用于第 3 层指标，可为 None。
	topk: 需要计算的 top-k 列表。

输出：
	Dict[str, Any]，扁平的指标字典；`per_class_global` / `per_action` / `per_task` 为嵌套字典，只写日志文件。
"""
def compute_adapter_metrics(
	global_logits: torch.Tensor,
	detail_logits: torch.Tensor,
	k_global: torch.Tensor,
	k_detail: torch.Tensor,
	actions: Optional[Sequence[str]] = None,
	tasks: Optional[Sequence[str]] = None,
	global_codebook: Optional[torch.Tensor] = None,
	detail_codebook: Optional[torch.Tensor] = None,
	topk: Sequence[int] = (1, 5),
) -> Dict[str, Any]:
	global_logits = global_logits.float().cpu()
	detail_logits = detail_logits.float().cpu()
	k_global = k_global.cpu()
	k_detail = k_detail.cpu()

	num_samples, num_detail, detail_codebook_size = detail_logits.shape
	global_predictions = global_logits.argmax(dim=-1)	# [N, K_g] -> [N]
	detail_predictions = detail_logits.argmax(dim=-1)	# [N, N_detail, K_d] -> [N, N_detail]

	metrics: Dict[str, Any] = {"num_samples": int(num_samples)}

	# —— 第 1 层：单索引准确率 ——
	for k in topk:
		metrics[f"global_top{int(k)}"] = compute_topk_accuracy(global_logits, k_global, topk=int(k))
	macro_accuracy, per_class_accuracy = compute_macro_accuracy(global_predictions, k_global)
	metrics["global_macro_top1"] = macro_accuracy
	metrics["per_class_global"] = per_class_accuracy

	frequency_tiers = build_frequency_tiers(k_global, codebook_size=global_logits.shape[-1])
	for tier_name, tier_classes in frequency_tiers.items():
		tier_mask = torch.isin(k_global, torch.tensor(tier_classes, dtype=k_global.dtype))
		metrics[f"global_top1_{tier_name}"] = (
			float((global_predictions[tier_mask] == k_global[tier_mask]).float().mean())
			if bool(tier_mask.any())
			else float("nan")
		)

	# —— 第 2 层：联合准确率 ——
	global_correct = global_predictions == k_global	# [N]
	detail_correct_per_slot = detail_predictions == k_detail	# [N, N_detail]
	metrics["detail_top1"] = float(detail_correct_per_slot.float().mean())
	metrics["detail_all_slots_top1"] = float(detail_correct_per_slot.all(dim=-1).float().mean())
	metrics["joint_top1"] = float((global_correct & detail_correct_per_slot.all(dim=-1)).float().mean())

	# 槽间预测多样性：9 个槽 top-1 的唯一值个数均值（标签恒为 1.0，见 §7 R1）。
	slot_diversity = [len(set(row.tolist())) for row in detail_predictions]
	metrics["detail_slot_diversity"] = sum(slot_diversity) / max(len(slot_diversity), 1)
	label_slot_diversity = [len(set(row.tolist())) for row in k_detail]
	metrics["detail_label_slot_diversity"] = sum(label_slot_diversity) / max(len(label_slot_diversity), 1)

	# —— 第 3 层：码向量误差 ——
	if global_codebook is not None:
		metrics["global_codeword_error"] = compute_codeword_error(global_predictions, k_global, global_codebook)
	if detail_codebook is not None:
		metrics["detail_codeword_error"] = compute_codeword_error(detail_predictions, k_detail, detail_codebook)

	# —— 分组诊断 ——
	if actions is not None:
		metrics["per_action"] = compute_group_accuracy(global_predictions, k_global, actions)
	if tasks is not None:
		metrics["per_task"] = compute_group_accuracy(global_predictions, k_global, tasks)
	return metrics
