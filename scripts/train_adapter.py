#!/usr/bin/env python3
"""VQAP Adapter（Stage 2）训练脚本。

任务：给定「子任务起始观测（front + wrist）+ 子任务指令」，预测冻结 VQAP 双码本的索引。

设计要点：
	1. **单卡训练**。实测 num_workers=32 / batch=64 时 2.1 min/epoch，100 epoch 约 3.5 小时，
	   引入 DDP 的复杂度与收益不成比例。
	2. **每 epoch 验证**。val 仅 2,793 条（约 3 秒），相对 2.1 min 的 epoch 只有 2% 开销。
	3. **best checkpoint 判据 = val 的 k_global micro top-1**（细节码只有 1 个自由度，不适合当判据）。
	4. **checkpoint 只存可训练参数**（约 54 MB 而非 625 MB），两个冻结塔在构建时从官方缓存加载。

用法：
	python scripts/train_adapter.py --overfit-batch 64 --max-steps 300   # 过拟合自检
	bash run/train_adapter.sh                                            # 正式训练
"""

import argparse
import hashlib
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from data.adapter_dataset import AdapterDataset, adapter_collate_fn
from model.adapter import Adapter
from utils.evaluate_func import compute_adapter_metrics
from utils.init_logger_tensorboard import finish_tensorboard, init_logger, init_tensorboard
from utils.loss_func import compute_adapter_classification_loss

# 写进 TensorBoard 的验证标量（其余指标只进日志文件）。
EVAL_TENSORBOARD_KEYS = (
	"global_top1",
	"global_top5",
	"global_macro_top1",
	"global_top1_head",
	"global_top1_mid",
	"global_top1_tail",
	"detail_top1",
	"joint_top1",
	"detail_slot_diversity",
	"global_codeword_error",
	"detail_codeword_error",
)


"""读取 YAML 配置文件，并校验顶层必须为字典。"""
def load_yaml_config(path: str) -> Dict[str, Any]:
	config_path = Path(path).expanduser()
	if not config_path.is_file():
		raise FileNotFoundError(f"Config file does not exist: {config_path}")

	with config_path.open("r", encoding="utf-8") as file:
		config = yaml.safe_load(file)
	if not isinstance(config, dict):
		raise ValueError(f"Config file must contain a top-level mapping: {config_path}")
	return config


"""设置随机种子。"""
def set_random_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


"""流式计算文件 sha256，用于把标签文件的身份写进 checkpoint。"""
def compute_file_sha256(file_path: Path, chunk_size: int = 1 << 20) -> str:
	digest = hashlib.sha256()
	with Path(file_path).open("rb") as file:
		for chunk in iter(lambda: file.read(chunk_size), b""):
			digest.update(chunk)
	return digest.hexdigest()


"""构造线性 warmup + cosine decay 的学习率比例函数。"""
def make_lr_lambda(warmup_epochs: int, total_epochs: int, min_ratio: float) -> Any:
	def lr_lambda(epoch: int) -> float:
		if warmup_epochs > 0 and epoch < warmup_epochs:
			return float(epoch + 1) / float(warmup_epochs)
		decay_epochs = max(total_epochs - warmup_epochs, 1)
		progress = min(max(epoch - warmup_epochs, 0) / decay_epochs, 1.0)
		cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
		return min_ratio + (1.0 - min_ratio) * cosine_ratio

	return lr_lambda


class AdapterTrainer:

	def __init__(self, args: argparse.Namespace):
		self.args = args
		self.model_args = load_yaml_config(args.model_config)
		self.train_args = load_yaml_config(args.config)

		experiment_cfg = self.train_args["experiment"]
		runtime_cfg = self.train_args["runtime"]
		self.exp_name = args.exp_name or str(experiment_cfg.get("exp_name", "vqap_adapter"))
		self.seed = int(experiment_cfg.get("seed", 42))
		set_random_seed(self.seed)

		self.device = torch.device(args.device or str(runtime_cfg.get("device", "cuda:0")))
		if self.device.type == "cuda":
			torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
			torch.backends.cuda.matmul.allow_tf32 = bool(runtime_cfg.get("allow_tf32", True))

		self.precision = str(runtime_cfg.get("precision", "bf16")).lower()
		if self.device.type != "cuda" or self.precision == "fp32":
			self.autocast_dtype: Optional[torch.dtype] = None
		elif self.precision == "bf16":
			self.autocast_dtype = torch.bfloat16
		elif self.precision == "fp16":
			self.autocast_dtype = torch.float16
		else:
			raise ValueError(f"Unsupported precision setting: {self.precision}")
		self.grad_scaler = torch.amp.GradScaler(
			device="cuda",
			enabled=self.device.type == "cuda" and self.precision == "fp16",
		)
		self.grad_clip_norm = float(runtime_cfg["grad_clip_norm"])

		# 过拟合自检模式：固定少量样本反复训练，用于验证「标签 -> 前向 -> 损失」接线正确。
		self.overfit_batch = int(args.overfit_batch) if args.overfit_batch else 0
		self.max_steps = int(args.max_steps) if args.max_steps else 0

		checkpoint_cfg = self.train_args["checkpoint"]
		self.ckpt_dir = Path(checkpoint_cfg["root_dir"]).expanduser() / self.exp_name
		self.ckpt_dir.mkdir(parents=True, exist_ok=True)
		self.save_every_epochs = int(checkpoint_cfg["save_every_epochs"])
		self.save_trainable_only = bool(checkpoint_cfg.get("save_trainable_only", True))

		self.resume_path = Path(args.resume).expanduser() if args.resume else None
		self.logger = init_logger(
			rank=0,
			exp_name=self.exp_name,
			log_dir=str(self.train_args["logging"].get("log_dir", "log")),
			is_resume=self.resume_path is not None,
		)

		tb_cfg = dict(self.train_args)
		if args.disable_tensorboard or self.overfit_batch > 0:
			tb_cfg["tensorboard"] = dict(tb_cfg.get("tensorboard", {}))
			tb_cfg["tensorboard"]["enable"] = False
		self.tb_writer = init_tensorboard(
			rank=0,
			exp_name=self.exp_name,
			cfg=tb_cfg,
			ckpt_dir=str(self.ckpt_dir),
			is_resume=self.resume_path is not None,
		)

		self.total_epochs = int(self.train_args["train"]["epochs"])
		self.eval_cfg = self.train_args["eval"]
		self.eval_every_epochs = int(self.eval_cfg.get("eval_every_epochs", 1))
		self.loss_cfg = self.train_args["loss"]

		self._init_datasets()
		self._init_codebooks()
		self.model = Adapter(model_args=self.model_args).to(device=self.device)
		self.optimizer = self._init_optimizer()
		self.scheduler = self._init_scheduler()

		self.start_epoch = 0
		self.global_step = 0
		self.best_val_top1 = 0.0
		self.epoch_history: List[Dict[str, Any]] = []
		self._load_resume_state()

		trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		total = sum(p.numel() for p in self.model.parameters())
		self.logger.info(
			f"Trainer initialized | device={self.device} | precision={self.precision} | "
			f"train={len(self.train_dataset)} val={len(self.val_dataset)} | "
			f"trainable={trainable:,} frozen={total - trainable:,} | "
			f"epochs={self.total_epochs} batch={self.batch_size} lr={float(self.train_args['lr']['base']):.1e} | "
			f"lambda_detail={float(self.loss_cfg['lambda_detail'])} label_smoothing={float(self.loss_cfg['label_smoothing'])}"
		)

	"""构建训练 / 验证数据集与 dataloader。"""
	def _init_datasets(self) -> None:
		data_cfg = self.train_args["data"]
		index_json = REPO_ROOT / str(data_cfg["index_json"])
		self.index_json = index_json
		self.batch_size = int(data_cfg["batch_size"])

		self.train_dataset = AdapterDataset(index_path=index_json, split="train")
		self.val_dataset = AdapterDataset(index_path=index_json, split="val")
		self.split_meta = {
			key: self.train_dataset.meta.get(key)
			for key in ("split_strategy", "val_ratio", "split_seed", "split_counts", "checkpoint_sha256")
		}

		num_workers = int(data_cfg["num_workers"])
		train_source: Any = self.train_dataset
		shuffle = True
		drop_last = bool(data_cfg.get("drop_last", True))
		if self.overfit_batch > 0:
			# 固定同一批样本反复训练，关掉 shuffle 与多进程，保证每步喂进去的完全一致。
			train_source = Subset(self.train_dataset, list(range(self.overfit_batch)))
			shuffle = False
			drop_last = False
			num_workers = 0

		self.train_loader = DataLoader(
			train_source,
			batch_size=self.batch_size,
			shuffle=shuffle,
			num_workers=num_workers,
			collate_fn=adapter_collate_fn,
			pin_memory=bool(data_cfg.get("pin_memory", True)) and self.device.type == "cuda",
			drop_last=drop_last,
			persistent_workers=bool(data_cfg.get("persistent_workers", True)) and num_workers > 0,
			prefetch_factor=int(data_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None,
		)
		self.val_loader = DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=int(data_cfg.get("val_num_workers", 8)),
			collate_fn=adapter_collate_fn,
			pin_memory=bool(data_cfg.get("pin_memory", True)) and self.device.type == "cuda",
			drop_last=False,
		)

	"""加载冻结码本，供第 3 层「码向量误差」指标查表。"""
	def _init_codebooks(self) -> None:
		codebook_path = self.eval_cfg.get("codebook_path")
		self.global_codebook: Optional[torch.Tensor] = None
		self.detail_codebook: Optional[torch.Tensor] = None
		if not codebook_path:
			return

		resolved_path = REPO_ROOT / str(codebook_path)
		if not resolved_path.is_file():
			self.logger.warning(f"codebook_path not found, skip codeword-error metrics: {resolved_path}")
			return

		payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
		self.global_codebook = payload["global_codebook"]["codebooks"].float()
		self.detail_codebook = payload["detail_codebook"]["codebooks"].float()
		self.logger.info(
			f"codebook loaded: global{tuple(self.global_codebook.shape)} detail{tuple(self.detail_codebook.shape)}"
		)

	"""判断某个参数是否应当跳过 weight decay（与 train_vqap.py 同一规则）。"""
	def _should_skip_weight_decay(self, name: str, parameter: torch.nn.Parameter) -> bool:
		name_lower = name.lower()
		if parameter.ndim < 2:
			return True
		if name.endswith(".bias"):
			return True
		if "norm" in name_lower:
			return True
		if "embedding" in name_lower:
			return True
		if "query_tokens" in name_lower:
			return True
		return False

	"""只对可训练参数构建 AdamW，冻结塔完全不进优化器。"""
	def _init_optimizer(self) -> AdamW:
		optimizer_cfg = self.train_args["optimizer"]
		base_lr = float(self.train_args["lr"]["base"])

		decay_params = []
		no_decay_params = []
		for name, parameter in self.model.named_parameters():
			if not parameter.requires_grad:
				continue
			if self._should_skip_weight_decay(name, parameter):
				no_decay_params.append(parameter)
			else:
				decay_params.append(parameter)

		param_groups = []
		for params, weight_decay, group_name in (
			(decay_params, float(optimizer_cfg["weight_decay"]), "decay"),
			(no_decay_params, 0.0, "nodecay"),
		):
			if params:
				param_groups.append(
					{
						"params": params,
						"lr": base_lr,
						"betas": (float(optimizer_cfg["beta1"]), float(optimizer_cfg["beta2"])),
						"eps": float(optimizer_cfg["eps"]),
						"weight_decay": weight_decay,
						"group_name": group_name,
					}
				)
		return AdamW(param_groups)

	"""构建 warmup + cosine scheduler。"""
	def _init_scheduler(self) -> LambdaLR:
		scheduler_cfg = self.train_args["scheduler"]
		lr_lambda = make_lr_lambda(
			warmup_epochs=int(scheduler_cfg["warmup_epochs"]),
			total_epochs=self.total_epochs,
			min_ratio=float(scheduler_cfg["min_lr_ratio"]),
		)
		return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

	"""按 precision 返回 autocast 上下文；fp32 时退化为空上下文。"""
	def _autocast(self) -> Any:
		if self.autocast_dtype is None:
			return nullcontext()
		return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

	"""把一个 batch 中需要上卡的张量移到当前 device。"""
	def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
		batch["camera_images"] = {
			camera: images.to(device=self.device, non_blocking=True)
			for camera, images in batch["camera_images"].items()
		}
		batch["k_global"] = batch["k_global"].to(device=self.device, non_blocking=True)
		batch["k_detail"] = batch["k_detail"].to(device=self.device, non_blocking=True)
		return batch

	"""执行一次前向并计算损失。"""
	def _forward_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		outputs = self.model(batch["camera_images"], batch["instructions"])
		losses = compute_adapter_classification_loss(
			global_logits=outputs["global_logits"],
			detail_logits=outputs["detail_logits"],
			k_global=batch["k_global"],
			k_detail=batch["k_detail"],
			lambda_detail=float(self.loss_cfg["lambda_detail"]),
			label_smoothing=float(self.loss_cfg["label_smoothing"]),
		)
		losses["global_logits"] = outputs["global_logits"]
		losses["detail_logits"] = outputs["detail_logits"]
		return losses

	"""执行单个 epoch 的训练循环，返回 epoch 级平均指标。"""
	def _train_epoch(self, epoch: int) -> Dict[str, float]:
		self.model.train()
		metric_sums = {"loss_total": 0.0, "loss_global": 0.0, "loss_detail": 0.0, "grad_norm": 0.0}
		correct_global = 0
		num_samples = 0
		num_steps = 0

		progress = tqdm(
			self.train_loader,
			desc=f"train e{epoch + 1}/{self.total_epochs}",
			dynamic_ncols=True,
			leave=False,
			disable=not sys.stderr.isatty(),
		)
		for batch in progress:
			batch = self._move_batch_to_device(batch)
			with self._autocast():
				losses = self._forward_loss(batch)

			self.optimizer.zero_grad(set_to_none=True)
			self.grad_scaler.scale(losses["loss_total"]).backward()
			self.grad_scaler.unscale_(self.optimizer)
			grad_norm = clip_grad_norm_(
				[p for p in self.model.parameters() if p.requires_grad],
				self.grad_clip_norm,
			)
			self.grad_scaler.step(self.optimizer)
			self.grad_scaler.update()

			batch_size = batch["k_global"].shape[0]
			with torch.no_grad():
				correct_global += int((losses["global_logits"].argmax(dim=-1) == batch["k_global"]).sum())
			num_samples += batch_size
			num_steps += 1
			self.global_step += 1
			for key in ("loss_total", "loss_global", "loss_detail"):
				metric_sums[key] += float(losses[key])
			metric_sums["grad_norm"] += float(grad_norm)

			progress.set_postfix(
				loss=f"{float(losses['loss_total']):.3f}",
				top1=f"{correct_global / max(num_samples, 1):.3f}",
			)
			if self.max_steps and num_steps >= self.max_steps:
				break

		epoch_metrics = {key: value / max(num_steps, 1) for key, value in metric_sums.items()}
		epoch_metrics["train_top1"] = correct_global / max(num_samples, 1)
		epoch_metrics["num_steps"] = float(num_steps)
		return epoch_metrics

	"""在 val 上评估，返回三层指标。"""
	@torch.no_grad()
	def _evaluate(self) -> Dict[str, Any]:
		self.model.eval()
		global_logits_list = []
		detail_logits_list = []
		k_global_list = []
		k_detail_list = []
		actions: List[str] = []
		tasks: List[str] = []

		for batch in tqdm(self.val_loader, desc="val", dynamic_ncols=True, leave=False, disable=not sys.stderr.isatty()):
			batch = self._move_batch_to_device(batch)
			with self._autocast():
				outputs = self.model(batch["camera_images"], batch["instructions"])
			global_logits_list.append(outputs["global_logits"].float().cpu())
			detail_logits_list.append(outputs["detail_logits"].float().cpu())
			k_global_list.append(batch["k_global"].cpu())
			k_detail_list.append(batch["k_detail"].cpu())
			actions.extend(batch["meta"]["action"])
			tasks.extend(batch["meta"]["task"])

		return compute_adapter_metrics(
			global_logits=torch.cat(global_logits_list, dim=0),
			detail_logits=torch.cat(detail_logits_list, dim=0),
			k_global=torch.cat(k_global_list, dim=0),
			k_detail=torch.cat(k_detail_list, dim=0),
			actions=actions,
			tasks=tasks,
			global_codebook=self.global_codebook,
			detail_codebook=self.detail_codebook,
			topk=tuple(int(k) for k in self.eval_cfg.get("topk", (1, 5))),
		)

	"""只保留可训练参数的 state_dict（冻结塔在构建时从官方缓存加载，无需入盘）。"""
	def _trainable_state_dict(self) -> Dict[str, torch.Tensor]:
		if not self.save_trainable_only:
			return self.model.state_dict()
		trainable_names = {name for name, parameter in self.model.named_parameters() if parameter.requires_grad}
		return {key: value for key, value in self.model.state_dict().items() if key in trainable_names}

	"""用先写临时文件再原子替换的方式保存 checkpoint。"""
	def _atomic_torch_save(self, payload: Dict[str, Any], path: Path) -> None:
		temp_path = path.with_suffix(path.suffix + ".tmp")
		torch.save(payload, temp_path)
		temp_path.replace(path)

	"""保存 checkpoint：best 每次评估都判定，latest 按间隔写。"""
	def _save_checkpoint(self, epoch: int, eval_metrics: Optional[Dict[str, Any]], save_latest: bool) -> None:
		best_improved = False
		if eval_metrics is not None and eval_metrics["global_top1"] > self.best_val_top1:
			self.best_val_top1 = float(eval_metrics["global_top1"])
			best_improved = True
		if not (save_latest or best_improved):
			return

		image_cfg = self.model_args["Adapter"]["image_tokenizer"]
		text_cfg = self.model_args["Adapter"]["text_tokenizer"]
		payload = {
			"epoch": epoch + 1,
			"global_step": self.global_step,
			"model": self._trainable_state_dict(),
			"best_val_top1": self.best_val_top1,
			"model_args": self.model_args,
			"train_args": self.train_args,
			"epoch_history": self.epoch_history,
			"save_trainable_only": self.save_trainable_only,
			"frozen_towers": {
				"dinov2": image_cfg["dinov2"]["model_name"],
				"clip": text_cfg["clip"]["pretrained_path"],
			},
			"label_index": {
				"path": str(self.index_json),
				"sha256": self.index_sha256,
				**self.split_meta,
			},
			"tb_log_dir": getattr(self.tb_writer, "log_dir", None),
		}

		# best.pth 是部署产物（插件 vqap_adapter.pth 的来源），不带优化器状态 -> 54 MB；
		# latest.pth 供续训，需要带 optimizer/scheduler -> 约 170 MB。
		if save_latest:
			resume_payload = dict(payload)
			resume_payload["optimizer"] = self.optimizer.state_dict()
			resume_payload["scheduler"] = self.scheduler.state_dict()
			# 保存 RNG 状态，使续训的 shuffle 顺序与不中断时一致。
			resume_payload["rng_state"] = {
				"python": random.getstate(),
				"torch": torch.get_rng_state(),
				"cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
			}
			self._atomic_torch_save(resume_payload, self.ckpt_dir / "latest.pth")
		if best_improved:
			self._atomic_torch_save(payload, self.ckpt_dir / "best.pth")
			self.logger.info(f"new best val global_top1={self.best_val_top1:.4f} -> best.pth")

	"""从 checkpoint 恢复训练状态；只加载可训练参数，冻结塔来自官方缓存。"""
	def _load_resume_state(self) -> None:
		self.index_sha256 = compute_file_sha256(self.index_json)
		if self.resume_path is None:
			return
		if not self.resume_path.is_file():
			raise FileNotFoundError(f"resume checkpoint does not exist: {self.resume_path}")

		state = torch.load(self.resume_path, map_location="cpu", weights_only=False)
		report = self.model.load_state_dict(state["model"], strict=False)
		if report.unexpected_keys:
			raise ValueError(f"resume checkpoint has unexpected keys: {report.unexpected_keys[:5]}")
		frozen_prefixes = ("image_tokenizer.feature_extractor.backbone", "text_tokenizer.backbone")
		unexpected_missing = [key for key in report.missing_keys if not key.startswith(frozen_prefixes)]
		if unexpected_missing:
			raise ValueError(f"resume checkpoint misses trainable keys: {unexpected_missing[:5]}")

		if "optimizer" not in state or "scheduler" not in state:
			raise ValueError(
				f"{self.resume_path} 不含 optimizer/scheduler 状态，无法续训。"
				"best.pth 是部署产物，续训请用 latest.pth。"
			)
		self.optimizer.load_state_dict(state["optimizer"])
		self.scheduler.load_state_dict(state["scheduler"])
		self.start_epoch = int(state["epoch"])
		self.global_step = int(state.get("global_step", 0))
		self.best_val_top1 = float(state.get("best_val_top1", 0.0))
		self.epoch_history = [dict(record) for record in state.get("epoch_history", [])]

		rng_state = state.get("rng_state")
		if rng_state is not None:
			random.setstate(rng_state["python"])
			torch.set_rng_state(rng_state["torch"])
			if rng_state.get("cuda") is not None and torch.cuda.is_available():
				torch.cuda.set_rng_state_all(rng_state["cuda"])

		checkpoint_sha256 = state.get("label_index", {}).get("sha256")
		if checkpoint_sha256 and checkpoint_sha256 != self.index_sha256:
			self.logger.warning(
				"label index sha256 changed since the checkpoint was written; "
				f"checkpoint={checkpoint_sha256[:12]} current={self.index_sha256[:12]}"
			)
		self.logger.info(
			f"Resumed from {self.resume_path} | epoch={self.start_epoch}/{self.total_epochs} | "
			f"global_step={self.global_step} | best_val_top1={self.best_val_top1:.4f}"
		)

	"""记录 epoch 级日志并同步 TensorBoard。"""
	def _log_epoch(
		self,
		epoch: int,
		train_metrics: Dict[str, float],
		eval_metrics: Optional[Dict[str, Any]],
		epoch_seconds: float,
	) -> None:
		log_step = epoch + 1
		current_lr = self.optimizer.param_groups[0]["lr"]
		message = (
			f"[epoch {log_step}/{self.total_epochs}] "
			f"loss={train_metrics['loss_total']:.4f} (g={train_metrics['loss_global']:.4f} "
			f"d={train_metrics['loss_detail']:.4f}) | train_top1={train_metrics['train_top1']:.4f} | "
			f"grad_norm={train_metrics['grad_norm']:.3f} | lr={current_lr:.2e} | {epoch_seconds:.0f}s"
		)
		baseline_global = float(self.eval_cfg.get("instruction_only_baseline_global", 0.0))
		baseline_joint = float(self.eval_cfg.get("instruction_only_baseline_joint", 0.0))
		if eval_metrics is not None:
			message += (
				f"\n  val: global_top1={eval_metrics['global_top1']:.4f} "
				f"(Δbaseline={eval_metrics['global_top1'] - baseline_global:+.4f}) "
				f"top5={eval_metrics['global_top5']:.4f} macro={eval_metrics['global_macro_top1']:.4f} "
				f"| head/mid/tail={eval_metrics['global_top1_head']:.3f}/"
				f"{eval_metrics['global_top1_mid']:.3f}/{eval_metrics['global_top1_tail']:.3f}"
				f"\n  val: joint_top1={eval_metrics['joint_top1']:.4f} "
				f"(Δbaseline={eval_metrics['joint_top1'] - baseline_joint:+.4f}) "
				f"| detail_top1={eval_metrics['detail_top1']:.4f} "
				f"slot_diversity={eval_metrics['detail_slot_diversity']:.2f}/9 "
				f"(标签 {eval_metrics['detail_label_slot_diversity']:.2f}/9, 仅 1 个自由度)"
				f"\n  val: codeword_error g={eval_metrics.get('global_codeword_error', float('nan')):.3f} "
				f"d={eval_metrics.get('detail_codeword_error', float('nan')):.3f} (0=全对, 1≈随机)"
			)
		self.logger.info(message)

		history_record = {
			"epoch": log_step,
			"epoch_time_seconds": epoch_seconds,
			"lr": current_lr,
			**{key: train_metrics[key] for key in ("loss_total", "loss_global", "loss_detail", "train_top1", "grad_norm")},
		}
		if eval_metrics is not None:
			history_record.update({f"val_{key}": eval_metrics[key] for key in EVAL_TENSORBOARD_KEYS if key in eval_metrics})
			# 分组明细只写日志文件，不进 TensorBoard。
			per_action = ", ".join(f"{name}={value:.3f}" for name, value in eval_metrics["per_action"].items())
			self.logger.info(f"  val per-action top1: {per_action}")
		self.epoch_history.append(history_record)

		if self.tb_writer is None:
			return
		self.tb_writer.add_scalar("train/loss_total", train_metrics["loss_total"], log_step)
		self.tb_writer.add_scalar("train/loss_global", train_metrics["loss_global"], log_step)
		self.tb_writer.add_scalar("train/loss_detail", train_metrics["loss_detail"], log_step)
		self.tb_writer.add_scalar("train/top1", train_metrics["train_top1"], log_step)
		self.tb_writer.add_scalar("train/grad_norm", train_metrics["grad_norm"], log_step)
		self.tb_writer.add_scalar("train/lr", current_lr, log_step)
		if eval_metrics is not None:
			for key in EVAL_TENSORBOARD_KEYS:
				if key in eval_metrics:
					self.tb_writer.add_scalar(f"val/{key}", eval_metrics[key], log_step)
			# 基线作为常量标量写入，在 TensorBoard 里形成一条水平参照线。
			self.tb_writer.add_scalar("val/baseline_global_top1", baseline_global, log_step)
			self.tb_writer.add_scalar("val/baseline_joint_top1", baseline_joint, log_step)

	"""外层训练循环。"""
	def train(self) -> None:
		if self.overfit_batch > 0:
			self._run_overfit_check()
			return

		for epoch in range(self.start_epoch, self.total_epochs):
			epoch_start = time.time()
			train_metrics = self._train_epoch(epoch)
			self.scheduler.step()

			should_evaluate = (epoch + 1) % self.eval_every_epochs == 0 or epoch + 1 == self.total_epochs
			eval_metrics = self._evaluate() if should_evaluate else None
			self._log_epoch(epoch, train_metrics, eval_metrics, time.time() - epoch_start)
			self._save_checkpoint(
				epoch=epoch,
				eval_metrics=eval_metrics,
				save_latest=(epoch + 1) % self.save_every_epochs == 0 or epoch + 1 == self.total_epochs,
			)

		self.logger.info(f"Training finished | best val global_top1={self.best_val_top1:.4f}")
		finish_tensorboard(rank=0, writer=self.tb_writer)

	"""过拟合自检：固定 overfit_batch 个样本反复训练 max_steps 步，train top-1 应逼近 100%。"""
	def _run_overfit_check(self) -> None:
		self.logger.info(
			f"Overfit check | samples={self.overfit_batch} | steps={self.max_steps} | "
			f"label_smoothing={float(self.loss_cfg['label_smoothing'])}"
		)
		self.model.train()
		batches = [self._move_batch_to_device(batch) for batch in self.train_loader]
		step = 0
		while step < self.max_steps:
			for batch in batches:
				with self._autocast():
					losses = self._forward_loss(batch)
				self.optimizer.zero_grad(set_to_none=True)
				self.grad_scaler.scale(losses["loss_total"]).backward()
				self.grad_scaler.unscale_(self.optimizer)
				clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.grad_clip_norm)
				self.grad_scaler.step(self.optimizer)
				self.grad_scaler.update()

				step += 1
				if step % 25 == 0 or step == self.max_steps:
					with torch.no_grad():
						global_top1 = float((losses["global_logits"].argmax(dim=-1) == batch["k_global"]).float().mean())
						detail_top1 = float((losses["detail_logits"].argmax(dim=-1) == batch["k_detail"]).float().mean())
					self.logger.info(
						f"  step {step:4d} | loss={float(losses['loss_total']):.4f} "
						f"(g={float(losses['loss_global']):.4f} d={float(losses['loss_detail']):.4f}) | "
						f"train_top1_g={global_top1:.4f} train_top1_d={detail_top1:.4f}"
					)
				if step >= self.max_steps:
					break
		self.logger.info("Overfit check finished")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train the VQAP Adapter (Stage 2)")
	parser.add_argument("--config", type=str, default=str(REPO_ROOT / "config" / "train_adapter.yaml"))
	parser.add_argument("--model-config", type=str, default=str(REPO_ROOT / "config" / "model.yaml"))
	parser.add_argument("--exp-name", type=str, default=None)
	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--resume", type=str, default=None)
	parser.add_argument("--disable-tensorboard", action="store_true")
	parser.add_argument("--overfit-batch", type=int, default=0, help="过拟合自检：固定前 N 个训练样本")
	parser.add_argument("--max-steps", type=int, default=0, help="限制步数，配合 --overfit-batch 使用")
	return parser.parse_args()


def main() -> None:
	trainer = AdapterTrainer(parse_args())
	trainer.train()


if __name__ == "__main__":
	main()
