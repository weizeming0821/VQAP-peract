from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn

from .module.encoder import AdapterEncoderLayer, AdapterImageTokenizer, AdapterTextTokenizer
from .module.model_init import apply_vqap_initialization


"""VQAP Adapter 顶层模型。

把「子任务起始观测 + 子任务指令」映射到冻结 VQAP 双码本的索引，是冻结动作编码器的摊还（amortization）。

输入：
	__init__:
		model_args: Dict[str, Any]，来自 model.yaml 的模型配置，至少包含 `Adapter` 段。
	forward:
		camera_images: Dict[str, torch.Tensor]，键为相机名称，值为 [B, 3, H, W]。
			uint8 张量按 [0, 255] 解释，浮点张量按 [0, 1] 解释。
		instructions: 长度为 B 的子任务指令字符串序列。

输出：
	forward:
		global_logits: [B, global_codebook_size]
		detail_logits: [B, num_detail_queries, detail_codebook_size]

序列布局（长度 S = N_cam * P + T_clip + 1 + N_detail）：
	[ 图像 token | 文本 token | 全局 query | 细节 query ]

注意力可见性（行为 query 位置，列为 key 位置）：
	图像 / 文本  -> 图像、文本
	全局 query   -> 图像、文本、自身
	细节 query   -> 全部 token
即观测段看不到任何 query token，全局码只由观测决定，细节码以全局码为条件。

⚠️ 槽位契约：第 n 个细节 query 预测的是 VQAP 细节支路第 n 个 learnable query 的码索引。
标签导出必须保持同一顺序，错位不会报错，只会让 9 路分类全部学成噪声。
"""
class Adapter(nn.Module):

	def __init__(self, model_args: Dict[str, Any]) -> None:
		super().__init__()
		self.model_args = model_args
		adapter_args = model_args["Adapter"]
		image_args = adapter_args["image_tokenizer"]
		text_args = adapter_args["text_tokenizer"]
		transformer_args = adapter_args["transformer"]

		self.hidden_dim = int(adapter_args["hidden_dim"])
		self.num_detail_queries = int(adapter_args["num_detail_queries"])
		self.global_codebook_size = int(adapter_args["global_codebook_size"])
		self.detail_codebook_size = int(adapter_args["detail_codebook_size"])
		self._validate_codebook_config(model_args)

		self.image_tokenizer = AdapterImageTokenizer(
			dinov2_cfg=image_args["dinov2"],
			hidden_dim=self.hidden_dim,
			camera_names=image_args["camera_names"],
			intermediate_size=int(image_args["intermediate_size"]),
			patch_pool=int(image_args["patch_pool"]),
			freeze=bool(image_args["freeze"]),
		)
		self.text_tokenizer = AdapterTextTokenizer(
			clip_cfg=text_args["clip"],
			hidden_dim=self.hidden_dim,
			freeze=bool(text_args["freeze"]),
		)

		self.num_cameras = self.image_tokenizer.num_cameras
		self.num_tokens_per_camera = self.image_tokenizer.num_tokens_per_camera
		self.num_image_tokens = self.num_cameras * self.num_tokens_per_camera
		self.num_text_tokens = self.text_tokenizer.num_tokens
		self.num_query_tokens = 1 + self.num_detail_queries
		self.sequence_length = self.num_image_tokens + self.num_text_tokens + self.num_query_tokens

		# 段边界：[图像 | 文本 | 全局 query | 细节 query]
		self.text_start_index = self.num_image_tokens
		self.query_start_index = self.num_image_tokens + self.num_text_tokens
		self.global_query_index = self.query_start_index
		self.detail_query_start_index = self.query_start_index + 1

		self.camera_embedding = nn.Embedding(self.num_cameras, self.hidden_dim)
		self.position_embedding_2d = nn.Parameter(torch.zeros(self.num_tokens_per_camera, self.hidden_dim))
		self.query_tokens = nn.Parameter(torch.zeros(self.num_query_tokens, self.hidden_dim))

		self.layers = nn.ModuleList(
			[
				AdapterEncoderLayer(
					hidden_dim=self.hidden_dim,
					num_heads=int(transformer_args["num_heads"]),
					ffn_dim=int(transformer_args["ffn_dim"]),
					dropout=float(transformer_args["dropout"]),
					norm_type=str(transformer_args["norm_type"]),
				)
				for _ in range(int(transformer_args["num_layers"]))
			]
		)

		self.global_head = nn.Sequential(
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.GELU(),
			nn.Linear(self.hidden_dim, self.global_codebook_size),
		)
		self.detail_head = nn.Sequential(
			nn.Linear(self.hidden_dim * 2, self.hidden_dim),
			nn.GELU(),
			nn.Linear(self.hidden_dim, self.detail_codebook_size),
		)

		self.register_buffer("block_mask", self._build_block_mask(), persistent=False)
		self._init_model_parameters()

	"""执行 Adapter 全模型统一初始化。"""
	def _init_model_parameters(self) -> None:
		apply_vqap_initialization(self)

	"""对 nn.Parameter 形式的 query token 与位置嵌入做项目特化初始化。"""
	def init_parameters(self) -> None:
		nn.init.normal_(self.position_embedding_2d, mean=0.0, std=0.02)
		nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

	"""交叉校验 Adapter 的码本规格与 AtomAction_NSVQ 段是否一致。

	当 model_args 中不含 AtomAction_NSVQ 段（例如只加载 Adapter 插件配置）时跳过校验。
	"""
	def _validate_codebook_config(self, model_args: Dict[str, Any]) -> None:
		nsvq_args = model_args.get("AtomAction_NSVQ")
		if not isinstance(nsvq_args, dict):
			return

		global_codebook_args = nsvq_args.get("global_codebook", {})
		detail_codebook_args = nsvq_args.get("detail_codebook", {})
		expected_values = {
			"num_detail_queries": (self.num_detail_queries, detail_codebook_args.get("num_queries")),
			"global_codebook_size": (self.global_codebook_size, global_codebook_args.get("codebook_size")),
			"detail_codebook_size": (self.detail_codebook_size, detail_codebook_args.get("codebook_size")),
		}
		for field_name, (adapter_value, nsvq_value) in expected_values.items():
			if nsvq_value is None:
				continue
			if int(nsvq_value) != int(adapter_value):
				raise ValueError(
					f"Adapter.{field_name} does not match AtomAction_NSVQ config. "
					f"Got: {adapter_value}, expected: {int(nsvq_value)}"
				)

	"""构建与 batch 无关的可见性矩阵。

	维度变化：
		block_mask: [S, S]，True 表示该 query 位置可以看到该 key 位置。
	"""
	def _build_block_mask(self) -> torch.Tensor:
		block_mask = torch.zeros(self.sequence_length, self.sequence_length, dtype=torch.bool)
		observation_end_index = self.query_start_index

		# 观测段（图像 + 文本）内部双向可见，且看不到任何 query token。
		block_mask[:observation_end_index, :observation_end_index] = True
		# 全局 query 可见观测段与自身，看不到细节 query。
		block_mask[self.global_query_index, :observation_end_index] = True
		block_mask[self.global_query_index, self.global_query_index] = True
		# 细节 query 可见全部 token。
		block_mask[self.detail_query_start_index:, :] = True

		# 每一行都至少能看到一个图像 token，而图像 token 恒为有效 key，
		# 因此与 padding 合成之后不可能出现整行被屏蔽（softmax 不会出 NaN）。
		if not bool(block_mask[:, : self.num_image_tokens].any(dim=-1).all()):
			raise ValueError("every query position must be able to attend to at least one image token")
		return block_mask

	"""拼装输入序列并合成可见性矩阵。

	维度变化：
		image_tokens: [B, N_cam, P, C] -> [B, N_cam * P, C]
		sequence: [B, N_cam * P, C] + [B, T_clip, C] + [B, N_query, C] -> [B, S, C]
		attention_mask: [S, S] AND [B, S] -> [B, S, S]
	"""
	def _build_sequence(
		self,
		image_tokens: torch.Tensor,
		image_mask: torch.Tensor,
		text_tokens: torch.Tensor,
		text_mask: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		batch_size = image_tokens.shape[0]
		device = image_tokens.device

		camera_indices = torch.arange(self.num_cameras, device=device)
		camera_embedding = self.camera_embedding(camera_indices)	# [N_cam] -> [N_cam, C]

		# [B, N_cam, P, C] + [1, N_cam, 1, C] + [1, 1, P, C] -> [B, N_cam, P, C]
		image_tokens = image_tokens + camera_embedding.unsqueeze(0).unsqueeze(2) + self.position_embedding_2d.unsqueeze(0).unsqueeze(0)
		image_tokens = image_tokens.reshape(batch_size, self.num_image_tokens, self.hidden_dim)
		image_mask = image_mask.reshape(batch_size, self.num_image_tokens)

		query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)	# [N_query, C] -> [B, N_query, C]
		query_mask = torch.ones(batch_size, self.num_query_tokens, dtype=torch.bool, device=device)

		sequence = torch.cat([image_tokens, text_tokens, query_tokens], dim=1)	# -> [B, S, C]
		key_valid_mask = torch.cat([image_mask, text_mask, query_mask], dim=1)	# -> [B, S]
		attention_mask = self.block_mask.unsqueeze(0) & key_valid_mask.unsqueeze(1)	# [1, S, S] & [B, 1, S] -> [B, S, S]
		return sequence, attention_mask

	def forward(
		self,
		camera_images: Dict[str, torch.Tensor],
		instructions: Sequence[str],
	) -> Dict[str, torch.Tensor]:
		image_tokens, image_mask = self.image_tokenizer(camera_images)	# -> [B, N_cam, P, C], [B, N_cam, P]
		text_tokens, text_mask = self.text_tokenizer(instructions)	# -> [B, T_clip, C], [B, T_clip]
		if image_tokens.shape[0] != text_tokens.shape[0]:
			raise ValueError("camera_images and instructions must describe the same batch size")

		sequence, attention_mask = self._build_sequence(
			image_tokens=image_tokens,
			image_mask=image_mask,
			text_tokens=text_tokens,
			text_mask=text_mask,
		)
		for layer in self.layers:
			sequence = layer(sequence, attention_mask)	# [B, S, C] -> [B, S, C]

		global_feature = sequence[:, self.global_query_index]	# [B, S, C] -> [B, C]
		detail_feature = sequence[:, self.detail_query_start_index:]	# [B, S, C] -> [B, N_detail, C]

		global_logits = self.global_head(global_feature)	# [B, C] -> [B, K_g]

		# 细节头显式拼接全局隐状态，实现「细节码以全局码为条件」。
		expanded_global_feature = global_feature.unsqueeze(1).expand(-1, self.num_detail_queries, -1)	# [B, C] -> [B, N_detail, C]
		detail_input = torch.cat([detail_feature, expanded_global_feature], dim=-1)	# -> [B, N_detail, 2C]
		detail_logits = self.detail_head(detail_input)	# [B, N_detail, 2C] -> [B, N_detail, K_d]

		return {
			"global_logits": global_logits,
			"detail_logits": detail_logits,
		}

	"""计算 Adapter 总参数量与可训练参数量。"""
	def print_module_params(self) -> Dict[str, int]:
		total_params = sum(param.numel() for param in self.parameters())
		trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
		print(f"Adapter total params: {total_params}")
		print(f"Adapter trainable params: {trainable_params}")
		return {
			"total": total_params,
			"trainable": trainable_params,
		}
