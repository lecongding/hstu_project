# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from generative_recommenders.research.modeling.sequential.autoregressive_losses import (
    AutoregressiveLoss,
    NegativesSampler,
)
from torch.utils.checkpoint import checkpoint


class SampledSoftmaxLoss(AutoregressiveLoss):
    def __init__(
        self,
        num_to_sample: int,
        softmax_temperature: float,
        model,
        activation_checkpoint: bool = False,
        # === 新增：流行度校正 ===
        item_logQ: torch.Tensor = None,   # shape [num_items+1]
        pop_alpha: float = 0.0,           # 0 表示不开启
    ) -> None:
        super().__init__()

        self._num_to_sample: int = num_to_sample
        self._softmax_temperature: float = softmax_temperature
        self._model = model
        self._activation_checkpoint: bool = activation_checkpoint

        self._pop_alpha: float = float(pop_alpha)
        if item_logQ is not None:
            # register_buffer 让它跟着 .to(device)、DDP 同步
            self.register_buffer("_item_logQ", item_logQ.float())
        else:
            self._item_logQ = None

    def _apply_pop_correction_pos(self, logits: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
        """logits: [N',1], pos_ids: [N']"""
        if (self._item_logQ is None) or (self._pop_alpha <= 0.0):
            return logits
        logQ = self._item_logQ[pos_ids].to(device=logits.device, dtype=logits.dtype).unsqueeze(1)  # [N',1]
        return logits - (self._pop_alpha * logQ)

    def _apply_pop_correction_neg(self, logits: torch.Tensor, neg_ids: torch.Tensor) -> torch.Tensor:
        """logits: [N',R], neg_ids: [N',R]"""
        if (self._item_logQ is None) or (self._pop_alpha <= 0.0):
            return logits
        logQ = self._item_logQ[neg_ids].to(device=logits.device, dtype=logits.dtype)  # [N',R]
        return logits - (self._pop_alpha * logQ)

    def jagged_forward(  # pyre-ignore [15]
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
        **kwargs,
    ):
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        assert supervision_ids.size() == supervision_weights.size()
    
        sampled_ids, sampled_negative_embeddings = negatives_sampler(
            positive_ids=supervision_ids,
            num_to_sample=self._num_to_sample,
        )
        positive_embeddings = negatives_sampler.normalize_embeddings(supervision_embeddings)

        positive_logits, aux_losses = self._model.similarity_fn(
            query_embeddings=output_embeddings,                    # [N', D]
            item_ids=supervision_ids.unsqueeze(1),                 # [N', 1]
            item_embeddings=positive_embeddings.unsqueeze(1),       # [N', 1, D]
            **kwargs,
        )
        # 先 /T，再 -alpha*logQ
        positive_logits = positive_logits / self._softmax_temperature
        positive_logits = self._apply_pop_correction_pos(positive_logits, supervision_ids)

        sampled_negatives_logits, _ = self._model.similarity_fn(
            query_embeddings=output_embeddings,                   # [N', D]
            item_ids=sampled_ids,                                 # [N', R]
            item_embeddings=sampled_negative_embeddings,          # [N', R, D]
            **kwargs,
        )
        sampled_negatives_logits = sampled_negatives_logits / self._softmax_temperature
        sampled_negatives_logits = self._apply_pop_correction_neg(sampled_negatives_logits, sampled_ids)

        # 屏蔽：负样本里出现正样本 id 的位置
        sampled_negatives_logits = torch.where(
            supervision_ids.unsqueeze(1) == sampled_ids,
            -5e4,
            sampled_negatives_logits,
        )

        jagged_loss = -F.log_softmax(
            torch.cat([positive_logits, sampled_negatives_logits], dim=1),
            dim=1,
        )[:, 0]

        return (jagged_loss * supervision_weights).sum() / supervision_weights.sum(), aux_losses


    def forward(  # pyre-ignore [15]
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            lengths: [B] x int32 representing number of non-zero elements per row.
            output_embeddings: [B, N, D] x float, embeddings for the current
                input sequence.
            supervision_ids: [B, N] x int64, (positive) supervision ids.
            supervision_embeddings: [B, N, D] x float.
            supervision_weights: Optional [B, N] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            Tuple of (loss for the current engaged sequence, str-keyed aux_losses).
        """
        torch._assert(
            output_embeddings.size() == supervision_embeddings.size(),
            "Invalid supervision embeddings size.",
        )
        torch._assert(
            supervision_ids.size() == supervision_embeddings.size()[:-1],
            "Invalid supervision ids size.",
        )

        jagged_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        jagged_supervision_ids = (
            torch.ops.fbgemm.dense_to_jagged(
                supervision_ids.unsqueeze(-1).float(), [jagged_id_offsets]
            )[0]
            .squeeze(1)
            .long()
        )
        if "user_ids" in kwargs:
            # expand to jagged.
            max_length: int = int(lengths.max())
            kwargs["user_ids"] = torch.ops.fbgemm.dense_to_jagged(
                kwargs["user_ids"]
                .unsqueeze(1)
                .expand(-1, max_length)
                .unsqueeze(2),  # (B, max_length, 1)
                [jagged_id_offsets],
            )[0].squeeze(1)

        args = OrderedDict(
            [
                (
                    "output_embeddings",
                    torch.ops.fbgemm.dense_to_jagged(
                        output_embeddings,
                        [jagged_id_offsets],
                    )[0],
                ),
                ("supervision_ids", jagged_supervision_ids),
                (
                    "supervision_embeddings",
                    torch.ops.fbgemm.dense_to_jagged(
                        supervision_embeddings,
                        [jagged_id_offsets],
                    )[0],
                ),
                (
                    "supervision_weights",
                    torch.ops.fbgemm.dense_to_jagged(
                        supervision_weights.unsqueeze(-1),
                        [jagged_id_offsets],
                    )[0].squeeze(1),
                ),
                ("negatives_sampler", negatives_sampler),
            ]
        )
        args.update(kwargs)
        if self._activation_checkpoint:
            return checkpoint(
                self.jagged_forward,
                *args.values(),
                use_reentrant=False,
            )
        else:
            return self.jagged_forward(
                output_embeddings=torch.ops.fbgemm.dense_to_jagged(
                    output_embeddings,
                    [jagged_id_offsets],
                )[0],
                supervision_ids=jagged_supervision_ids,
                supervision_embeddings=torch.ops.fbgemm.dense_to_jagged(
                    supervision_embeddings,
                    [jagged_id_offsets],
                )[0],
                supervision_weights=torch.ops.fbgemm.dense_to_jagged(
                    supervision_weights.unsqueeze(-1),
                    [jagged_id_offsets],
                )[0].squeeze(1),
                negatives_sampler=negatives_sampler,
                **kwargs,
            )
