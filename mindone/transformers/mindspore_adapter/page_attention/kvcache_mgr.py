# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""KVCache Manager for inference."""
import numpy as np

import mindspore as ms
from mindspore import mint, nn, ops


class KVCacheMgr(nn.Cell):
    """KVCache Manager."""

    def __init__(
        self,
        n_head,
        head_dim,
        max_batch_size=8,
        max_seq_length=4096,
        compute_dtype=ms.float16,
        is_dynamic=False,
        use_kvcache_op=True,
        is_flexible_shape=False,
    ):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.dtype = compute_dtype
        self.use_kvcache_op = use_kvcache_op
        self.is_dynamic = is_dynamic
        self.is_flexible_shape = is_flexible_shape
        self.is_first_iteration = True

        self.cache_length_tensor = ms.Tensor([max_batch_size * max_seq_length], dtype=ms.int32)
        self.cache_pad_tensor = ms.Tensor([3], dtype=ms.int64)
        self.seq_length_tensor = ms.Tensor([max_seq_length], dtype=ms.int32)
        self.seq_length_tensor_pad = ms.Tensor([max_seq_length, 3], dtype=ms.int64)
        self.seqlen_axis_tensor_pad = ms.Tensor([2, 3], dtype=ms.int64)
        self.pad_before = ms.Tensor([0, 0, 0, 0, 0], ms.int32)
        self.pad_after = ms.Tensor([0, 0], ms.int32)
        self.pad_zero = ms.Tensor(0, compute_dtype)
        self.pad = ops.PadV3()
        self.assign = ops.Assign()

        if self.use_kvcache_op:
            # pylint: disable=W0212
            self.prompt_kvcache = ops._inner_ops.PromptKVCache()
            # pylint: disable=W0212
            self.decoder_kvcache = ops._inner_ops.DecoderKVCache()

        kv_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.key_past = ms.Parameter(ms.Tensor(np.zeros(kv_shape), compute_dtype), name="key_past", requires_grad=False)
        self.value_past = ms.Parameter(
            ms.Tensor(np.zeros(kv_shape), compute_dtype), name="value_past", requires_grad=False
        )

    def padding(self, key, value, seq_length):
        """padding key, value"""
        pad_length = self.seq_length_tensor - seq_length
        # calculate padding parameter: (0, 0),(0,0),(0,pad_length),(0,0), append values of 'pad_length' in axis
        pad_config = mint.cat((self.pad_before, pad_length, self.pad_after), dim=0)
        key_padding = self.pad(key, pad_config, self.pad_zero)
        value_padding = self.pad(value, pad_config, self.pad_zero)
        return key_padding, value_padding

    def trimming(self, key, value, zactivate_len, batch_size):
        """tramming key, value"""
        if self.is_flexible_shape:
            key = key.reshape(batch_size, self.n_head, -1, self.head_dim)
            value = value.reshape(batch_size, self.n_head, -1, self.head_dim)
        if zactivate_len is not None:
            act_len = zactivate_len.shape[0]
            key = key[:batch_size, : self.n_head, :act_len, : self.head_dim]
            value = value[:batch_size, : self.n_head, :act_len, : self.head_dim]
        elif not self.is_flexible_shape:
            key = key[:batch_size, : self.n_head, : self.max_seq_length, : self.head_dim]
            value = value[:batch_size, : self.n_head, : self.max_seq_length, : self.head_dim]
        return key, value

    def auto_caching(self, key_update, value_update, batch_valid_length, seq_length_tensor_pad, batch_index_pad=None):
        """use kvcache op to cache key, value"""
        # key_update shape: [real_bs, n_head, max_seqlen, head_dim]
        if self.is_first_iteration:
            batch_valid_length = batch_valid_length * 0
            self.prompt_kvcache(
                self.key_past,
                key_update,
                batch_valid_length,
                batch_index_pad,
                self.seqlen_axis_tensor_pad,
                seq_length_tensor_pad,
                seq_length_tensor_pad,
            )
            self.prompt_kvcache(
                self.value_past,
                value_update,
                batch_valid_length,
                batch_index_pad,
                self.seqlen_axis_tensor_pad,
                seq_length_tensor_pad,
                seq_length_tensor_pad,
            )
            return None

        key_cache = self.key_past
        value_cache = self.value_past
        key_update = self.decoder_kvcache(
            self.key_past,
            key_update,
            batch_valid_length,
            batch_index_pad,
            self.seqlen_axis_tensor_pad,
            seq_length_tensor_pad,
            seq_length_tensor_pad,
        )
        value_update = self.decoder_kvcache(
            self.value_past,
            value_update,
            batch_valid_length,
            batch_index_pad,
            self.seqlen_axis_tensor_pad,
            seq_length_tensor_pad,
            seq_length_tensor_pad,
        )
        key_cache = ops.depend(key_cache, key_update)
        value_cache = ops.depend(value_cache, value_update)
        return key_cache, value_cache

    def manual_caching(self, key_update, value_update, valid_length_vector, batch_size):
        """use assign to cache key, value"""
        # key_update shape: [real_bs, n_head, 1, head_dim]
        if self.is_first_iteration:
            if self.is_dynamic:
                self.assign(self.key_past, key_update.reshape((self.max_batch_size, self.n_head, -1, self.head_dim)))
                self.assign(
                    self.value_past, value_update.reshape((self.max_batch_size, self.n_head, -1, self.head_dim))
                )
            else:
                self.assign(self.key_past, key_update * valid_length_vector)
                self.assign(self.value_past, value_update * valid_length_vector)
            return None

        if self.is_dynamic:
            key = self.key_past.reshape((batch_size, self.n_head, -1, self.head_dim)) + key_update * valid_length_vector
            value = (
                self.value_past.reshape((batch_size, self.n_head, -1, self.head_dim))
                + value_update * valid_length_vector
            )
            self.assign(self.key_past, key.reshape((self.max_batch_size, self.n_head, -1, self.head_dim)))
            self.assign(self.value_past, value.reshape((self.max_batch_size, self.n_head, -1, self.head_dim)))
        else:
            key = self.key_past + key_update * valid_length_vector
            value = self.value_past + value_update * valid_length_vector
            self.assign(self.key_past, key)
            self.assign(self.value_past, value)
        # key shape: [real_bs, n_head, max_cache_len // real_bs, head_dim]
        return key, value

    def construct(self, key, value, kvcache_inputs=None):
        """The forward compute of KVCacheMgr."""
        # add inputs check
        batch_valid_length, zactivate_len, batch_index_pad, seq_length_tensor_pad = kvcache_inputs
        if not self.use_kvcache_op:
            batch_valid_length = batch_valid_length.to(self.dtype)
        batch_size, _, seq_length, _ = key.shape
        if self.is_first_iteration:
            if self.is_dynamic:
                key_padding, value_padding = self.padding(key, value, seq_length=seq_length)
            else:
                key_padding, value_padding = key, value
            if self.use_kvcache_op:
                self.auto_caching(
                    key_padding, value_padding, batch_valid_length, seq_length_tensor_pad, batch_index_pad
                )
            else:
                self.manual_caching(key_padding, value_padding, batch_valid_length, batch_size=batch_size)
        else:
            if self.use_kvcache_op:
                key, value = self.auto_caching(key, value, batch_valid_length, seq_length_tensor_pad, batch_index_pad)
            else:
                key, value = self.manual_caching(key, value, batch_valid_length, batch_size=batch_size)
            key, value = self.trimming(key, value, zactivate_len, batch_size=batch_size)

        return key, value


class KVCachePreprocess(nn.Cell):
    """KVCache Manager."""

    def __init__(
        self, max_batch_size=8, max_seq_length=4096, is_dynamic=False, use_kvcache_op=False, is_flexible_shape=False
    ):
        super().__init__()
        self.is_dynamic = is_dynamic
        self.use_kvcache_op = use_kvcache_op
        self.is_flexible_shape = is_flexible_shape

        self.max_cache_length = max_batch_size * max_seq_length
        range_len = self.max_cache_length if self.is_flexible_shape else max_seq_length
        self.range = ms.Tensor(np.arange(range_len).reshape((1, 1, -1)), ms.int32)
        self.cache_length_tensor = ms.Tensor([max_batch_size * max_seq_length], dtype=ms.int32)
        self.cache_pad_tensor = ms.Tensor([3], dtype=ms.int64)
        self.seq_length_tensor = ms.Tensor([max_seq_length], dtype=ms.int32)
        self.seq_length_tensor_pad = ms.Tensor([max_seq_length, 3], dtype=ms.int64)
        self.is_first_iteration = ms.Parameter(ms.Tensor(True))

    def construct(self, batch_size, batch_valid_length=None, batch_index=None, zactivate_len=None):
        """precompute kvcache inputs"""
        seq_range = self.range
        if self.is_dynamic and self.is_flexible_shape and not self.use_kvcache_op:
            seq_range = seq_range[:1, :1, : self.max_cache_length // batch_size]

        if self.use_kvcache_op:
            if batch_index is None:
                batch_index = ops.arange(0, batch_size, 1)
            batch_index_pad = mint.cat((batch_index, self.cache_pad_tensor), dim=0)
            seq_length_tensor_pad = self.get_seq_length_tensor_pad(batch_size=batch_size)
            batch_valid_length = batch_valid_length.reshape(
                -1,
            ).to(ms.int64)
            kvcache_inputs = (batch_valid_length, zactivate_len, batch_index_pad, seq_length_tensor_pad)
        else:
            if self.is_first_iteration:
                self.is_first_iteration = False
                valid_length_vector = seq_range < batch_valid_length.reshape(-1, 1, 1)
            else:
                valid_length_vector = seq_range == batch_valid_length.reshape(-1, 1, 1)
            valid_length_vector = ops.expand_dims(valid_length_vector, 3)
            kvcache_inputs = (valid_length_vector, zactivate_len, None, None)
        return kvcache_inputs

    def get_seq_length_tensor_pad(self, batch_size):
        """get seq_length_tensor_pad"""
        if self.is_flexible_shape:
            max_seq_length = (self.cache_length_tensor / batch_size).to(ms.int64)
            return mint.cat((max_seq_length, self.cache_pad_tensor), dim=0)
        return self.seq_length_tensor_pad
