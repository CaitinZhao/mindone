import numpy as np

import mindspore as ms
from mindspore import mint, nn, ops


class LowerTriangularMaskWithDynamic(nn.Cell):
    r"""
    Get the Strictly Lower triangular matrix from the input_ids.
    """

    def __init__(
        self,
        seq_length,
        batch_size=1,
        compute_type=ms.float16,
        is_dynamic=False,
        pad_token_id=0,
        use_flash_attention=False,
        use_attn_mask_compression=False,
        use_past=False,
        seq_split_num=1,
        chunk_prefill=False,
    ):
        super().__init__()
        self.dtype = compute_type
        self.is_dynamic = is_dynamic
        self.pad_token_id = pad_token_id
        self.use_flash_attention = use_flash_attention
        self.use_attn_mask_compression = use_attn_mask_compression
        self.seq_length = seq_length
        self.is_first_iteration = True
        self.multiply_data = ms.Tensor([-10000.0], dtype=compute_type)
        self.one = ms.Tensor([1.0], dtype=compute_type)
        self.is_pynative = ms.get_context("mode") == ms.PYNATIVE_MODE
        self.chunk_prefill = chunk_prefill
        self.assign_mask = ops.Assign()
        if use_past and chunk_prefill:
            self.lower_triangle_mask = ms.Tensor(
                np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)), dtype=compute_type
            )
        elif use_past and not self.is_pynative:
            if not self.use_flash_attention:
                self.lower_triangle_mask = ms.Tensor(
                    np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)), dtype=compute_type
                )
            elif self.is_dynamic:
                mask_coeff = 1.0 if compute_type is ms.bfloat16 else -10000.0
                self.lower_triangle_mask = ms.Tensor(
                    np.triu(np.ones(shape=(128, 128), dtype=np.float16), 1) * mask_coeff, dtype=compute_type
                )
            else:
                self.lower_triangle_mask = None
        else:
            if use_attn_mask_compression:
                if seq_length < 2048:
                    raise ValueError("seq_length should be larger than 2048 when use mask_compression")
                self.lower_triangle_mask = ms.Tensor(np.triu(np.ones((2048, 2048), dtype=np.int8), k=1), dtype=ms.uint8)
            else:
                self.lower_triangle_mask = ms.Tensor(
                    np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)), dtype=compute_type
                )

        self.seq_split_num = seq_split_num
        self.seq_pipe = seq_split_num > 1
        if self.seq_pipe:
            self.mask_cache = ms.Parameter(
                ms.Tensor(shape=(batch_size, seq_length), dtype=ms.float32, init=ms.common.initializer.Zero()),
                name="mask_cache",
                requires_grad=False,
                parallel_optimizer=False,
            )
            mask_mask = np.zeros((batch_size, seq_length), dtype=np.int32)
            self.seq_seg_len = seq_length // self.seq_split_num
            for s in range(self.seq_split_num):
                mask_mask[:, s * self.seq_seg_len : (s + 1) * self.seq_seg_len] = s
            self.mask_mask = ms.Tensor(mask_mask)
            np_range = np.arange(seq_length // self.seq_split_num)
            self.seq_seg_range = ms.Tensor(np_range, dtype=ms.int32)
            self.seq_seg_len = ms.Tensor(seq_length // self.seq_split_num, dtype=ms.int32)

    def construct(self, tokens=None, masks=None, seq_chunk=None):
        """Forward process of the CausalMask"""
        if self.use_attn_mask_compression:
            attention_mask = self.lower_triangle_mask
            return attention_mask
        if tokens is not None:
            bs = tokens.shape[0]
            seq_len = tokens.shape[1]
            input_mask = (tokens != self.pad_token_id).to(self.dtype)
        else:
            bs = masks.shape[0]
            seq_len = masks.shape[1]
            input_mask = masks.to(self.dtype)
        shape_right = (bs, 1, seq_len)

        # Mask the padded inputs
        mask_right = input_mask.reshape(shape_right)
        attention_mask = mask_right

        lower_triangle_mask = self.lower_triangle_mask
        if self.is_pynative or self.is_dynamic:
            lower_triangle_mask = self.lower_triangle_mask[:seq_len, :seq_len]
        lower_triangle = ops.expand_dims(lower_triangle_mask, 0)

        if self.seq_pipe:
            seq_seg_range = self.seq_seg_range + self.seq_seg_len * seq_chunk
            attention_mask_chunk = mint.gather(lower_triangle, seq_seg_range, 1)
            mask_mask = (self.mask_mask == seq_chunk).to(self.dtype)
            input_mask = input_mask.tile((1, self.seq_split_num))
            input_mask = input_mask * mask_mask
            input_mask_update = input_mask + self.mask_cache
            mask_update = self.assign_mask(self.mask_cache, input_mask_update)
            mask_reshape = input_mask_update.reshape((bs, 1, seq_len * self.seq_split_num))
            mask_reshape = ops.depend(mask_reshape, mask_update)
            attention_mask = mask_reshape * attention_mask_chunk
            attention_mask = self.one - attention_mask
            attention_mask = ops.expand_dims(attention_mask, 1)
            attention_mask = attention_mask.to(ms.uint8)
            return attention_mask
        # the returned shape is [bs, 1, seq_length, seq_length]
        attention_mask = attention_mask * lower_triangle
        attention_mask = self.one - attention_mask
        attention_mask = ops.expand_dims(attention_mask, 1)
        if self.use_flash_attention:
            attention_mask = attention_mask.to(ms.uint8)
        else:
            attention_mask = attention_mask * self.multiply_data
        return attention_mask

    def prefill(self):
        return self.lower_triangle_mask

    def chunk_masks(self, seq_range):
        masks = mint.gather(self.lower_triangle_mask, seq_range, 0)
        return 1 - masks
