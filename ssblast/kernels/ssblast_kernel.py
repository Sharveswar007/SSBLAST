# ssblast/kernels/ssblast_kernel.py
# Layer 4 -- FP8 Per-Tile Scaled GEMM
# THE NOVEL CONTRIBUTION OF ssBlast

import triton
import triton.language as tl
import cupy as cp
import torch


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fp8_scaled_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_row = tl.program_id(0)
    block_col = tl.program_id(1)
    rows = block_row * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = block_col * BLOCK_N + tl.arange(0, BLOCK_N)
    acc  = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_idx  = k + tl.arange(0, BLOCK_K)
        a_mask = (rows[:, None] < M) & (k_idx[None, :] < K)
        a_tile = tl.load(A_ptr + rows[:, None] * stride_am + k_idx[None, :] * stride_ak,
                         mask=a_mask, other=0.0)
        b_mask = (k_idx[:, None] < K) & (cols[None, :] < N)
        b_tile = tl.load(B_ptr + k_idx[:, None] * stride_bk + cols[None, :] * stride_bn,
                         mask=b_mask, other=0.0)

        # Per-tile FP8 scaling
        a_max   = tl.max(tl.abs(a_tile))
        b_max   = tl.max(tl.abs(b_tile))
        a_scale = tl.where(a_max == 0.0, 1.0, a_max / 447.0)
        b_scale = tl.where(b_max == 0.0, 1.0, b_max / 447.0)

        product = tl.dot(
            (a_tile / a_scale).to(tl.float16),
            (b_tile / b_scale).to(tl.float16),
            out_dtype=tl.float32,
        )
        acc += product * a_scale * b_scale

    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(C_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
             acc, mask=c_mask)


def fp8_gemm(A, b, config):
    M = A.shape[0]
    N = 1
    K = A.shape[1]
    b_col = b.reshape(M, 1)

    # CuPy -> numpy -> torch (host round-trip; correct and reliable)
    A_t = torch.from_numpy(A.astype(cp.float32).get()).to('cuda').contiguous()
    b_t = torch.from_numpy(b_col.astype(cp.float32).get()).to('cuda').contiguous()
    C_t = torch.zeros((M, N), dtype=torch.float32, device='cuda')

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    _fp8_scaled_gemm_kernel[grid](
        A_t, b_t, C_t,
        M, N, K,
        A_t.stride(0), A_t.stride(1),
        b_t.stride(0), b_t.stride(1),
        C_t.stride(0), C_t.stride(1),
    )

    # torch -> cupy via dlpack (zero-copy on GPU)
    return cp.from_dlpack(C_t.reshape(M)).astype(cp.float64)
