# HIP decode-attention resource usage

Compiled `engine/src/kernels/cuda/qsa_decode_attn.cu` for `gfx1201` with `/opt/rocm/lib/llvm/bin/clang++` and `-Rpass-analysis=resource-usage`. The translation unit emitted all six `attn_merge_v2_kernel<U, CHUNK>` specializations in one compile. The row-kernel measurements below use the default gfx1201 INT8 path (`KV_MODE=1`, lane-cell off, V4 on); that kernel does not depend on U, so its values repeat for each U. VGPR, SGPR, and occupancy values are the compiler's diagnostics; LDS is the static shared-memory size from the kernel arrays.

| U | CHUNK | Kernel | VGPRs | SGPRs | LDS (bytes) | Occupancy (waves/SIMD) |
|---:|---:|---|---:|---:|---:|---:|
| 16 | 64 | `attn_chunk_rows_kernel<1,false,true,64>` | 73 | 40 | 15,872 | 16 |
| 16 | 64 | `attn_merge_v2_kernel<16,64>` | 21 | 78 | 0 | 16 |
| 8 | 64 | `attn_chunk_rows_kernel<1,false,true,64>` | 73 | 40 | 15,872 | 16 |
| 8 | 64 | `attn_merge_v2_kernel<8,64>` | 13 | 55 | 0 | 16 |
| 4 | 64 | `attn_chunk_rows_kernel<1,false,true,64>` | 73 | 40 | 15,872 | 16 |
| 4 | 64 | `attn_merge_v2_kernel<4,64>` | 9 | 43 | 0 | 16 |
| 16 | 128 | `attn_chunk_rows_kernel<1,false,true,128>` | 73 | 40 | 19,456 | 12 |
| 16 | 128 | `attn_merge_v2_kernel<16,128>` | 21 | 78 | 0 | 16 |
| 8 | 128 | `attn_chunk_rows_kernel<1,false,true,128>` | 73 | 40 | 19,456 | 12 |
| 8 | 128 | `attn_merge_v2_kernel<8,128>` | 13 | 55 | 0 | 16 |
| 4 | 128 | `attn_chunk_rows_kernel<1,false,true,128>` | 73 | 40 | 19,456 | 12 |
| 4 | 128 | `attn_merge_v2_kernel<4,128>` | 9 | 43 | 0 | 16 |

For wave32 on gfx1201, use the 512-VGPR/SIMD-thread-slot budget as the reference; the compiler reports 73 VGPRs for the representative row kernel and 9/13/21 for merge U=4/8/16. The reported maximum occupancy is 16 waves/SIMD at CHUNK=64 and 12 at CHUNK=128 for the row kernel, while all merge instances report 16 waves/SIMD and no spills. The wider chunk therefore reduces compiler-estimated row-kernel residency even though VGPR use is unchanged; static shared memory rises from 15,872 to 19,456 bytes, so resource pressure is not VGPR count alone.
