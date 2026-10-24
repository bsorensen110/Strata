// metal/kernels/hadamard.metal - Gyro (APR): the block-Hadamard rotation of the routed experts' input activations,
// y = H_block (signs * x) per block of `block` values of each row (include/strata/kernels/cpu/hadamard.hpp;
// src/kernels/cuda/iq_kernels.cu's hadamard_rows_kernel).  The butterflies are hadamard.cpp's in the same order, so the
// floats are the CPU's.  No dependency on the other kernel files.

// hadamard_rows: f32 rows.  Grid (width / block, rows), block threads (a power of two <= 1024); y may be x (each
// threadgroup reads its block, then writes it).
struct HadArgs {
    uint width;
    uint has_signs;
    float scale;     // 1 / sqrt(block), as hadamard_rotate computes it
    uint pad;
};
kernel void hadamard_rows(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                          device const float* signs [[buffer(2)]], constant HadArgs& a [[buffer(3)]],
                          uint3 tg [[threadgroup_position_in_grid]], ushort i [[thread_index_in_threadgroup]],
                          uint3 nt [[threads_per_threadgroup]]) {
    threadgroup float hb[1024];
    const ushort n = ushort(nt.x);
    const uint col = tg.x * n + i, at = tg.y * a.width + col;
    float v = x[at];
    if (a.has_signs) v *= signs[col];
    hb[i] = v;
    for (ushort len = 1; len < n; len <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((i & len) == 0) {
            const float p = hb[i], q = hb[i + len];
            hb[i] = p + q;
            hb[i + len] = p - q;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    y[at] = hb[i] * a.scale;
}

// hadamard_rows_h: the fp16 rows of the prompt path (HS), in place, computed in float.  Grid (rows), block threads: the
// threadgroup walks the row's blocks.  (The grid may come from the plan's GPU-side row count: indirect dispatch.)
kernel void hadamard_rows_h(device half* x [[buffer(0)]], device const float* signs [[buffer(1)]],
                            constant HadArgs& a [[buffer(2)]], uint3 tg [[threadgroup_position_in_grid]],
                            ushort i [[thread_index_in_threadgroup]], uint3 nt [[threads_per_threadgroup]]) {
    threadgroup float hb[1024];
    const ushort n = ushort(nt.x);
    for (uint b0 = 0; b0 < a.width; b0 += n) {
        const uint col = b0 + i, at = tg.x * a.width + col;
        float v = float(x[at]);
        if (a.has_signs) v *= signs[col];
        hb[i] = v;
        for (ushort len = 1; len < n; len <<= 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if ((i & len) == 0) {
                const float p = hb[i], q = hb[i + len];
                hb[i] = p + q;
                hb[i + len] = p - q;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        x[at] = half(hb[i] * a.scale);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
