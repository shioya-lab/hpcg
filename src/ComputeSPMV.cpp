
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"

#include <cassert>

#ifdef USE_RISCV_VECTOR
#include <riscv_vector.h>
#include <cstdint>

// One-row sparse dot:  y[i] = sum_{j=0..nnz-1} cur_vals[j] * xv[cur_inds[j]]
//
// Vectorization strategy:
//   - LMUL=2 (32 doubles at vlen=1024) covers a typical HPCG row (<=27 nnz)
//     in one stripmine iteration; the outer while-loop handles arbitrary nnz.
//   - 32-bit column indices loaded with vle32, scaled to byte offsets (<<3),
//     reinterpreted as unsigned, then fed to vluxei32 for gathered x loads.
//   - vfmacc.vv accumulates into a per-lane partial-sum vector; one
//     vfredusum.vs at the end collapses to the scalar row result.
static inline void spmv_one_row(const double * __restrict__ cur_vals,
                                const local_int_t * __restrict__ cur_inds,
                                int cur_nnz,
                                const double * __restrict__ xv,
                                double * __restrict__ y_row) {
  const size_t vlmax = __riscv_vsetvlmax_e64m2();
  vfloat64m2_t vacc = __riscv_vfmv_v_f_f64m2(0.0, vlmax);

  size_t j = 0;
  size_t avl = static_cast<size_t>(cur_nnz);
  while (avl > 0) {
    size_t vl = __riscv_vsetvl_e64m2(avl);

    // local_int_t is 32-bit signed; values are non-negative array indices.
    vint32m1_t vidx_s = __riscv_vle32_v_i32m1(
        reinterpret_cast<const int32_t *>(cur_inds + j), vl);
    // byte_offset = index * sizeof(double)
    vint32m1_t vidx_bytes = __riscv_vsll_vx_i32m1(vidx_s, 3, vl);
    // vluxei32 wants unsigned 32-bit byte offsets.
    vuint32m1_t vidx_u = __riscv_vreinterpret_v_i32m1_u32m1(vidx_bytes);

    vfloat64m2_t vvals    = __riscv_vle64_v_f64m2(cur_vals + j, vl);
    vfloat64m2_t vxgather = __riscv_vluxei32_v_f64m2(xv, vidx_u, vl);
    vacc = __riscv_vfmacc_vv_f64m2(vacc, vvals, vxgather, vl);

    j += vl; avl -= vl;
  }

  vfloat64m1_t vzero = __riscv_vfmv_s_f_f64m1(0.0, 1);
  vfloat64m1_t vred  = __riscv_vfredusum_vs_f64m2_f64m1(vacc, vzero, vlmax);
  *y_row = __riscv_vfmv_f_s_f64m1_f64(vred);
}

#endif // USE_RISCV_VECTOR

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {

  assert(x.localLength >= A.localNumberOfColumns);
  assert(y.localLength >= A.localNumberOfRows);

#ifdef USE_RISCV_VECTOR
  A.isSpmvOptimized = true;
  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
  for (local_int_t i = 0; i < nrow; ++i) {
    spmv_one_row(A.matrixValues[i], A.mtxIndL[i],
                 static_cast<int>(A.nonzerosInRow[i]),
                 xv, yv + i);
  }
  return 0;
#else
  A.isSpmvOptimized = false;
  return ComputeSPMV_ref(A, x, y);
#endif
}
