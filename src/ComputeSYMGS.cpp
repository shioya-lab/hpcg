
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
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"

#include <cassert>

#ifdef USE_RISCV_VECTOR
#include <riscv_vector.h>
#include <cstdint>

// Row-local sparse dot:  D = sum_{j=0..nnz-1} cur_vals[j] * xv[cur_inds[j]]
//
// Identical structure to ComputeSPMV's row reduction (indexed gather of x via
// vluxei32, vfmacc.vv accumulation, vfredusum.vs at end). Defined here to
// keep ComputeSYMGS.cpp self-contained; refactoring into a shared header is
// a possible follow-up.
//
// IMPORTANT: SYMGS sequentiality is preserved by keeping the OUTER row loop
// sequential. Only the row-inner reduction is vectorized, so xv[i] updates
// still feed forward to subsequent rows exactly like the reference impl.
static inline double symgs_row_dot(const double * __restrict__ cur_vals,
                                   const local_int_t * __restrict__ cur_inds,
                                   int cur_nnz,
                                   const double * __restrict__ xv) {
  const size_t vlmax = __riscv_vsetvlmax_e64m2();
  vfloat64m2_t vacc = __riscv_vfmv_v_f_f64m2(0.0, vlmax);

  size_t j = 0;
  size_t avl = static_cast<size_t>(cur_nnz);
  while (avl > 0) {
    size_t vl = __riscv_vsetvl_e64m2(avl);

    vint32m1_t vidx_s = __riscv_vle32_v_i32m1(
        reinterpret_cast<const int32_t *>(cur_inds + j), vl);
    vint32m1_t vidx_bytes = __riscv_vsll_vx_i32m1(vidx_s, 3, vl);
    vuint32m1_t vidx_u = __riscv_vreinterpret_v_i32m1_u32m1(vidx_bytes);

    vfloat64m2_t vvals    = __riscv_vle64_v_f64m2(cur_vals + j, vl);
    vfloat64m2_t vxgather = __riscv_vluxei32_v_f64m2(xv, vidx_u, vl);
    vacc = __riscv_vfmacc_vv_f64m2(vacc, vvals, vxgather, vl);

    j += vl; avl -= vl;
  }

  vfloat64m1_t vzero = __riscv_vfmv_s_f_f64m1(0.0, 1);
  vfloat64m1_t vred  = __riscv_vfredusum_vs_f64m2_f64m1(vacc, vzero, vlmax);
  return __riscv_vfmv_f_s_f64m1_f64(vred);
}

#endif // USE_RISCV_VECTOR

/*!
  Routine to compute one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {

  assert(x.localLength == A.localNumberOfColumns);

#ifdef USE_RISCV_VECTOR
  const local_int_t nrow = A.localNumberOfRows;
  double ** matrixDiagonal = A.matrixDiagonal;
  const double * const rv = r.values;
  double * const xv = x.values;

  // Forward sweep: i = 0 .. nrow-1
  for (local_int_t i = 0; i < nrow; ++i) {
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = static_cast<int>(A.nonzerosInRow[i]);
    const double diag = matrixDiagonal[i][0];

    // D = sum_j cur_vals[j] * xv[cur_inds[j]]   (includes the diagonal term)
    const double D = symgs_row_dot(cur_vals, cur_inds, cur_nnz, xv);
    // sum = rv[i] - (D - xv[i]*diag), i.e. remove the diagonal contribution
    const double sum = rv[i] - D + xv[i] * diag;
    xv[i] = sum / diag;
  }

  // Backward sweep: i = nrow-1 .. 0
  for (local_int_t i = nrow - 1; i >= 0; --i) {
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = static_cast<int>(A.nonzerosInRow[i]);
    const double diag = matrixDiagonal[i][0];

    const double D = symgs_row_dot(cur_vals, cur_inds, cur_nnz, xv);
    const double sum = rv[i] - D + xv[i] * diag;
    xv[i] = sum / diag;
  }

  return 0;
#else
  return ComputeSYMGS_ref(A, r, x);
#endif
}
