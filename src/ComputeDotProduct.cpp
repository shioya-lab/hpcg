
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"

#include <cassert>

#ifdef USE_RISCV_VECTOR
#include <riscv_vector.h>

// Dot product:  result = sum_{i=0}^{n-1} xv[i] * yv[i]
//
// Strategy: keep a per-lane partial-sum vector accumulator (LMUL=8) across all
// stripmine iterations, then do ONE horizontal sum reduction at the very end.
// This avoids paying the reduction latency on every iteration. When the final
// iteration uses vl < VLMAX, lanes [vl..VLMAX-1] of the accumulator retain
// their already-correct partial sums from prior iterations, so the final
// vfredusum still sees every contribution exactly once.
//
// Special case yv == xv (sum of squares) uses a single load per iteration.
static inline double ComputeDotProduct_rvv(const local_int_t n,
                                           const double * __restrict__ xv,
                                           const double * __restrict__ yv) {
  const size_t vlmax = __riscv_vsetvlmax_e64m8();
  vfloat64m8_t vacc = __riscv_vfmv_v_f_f64m8(0.0, vlmax);

  size_t i = 0;
  size_t avl = static_cast<size_t>(n);

  if (xv == yv) {
    // result = sum xv[i]^2
    while (avl > 0) {
      size_t vl = __riscv_vsetvl_e64m8(avl);
      vfloat64m8_t vx = __riscv_vle64_v_f64m8(xv + i, vl);
      vacc = __riscv_vfmacc_vv_f64m8(vacc, vx, vx, vl);
      i += vl; avl -= vl;
    }
  } else {
    // result = sum xv[i] * yv[i]
    while (avl > 0) {
      size_t vl = __riscv_vsetvl_e64m8(avl);
      vfloat64m8_t vx = __riscv_vle64_v_f64m8(xv + i, vl);
      vfloat64m8_t vy = __riscv_vle64_v_f64m8(yv + i, vl);
      vacc = __riscv_vfmacc_vv_f64m8(vacc, vx, vy, vl);
      i += vl; avl -= vl;
    }
  }

  // Horizontal sum: vacc[0..vlmax-1]  ->  scalar.
  // vfredusum is unordered (faster than ordered, FP non-associativity makes
  // the precise sum implementation-defined). HPCG's spectral convergence
  // test passes with this approach in practice.
  vfloat64m1_t vzero = __riscv_vfmv_s_f_f64m1(0.0, 1);
  vfloat64m1_t vred  = __riscv_vfredusum_vs_f64m8_f64m1(vacc, vzero, vlmax);
  return __riscv_vfmv_f_s_f64m1_f64(vred);
}

#endif // USE_RISCV_VECTOR

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

  assert(x.localLength >= n);
  assert(y.localLength >= n);

#ifdef USE_RISCV_VECTOR
  isOptimized = true;
  // HPCG_NO_MPI is forced on for our single-process RVV builds, so no allreduce.
  result = ComputeDotProduct_rvv(n, x.values, y.values);
  time_allreduce += 0.0;
  return 0;
#else
  isOptimized = false;
  return ComputeDotProduct_ref(n, x, y, result, time_allreduce);
#endif
}
