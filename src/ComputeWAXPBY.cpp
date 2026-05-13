
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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"

#include <cassert>

#ifdef USE_RISCV_VECTOR
#include <riscv_vector.h>

// w[i] = alpha * x[i] + beta * y[i]
//
// All four sub-paths use LMUL=8 grouping for maximum throughput at large n.
// vfmacc_vf computes  acc + scalar * v  in one instruction (FMA), so the
// alpha==1 and beta==1 specializations save one multiply per element.
static inline int ComputeWAXPBY_rvv(const local_int_t n,
                                    const double alpha,
                                    const double * __restrict__ xv,
                                    const double beta,
                                    const double * __restrict__ yv,
                                    double * __restrict__ wv) {
  size_t i = 0;
  size_t avl = static_cast<size_t>(n);

  if (alpha == 1.0 && beta == 1.0) {
    while (avl > 0) {
      size_t vl = __riscv_vsetvl_e64m8(avl);
      vfloat64m8_t vx = __riscv_vle64_v_f64m8(xv + i, vl);
      vfloat64m8_t vy = __riscv_vle64_v_f64m8(yv + i, vl);
      vfloat64m8_t vw = __riscv_vfadd_vv_f64m8(vx, vy, vl);
      __riscv_vse64_v_f64m8(wv + i, vw, vl);
      i += vl; avl -= vl;
    }
  } else if (alpha == 1.0) {
    // w = x + beta * y
    while (avl > 0) {
      size_t vl = __riscv_vsetvl_e64m8(avl);
      vfloat64m8_t vx = __riscv_vle64_v_f64m8(xv + i, vl);
      vfloat64m8_t vy = __riscv_vle64_v_f64m8(yv + i, vl);
      vfloat64m8_t vw = __riscv_vfmacc_vf_f64m8(vx, beta, vy, vl);
      __riscv_vse64_v_f64m8(wv + i, vw, vl);
      i += vl; avl -= vl;
    }
  } else if (beta == 1.0) {
    // w = alpha * x + y
    while (avl > 0) {
      size_t vl = __riscv_vsetvl_e64m8(avl);
      vfloat64m8_t vx = __riscv_vle64_v_f64m8(xv + i, vl);
      vfloat64m8_t vy = __riscv_vle64_v_f64m8(yv + i, vl);
      vfloat64m8_t vw = __riscv_vfmacc_vf_f64m8(vy, alpha, vx, vl);
      __riscv_vse64_v_f64m8(wv + i, vw, vl);
      i += vl; avl -= vl;
    }
  } else {
    // General: w = alpha * x + beta * y
    while (avl > 0) {
      size_t vl = __riscv_vsetvl_e64m8(avl);
      vfloat64m8_t vx = __riscv_vle64_v_f64m8(xv + i, vl);
      vfloat64m8_t vy = __riscv_vle64_v_f64m8(yv + i, vl);
      vfloat64m8_t vw = __riscv_vfmul_vf_f64m8(vx, alpha, vl);
      vw = __riscv_vfmacc_vf_f64m8(vw, beta, vy, vl);
      __riscv_vse64_v_f64m8(wv + i, vw, vl);
      i += vl; avl -= vl;
    }
  }
  return 0;
}

#endif // USE_RISCV_VECTOR

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {

  assert(x.localLength >= n);
  assert(y.localLength >= n);

#ifdef USE_RISCV_VECTOR
  isOptimized = true;
  return ComputeWAXPBY_rvv(n, alpha, x.values, beta, y.values, w.values);
#else
  isOptimized = false;
  return ComputeWAXPBY_ref(n, alpha, x, beta, y, w);
#endif
}
