
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
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"

#include <cassert>

// Multigrid V-cycle.
//
// Structure mirrors ComputeMG_ref exactly. The only changes are that
// the SYMGS smoother and the SPMV residual step call the optimized
// wrappers (ComputeSYMGS / ComputeSPMV) instead of the *_ref
// reference implementations, and the recursion goes through this same
// optimized ComputeMG so coarser levels also benefit. Restriction and
// Prolongation have no non-reference wrapper available, so they
// continue to call the *_ref versions.

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix & A, const Vector & r, Vector & x) {
  assert(x.localLength == A.localNumberOfColumns); // halo space sanity

#ifdef USE_RISCV_VECTOR
  A.isMgOptimized = true;

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData != 0) {
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i = 0; i < numberOfPresmootherSteps; ++i)
      ierr += ComputeSYMGS(A, r, x);
    if (ierr != 0) return ierr;

    ierr = ComputeSPMV(A, x, *A.mgData->Axf);     if (ierr != 0) return ierr;
    // Restriction / Prolongation have no optimized wrapper; use *_ref.
    ierr = ComputeRestriction_ref(A, r);          if (ierr != 0) return ierr;
    // Recurse with optimized ComputeMG so coarser levels are also vectorized.
    ierr = ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc);
                                                  if (ierr != 0) return ierr;
    ierr = ComputeProlongation_ref(A, x);         if (ierr != 0) return ierr;

    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i = 0; i < numberOfPostsmootherSteps; ++i)
      ierr += ComputeSYMGS(A, r, x);
    if (ierr != 0) return ierr;
  } else {
    ierr = ComputeSYMGS(A, r, x);
    if (ierr != 0) return ierr;
  }
  return 0;
#else
  A.isMgOptimized = false;
  return ComputeMG_ref(A, r, x);
#endif
}
