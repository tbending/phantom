!--------------------------------------------------------------------------!
! The Phantom Smoothed Particle Hydrodynamics code, by Daniel Price et al.  !
! Copyright (c) 2007-2025 The Authors (see AUTHORS)                         !
! See LICENCE file for usage and distribution conditions                    !
! http://phantomsph.github.io/                                              !
!--------------------------------------------------------------------------!
module gpu_force_iface
!
! Fortran interface to the cosmoSPHere GPU force pass.
!
! Separate from gpu_dens_iface because phantom runs density and force as two
! passes (deriv.f90 calls densityiterate_gpu, then force), and cosmoSPHere
! mirrors that with two C entry points.
!
! STATUS: the GPU force kernel does not exist yet.  force_gpu currently builds
! only the SYMMETRIC (gather+scatter) j-leaf list that the force sum will need
! — the list the density walk cannot produce, because density asks only "is j
! inside my kernel?" while force also needs "am I inside theirs?".  It computes
! no forces and writes nothing back, so phantom's results are unaffected.  It
! is called so the cost of the walk can be measured before the kernel lands.
!
! It consumes the octree and per-leaf hmax that densityiterate_gpu leaves in the
! GPU state, so it is only valid immediately after a GPU density solve; the C
! side aborts if that is not the case.
!
! :References: None
!
! :Owner: Not Committed Yet
!
! :Runtime parameters: None
!
! :Dependencies: iso_c_binding
!
 implicit none

#ifdef GPU
!--C interface to cosmoSPHere/src/force_c_api.cu
 interface
  subroutine force_gpu_c(n, pmass) bind(C)
   use iso_c_binding, only:c_double,c_int
   integer(c_int), value :: n
   real(c_double), value :: pmass
  end subroutine force_gpu_c
 end interface
#endif

 public :: force_gpu
 private

contains

!-----------------------------------------------------------------------
!+
!  Build the symmetric j-leaf list on the GPU, and time it.
!+
!-----------------------------------------------------------------------
subroutine force_gpu(npart)
 use part, only:massoftype,igas
#ifdef GPU
 use iso_c_binding, only:c_double
#endif
 integer, intent(in) :: npart

#ifdef GPU
 call force_gpu_c(npart, real(massoftype(igas), kind=c_double))
#else
 print *, 'ERROR: force_gpu called but phantom not compiled with GPU=yes'
 stop
#endif

end subroutine force_gpu

end module gpu_force_iface
