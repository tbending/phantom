!--------------------------------------------------------------------------!
! The Phantom Smoothed Particle Hydrodynamics code, by Daniel Price et al. !
! Copyright (c) 2007-2026 The Authors (see AUTHORS)                        !
! See LICENCE file for usage and distribution conditions                   !
! http://phantomsph.github.io/                                             !
!--------------------------------------------------------------------------!
module gpu_dens_iface
!
! Fortran interface to the cosmoSPHere GPU density solver.
!
! Provides densityiterate_gpu(), which replaces densityiterate(icall=1,...)
! in deriv.f90 when phantom is compiled with GPU=yes.
!
! Compile-time control:
!   GPU=yes  — includes the bind(C) interface to densityiterate_gpu_c and
!              enables use_gpu_dens = .true. by default.
!   GPU=no   — module builds cleanly but densityiterate_gpu is a stub
!              (stop-on-call guard; use_gpu_dens is .false. so it will
!              never be called from deriv.f90 in normal builds).
!
! Runtime control:
!   gpu_dens_iface::use_gpu_dens  — set .false. to fall back to CPU density
!                                   without recompiling (useful for testing).
!
! Outputs written back to phantom particle arrays:
!   xyzh(4,i)  — converged smoothing length h_i
!   gradh(1,i) — 1/omega_i  (phantom's Grad-h correction factor)
!              where omega = 1 + (h/3*rho) * d(rho)/d(h)
!
! :Dependencies: iso_c_binding, part
!
 implicit none

#ifdef GPU
 logical, public :: use_gpu_dens = .true.
#else
 logical, public :: use_gpu_dens = .false.
#endif

#ifdef GPU
!--C interface to cosmoSPHere/src/dens_c_api.cu
 interface
  subroutine densityiterate_gpu_c(h, rho, gradh_out, x, y, z, n, pmass) bind(C)
   use iso_c_binding, only:c_double,c_int
   real(c_double), intent(inout) :: h(*)
   real(c_double), intent(out)   :: rho(*), gradh_out(*)
   real(c_double), intent(in)    :: x(*), y(*), z(*)
   integer(c_int), value         :: n
   real(c_double), value         :: pmass
  end subroutine densityiterate_gpu_c
 end interface
#endif

 public :: densityiterate_gpu
 private

contains

!-------------------------------------------------------------
!+
!  GPU density iteration: replaces densityiterate(icall=1,...).
!
!  Calls the cosmoSPHere Newton-Raphson + Cornerstone GPU solver,
!  then converts outputs to phantom's array conventions:
!    xyzh(4,i)  <- h_i   (converged smoothing length)
!    gradh(1,i) <- 1/omega_i
!
!  Only gas particles (igas type) are passed to the GPU.
!  pmass = massoftype(igas) from the part module.
!
!  NOTE: hfact is hardcoded to 1.2 (cubic spline) inside cosmoSPHere.
!  If a different kernel is selected in phantom, ensure the same hfact
!  is used in cosmoSPHere/include/kernel.hpp before compiling libcosmoSPHere.a.
!+
!-------------------------------------------------------------
subroutine densityiterate_gpu(npart, xyzh, gradh)
 use part, only:massoftype,igas
#ifdef GPU
 use iso_c_binding, only:c_double,c_int
#endif
 integer,      intent(in)    :: npart
 real,         intent(inout) :: xyzh(:,:)
 real(kind=4), intent(inout) :: gradh(:,:)

#ifdef GPU
 real(c_double), allocatable :: x8(:), y8(:), z8(:), h8(:)
 real(c_double), allocatable :: rho8(:), drhofh8(:)
 real    :: hi, rhoi, drhoi, omega
 integer :: i

 allocate(x8(npart), y8(npart), z8(npart), h8(npart))
 allocate(rho8(npart), drhofh8(npart))

 do i = 1, npart
    x8(i) = real(xyzh(1,i), kind=c_double)
    y8(i) = real(xyzh(2,i), kind=c_double)
    z8(i) = real(xyzh(3,i), kind=c_double)
    h8(i) = real(abs(xyzh(4,i)), kind=c_double)
 enddo

 call densityiterate_gpu_c(h8, rho8, drhofh8, x8, y8, z8, &
                            int(npart, kind=c_int), &
                            real(massoftype(igas), kind=c_double))

 !--write results back to phantom arrays
 !  Skip inactive/dead particles (xyzh(4,i) < 0 in phantom convention)
 do i = 1, npart
    if (xyzh(4,i) < 0.) cycle   ! preserve negative h for inactive particles
    hi    = real(h8(i))
    rhoi  = real(rho8(i))
    drhoi = real(drhofh8(i))    ! d(rho)/d(h), normalised
    xyzh(4,i) = hi
    !--convert to phantom's gradh(1,i) = 1/omega
    !  omega = 1 + (h/3*rho) * d(rho)/d(h)
    if (rhoi > 0. .and. hi > 0.) then
       omega = 1.0 + (hi / (3.0 * rhoi)) * drhoi
       if (omega > 0.) then
          gradh(1,i) = real(1.0 / omega, kind=4)
       else
          gradh(1,i) = 1.0_4
       endif
    else
       gradh(1,i) = 1.0_4
    endif
 enddo

 deallocate(x8, y8, z8, h8, rho8, drhofh8)

#else
 !--stub: should never be reached (use_gpu_dens is .false. without GPU)
 print *, 'ERROR: densityiterate_gpu called but phantom not compiled with GPU=yes'
 stop
#endif

end subroutine densityiterate_gpu

end module gpu_dens_iface
