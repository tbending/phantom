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
! inside my kernel?" while force also needs "am I inside theirs?" (pun intended).  It computes
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
!update 0829
#ifdef GPU
 interface
  subroutine force_gpu_c(n,pmass,pro2,fx,fy,fz,f4) bind(C)
   use iso_c_binding, only:c_double,c_int
   integer(c_int), value       :: n
   real(c_double), value       :: pmass
   real(c_double), intent(in)  :: pro2(*)
   real(c_double), intent(out) :: fx(*),fy(*),fz(*),f4(*)
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
subroutine force_gpu(npart,pro2,fxyzu)
 use part, only:massoftype,igas
#ifdef GPU
 use iso_c_binding, only:c_double,c_int
#endif
 integer, intent(in)    :: npart
 real,    intent(in)    :: pro2(:)
 real,    intent(inout) :: fxyzu(:,:)

#ifdef GPU
 real(c_double), allocatable :: pro2_8(:)
 real(c_double), allocatable :: fx8(:),fy8(:),fz8(:), f48(:)
 integer :: i

 if (npart <= 0) return

 allocate(pro2_8(npart))

 allocate(fx8(npart),fy8(npart),fz8(npart),f48(npart))

 pro2_8 = real(pro2(1:npart),kind=c_double)

 call force_gpu_c(int(npart,kind=c_int),                               &
                  real(massoftype(igas),kind=c_double),                &
                  pro2_8,fx8,fy8,fz8,f48)

 do i = 1,npart
    fxyzu(1,i) = real(fx8(i),kind=kind(fxyzu))
    fxyzu(2,i) = real(fy8(i),kind=kind(fxyzu))
    fxyzu(3,i) = real(fz8(i),kind=kind(fxyzu))
    fxyzu(4,i) = real(f48(i),kind=kind(fxyzu))
 enddo

 deallocate(pro2_8,fx8,fy8,fz8,f48)
#else
 print *, 'ERROR: force_gpu called but phantom not compiled with GPU=yes'
 stop
#endif

end subroutine force_gpu

end module gpu_force_iface
