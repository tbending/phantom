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
  subroutine force_gpu_c(n,pmass,pro2,spsound,alphaAV,u,beta,alphau, &
                       fx,fy,fz,f4,vsigmax) bind(C)
  use iso_c_binding, only:c_double,c_int

  integer(c_int), value       :: n
  real(c_double), value       :: pmass
  real(c_double), intent(in)  :: pro2(*)
  real(c_double), intent(in)  :: spsound(*)
  real(c_double), intent(in)  :: alphaAV(*)
  real(c_double), intent(in)  :: u(*)
  real(c_double), value       :: beta
  real(c_double), value       :: alphau
  real(c_double), intent(out) :: fx(*),fy(*),fz(*),f4(*)
  real(c_double), intent(out) :: vsigmax(*)
  end subroutine force_gpu_c

 end interface
#endif

 public :: force_gpu, finish_gpu_force_timesteps
 private

contains

!-----------------------------------------------------------------------
!+
!  Build the symmetric j-leaf list on the GPU, and time it.
!+
!-----------------------------------------------------------------------
subroutine force_gpu(npart,pro2,spsound,alphaAV,u,beta,alphau, &
                     fxyzu,vsigmax)
 use part, only:massoftype,igas
#ifdef GPU
 use iso_c_binding, only:c_double,c_int
#endif
 integer, intent(in)    :: npart
 real,    intent(in)    :: pro2(:)
 real,    intent(in)    :: spsound(:)
 real,    intent(in)    :: alphaAV(:)
 real,    intent(in)    :: u(:)
 real,    intent(in)    :: beta,alphau
 real,    intent(inout) :: fxyzu(:,:)
 real,    intent(out)   :: vsigmax(:)

#ifdef GPU
 real(c_double), allocatable :: pro2_8(:)
 real(c_double), allocatable :: spsound_8(:)
 real(c_double), allocatable :: alphaAV_8(:)
 real(c_double), allocatable :: u_8(:)
 real(c_double), allocatable :: fx8(:),fy8(:),fz8(:),f48(:)
 real(c_double), allocatable :: vsigmax8(:)
 integer :: i
 if (npart <= 0) return

allocate(pro2_8(npart))
allocate(spsound_8(npart))
allocate(alphaAV_8(npart))
allocate(u_8(npart))

allocate(fx8(npart))
allocate(fy8(npart))
allocate(fz8(npart))
allocate(f48(npart))
allocate(vsigmax8(npart))

pro2_8    = real(pro2(1:npart),kind=c_double)
spsound_8 = real(spsound(1:npart),kind=c_double)
alphaAV_8 = real(alphaAV(1:npart),kind=c_double)
u_8       = real(u(1:npart),kind=c_double)

call force_gpu_c(int(npart,kind=c_int),                       &
                 real(massoftype(igas),kind=c_double),        &
                 pro2_8,spsound_8,alphaAV_8,u_8,              &
                 real(beta,kind=c_double),                    &
                 real(alphau,kind=c_double),                  &
                 fx8,fy8,fz8,f48,vsigmax8)

do i = 1,npart
   fxyzu(1,i) = real(fx8(i),kind=kind(fxyzu))
   fxyzu(2,i) = real(fy8(i),kind=kind(fxyzu))
   fxyzu(3,i) = real(fz8(i),kind=kind(fxyzu))
   fxyzu(4,i) = real(f48(i),kind=kind(fxyzu))
   vsigmax(i) = real(vsigmax8(i),kind=kind(vsigmax))
enddo

deallocate(pro2_8,spsound_8,alphaAV_8,u_8)
deallocate(fx8,fy8,fz8,f48,vsigmax8)

#else
 print *, 'ERROR: force_gpu called but phantom not compiled with GPU=yes'
 stop
#endif

end subroutine force_gpu


subroutine finish_gpu_force_timesteps(npart,xyzh,fxyzu, &
                                      spsound,vsigmax)
 use options,  only:alpha
 use timestep, only:C_cour,C_force,bignumber, &
                    dtcourant,dtforce,dtrad

 integer, intent(in) :: npart
 real,    intent(in) :: xyzh(:,:)
 real,    intent(in) :: fxyzu(:,:)
 real,    intent(in) :: spsound(:)
 real,    intent(in) :: vsigmax(:)

 integer :: i
 real    :: hi,vsigdtc,f2i,dtc,dtf

 dtcourant = bignumber
 dtforce   = bignumber
 dtrad     = bignumber

 do i = 1,npart
    hi = xyzh(4,i)

    vsigdtc = max(vsigmax(i),spsound(i))

    dtc = C_cour*hi / &
          (vsigdtc*max(alpha,1.0))

    f2i = fxyzu(1,i)*fxyzu(1,i) + &
          fxyzu(2,i)*fxyzu(2,i) + &
          fxyzu(3,i)*fxyzu(3,i)

    dtf = bignumber

    if (f2i > 0.0) then
       dtf = C_force*sqrt(hi/sqrt(f2i))
    endif

    dtcourant = min(dtcourant,dtc)
    dtforce   = min(dtforce,dtf)
 enddo

end subroutine finish_gpu_force_timesteps



end module gpu_force_iface


