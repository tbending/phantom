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
!   xyzh(4,i)     — converged smoothing length h_i
!   gradh(1,i)    — 1/omega_i  (phantom's Grad-h correction factor)
!                   where omega = 1 + (h/3*rho) * d(rho)/d(h)
!   divcurlv(1,i) — div v
!   dvdx(1:9,i)   — velocity gradient tensor
!   alphaind(3,i) — d(div v)/dt, source term of the Cullen & Dehnen switch
!
! The last three used to be produced by a CPU densityiterate(icall=3) sweep
! run after the GPU solve; they are now computed on the GPU in one sweep at
! the converged h, so the CPU no longer re-walks the kd-tree.
!
! :Dependencies: dim, iso_c_binding, kdtree, neighkdtree, part
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
  subroutine densityiterate_gpu_c(h, rho, gradh_out, divv, dvdx_out, ddivvdt, &
                                  x, y, z, vx, vy, vz, ax, ay, az, n, pmass) bind(C)
   use iso_c_binding, only:c_double,c_int
   real(c_double), intent(inout) :: h(*)
   real(c_double), intent(out)   :: rho(*), gradh_out(*)
   real(c_double), intent(out)   :: divv(*), dvdx_out(*), ddivvdt(*)
   real(c_double), intent(in)    :: x(*), y(*), z(*)
   real(c_double), intent(in)    :: vx(*), vy(*), vz(*)
   real(c_double), intent(in)    :: ax(*), ay(*), az(*)
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
!  then converts outputs to phantom's array conventions.
!
!  Only gas particles (igas type) are passed to the GPU.
!  pmass = massoftype(igas) from the part module.
!
!  NOTE: hfact is hardcoded to 1.2 (cubic spline) inside cosmoSPHere.
!  If a different kernel is selected in phantom, ensure the same hfact
!  is used in cosmoSPHere/include/kernel.hpp before compiling libcosmoSPHere.a.
!+
!-------------------------------------------------------------
subroutine densityiterate_gpu(npart, xyzh, vxyzu, fxyzu, fext, gradh, divcurlv, dvdx, alphaind)
 use part, only:massoftype,igas
 use dim,  only:nalpha,maxdvdx,maxp
#ifdef GPU
 use io,   only:fatal
 use dim,  only:curlv,mhd,use_dust,do_radiation,gravity
 use iso_c_binding, only:c_double,c_int
#endif
 integer,      intent(in)    :: npart
 real,         intent(inout) :: xyzh(:,:)
 real,         intent(in)    :: vxyzu(:,:),fxyzu(:,:),fext(:,:)
 real(kind=4), intent(inout) :: gradh(:,:)
 real(kind=4), intent(inout) :: divcurlv(:,:)
 real(kind=4), intent(inout) :: dvdx(:,:)
 real(kind=4), intent(inout) :: alphaind(:,:)

#ifdef GPU
 real(c_double), allocatable :: x8(:), y8(:), z8(:), h8(:)
 real(c_double), allocatable :: vx8(:), vy8(:), vz8(:)
 real(c_double), allocatable :: ax8(:), ay8(:), az8(:)
 real(c_double), allocatable :: rho8(:), drhofh8(:)
 real(c_double), allocatable :: divv8(:), dvdx8(:), ddivvdt8(:)
 real    :: hi, rhoi, drhoi, omega
 integer :: i, c
 integer(kind=8) :: ic0,ic1,ic2,ic3,ic4,crate
 character(len=8) :: statsenv
 logical, save    :: stats = .false.
 logical, save    :: stats_checked = .false.

 if (npart <= 0) return

 !--the GPU sweep produces div v, dv/dx and d(div v)/dt for a single gas type.
 !  Anything that needs a quantity it does not compute would silently get stale
 !  values, so refuse rather than run: curl v (divcurlv(2:4)), div/curl B,
 !  the 2-fluid dust density, radiation fluxes, and gradsoft in gradh(2,:).
 if (curlv)        call fatal('densityiterate_gpu','curl v not computed on GPU (set curlv=F)')
 if (mhd)          call fatal('densityiterate_gpu','divcurlB not computed on GPU')
 if (use_dust)     call fatal('densityiterate_gpu','dust density not computed on GPU')
 if (do_radiation) call fatal('densityiterate_gpu','radiation flux not computed on GPU')
 if (gravity)      call fatal('densityiterate_gpu','gradsoft not computed on GPU')

 !--COSMO_DENS_STATS=1 also reports the phantom-side cost of the GPU call
 if (.not. stats_checked) then
    call get_environment_variable('COSMO_DENS_STATS', statsenv)
    stats = (len_trim(statsenv) > 0)
    stats_checked = .true.
 endif
 call system_clock(ic0, crate)

 allocate(x8(npart), y8(npart), z8(npart), h8(npart))
 allocate(vx8(npart), vy8(npart), vz8(npart))
 allocate(ax8(npart), ay8(npart), az8(npart))
 allocate(rho8(npart), drhofh8(npart))
 allocate(divv8(npart), dvdx8(9*npart), ddivvdt8(npart))

 !$omp parallel do default(none) private(i) &
 !$omp shared(npart,xyzh,vxyzu,fxyzu,fext,x8,y8,z8,h8,vx8,vy8,vz8,ax8,ay8,az8)
 do i = 1, npart
    x8(i) = real(xyzh(1,i), kind=c_double)
    y8(i) = real(xyzh(2,i), kind=c_double)
    z8(i) = real(xyzh(3,i), kind=c_double)
    h8(i) = real(abs(xyzh(4,i)), kind=c_double)
    vx8(i) = real(vxyzu(1,i), kind=c_double)
    vy8(i) = real(vxyzu(2,i), kind=c_double)
    vz8(i) = real(vxyzu(3,i), kind=c_double)
    !--the Cullen & Dehnen switch differentiates the TOTAL acceleration
    ax8(i) = real(fxyzu(1,i) + fext(1,i), kind=c_double)
    ay8(i) = real(fxyzu(2,i) + fext(2,i), kind=c_double)
    az8(i) = real(fxyzu(3,i) + fext(3,i), kind=c_double)
 enddo
 !$omp end parallel do
 call system_clock(ic1)

 call densityiterate_gpu_c(h8, rho8, drhofh8, divv8, dvdx8, ddivvdt8, &
                            x8, y8, z8, vx8, vy8, vz8, ax8, ay8, az8, &
                            int(npart, kind=c_int), &
                            real(massoftype(igas), kind=c_double))
 call system_clock(ic2)

 !--write results back to phantom arrays
 !  Skip inactive/dead particles (xyzh(4,i) < 0 in phantom convention)
 !$omp parallel do default(none) private(i,c,hi,rhoi,drhoi,omega) &
 !$omp shared(npart,xyzh,gradh,divcurlv,dvdx,alphaind) &
 !$omp shared(h8,rho8,drhofh8,divv8,dvdx8,ddivvdt8,maxdvdx,maxp)
 do i = 1, npart
    if (xyzh(4,i) < 0.) cycle   ! preserve negative h for inactive particles
    hi    = real(h8(i))
    rhoi  = real(rho8(i))
    drhoi = real(drhofh8(i))    ! d(rho)/d(h), normalised
    xyzh(4,i) = hi
    !--convert to phantom's gradh(1,i) = 1/omega
    !  omega = 1 + (h/3*rho) * d(rho)/d(h)
    !  NB: the GPU forms the same omega internally to normalise the gradients
    !  below (sphGradientsKernel) -- keep the two in step.
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
    divcurlv(1,i) = real(divv8(i), kind=4)
    if (maxdvdx == maxp) then
       do c = 1, 9
          dvdx(c,i) = real(dvdx8(9*(i-1) + c), kind=4)
       enddo
    endif
    if (nalpha >= 3) alphaind(3,i) = real(ddivvdt8(i), kind=4)
 enddo
 !$omp end parallel do

 call system_clock(ic3)

 deallocate(x8, y8, z8, h8, vx8, vy8, vz8, ax8, ay8, az8)
 deallocate(rho8, drhofh8, divv8, dvdx8, ddivvdt8)

 !--the CPU kd-tree still serves the force loop, and it caches h
 call sync_tree_h(xyzh)
 call system_clock(ic4)

 if (stats) then
    write(0,'(a,f8.2,a,f8.2,a,f8.2,a,f8.2,a,f8.2)') &
       'COSMO_FORT stage=', 1.e3*real(ic1-ic0)/real(crate), &
       ' capi=',            1.e3*real(ic2-ic1)/real(crate), &
       ' writeback=',       1.e3*real(ic3-ic2)/real(crate), &
       ' treesync=',        1.e3*real(ic4-ic3)/real(crate), &
       ' total=',           1.e3*real(ic4-ic0)/real(crate)
 endif

#else
 !--stub: should never be reached (use_gpu_dens is .false. without GPU)
 print *, 'ERROR: densityiterate_gpu called but phantom not compiled with GPU=yes'
 stop
#endif

end subroutine densityiterate_gpu

#ifdef GPU
!-------------------------------------------------------------
!+
!  Push GPU-updated smoothing lengths back into the kd-tree.
!
!  The force loop reads neighbour smoothing lengths from treecache
!  (kdtree.F90 fills xyzcache(4,:) = 1/treecache(4,:), which force.F90
!  then uses as hj1), and sizes its neighbour search from each node's
!  hmax.  Both were set when the tree was built, i.e. from the PREDICTED
!  h, so without this the force loop would silently mix stale cached h
!  with the converged h.  On the CPU path store_results/set_hmaxcell do
!  the same refresh as the density iteration proceeds.
!+
!-------------------------------------------------------------
subroutine sync_tree_h(xyzh)
 use dim,         only:maxpsph
 use part,        only:treecache
 use kdtree,      only:inodeparts,inoderange
 use neighkdtree, only:ncells,leaf_is_active,set_hmaxcell,get_hmaxcell
 real, intent(in) :: xyzh(:,:)
 integer :: icell,ip,i
 real    :: hmaxcell,hmaxold

 !$omp parallel do default(none) schedule(runtime) private(icell,ip,i,hmaxcell,hmaxold) &
 !$omp shared(xyzh,treecache,inodeparts,inoderange,ncells,leaf_is_active,maxpsph)
 do icell = 1, int(ncells)
    if (leaf_is_active(icell) == 0) cycle   ! internal node or empty cell
    hmaxcell = 0.
    do ip = inoderange(1,icell), inoderange(2,icell)
       i = inodeparts(ip)
       if (i < 0 .or. i > maxpsph) cycle
       treecache(4,ip) = xyzh(4,i)
       hmaxcell = max(hmaxcell, xyzh(4,i))
    enddo
    !--only widen the search radius, never narrow it: hmax has to be an upper
    !  bound on h in the cell, and set_hmaxcell walks to the root through a
    !  critical section, so doing it for every cell every step serialises.
    !  A cell whose h shrank keeps a conservative (too large) hmax, exactly as
    !  on the CPU path, where set_hmaxcell only fires when h outgrows the cell.
    if (hmaxcell > 0.) then
       call get_hmaxcell(icell, hmaxold)
       if (hmaxcell > hmaxold) call set_hmaxcell(icell, hmaxcell)
    endif
 enddo
 !$omp end parallel do

end subroutine sync_tree_h
#endif

end module gpu_dens_iface
