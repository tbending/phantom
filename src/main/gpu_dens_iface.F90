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
!   COSMO_GPU=0 (or n/N/f/F) in the environment runs a GPU=yes binary on the
!   CPU path instead; any other value, or unset, keeps the compiled-in default.
!   One binary can then run both arms of a GPU-vs-CPU comparison, which a pair
!   of binaries would confound with build differences.  Ignored when GPU=no.
!
! Outputs written back to phantom particle arrays:
!   xyzh(4,i)     — converged smoothing length h_i
!   gradh(1,i)    — 1/omega_i  (phantom's Grad-h correction factor)
!                   where omega = 1 + (h/3*rho) * d(rho)/d(h)
!   divcurlv(1,i) — div v
!   xi_gpu(i)     — Cullen & Dehnen xi limiter, from the velocity gradient
!                   tensor, which never leaves the GPU (see cons2prim)
!   alphaind(3,i) — d(div v)/dt, source term of the Cullen & Dehnen switch
!
! The last three used to be produced by a CPU densityiterate(icall=3) sweep
! run after the GPU solve; they are now computed on the GPU in one sweep at
! the converged h, so the CPU no longer re-walks the kd-tree.
!
! :Dependencies: dim, HIIRegion, io, iso_c_binding, part, ptmass, ptmass_radiation,
!   viscosity
!
 use iso_c_binding, only:c_double
 implicit none

#ifdef GPU
 logical, public :: use_gpu_dens = .true.
#else
 logical, public :: use_gpu_dens = .false.
#endif

#ifdef GPU
!--C interface to cosmoSPHere/src/dens_c_api.cu
 interface
  subroutine densityiterate_gpu_c(h, rho, gradh_out, divv, xi_out, ddivvdt, &
                                  x, y, z, vx, vy, vz, ax, ay, az, n, pmass) bind(C)
   use iso_c_binding, only:c_double,c_int
   real(c_double), intent(inout) :: h(*)
   real(c_double), intent(out)   :: rho(*), gradh_out(*)
   real(c_double), intent(out)   :: divv(*), xi_out(*), ddivvdt(*)
   real(c_double), intent(in)    :: x(*), y(*), z(*)
   real(c_double), intent(in)    :: vx(*), vy(*), vz(*)
   real(c_double), intent(in)    :: ax(*), ay(*), az(*)
   integer(c_int), value         :: n
   real(c_double), value         :: pmass
  end subroutine densityiterate_gpu_c

!--C interface to cosmoSPHere/src/pin_c_api.cu
  subroutine cosmo_pin_host(ptr, nbytes) bind(C)
   use iso_c_binding, only:c_ptr,c_size_t
   type(c_ptr),       value :: ptr
   integer(c_size_t), value :: nbytes
  end subroutine cosmo_pin_host

  subroutine cosmo_unpin_host(ptr) bind(C)
   use iso_c_binding, only:c_ptr
   type(c_ptr), value :: ptr
  end subroutine cosmo_unpin_host
 end interface
#endif

 public :: densityiterate_gpu, init_gpu_switch, pin_buffer, unpin_buffer
 private

!
! xi limiter per particle from the last GPU density solve, for cons2prim.  On the
! GPU path the Cullen & Dehnen switch is the only reader of dv/dx (everything else
! that needs it is refused), so the GPU forms xi and the 9-component tensor stays
! on the device.  Set to 1 (xi of a zero tensor) until first computed.
!
 real, allocatable, public :: xi_gpu(:)

#ifdef GPU
!
! Staging buffers for the C call, module level and reused across calls:
! allocated on the first solve and grown only if npart rises.  Allocating and
! freeing ~15 arrays of npart every solve made every
! write into them a first touch of fresh pages.  They are registered with the
! driver for as long as they are allocated (see pin_buffer).
!
 integer :: nbuf = 0
 real(c_double), allocatable, target :: x8(:), y8(:), z8(:), h8(:)
 real(c_double), allocatable, target :: vx8(:), vy8(:), vz8(:)
 real(c_double), allocatable, target :: ax8(:), ay8(:), az8(:)
 real(c_double), allocatable, target :: rho8(:), drhofh8(:)
 real(c_double), allocatable, target :: divv8(:), xi8(:), ddivvdt8(:)
#endif

contains

!-------------------------------------------------------------
!+
!  Apply COSMO_GPU from the environment to use_gpu_dens, once.
!  Cheap to call on every derivs: only the first call reads the
!  environment.
!+
!-------------------------------------------------------------
subroutine init_gpu_switch()
#ifdef GPU
 use io, only:iprint
 character(len=16) :: val
 integer           :: ln,ierr
 logical, save     :: done = .false.

 if (done) return
 done = .true.
 call get_environment_variable('COSMO_GPU',val,ln,ierr)
 if (ierr == 0 .and. ln > 0) then
    select case(val(1:1))
    case('0','n','N','f','F')
       use_gpu_dens = .false.
    end select
 endif
 write(iprint,'(a,l1)') ' cosmoSPHere: use_gpu_dens = ',use_gpu_dens
#endif

end subroutine init_gpu_switch

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
subroutine densityiterate_gpu(npart, xyzh, vxyzu, fxyzu, fext, gradh, divcurlv, alphaind)
 use part, only:massoftype,igas
 use dim,  only:nalpha
#ifdef GPU
 use io,   only:fatal
 use dim,  only:curlv,mhd,use_dust,do_radiation,gravity,ind_timesteps,use_apr,gr
 use viscosity,        only:irealvisc
 use ptmass,           only:icreate_sinks
 use HIIRegion,        only:iH2R
 use ptmass_radiation, only:iget_tdust
 use iso_c_binding, only:c_double,c_int
#endif
 integer,      intent(in)    :: npart
 real,         intent(inout) :: xyzh(:,:)
 real,         intent(in)    :: vxyzu(:,:),fxyzu(:,:),fext(:,:)
 real(kind=4), intent(inout) :: gradh(:,:)
 real(kind=4), intent(inout) :: divcurlv(:,:)
 real(kind=4), intent(inout) :: alphaind(:,:)

#ifdef GPU
 real    :: hi, rhoi, drhoi, omega
 integer :: i
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
 !--the GPU force pass evaluates and overwrites every particle, not just the
 !  active ones, so individual timesteps would advance inactive particles
 if (ind_timesteps) call fatal('densityiterate_gpu','individual timesteps not supported on GPU (IND_TIMESTEPS=no)')
 !--the GPU path builds no kd-tree, so refuse everything that queries it
 !  during a step: sink creation, HII regions, APR's merge tree, and the
 !  sink-radiation ray tracer (iget_tdust 3 and 4)
 if (icreate_sinks > 0) call fatal('densityiterate_gpu','sink creation needs the kd-tree (icreate_sinks=0)')
 if (iH2R > 0)          call fatal('densityiterate_gpu','HII regions need the kd-tree (iH2R=0)')
 if (use_apr)           call fatal('densityiterate_gpu','APR needs the kd-tree')
 if (iget_tdust >= 3)   call fatal('densityiterate_gpu','ray-traced dust temperature needs the kd-tree (iget_tdust<3)')
 !--the GPU force pass has neither physical viscosity nor general relativity
 if (irealvisc > 0)     call fatal('densityiterate_gpu','physical viscosity not computed on GPU (irealvisc=0)')
 if (gr)                call fatal('densityiterate_gpu','general relativity not supported on GPU')

 !--COSMO_DENS_STATS=1 also reports the phantom-side cost of the GPU call
 if (.not. stats_checked) then
    call get_environment_variable('COSMO_DENS_STATS', statsenv)
    stats = (len_trim(statsenv) > 0)
    stats_checked = .true.
 endif
 call system_clock(ic0, crate)

 call ensure_buffers(npart)

 !$omp parallel do default(none) private(i) &
 !$omp shared(npart,xyzh,vxyzu,fxyzu,fext,x8,y8,z8,h8,vx8,vy8,vz8,ax8,ay8,az8)
 do i = 1, npart
    x8(i) = real(xyzh(1,i), kind=c_double)
    y8(i) = real(xyzh(2,i), kind=c_double)
    z8(i) = real(xyzh(3,i), kind=c_double)
    !--h passed SIGNED: h <= 0 marks dead and accreted particles
    !  (isdead_or_accreted), which the GPU sorts out of the tree.  abs() here
    !  used to make them look alive.
    h8(i) = real(xyzh(4,i), kind=c_double)
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

 call densityiterate_gpu_c(h8, rho8, drhofh8, divv8, xi8, ddivvdt8, &
                            x8, y8, z8, vx8, vy8, vz8, ax8, ay8, az8, &
                            int(npart, kind=c_int), &
                            real(massoftype(igas), kind=c_double))
 call system_clock(ic2)

 !--write results back to phantom arrays
 !  Skip inactive/dead particles (xyzh(4,i) < 0 in phantom convention)
 !$omp parallel do default(none) private(i,hi,rhoi,drhoi,omega) &
 !$omp shared(npart,xyzh,gradh,divcurlv,alphaind,xi_gpu) &
 !$omp shared(h8,rho8,drhofh8,divv8,xi8,ddivvdt8)
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
    xi_gpu(i) = real(xi8(i))
    if (nalpha >= 3) alphaind(3,i) = real(ddivvdt8(i), kind=4)
 enddo
 !$omp end parallel do

 call system_clock(ic3)

 call system_clock(ic4)

 if (stats) then
    write(0,'(a,f8.2,a,f8.2,a,f8.2,a,f8.2)') &
       'COSMO_FORT stage=', 1.e3*real(ic1-ic0)/real(crate), &
       ' capi=',            1.e3*real(ic2-ic1)/real(crate), &
       ' writeback=',       1.e3*real(ic3-ic2)/real(crate), &
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
!  Grow the staging buffers to hold at least n particles.  No-op once
!  they are big enough, so the allocation happens on the first solve only.
!+
!-------------------------------------------------------------
subroutine ensure_buffers(n)
 integer, intent(in) :: n

 if (nbuf >= n) return

 if (allocated(x8)) then
    call unpin_buffer(x8);   call unpin_buffer(y8);   call unpin_buffer(z8)
    call unpin_buffer(h8);   call unpin_buffer(vx8);  call unpin_buffer(vy8)
    call unpin_buffer(vz8);  call unpin_buffer(ax8);  call unpin_buffer(ay8)
    call unpin_buffer(az8);  call unpin_buffer(rho8); call unpin_buffer(drhofh8)
    call unpin_buffer(divv8); call unpin_buffer(xi8); call unpin_buffer(ddivvdt8)
    deallocate(x8, y8, z8, h8, vx8, vy8, vz8, ax8, ay8, az8, &
               rho8, drhofh8, divv8, xi8, ddivvdt8, xi_gpu)
 endif

 allocate(x8(n), y8(n), z8(n), h8(n), vx8(n), vy8(n), vz8(n), &
          ax8(n), ay8(n), az8(n), rho8(n), drhofh8(n), &
          divv8(n), xi8(n), ddivvdt8(n), xi_gpu(n))
 xi_gpu = 1.

 call pin_buffer(x8);   call pin_buffer(y8);   call pin_buffer(z8)
 call pin_buffer(h8);   call pin_buffer(vx8);  call pin_buffer(vy8)
 call pin_buffer(vz8);  call pin_buffer(ax8);  call pin_buffer(ay8)
 call pin_buffer(az8);  call pin_buffer(rho8); call pin_buffer(drhofh8)
 call pin_buffer(divv8); call pin_buffer(xi8); call pin_buffer(ddivvdt8)

 nbuf = n

end subroutine ensure_buffers
#endif

!-------------------------------------------------------------
!+
!  Register a staging buffer with the GPU driver, so the device
!  copies into and out of it are not slowed by on-demand page
!  population (catastrophically so on GH200 at 11.3M particles).
!  Must be undone with unpin_buffer before the buffer is freed.
!  Failure is silent: the copies still work, unpinned.  No-ops in a
!  GPU=no build.
!+
!-------------------------------------------------------------
subroutine pin_buffer(a)
 use iso_c_binding, only:c_loc,c_size_t,c_sizeof
 real(c_double), intent(in), target :: a(:)

#ifdef GPU
 if (size(a) > 0) call cosmo_pin_host(c_loc(a(1)), int(size(a),kind=c_size_t)*c_sizeof(a(1)))
#endif

end subroutine pin_buffer

subroutine unpin_buffer(a)
 use iso_c_binding, only:c_loc
 real(c_double), intent(in), target :: a(:)

#ifdef GPU
 if (size(a) > 0) call cosmo_unpin_host(c_loc(a(1)))
#endif

end subroutine unpin_buffer

end module gpu_dens_iface
