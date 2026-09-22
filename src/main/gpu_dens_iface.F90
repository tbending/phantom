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
! Periodic boundaries (PERIODIC=yes) are handled on the GPU: pairs and tree
! nodes are taken at their nearest image, as in dens.F90 and force.F90.  All
! that is needed from here is the switch, the box, and wrapping the particles
! into it, which the kd-tree build does on the CPU path.
!
! Host staging (the pinned arena the C call reads and writes) belongs to gpu_arrays,
! not to this module: the force interface stages the same particle data, and both
! used to keep their own copy of it.
!
! :Dependencies: boundary, dim, gpu_arrays, HIIRegion, io, iso_c_binding, kernel,
!   mpidomain, options, part, ptmass, ptmass_radiation, viscosity
!
 use iso_c_binding, only:c_double
 use gpu_arrays,    only:gpu_arrays_init,gpu_arrays_comp,gpu_arrays_nbuf, &
                         gpu_arrays_mark_packed,gpu_arrays_upload,gpu_arrays_download, &
                         gpu_arrays_positions_moved,gpu_arrays_order_rebuilt, &
                         ibun_pos,ibun_hsml,ibun_vel,ibun_accel, &
                         ibun_dens_out,ibun_grad_out
 implicit none

#ifdef GPU
 logical, public :: use_gpu_dens = .true.
#else
 logical, public :: use_gpu_dens = .false.
#endif

#ifdef GPU
!--C interface to cosmoSPHere/src/dens_c_api.cu
 interface
  subroutine densityiterate_gpu_c(n, pmass, &
                                  periodic, box, tolh, hfact) bind(C)
   use iso_c_binding, only:c_double,c_int
   integer(c_int), value         :: n
   real(c_double), value         :: pmass
   integer(c_int), value         :: periodic
   real(c_double), intent(in)    :: box(6)
   real(c_double), value         :: tolh
   real(c_double), value         :: hfact
  end subroutine densityiterate_gpu_c

!--C interface to cosmo_kernel_radius in cosmoSPHere/src/dens_c_api.cu
  real(c_double) function cosmo_kernel_radius() bind(C)
   use iso_c_binding, only:c_double
  end function cosmo_kernel_radius

 end interface
#endif

 public :: densityiterate_gpu, init_gpu_switch
 private

!
! xi limiter per particle from the last GPU density solve, for cons2prim.  On the
! GPU path the Cullen & Dehnen switch is the only reader of dv/dx (everything else
! that needs it is refused), so the GPU forms xi and the 9-component tensor stays
! on the device.  Set to 1 (xi of a zero tensor) until first computed.
!
 real, allocatable, public :: xi_gpu(:)

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
 use dim,  only:periodic
 use part, only:isdead_or_accreted
 use boundary,  only:cross_boundary,xmin,xmax,ymin,ymax,zmin,zmax
 use mpidomain, only:isperiodic
 use options,   only:tolh
 use dim,  only:curlv,mhd,use_dust,do_radiation,gravity,ind_timesteps,use_apr,gr,use_sinktree
 use viscosity,        only:irealvisc
 use ptmass,           only:icreate_sinks
 use HIIRegion,        only:iH2R
 use ptmass_radiation, only:iget_tdust
 use kernel,           only:kernelname,radkern
 use part,             only:hfact
 use part,             only:iphase,iamtype,iamboundary,igas
 use dim,              only:maxphase,maxp
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
 integer :: i, ncross, nbound, nother
 !--slices of the gpu_arrays arena, re-associated each call because the arena moves
 !  when it grows.  contiguous, so these pass straight to the bind(C) call.
 real(c_double), pointer, contiguous :: x8(:),y8(:),z8(:),h8(:)
 real(c_double), pointer, contiguous :: vx8(:),vy8(:),vz8(:)
 real(c_double), pointer, contiguous :: ax8(:),ay8(:),az8(:)
 real(c_double), pointer, contiguous :: rho8(:),drhofh8(:)
 real(c_double), pointer, contiguous :: divv8(:),xi8(:),ddivvdt8(:)
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
 !--sinks in the tree add sink-gas forces during the force walk, which the GPU does not do
 if (use_sinktree)      call fatal('densityiterate_gpu','sinks in the tree not supported on GPU (use_sinktree)')
 !--cosmoSPHere implements the M_4 cubic and M_6 quintic splines, chosen when it is
 !  built (KERNEL is passed through); any other kernel, or a library built for a
 !  different one, would silently use the wrong kernel
 if (trim(kernelname) /= 'M_4 cubic' .and. trim(kernelname) /= 'M_6 quintic') &
    call fatal('densityiterate_gpu','kernel not implemented on GPU (use KERNEL=cubic or quintic): '//trim(kernelname))
 if (abs(real(cosmo_kernel_radius()) - radkern) > 1.e-6) call fatal('densityiterate_gpu', &
    'cosmoSPHere library was built for a different kernel: rebuild it',var='radkern',val=real(cosmo_kernel_radius()))

 !--the GPU solves every particle as gas of mass massoftype(igas).  Boundary particles
 !  (wind shells, BHL inflow) are inactive after the first call on the CPU path, and
 !  stars, dark matter and other types have their own mass and physics, so refuse any
 !  particle that is not gas.  Checked every call, as injection adds particles.
 if (maxphase == maxp) then
    nbound = 0
    nother = 0
    !$omp parallel do default(none) shared(npart,iphase) private(i) reduction(+:nbound,nother)
    do i = 1, npart
       if (iamboundary(iamtype(iphase(i)))) then
          nbound = nbound + 1
       elseif (iamtype(iphase(i)) /= igas) then
          nother = nother + 1
       endif
    enddo
    !$omp end parallel do
    if (nbound > 0) call fatal('densityiterate_gpu','boundary particles not supported on GPU',ival=nbound)
    if (nother > 0) call fatal('densityiterate_gpu','only gas particles are supported on GPU',ival=nother)
 endif

 !--COSMO_DENS_STATS=1 also reports the phantom-side cost of the GPU call
 if (.not. stats_checked) then
    call get_environment_variable('COSMO_DENS_STATS', statsenv)
    stats = (len_trim(statsenv) > 0)
    stats_checked = .true.
 endif
 call system_clock(ic0, crate)

 call gpu_arrays_init(npart)
 !--xi_gpu is a phantom-side result for cons2prim, not staging, so it is not in the
 !  arena.  It grows with it, and starts at the xi of a zero tensor.
 if (.not.allocated(xi_gpu)) then
    allocate(xi_gpu(gpu_arrays_nbuf()))
    xi_gpu = 1.
 elseif (size(xi_gpu) < npart) then
    deallocate(xi_gpu)
    allocate(xi_gpu(gpu_arrays_nbuf()))
    xi_gpu = 1.
 endif

 x8      => gpu_arrays_comp(ibun_pos,1)
 y8      => gpu_arrays_comp(ibun_pos,2)
 z8      => gpu_arrays_comp(ibun_pos,3)
 h8      => gpu_arrays_comp(ibun_hsml,1)
 vx8     => gpu_arrays_comp(ibun_vel,1)
 vy8     => gpu_arrays_comp(ibun_vel,2)
 vz8     => gpu_arrays_comp(ibun_vel,3)
 ax8     => gpu_arrays_comp(ibun_accel,1)
 ay8     => gpu_arrays_comp(ibun_accel,2)
 az8     => gpu_arrays_comp(ibun_accel,3)
 rho8    => gpu_arrays_comp(ibun_dens_out,1)
 drhofh8 => gpu_arrays_comp(ibun_dens_out,2)
 divv8   => gpu_arrays_comp(ibun_grad_out,1)
 xi8     => gpu_arrays_comp(ibun_grad_out,2)
 ddivvdt8=> gpu_arrays_comp(ibun_grad_out,3)

 ncross = 0
 !$omp parallel do default(none) private(i) &
 !$omp shared(npart,xyzh,vxyzu,fxyzu,fext,x8,y8,z8,h8,vx8,vy8,vz8,ax8,ay8,az8) &
 !$omp shared(isperiodic) reduction(+:ncross)
 do i = 1, npart
    !--the GPU tree is built in the periodic box, so particles must be inside it
    if (periodic) then
       if (.not.isdead_or_accreted(xyzh(4,i))) call cross_boundary(isperiodic,xyzh(:,i),ncross)
    endif
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
 !--the force pass that follows uses the same velocities, from the same bundle, so
 !  tell it not to pack them again (gpu_arrays `packed`)
 call gpu_arrays_mark_packed(ibun_vel)
 call system_clock(ic1)

 !--these positions are for a new step, so the device's ordering is now stale; it
 !  describes where the particles were.  The solve below rebuilds it.
 call gpu_arrays_positions_moved()
 call gpu_arrays_upload(ibun_pos,   npart)
 call gpu_arrays_upload(ibun_hsml,  npart)
 call gpu_arrays_upload(ibun_vel,   npart)
 call gpu_arrays_upload(ibun_accel, npart)

 call densityiterate_gpu_c(int(npart, kind=c_int), &
                            real(massoftype(igas), kind=c_double), &
                            merge(1_c_int, 0_c_int, periodic), &
                            real([xmin,xmax,ymin,ymax,zmin,zmax], kind=c_double), &
                            real(tolh, kind=c_double), &
                            real(hfact, kind=c_double))
 call gpu_arrays_order_rebuilt()

 !--the solve leaves its results on the device; fetch the three bundles it filled
 call gpu_arrays_download(ibun_hsml,     npart)
 call gpu_arrays_download(ibun_dens_out, npart)
 call gpu_arrays_download(ibun_grad_out, npart)
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


end module gpu_dens_iface
