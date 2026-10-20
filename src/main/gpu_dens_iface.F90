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
!   rho(i)        — the kernel-summed density (same type only)
!   gradh(igradomega,i) — 1/Omega~_i, Omega~ = 1 - dh/dn(h) dn/dh   (number-density
!   gradh(igradzeta,i)  — zeta_i = dh/dn(h) d(rho)/dh                grad-h terms, as
!                   dens.F90 finish_rhosum; dh/dn = -h/(3 n(h)), n(h) = (hfact/h)^3)
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
 use gpu_arrays,    only:resident_feed,resident_mode
 use gpu_arrays,    only:gpu_arrays_init,gpu_arrays_comp,gpu_arrays_nbuf, &
                         gpu_arrays_mark_packed,gpu_arrays_upload,gpu_arrays_download, &
                         gpu_arrays_positions_moved,gpu_arrays_order_rebuilt, &
                         ibun_pos,ibun_hsml,ibun_vel,ibun_accel, &
                         ibun_dens_out,ibun_grad_out,ibun_ptype,ibun_dustgas,ibun_active
 implicit none

#ifdef GPU
 logical, public :: use_gpu_dens = .true.
#else
 logical, public :: use_gpu_dens = .false.
#endif

#ifdef GPU
!--C interface to cosmoSPHere/src/dens_c_api.cu
 interface
  subroutine densityiterate_gpu_c(n, pmass, pmass_dust, &
                                  periodic, box, tolh, hfact, keep_dvdx, use_active) bind(C)
   use iso_c_binding, only:c_double,c_int
   integer(c_int), value         :: n
   real(c_double), value         :: pmass
   real(c_double), value         :: pmass_dust
   integer(c_int), value         :: periodic
   real(c_double), intent(in)    :: box(6)
   real(c_double), value         :: tolh
   real(c_double), value         :: hfact
   integer(c_int), value         :: keep_dvdx
   integer(c_int), value         :: use_active
  end subroutine densityiterate_gpu_c

!--C interfaces to cosmoSPHere/src/resident.cu: the solve's inputs from the device's
!  resident state (resident_feed)
  subroutine cosmo_rs_wrap(n,box,dbound) bind(C)
   use iso_c_binding, only:c_int,c_double
   integer(c_int), value :: n
   real(c_double), intent(in) :: box(6),dbound(3)
  end subroutine cosmo_rs_wrap
  subroutine cosmo_rs_feed_density(n,usedust,useactive) bind(C)
   use iso_c_binding, only:c_int
   integer(c_int), value :: n,usedust,useactive
  end subroutine cosmo_rs_feed_density

!--C interface to cosmo_kernel_radius in cosmoSPHere/src/dens_c_api.cu
  real(c_double) function cosmo_kernel_radius() bind(C)
   use iso_c_binding, only:c_double
  end function cosmo_kernel_radius

 end interface
#endif

 public :: densityiterate_gpu, init_gpu_switch
 public :: gpu_dens_mirror_before, gpu_dens_mirror_after

 !--DEV HATCH (GPU_IND_TIMESTEPS_PLAN Phase 1.1, removed before the phase's last
 !  commit): COSMO_DENS_MIRROR=1 with COSMO_GPU=0 runs the GPU density solve beside the
 !  CPU one on the same inputs, every derivs, and compares them per particle.  The run
 !  itself stays on the CPU path.
 logical, public :: gpu_dens_mirror = .false.
 !--and COSMO_FORCE_MIRROR=1 (implies the density one, whose solve the GPU force needs)
 !  does the same for the force pass (gpu_force_iface gpu_force_mirror_*)
 logical, public :: gpu_force_mirror = .false.
 real,         allocatable :: mir_h0(:),mir_rho0(:),mir_hg(:),mir_rhog(:)
 real(kind=4), allocatable :: mir_gradh0(:,:),mir_divv0(:),mir_alpha0(:)
 real(kind=4), allocatable :: mir_gradhg(:,:),mir_divvg(:),mir_alphag(:)
 integer, save :: mir_ncall = 0
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
 call get_environment_variable('COSMO_DENS_MIRROR',val,ln,ierr)
 gpu_dens_mirror = (ierr == 0 .and. ln > 0 .and. .not.use_gpu_dens)
 call get_environment_variable('COSMO_FORCE_MIRROR',val,ln,ierr)
 gpu_force_mirror = (ierr == 0 .and. ln > 0 .and. .not.use_gpu_dens)
 if (gpu_force_mirror) gpu_dens_mirror = .true.
 if (gpu_force_mirror) write(iprint,'(a)') ' cosmoSPHere: COSMO_FORCE_MIRROR, GPU force checked against the CPU'
 if (gpu_dens_mirror) write(iprint,'(a)') ' cosmoSPHere: COSMO_DENS_MIRROR, GPU density checked against the CPU'
#endif

end subroutine init_gpu_switch

!-------------------------------------------------------------
!+
!  GPU density iteration: replaces densityiterate(icall=1,...).
!
!  Calls the cosmoSPHere Newton-Raphson + Cornerstone GPU solver,
!  then converts outputs to phantom's array conventions.
!
!  Gas particles, and with two-fluid dust (dust as particles, one
!  large grain type) the dust particles too: the GPU builds a tree per
!  type and solves each with its own mass, massoftype(igas) and
!  massoftype(idust).
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
 use boundary,  only:cross_boundary,xmin,xmax,ymin,ymax,zmin,zmax,dxbound,dybound,dzbound
 use mpidomain, only:isperiodic
 use options,   only:tolh
 use dim,  only:curlv,mhd,use_dust,do_radiation,gravity,ind_timesteps,use_apr,gr,use_sinktree
 use viscosity,        only:irealvisc
 use ptmass,           only:icreate_sinks
 use HIIRegion,        only:iH2R
 use ptmass_radiation, only:iget_tdust
 use kernel,           only:kernelname,radkern
 use part,             only:hfact
 use part,             only:iphase,iamtype,iamboundary,igas,idust,iactive
 use part,             only:ndustsmall,ndustlarge,dustfrac,ndusttypes
 use dim,              only:use_dustgrowth
 use options,          only:use_dustfrac,use_hybrid
 use dust,             only:irecon
 use dim,              only:maxphase,maxp
 use part,             only:rho,dhdn,rhoh
 use dim,              only:igradomega,igradzeta
 use options,          only:two_kernel
 use iso_c_binding, only:c_double,c_int
#endif
 integer,      intent(in)    :: npart
 real,         intent(inout) :: xyzh(:,:)
 real,         intent(in)    :: vxyzu(:,:),fxyzu(:,:),fext(:,:)
 real(kind=4), intent(inout) :: gradh(:,:)
 real(kind=4), intent(inout) :: divcurlv(:,:)
 real(kind=4), intent(inout) :: alphaind(:,:)

#ifdef GPU
 real    :: hi, rhoi, drhoi, omega, dhdni, pmassi
 integer :: i, ncross, nbound, nother
 !--slices of the gpu_arrays arena, re-associated each call because the arena moves
 !  when it grows.  contiguous, so these pass straight to the bind(C) call.
 real(c_double), pointer, contiguous :: x8(:),y8(:),z8(:),h8(:)
 real(c_double), pointer, contiguous :: vx8(:),vy8(:),vz8(:)
 real(c_double), pointer, contiguous :: ax8(:),ay8(:),az8(:)
 real(c_double), pointer, contiguous :: rho8(:),drhofh8(:)
 real(c_double), pointer, contiguous :: divv8(:),xi8(:),ddivvdt8(:)
 real(c_double), pointer, contiguous :: ptype8(:),dustgas8(:),active8(:)
 real(c_double) :: pmass_dust
 integer(kind=8) :: ic0,ic1,ic2,ic3,ic4,crate
 character(len=8) :: statsenv
 logical, save    :: stats = .false.
 logical, save    :: stats_checked = .false.

 if (npart <= 0) return

 !--the GPU sweep produces div v, dv/dx and d(div v)/dt for a single gas type.
 !  Anything that needs a quantity it does not compute would silently get stale
 !  values, so refuse rather than run: curl v (divcurlv(2:4)), div/curl B,
 !  radiation fluxes, and gradsoft in gradh(2,:).
 if (curlv)        call fatal('densityiterate_gpu','curl v not computed on GPU (set curlv=F)')
 if (mhd)          call fatal('densityiterate_gpu','divcurlB not computed on GPU')
 !--two-fluid dust is solved on its own tree.  Dust as a mixture (dustfrac on the gas),
 !  small grains, more than one large grain size (one tree per size is not decided:
 !  force/GPU_DUST_TWO_TREE_PLAN.md) and dust growth are not.
 if (use_dust) then
    if (use_dustfrac .or. use_hybrid .or. ndustsmall > 0) &
       call fatal('densityiterate_gpu','dust as a mixture not supported on GPU (use dust_method=2)')
    if (ndustlarge > 1) &
       call fatal('densityiterate_gpu','more than one large grain size not supported on GPU',ival=ndustlarge)
    if (use_dustgrowth) call fatal('densityiterate_gpu','dust growth not supported on GPU')
 endif
 if (do_radiation) call fatal('densityiterate_gpu','radiation flux not computed on GPU')
 if (gravity)      call fatal('densityiterate_gpu','gradsoft not computed on GPU')
 if (two_kernel)   call fatal('densityiterate_gpu','two_kernel (n from W~) not computed on GPU')
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

 !--the GPU solves gas with massoftype(igas) and, with two-fluid dust, dust with
 !  massoftype(idust).  Boundary particles (wind shells, BHL inflow) are inactive after
 !  the first call on the CPU path, and stars, dark matter and other types have their
 !  own mass and physics, so refuse any other particle.  Checked every call, as
 !  injection adds particles.  (COSMO_RESIDENT: the types were checked on the first
 !  step, and the resident path refuses injection.)
 if (maxphase == maxp .and. .not.resident_mode) then
    nbound = 0
    nother = 0
    !$omp parallel do default(none) shared(npart,iphase) private(i) reduction(+:nbound,nother)
    do i = 1, npart
       if (iamboundary(iamtype(iphase(i)))) then
          nbound = nbound + 1
       elseif (iamtype(iphase(i)) /= igas .and. .not.(use_dust .and. iamtype(iphase(i)) == idust)) then
          nother = nother + 1
       endif
    enddo
    !$omp end parallel do
    if (nbound > 0) call fatal('densityiterate_gpu','boundary particles not supported on GPU',ival=nbound)
    if (nother > 0) call fatal('densityiterate_gpu','only gas and two-fluid dust particles are supported on GPU',ival=nother)
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
 if (use_dust) ptype8   => gpu_arrays_comp(ibun_ptype,1)
 if (use_dust) dustgas8 => gpu_arrays_comp(ibun_dustgas,1)
 if (ind_timesteps) active8 => gpu_arrays_comp(ibun_active,1)

 ncross = 0
 !--COSMO_RESIDENT: the device stages its own copy (cosmo_rs_feed_density below)
 if (.not.resident_mode) then
 !$omp parallel do default(none) private(i) &
 !$omp shared(npart,xyzh,vxyzu,fxyzu,fext,x8,y8,z8,h8,vx8,vy8,vz8,ax8,ay8,az8) &
 !$omp shared(isperiodic,ptype8,active8,iphase) reduction(+:ncross)
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
    !--which tree the particle goes in (the type check above admits only these two)
    if (use_dust) ptype8(i) = merge(1._c_double, 0._c_double, iamtype(iphase(i)) == idust)
    !--individual timesteps: the solve iterates only the particles active this step
    if (ind_timesteps) active8(i) = merge(1._c_double, 0._c_double, iactive(iphase(i)))
 enddo
 !$omp end parallel do
 endif
 !--the force pass that follows uses the same velocities, from the same bundle, so
 !  tell it not to pack them again (gpu_arrays `packed`)
 call gpu_arrays_mark_packed(ibun_vel)
 call system_clock(ic1)

 !--these positions are for a new step, so the device's ordering is now stale; it
 !  describes where the particles were.  The solve below rebuilds it.
 call gpu_arrays_positions_moved()
 if (resident_feed) then
    !--the same inputs from the device's own copy, device to device (gpu_resident)
    if (periodic) call cosmo_rs_wrap(int(npart,kind=c_int), &
                                     real([xmin,xmax,ymin,ymax,zmin,zmax],kind=c_double), &
                                     real([dxbound,dybound,dzbound],kind=c_double))
    call cosmo_rs_feed_density(int(npart,kind=c_int),merge(1_c_int,0_c_int,use_dust), &
                               merge(1_c_int,0_c_int,ind_timesteps))
 else
    call gpu_arrays_upload(ibun_pos,   npart)
    call gpu_arrays_upload(ibun_hsml,  npart)
    call gpu_arrays_upload(ibun_vel,   npart)
    call gpu_arrays_upload(ibun_accel, npart)
    if (use_dust) call gpu_arrays_upload(ibun_ptype, npart)
    if (ind_timesteps) call gpu_arrays_upload(ibun_active, npart)
 endif
 pmass_dust = 0._c_double
 if (use_dust) pmass_dust = real(massoftype(idust), kind=c_double)

 call densityiterate_gpu_c(int(npart, kind=c_int), &
                            real(massoftype(igas), kind=c_double), &
                            pmass_dust, &
                            merge(1_c_int, 0_c_int, periodic), &
                            real([xmin,xmax,ymin,ymax,zmin,zmax], kind=c_double), &
                            real(tolh, kind=c_double), &
                            real(hfact, kind=c_double), &
                            !--drag reconstruction reads both particles' dv/dx in the
                            !  force pass, so the device keeps it (gpu_arrays ibun_dvdx)
                            merge(1_c_int, 0_c_int, use_dust .and. irecon >= 0), &
                            merge(1_c_int, 0_c_int, ind_timesteps))
 call gpu_arrays_order_rebuilt()

 !--COSMO_RESIDENT: the results stay on the device (gpu_resident's density writeback)
 if (resident_mode) then
    call system_clock(ic2)
    if (stats) write(0,'(a,f8.2,a,f8.2)') 'COSMO_FORT stage=',1.e3*real(ic1-ic0)/real(crate), &
                                         ' capi=',1.e3*real(ic2-ic1)/real(crate)
    return
 endif
 !--the solve leaves its results on the device; fetch the three bundles it filled
 call gpu_arrays_download(ibun_hsml,     npart)
 call gpu_arrays_download(ibun_dens_out, npart)
 call gpu_arrays_download(ibun_grad_out, npart)
 if (use_dust) call gpu_arrays_download(ibun_dustgas, npart)
 call system_clock(ic2)

 !--write results back to phantom arrays, for the particles the solve touched: not the
 !  dead and accreted ones (h < 0), and under individual timesteps only the active ones,
 !  whose results phantom's densityiterate would have written.  The inactive keep their
 !  gradh, div v, xi and d(div v)/dt from their last active step and the h predicted in
 !  step, which the device did not change either; their rho is rhoh(h), as the CPU
 !  densityiterate sets it for every live particle first (init_rho_from_h).
 !$omp parallel do default(none) private(i,hi,rhoi,drhoi,omega,dhdni,pmassi) &
 !$omp shared(npart,xyzh,gradh,divcurlv,alphaind,xi_gpu,dustfrac,ndustsmall,ndusttypes,rho) &
 !$omp shared(h8,rho8,drhofh8,divv8,xi8,ddivvdt8,dustgas8,iphase,massoftype,maxp,maxphase)
 do i = 1, npart
    if (xyzh(4,i) < 0.) cycle   ! dead or accreted: keep the negative h
    if (ind_timesteps) then
       if (.not.iactive(iphase(i))) then
          pmassi = massoftype(iamtype(iphase(i)))
          rho(i) = rhoh(xyzh(4,i),pmassi)
          cycle
       endif
    endif
    hi    = real(h8(i))
    rhoi  = real(rho8(i))
    drhoi = real(drhofh8(i))    ! d(rho)/d(h), normalised
    xyzh(4,i) = hi
    rho(i)    = rhoi
    !--phantom's number-density grad-h terms (dens.F90 finish_rhosum), sums over the
    !  particle's own type: Omega~ = 1 - dh/dn dn/dh with dn/dh = (drho/dh)/m, and
    !  zeta = dh/dn drho/dh.  NB: the GPU forms the same Omega~ internally to normalise
    !  the gradients (sphGradientsKernel) and the force (forcePrepKernel) -- keep in step.
    if (rhoi > 0. .and. hi > 0.) then
       pmassi = massoftype(igas)
       if (maxphase==maxp) pmassi = massoftype(iamtype(iphase(i)))
       dhdni = dhdn(hi)
       omega = 1.0 - dhdni * drhoi / pmassi
       if (omega > 0.) then
          gradh(igradomega,i) = real(1.0 / omega, kind=4)
       else
          gradh(igradomega,i) = 1.0_4
       endif
       gradh(igradzeta,i) = real(dhdni * drhoi, kind=4)
    else
       gradh(igradomega,i) = 1.0_4
       gradh(igradzeta,i)  = 0.0_4
    endif
    divcurlv(1,i) = real(divv8(i), kind=4)
    xi_gpu(i) = real(xi8(i))
    if (nalpha >= 3) alphaind(3,i) = real(ddivvdt8(i), kind=4)
    !--two-fluid dust, as dens.F90: dustfrac is cleared on every particle the solve
    !  touched, then holds the dust-to-gas ratio on gas (zero from the device on dust).
    !  Only the dust types in use: dustfrac has maxdusttypes rows (22 unless MAXDUST* is
    !  set at build time), and clearing all of them on every particle was most of this loop.
    if (use_dust) then
       dustfrac(1:ndusttypes,i) = 0.
       dustfrac(ndustsmall+1,i) = real(dustgas8(i))
    endif
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


!-------------------------------------------------------------
!+
!  DEV HATCH, Phase 1.1: before the CPU density pass, run the GPU one on
!  the same inputs, keep its results, and put the inputs back.
!+
!-------------------------------------------------------------
subroutine gpu_dens_mirror_before(npart,xyzh,vxyzu,fxyzu,fext,gradh,divcurlv,alphaind)
 use part, only:rho
 use dim,  only:nalpha
 integer,      intent(in)    :: npart
 real,         intent(inout) :: xyzh(:,:)
 real,         intent(in)    :: vxyzu(:,:),fxyzu(:,:),fext(:,:)
 real(kind=4), intent(inout) :: gradh(:,:),divcurlv(:,:),alphaind(:,:)
 integer :: ng

 if (allocated(mir_h0)) then
    if (size(mir_h0) < npart) deallocate(mir_h0,mir_rho0,mir_hg,mir_rhog,mir_gradh0,mir_divv0, &
                                         mir_alpha0,mir_gradhg,mir_divvg,mir_alphag)
 endif
 ng = size(gradh,1)
 if (.not.allocated(mir_h0)) allocate(mir_h0(npart),mir_rho0(npart),mir_hg(npart),mir_rhog(npart), &
                                      mir_gradh0(ng,npart),mir_divv0(npart),mir_alpha0(npart), &
                                      mir_gradhg(ng,npart),mir_divvg(npart),mir_alphag(npart))
 mir_h0(1:npart)       = xyzh(4,1:npart)
 mir_rho0(1:npart)     = rho(1:npart)
 mir_gradh0(:,1:npart) = gradh(:,1:npart)
 mir_divv0(1:npart)    = divcurlv(1,1:npart)
 if (nalpha >= 3) mir_alpha0(1:npart) = alphaind(3,1:npart)

 call densityiterate_gpu(npart,xyzh,vxyzu,fxyzu,fext,gradh,divcurlv,alphaind)

 mir_hg(1:npart)       = xyzh(4,1:npart)
 mir_rhog(1:npart)     = rho(1:npart)
 mir_gradhg(:,1:npart) = gradh(:,1:npart)
 mir_divvg(1:npart)    = divcurlv(1,1:npart)
 if (nalpha >= 3) mir_alphag(1:npart) = alphaind(3,1:npart)

 xyzh(4,1:npart)       = mir_h0(1:npart)
 rho(1:npart)          = mir_rho0(1:npart)
 gradh(:,1:npart)      = mir_gradh0(:,1:npart)
 divcurlv(1,1:npart)   = mir_divv0(1:npart)
 if (nalpha >= 3) alphaind(3,1:npart) = mir_alpha0(1:npart)

end subroutine gpu_dens_mirror_before

!-------------------------------------------------------------
!+
!  DEV HATCH, Phase 1.1: after the CPU density pass, compare.  One line
!  per call on stdout, COSMO_DENS_MIRROR, with: the number of active and
!  inactive live particles; over the active ones the largest relative
!  difference in h, rho and 1/Omega, the rms difference in div v over the
!  rms div v, and the same for d(div v)/dt; and, of the inactive particles,
!  how many the GPU pass left with a different h, rho, gradh, div v or
!  alphaind(3) from the CPU pass (must be 0: the same formula or no change),
!  and how many either pass changed from what it was given (rho = rhoh(h)).
!+
!-------------------------------------------------------------
subroutine gpu_dens_mirror_after(npart,xyzh,gradh,divcurlv,alphaind)
 use part, only:rho,iphase,iactive,maxphase,maxp
 use dim,  only:nalpha,ind_timesteps
 use io,   only:iprint
 integer,      intent(in) :: npart
 real,         intent(in) :: xyzh(:,:)
 real(kind=4), intent(in) :: gradh(:,:),divcurlv(:,:),alphaind(:,:)
 integer :: i,nact,ninact,ntouch_gpu,ntouch_cpu
 real    :: dh,drho,domega,sdv,sv,sda,sa
 logical :: act

 nact = 0; ninact = 0; ntouch_gpu = 0; ntouch_cpu = 0
 dh = 0.; drho = 0.; domega = 0.; sdv = 0.; sv = 0.; sda = 0.; sa = 0.
 !$omp parallel do default(none) private(i,act) &
 !$omp shared(npart,xyzh,gradh,divcurlv,alphaind,rho,iphase,mir_h0,mir_rho0,mir_gradh0) &
 !$omp shared(mir_divv0,mir_alpha0,mir_hg,mir_rhog,mir_gradhg,mir_divvg,mir_alphag,maxphase,maxp) &
 !$omp reduction(+:nact,ninact,ntouch_gpu,ntouch_cpu,sdv,sv,sda,sa) reduction(max:dh,drho,domega)
 do i = 1, npart
    if (mir_h0(i) <= 0.) cycle
    act = .true.
    if (ind_timesteps .and. maxphase == maxp) act = iactive(iphase(i))
    if (act) then
       nact = nact + 1
       if (xyzh(4,i) > 0.) dh = max(dh, abs(mir_hg(i)/xyzh(4,i) - 1.))
       if (rho(i) > 0.) drho = max(drho, abs(mir_rhog(i)/rho(i) - 1.))
       domega = max(domega, abs(real(mir_gradhg(1,i))/real(gradh(1,i)) - 1.))
       sdv = sdv + (real(mir_divvg(i)) - real(divcurlv(1,i)))**2
       sv  = sv  + real(divcurlv(1,i))**2
       if (nalpha >= 3) then
          sda = sda + (real(mir_alphag(i)) - real(alphaind(3,i)))**2
          sa  = sa  + real(alphaind(3,i))**2
       endif
    else
       ninact = ninact + 1
       if (mir_hg(i) /= xyzh(4,i) .or. mir_rhog(i) /= rho(i) .or. &
           any(mir_gradhg(:,i) /= gradh(:,i)) .or. mir_divvg(i) /= divcurlv(1,i)) &
          ntouch_gpu = ntouch_gpu + 1
       if (nalpha >= 3) then
          if (mir_alphag(i) /= alphaind(3,i)) ntouch_gpu = ntouch_gpu + 1
       endif
       if (xyzh(4,i) /= mir_h0(i) .or. rho(i) /= mir_rho0(i) .or. &
           any(gradh(:,i) /= mir_gradh0(:,i)) .or. divcurlv(1,i) /= mir_divv0(i)) &
          ntouch_cpu = ntouch_cpu + 1
    endif
 enddo
 !$omp end parallel do
 mir_ncall = mir_ncall + 1
 write(iprint,'(a,i6,a,i10,a,i10,4(a,es9.2),a,es9.2,a,2i8)') &
    ' COSMO_DENS_MIRROR call',mir_ncall,' active',nact,' inactive',ninact, &
    ' dh',dh,' drho',drho,' domega',domega,' ddivv',sqrt(sdv/max(sv,tiny(sv))), &
    ' dddivvdt',sqrt(sda/max(sa,tiny(sa))),' inactive_gpu_ne_cpu/cpu_changed',ntouch_gpu,ntouch_cpu

end subroutine gpu_dens_mirror_after

end module gpu_dens_iface
