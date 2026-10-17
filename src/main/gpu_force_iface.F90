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
! force_gpu is the whole of the GPU force call as deriv.f90 sees it: it derives
! the per-particle inputs the kernel needs, gathers them into contiguous
! C-interoperable buffers, calls force_gpu_c, writes fxyzu and divcurlv back,
! and sets the timestep constraints.  deriv.f90 therefore calls it exactly the
! way it calls force, with no GPU-specific scaffolding of its own.
!
! Host staging belongs to gpu_arrays, which this shares with the density pass:
! velocity is the same bundle for both, so whichever packs it first is the only
! one that packs it.
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
! :Dependencies: dim, gpu_arrays, iso_c_binding, options, part, timestep
!
 use iso_c_binding, only:c_double,c_int
 use gpu_arrays,    only:gpu_arrays_init,gpu_arrays_comp,gpu_arrays_claim_packed, &
                         gpu_arrays_upload, &
                         gpu_arrays_download, &
                         ibun_vel,ibun_thermo,ibun_force_out,ibun_drag_out,ibun_dragold
 implicit none

#ifdef GPU
 interface
    subroutine force_gpu_c(n,pmass,beta,alphau,disc_viscosity, &
                         pdv_heating,shock_heating,idrag,dust,drag_tally) bind(C)
    use iso_c_binding, only:c_double,c_int,c_long_long

    integer(c_int), value       :: n
    real(c_double), value       :: pmass
    real(c_double), value       :: beta
    real(c_double), value       :: alphau
    integer(c_int), value       :: disc_viscosity
    integer(c_int), value       :: pdv_heating,shock_heating
    integer(c_int), value       :: idrag             ! 0: no two-fluid drag
    real(c_double), intent(in)  :: dust(11)          ! see dust_params below
    integer(c_long_long), intent(out) :: drag_tally(3)  ! drag, Stokes, supersonic pairs
    end subroutine force_gpu_c

 end interface
#endif

 public :: force_gpu
 private

contains

!-----------------------------------------------------------------------
!+
!  Evaluate the SPH force on the GPU and write the results back into
!  phantom's arrays.  Drop-in replacement for a call to force.
!+
!-----------------------------------------------------------------------
subroutine force_gpu(npart,xyzh,vxyzu,eos_vars,alphaind,fxyzu,divcurlv,dt)
 use part,    only:massoftype,igas
 use options, only:beta,alphau
 use dim,     only:maxvxyzu,driving,disc_viscosity,track_lum,h2chemistry,store_dust_temperature
 use io,      only:fatal
 use part,    only:ien_type,ien_entropy,ien_entropy_s,rho,isdead_or_accreted
 use eos,     only:icooling,ipdv_heating,ishock_heating
 use cooling, only:energ_cooling,cooling_in_step
 use part,    only:nptmass,xyzmh_ptmass
 use part,    only:sinks_have_heating
 use dim,     only:use_dust
 use part,    only:iphase,iamtype,idust,grainsize,graindens,tstop,fxyz_drag,fxyz_dragold
 use dust,    only:idrag,irecon,drag_implicit,icut_backreaction,K_code,seff,coeff_gei_1,cste_mu
 use timestep,only:bignumber
 use io,      only:iverbose,warning,id,master
 use io_summary, only:summary_variable,iosumdgs,iosumdge
 use iso_c_binding, only:c_long_long

 integer,      intent(in)    :: npart
 real,         intent(in)    :: xyzh(:,:),vxyzu(:,:)
 real,         intent(in)    :: eos_vars(:,:)
 real(kind=4), intent(in)    :: alphaind(:,:)
 real,         intent(inout) :: fxyzu(:,:)
 real(kind=4), intent(inout) :: divcurlv(:,:)
 real,         intent(in)    :: dt

#ifdef GPU
 integer :: i
 real    :: dudtcool
 logical :: add_cooling
 !--slices of the gpu_arrays arena, re-associated each call because the arena moves
 !  when it grows.  contiguous, so these pass straight to the bind(C) call.
 real(c_double), pointer, contiguous :: vx8(:),vy8(:),vz8(:)
 real(c_double), pointer, contiguous :: pro2_8(:),spsound_8(:),alphaAV_8(:),u_8(:)
 real(c_double), pointer, contiguous :: fx8(:),fy8(:),fz8(:),f48(:)
 real(c_double), pointer, contiguous :: vsigmax8(:),divv8(:)
 real(c_double), pointer, contiguous :: fdx8(:),fdy8(:),fdz8(:),tsmin8(:)
 real(c_double), pointer, contiguous :: fdox8(:),fdoy8(:),fdoz8(:)
 real(c_double)       :: dust_params(11)
 integer(c_long_long) :: drag_tally(3)
 integer(c_int)       :: idrag_gpu
 real                 :: frac_stokes,frac_super
 !--host-side cost of the force pass, under the same COSMO_DENS_STATS gate as the
 !  density one.  Without it the sweeps below are invisible: the GPU stats cover only
 !  the density solve, and phantom's own force timer is quantised to 1/8 s by the
 !  real*4 in utils_timing.
 integer(kind=8)  :: ic0,ic1,ic2,ic3,ic4,ic5,crate
 character(len=8) :: statsenv
 logical, save    :: stats = .false.
 logical, save    :: stats_checked = .false.

 if (npart <= 0) return

 !--two-fluid dust: gas-dust drag, explicit or implicit (drag_implicit), with or without
 !  reconstruction (irecon), as force.F90; idrag = 0 would still need the gas-dust pairs
 !  for vsig, which the GPU only walks when there is drag.
 if (use_dust) then
    if (idrag < 1 .or. idrag > 3) &
       call fatal('force_gpu','two-fluid drag on the GPU needs idrag = 1, 2 or 3',ival=idrag)
 endif

 if (.not. stats_checked) then
    call get_environment_variable('COSMO_DENS_STATS', statsenv)
    stats = (len_trim(statsenv) > 0)
    stats_checked = .true.
 endif
 call system_clock(ic0, crate)

 !--the GPU returns du/dt as p dV work + shock heating (each switchable) + conductivity,
 !  as force.F90 assembles it for the internal energy; refuse what would change that sum
 if (maxvxyzu >= 4) then
    if (ien_type == ien_entropy .or. ien_type == ien_entropy_s) &
       call fatal('force_gpu','entropy as the energy variable is not supported on GPU')
    if (track_lum) call fatal('force_gpu','track_lum not supported on GPU')
    if (sinks_have_heating(nptmass,xyzmh_ptmass)) call fatal('force_gpu','sink heating not supported on GPU')
    if (icooling == 9) call fatal('force_gpu','icooling = 9 not supported on GPU')
    if (icooling > 0 .and. .not.cooling_in_step .and. (h2chemistry .or. store_dust_temperature)) &
       call fatal('force_gpu','cooling with chemistry or dust temperature in the force pass not supported on GPU')
 endif
 !--cooling that force.F90 applies in the force pass (not in the step), added below
 add_cooling = (maxvxyzu >= 4 .and. icooling > 0 .and. dt > 0. .and. .not.cooling_in_step)

 call gpu_arrays_init(npart)
 vx8      => gpu_arrays_comp(ibun_vel,1)
 vy8      => gpu_arrays_comp(ibun_vel,2)
 vz8      => gpu_arrays_comp(ibun_vel,3)
 pro2_8   => gpu_arrays_comp(ibun_thermo,1)
 spsound_8=> gpu_arrays_comp(ibun_thermo,2)
 alphaAV_8=> gpu_arrays_comp(ibun_thermo,3)
 u_8      => gpu_arrays_comp(ibun_thermo,4)
 fx8      => gpu_arrays_comp(ibun_force_out,1)
 fy8      => gpu_arrays_comp(ibun_force_out,2)
 fz8      => gpu_arrays_comp(ibun_force_out,3)
 f48      => gpu_arrays_comp(ibun_force_out,4)
 vsigmax8 => gpu_arrays_comp(ibun_force_out,5)
 divv8    => gpu_arrays_comp(ibun_force_out,6)
 if (use_dust) then
    fdx8   => gpu_arrays_comp(ibun_drag_out,1)
    fdy8   => gpu_arrays_comp(ibun_drag_out,2)
    fdz8   => gpu_arrays_comp(ibun_drag_out,3)
    tsmin8 => gpu_arrays_comp(ibun_drag_out,4)
 endif

 call prepare_pro2_gpu(npart,xyzh,vxyzu,eos_vars,alphaind, &
                       pro2_8,spsound_8,alphaAV_8,u_8)

 !--positions and h are not sent: the GPU uses its copies from the density solve.
 !  Velocity is shared with the density pass, so if that pass packed it for this
 !  same set of positions the slice already holds what this pass would write.  A
 !  later force pass on the same tree is the leapfrog corrector, whose velocities
 !  have changed, and the mark is gone by then, so it packs.
 if (.not. gpu_arrays_claim_packed(ibun_vel)) then
    !$omp parallel do default(none) schedule(static) private(i) shared(npart,vxyzu,vx8,vy8,vz8)
    do i = 1,npart
       vx8(i) = real(vxyzu(1,i),kind=c_double)
       vy8(i) = real(vxyzu(2,i),kind=c_double)
       vz8(i) = real(vxyzu(3,i),kind=c_double)
    enddo
    !$omp end parallel do
    !--packed here, so the device's copy is stale: send it
    call gpu_arrays_upload(ibun_vel,npart)
 endif

 call gpu_arrays_upload(ibun_thermo,npart)

 !--implicit drag, as force.F90: this pass reads the drag acceleration of the previous
 !  one, so keep it as fxyz_dragold and send it; fxyz_drag is refilled after the pass
 if (use_dust .and. drag_implicit) then
    fdox8 => gpu_arrays_comp(ibun_dragold,1)
    fdoy8 => gpu_arrays_comp(ibun_dragold,2)
    fdoz8 => gpu_arrays_comp(ibun_dragold,3)
    !$omp parallel do default(none) schedule(static) private(i) &
    !$omp shared(npart,fxyz_drag,fxyz_dragold,fdox8,fdoy8,fdoz8)
    do i = 1,npart
       fxyz_dragold(1:3,i) = fxyz_drag(1:3,i)
       fdox8(i) = real(fxyz_dragold(1,i),kind=c_double)
       fdoy8(i) = real(fxyz_dragold(2,i),kind=c_double)
       fdoz8(i) = real(fxyz_dragold(3,i),kind=c_double)
    enddo
    !$omp end parallel do
    call gpu_arrays_upload(ibun_dragold,npart)
 endif
 call system_clock(ic1)

 !--the drag parameters, in the order force_c_api.cu reads them: one large grain size
 !  (densityiterate_gpu refuses more), and dust.f90's constants for the stopping time
 idrag_gpu   = 0_c_int
 dust_params = 0._c_double
 drag_tally  = 0_c_long_long
 if (use_dust) then
    idrag_gpu   = int(idrag, kind=c_int)
    dust_params = real([grainsize(1), graindens(1), K_code(1), seff, coeff_gei_1, cste_mu, &
                        massoftype(idust), merge(1., 0., icut_backreaction == 1), real(irecon), &
                        merge(1., 0., drag_implicit), dt], kind=c_double)
 endif

 call force_gpu_c(int(npart,kind=c_int),                &
                  real(massoftype(igas),kind=c_double), &
                  real(beta,kind=c_double),             &
                  real(alphau,kind=c_double),           &
                  merge(1_c_int,0_c_int,disc_viscosity), &
                  int(ipdv_heating,kind=c_int),int(ishock_heating,kind=c_int), &
                  idrag_gpu,dust_params,drag_tally)

 !--the pass leaves its results on the device; fetch them
 call system_clock(ic2)

 call gpu_arrays_download(ibun_force_out,npart)
 if (use_dust) call gpu_arrays_download(ibun_drag_out,npart)

 call system_clock(ic3)

 !--as force.F90: with driving, fxyzu already holds the driving force (forceit
 !  runs first), so the SPH force is added to it.  Isothermal builds have no
 !  u, so fxyzu has no fourth row and du/dt is discarded.
 !$omp parallel do default(none) schedule(static) private(i) &
 !$omp shared(npart,fxyzu,divcurlv,fx8,fy8,fz8,f48,divv8,iphase,fdx8,fdy8,fdz8,tsmin8,tstop,xyzh) &
 !$omp shared(fxyz_drag,drag_implicit)
 do i = 1,npart
    if (driving) then
       fxyzu(1,i) = fxyzu(1,i) + real(fx8(i),kind=kind(fxyzu))
       fxyzu(2,i) = fxyzu(2,i) + real(fy8(i),kind=kind(fxyzu))
       fxyzu(3,i) = fxyzu(3,i) + real(fz8(i),kind=kind(fxyzu))
    else
       fxyzu(1,i) = real(fx8(i),kind=kind(fxyzu))
       fxyzu(2,i) = real(fy8(i),kind=kind(fxyzu))
       fxyzu(3,i) = real(fz8(i),kind=kind(fxyzu))
    endif
    if (maxvxyzu >= 4) fxyzu(4,i) = real(f48(i),kind=kind(fxyzu))
    if (use_dust) then
       !--as force.F90: the drag is added to fxyzu after the SPH force, and tstop is the
       !  smallest pair stopping time, bignumber for a particle with no partner
       fxyzu(1,i) = fxyzu(1,i) + real(fdx8(i),kind=kind(fxyzu))
       fxyzu(2,i) = fxyzu(2,i) + real(fdy8(i),kind=kind(fxyzu))
       fxyzu(3,i) = fxyzu(3,i) + real(fdz8(i),kind=kind(fxyzu))
       if (.not.isdead_or_accreted(xyzh(4,i))) then
          tstop(:,i) = min(real(tsmin8(i)),bignumber)
          !--as force.F90 with drag_implicit: this pass's drag, read back by the next one
          if (drag_implicit) fxyz_drag(1:3,i) = [real(fdx8(i)),real(fdy8(i)),real(fdz8(i))]
       endif
    endif
    !--as force.F90 (isgas): the force pass's div v replaces the density pass's for gas
    !  only.  Dust keeps the one from its own density solve.
    if (use_dust) then
       if (iamtype(iphase(i)) /= igas) cycle
    endif
    divcurlv(1,i) = real(divv8(i),kind=kind(divcurlv))
 enddo
 !$omp end parallel do

 call system_clock(ic4)

 !--as force.F90: cooling evaluated in the force pass, from div v of this pass
 if (add_cooling) then
    !$omp parallel do default(none) schedule(static) private(i,dudtcool) &
    !$omp shared(npart,xyzh,vxyzu,fxyzu,divcurlv,rho,dt,iphase)
    do i = 1,npart
       if (isdead_or_accreted(xyzh(4,i))) cycle
       !--as force.F90, cooling is for gas
       if (use_dust) then
          if (iamtype(iphase(i)) /= igas) cycle
       endif
       call energ_cooling(xyzh(1,i),xyzh(2,i),xyzh(3,i),vxyzu(4,i), &
                          rho(i),dt,divcurlv(1,i),dudtcool)
       fxyzu(4,i) = fxyzu(4,i) + dudtcool
    enddo
    !$omp end parallel do
 endif

 call finish_gpu_force_timesteps(npart,xyzh,vxyzu,fxyzu, &
                                 fx8,fy8,fz8,vsigmax8,spsound_8,tsmin8)

 !--as force.F90: the share of drag pairs in the Stokes and supersonic Epstein regimes
 if (use_dust) then
    if (drag_tally(1) > 0) then
       frac_stokes = real(drag_tally(2))/real(drag_tally(1))
       frac_super  = real(drag_tally(3))/real(drag_tally(1))
       if (iverbose >= 1 .and. id==master) then
          if (drag_tally(2) > 0) call warning('force','using Stokes drag regime',var='%Stokes',val=100.*frac_stokes)
          if (drag_tally(3) > 0) call warning('force','supersonic Epstein regime',val=100.*frac_super,var='%super')
       endif
       if (drag_tally(2) > 0) call summary_variable('dust',iosumdgs,int(drag_tally(2)),100.*frac_stokes)
       if (drag_tally(3) > 0) call summary_variable('dust',iosumdge,int(drag_tally(3)),100.*frac_super)
    endif
 endif
 call system_clock(ic5)

 if (stats) then
    write(0,'(a,f8.2,a,f8.2,a,f8.2,a,f8.2,a,f8.2)') &
       'COSMO_FFORT pack=',  1.e3*real(ic1-ic0)/real(crate), &
       ' capi=',             1.e3*real(ic2-ic1)/real(crate), &
       ' download=',         1.e3*real(ic3-ic2)/real(crate), &
       ' unpack+cool=',      1.e3*real(ic4-ic3)/real(crate), &
       ' dt=',               1.e3*real(ic5-ic4)/real(crate)
 endif
#else
 print *, 'ERROR: force_gpu called but phantom not compiled with GPU=yes'
 stop
#endif

end subroutine force_gpu

!-----------------------------------------------------------------------
!+
!  Derive the per-particle inputs the force kernel needs, into the staging
!  buffers.  No MHD/radiation/physical viscosity branch of get_stress.
!+
!-----------------------------------------------------------------------
subroutine prepare_pro2_gpu(npart,xyzh,vxyzu,eos_vars,alphaind, &
                            pro2_8,spsound_8,alphaAV_8,u_8)
 use dim,     only:maxalpha,maxp,maxvxyzu
 use options, only:alpha
 use part,    only:igasP,ics,rho

 integer,      intent(in)  :: npart
 real,         intent(in)  :: xyzh(:,:)
 real,         intent(in)  :: vxyzu(:,:)
 real,         intent(in)  :: eos_vars(:,:)
 real(kind=4), intent(in)  :: alphaind(:,:)
 real(c_double), intent(out) :: pro2_8(:),spsound_8(:),alphaAV_8(:),u_8(:)

 integer :: i
 real    :: rhoi,rho1i

 !$omp parallel do default(none) schedule(static) private(i,rhoi,rho1i) &
 !$omp shared(npart,xyzh,vxyzu,eos_vars,alphaind,rho,alpha,maxalpha,maxp) &
 !$omp shared(pro2_8,spsound_8,u_8,alphaAV_8)
 do i = 1,npart
    !--the kernel-summed density, as force.F90 and cons2prim since the number-density
    !  formulation; zero on particles the solve has never touched
    rhoi         = rho(i)
    rho1i        = 0.
    if (rhoi > 0.) rho1i = 1.0/rhoi

    pro2_8(i)    = eos_vars(igasP,i)*rho1i*rho1i
    spsound_8(i) = eos_vars(ics,i)
    !--isothermal: no u to read; u enters only the conductivity term of du/dt,
    !  which is discarded
    if (maxvxyzu >= 4) then
       u_8(i) = vxyzu(4,i)
    else
       u_8(i) = 0.
    endif

    if (maxalpha == maxp) then
       alphaAV_8(i) = real(alphaind(1,i),kind=c_double)
    else
       alphaAV_8(i) = alpha
    endif
 enddo
 !$omp end parallel do

end subroutine prepare_pro2_gpu

!-----------------------------------------------------------------------
!+
!  Courant and force timestep constraints from the GPU results.  vsigmax
!  and the sound speed come from the staging buffers the kernel just filled.
!+
!-----------------------------------------------------------------------
subroutine finish_gpu_force_timesteps(npart,xyzh,vxyzu,fxyzu, &
                                      fx8,fy8,fz8,vsigmax8,spsound_8,tsmin8)
 use options,  only:alpha
 use timestep, only:C_cour,C_force,bignumber,dtmax, &
                    dtcourant,dtforce,dtrad
 use part,     only:isdead_or_accreted,iphase,iamtype,igas
 use dim,      only:maxvxyzu,gr,use_dust
 use eos,      only:ieos
 use io_summary, only:summary_variable,iosumdgr
 use dust,     only:drag_implicit

 integer,        intent(in)    :: npart
 real,           intent(in)    :: xyzh(:,:),vxyzu(:,:)
 real,           intent(inout) :: fxyzu(:,:)
 real(c_double), intent(in)    :: fx8(:),fy8(:),fz8(:),vsigmax8(:),spsound_8(:)
 real(c_double), pointer, contiguous, intent(in) :: tsmin8(:)   ! associated only with use_dust

 integer :: i,ndustres
 real    :: hi,vsigdtc,f2i,dtc,dtf,dtcmin,dtfmin,eni,ts_min,dtdrag
 real    :: dustresfacmean,dustresfacmax
 logical :: limit_u,isgas

 dtcmin = bignumber
 dtfmin = bignumber
 dtrad  = bignumber
 ndustres       = 0
 dustresfacmean = 0.
 dustresfacmax  = 0.
 !--as force.F90: du/dt is limited so a Courant step cannot make u negative
 limit_u = (maxvxyzu >= 4 .and. .not.gr .and. ieos /= 23)

 !--min is exact under reduction, so this is bit-identical to the serial loop
 !$omp parallel do default(none) schedule(static) private(i,hi,vsigdtc,f2i,dtc,dtf,eni,isgas,ts_min,dtdrag) &
 !$omp shared(npart,xyzh,vxyzu,fxyzu,limit_u,fx8,fy8,fz8,vsigmax8,spsound_8,dtmax,C_cour,C_force,alpha) &
 !$omp shared(iphase,tsmin8,drag_implicit) &
 !$omp reduction(min:dtcmin,dtfmin) reduction(+:ndustres,dustresfacmean) reduction(max:dustresfacmax)
 do i = 1,npart
    hi = xyzh(4,i)
    !--as force.F90: dead and accreted particles (h <= 0) set no constraint.
    !  Left in, one of them makes dtc negative and dtf the sqrt of a negative
    !  number, which poisons the global timestep.
    if (isdead_or_accreted(hi)) cycle
    isgas = .true.
    if (use_dust) isgas = (iamtype(iphase(i)) == igas)

    !--as force.F90: no signal speed means no Courant constraint, not a
    !  division by zero.  Gas uses its own sound speed and alpha too; other particles
    !  (two-fluid dust) only the signal speed from their neighbours.
    dtc = dtmax
    if (isgas) then
       vsigdtc = max(vsigmax8(i),spsound_8(i))
       if (vsigdtc > tiny(vsigdtc)) dtc = C_cour*hi/(vsigdtc*max(alpha,1.0))
    else
       vsigdtc = vsigmax8(i)
       if (vsigdtc > tiny(vsigdtc)) dtc = C_cour*hi/vsigdtc
    endif

    !--as force.F90, after every heating and cooling term is in du/dt: change du/dt
    !  rather than let u + dtc*du/dt go negative
    if (limit_u .and. isgas) then
       eni = vxyzu(4,i)
       if (eni + dtc*fxyzu(4,i) < epsilon(0.) .and. eni > epsilon(0.)) then
          fxyzu(4,i) = fxyzu(4,i)/(1.-dtc*fxyzu(4,i)/eni)
       endif
    endif

    !--as force.F90, from the SPH force alone (before any driving force is added)
    f2i = fx8(i)*fx8(i) + &
          fy8(i)*fy8(i) + &
          fz8(i)*fz8(i)

    dtf = bignumber

    if (f2i > 0.0) then
       dtf = C_force*sqrt(hi/sqrt(f2i))
    endif

    !--as force.F90 for dust as particles: the drag timestep from the smallest pair
    !  stopping time, and the check that gas resolves it (h < c_s t_s)
    if (use_dust) then
       ts_min = min(real(tsmin8(i)),bignumber)
       if (drag_implicit) then
          dtdrag = 90.*ts_min
       else
          dtdrag = 0.9*ts_min
       endif
       dtf    = min(dtf,dtdrag)
       if (isgas .and. ts_min < bignumber) then
          if (hi > ts_min*spsound_8(i)) then
             ndustres       = ndustres + 1
             dustresfacmean = dustresfacmean + hi/(ts_min*spsound_8(i))
             dustresfacmax  = max(dustresfacmax, hi/(ts_min*real(spsound_8(i))))
          endif
       endif
    endif

    dtcmin = min(dtcmin,dtc)
    dtfmin = min(dtfmin,dtf)
 enddo
 !$omp end parallel do

 dtcourant = dtcmin
 dtforce   = dtfmin
 if (ndustres > 0) call summary_variable('dust',iosumdgr,ndustres,dustresfacmean/real(ndustres),dustresfacmax)

end subroutine finish_gpu_force_timesteps

end module gpu_force_iface
