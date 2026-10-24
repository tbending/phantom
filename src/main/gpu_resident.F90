!--------------------------------------------------------------------------!
! The Phantom Smoothed Particle Hydrodynamics code, by Daniel Price et al. !
! Copyright (c) 2007-2026 The Authors (see AUTHORS)                        !
! See LICENCE file for usage and distribution conditions                   !
! http://phantomsph.github.io/                                             !
!--------------------------------------------------------------------------!
module gpu_resident
!
! The device's own copy of the per-particle integrator state, and its twins of the
! loops in step that touch every particle every substep (cosmoSPHere resident.cu).
!
! Individual timesteps on the GPU, Phase 2 of GPU_IND_TIMESTEPS_PLAN.md: the device
! will own every particle's evolving state between dumps.  For now the host runs its
! own loops and stays the truth.  With COSMO_MIRROR=1 (GPU build, IND_TIMESTEPS):
!
!   - the whole state is sent once, at the first step;
!   - after each host loop that has a twin, the device runs the twin on its own copy
!     and the result is compared with the host's, particle by particle, bit for bit;
!     a field that differs is counted and re-sent, so later checks start level;
!   - after each host operation that has no twin yet (the substep with sinks or
!     external forces, the density and force passes, cons2prim, the new bins), the
!     fields it changed are re-sent.
!
! One line per step, COSMO_MIRROR, gives the mismatch counts; all zero is the gate.
! Without COSMO_MIRROR nothing here runs.
!
! :References: None
!
! :Owner: Not Committed Yet
!
! :Runtime parameters: None
!
! :Dependencies: dim, io, part, shock_capturing, iso_c_binding
!
 use iso_c_binding, only:c_double,c_int,c_ptr,c_size_t
 use gpu_arrays,    only:resident_feed,resident_mode,gpu_domain_on
 implicit none

 logical, public :: gpu_mirror = .false.
 logical, save   :: gpu_feed = .false.     ! COSMO_RESIDENT_FEED: the pair passes read the resident state
 logical, public :: gpu_resident_on = .false.  ! COSMO_RESIDENT: the device is the truth (implies both)
 public :: rs_init_switch,rs_start_step,rs_after_predictor,rs_after_substep
 public :: rs_after_predict_sph,rs_after_derivs,rs_after_corrector,rs_after_cons2prim
 public :: rs_after_force,rs_download_all
 public :: rs_after_kick,rs_after_subdrift,rs_after_get_force,rs_after_accretion,rs_before_gas_loop
 public :: rs_prime,rs_put_alpha1,rs_resync

 private

 !--field ids, as cosmoSPHere/src/resident.cu RsField
 integer, parameter :: rs_pos = 1, rs_h = 2, rs_v = 3, rs_vpred = 4, rs_f = 5, rs_twas = 6, &
                       rs_ibin = 7, rs_ibinold = 8, rs_ibinwake = 9, rs_active = 10, &
                       rs_divv = 11, rs_cs = 12, rs_alpha1 = 13, rs_alphaloc = 14, rs_fext = 15, &
                       rs_type = 16, rs_rho = 17, rs_gradh = 18, rs_xi = 19, rs_ddivvdt = 20, rs_pres = 21, &
                       rs_dustgas = 22, rs_iorig = 23, rs_last = 23

 !--checks counted per step: set_active, predictor, drift (plain and in the substep),
 !  predict_sph, corrector, kick, sink-gas force and accretion (in the substep), and the
 !  new bins after the force pass (GPU path)
 !  and the density and force passes' writebacks (GPU path)
 !  and cons2prim (GPU path; in ulps for ieos 3, whose pow may differ in the last bit)
 !  and the periodic wrap (with COSMO_RESIDENT_FEED, where the device wraps its own copy)
 integer, parameter :: nchecks = 13
 character(len=10), parameter :: checkname(nchecks) = &
    (/'setactive ','predictor ','drift     ','predictsph','corrector ','kick      ','sinkgas   ', &
      'accrete   ','newbins   ','denswb    ','forcewb   ','cons2prim ','wrap      '/)
 real, save :: maxulp_step = 0.   ! largest difference in ulps among tolerant comparisons
 integer(kind=1), save :: nbinmax_before_derivs = 0
 real, allocatable, save :: fsink_before(:,:)   ! fxyz_ptmass before get_force's gas loop
 integer(kind=8), save :: nbad(nchecks) = 0, nbadtot(nchecks) = 0
 integer, save :: nsync = 0          ! device Domain syncs so far (MPI)
 logical, save :: priming = .false.  ! rs_prime sent the state; the first step's bookkeeping is still to come
 !--MPI: a full Domain sync only when the coarsest active bin is within sync_levels of the
 !  finest (COSMO_SYNC_LEVELS, default 4: at most every 16 substeps), or the guard fires
 integer, save :: sync_levels = 4
 real(kind=8), save :: wall_syncstep = 0.d0   ! host wall in the sync decision + syncs (COSMO_DOMTIME)
#ifdef MPI
 interface
  integer(c_int) function cosmo_dom_need_sync() bind(C)
   import c_int
  end function cosmo_dom_need_sync
  subroutine cosmo_dom_exchange_after_force(full) bind(C)
   import c_int
   integer(c_int), value :: full
  end subroutine cosmo_dom_exchange_after_force
  subroutine cosmo_dom_report() bind(C)
  end subroutine cosmo_dom_report
  subroutine cosmo_dom_after_force_all() bind(C)
  end subroutine cosmo_dom_after_force_all
  subroutine cosmo_dom_check_halos(sub) bind(C)
   import c_int
   integer(c_int), value :: sub
  end subroutine cosmo_dom_check_halos
 end interface
#endif
 real,    save :: tsync_tot = 0.     ! and the seconds they took, this rank
 integer, save :: nrun(nchecks) = 0   ! comparisons made this step, per check
 integer, save :: nstep = 0
 logical, save :: started = .false., checked = .false.
 real(c_double), allocatable, target :: hbuf(:),dbuf(:)

#ifdef GPU
 interface
  subroutine cosmo_rs_init(n,nvu) bind(C)
   import :: c_int
   integer(c_int), value :: n,nvu
  end subroutine cosmo_rs_init
  subroutine cosmo_pin_host(ptr,nbytes) bind(C)
   import :: c_ptr,c_size_t
   type(c_ptr),       value :: ptr
   integer(c_size_t), value :: nbytes
  end subroutine cosmo_pin_host
  subroutine cosmo_unpin_host(ptr) bind(C)
   import :: c_ptr
   type(c_ptr), value :: ptr
  end subroutine cosmo_unpin_host
  subroutine cosmo_rs_use_slots(n) bind(C)
   import :: c_int
   integer(c_int), value :: n
  end subroutine cosmo_rs_use_slots
  subroutine cosmo_rs_put(field,host,n) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: field,n
   real(c_double), intent(in) :: host(*)
  end subroutine cosmo_rs_put
  subroutine cosmo_rs_get(field,host,n) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: field,n
   real(c_double), intent(out) :: host(*)
  end subroutine cosmo_rs_get
  integer(c_int) function cosmo_rs_set_active(n,istepfrac,nbinmax,nalive) bind(C)
   import :: c_int
   integer(c_int), value :: n,istepfrac,nbinmax
   integer(c_int), intent(out) :: nalive
  end function cosmo_rs_set_active
  subroutine cosmo_rs_predictor(n,timei,ufloor,icooling) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n,icooling
   real(c_double), value :: timei,ufloor
  end subroutine cosmo_rs_predictor
  subroutine cosmo_rs_kick(n,dkdt) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n
   real(c_double), value :: dkdt
  end subroutine cosmo_rs_kick
  real(c_double) function cosmo_rs_sink_gas(n,sink,nsink,massoftype,ntype,extrap,extrapfac,sinkacc, &
                                             dtfacphi2,dtphi2min) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n,nsink,ntype,extrap
   real(c_double), intent(in) :: sink(*),massoftype(*)
   real(c_double), value :: extrapfac,dtfacphi2
   real(c_double), intent(out) :: sinkacc(*),dtphi2min
  end function cosmo_rs_sink_gas
  integer(c_int) function cosmo_rs_accrete(n,sink,nsink,facc,nbinmax,radkern2,massoftype,ntype,dptmass) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n,nsink,nbinmax,ntype
   real(c_double), intent(in) :: sink(*),massoftype(*)
   real(c_double), value :: facc,radkern2
   real(c_double), intent(out) :: dptmass(*)
  end function cosmo_rs_accrete
  integer(c_int) function cosmo_rs_new_bins(n,dtmax,ccour,cforce,alpha,istepfrac,nbinmaxold, &
                                            ibinnow,usedust,limitu) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n,istepfrac,nbinmaxold,ibinnow,usedust,limitu
   real(c_double), value :: dtmax,ccour,cforce,alpha
  end function cosmo_rs_new_bins
  subroutine cosmo_rs_dens_writeback(n,massoftype,ntype,hfact,nalpha3,usedust) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n,ntype,nalpha3,usedust
   real(c_double), intent(in) :: massoftype(*)
   real(c_double), value :: hfact
  end subroutine cosmo_rs_dens_writeback
  subroutine cosmo_rs_force_writeback(n,usedust) bind(C)
   import :: c_int
   integer(c_int), value :: n,usedust
  end subroutine cosmo_rs_force_writeback
  subroutine cosmo_rs_cons2prim(n,ieos,polyk,gamma,qfacdisc,csmin,alphaloc,alpha,alphamax,sinkxyz) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n,ieos,alphaloc
   real(c_double), value :: polyk,gamma,qfacdisc,csmin,alpha,alphamax
   real(c_double), intent(in) :: sinkxyz(3)
  end subroutine cosmo_rs_cons2prim
  subroutine cosmo_rs_drift(n,dt) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n
   real(c_double), value :: dt
  end subroutine cosmo_rs_drift
  subroutine cosmo_rs_predict_sph(n,dtsph,timei,ufloor,alphamode,avdecay,alpha,alphamax) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n,alphamode
   real(c_double), value :: dtsph,timei,ufloor,avdecay,alpha,alphamax
  end subroutine cosmo_rs_predict_sph
  subroutine cosmo_rs_corrector(n,timei,ibin_dts,nbins,ithdt,ittwas,nbinmax,ufloor,icooling) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n,nbins,ithdt,ittwas,nbinmax,icooling
   real(c_double), value :: timei,ufloor
   real(c_double), intent(in) :: ibin_dts(*)
  end subroutine cosmo_rs_corrector
 end interface
#endif

contains

!-----------------------------------------------------------------------
!+
!  COSMO_MIRROR from the environment, once.  Only with a GPU build and
!  individual timesteps.
!+
!-----------------------------------------------------------------------
subroutine rs_init_switch()
 use dim, only:ind_timesteps
 use io,  only:iprint
 character(len=16) :: val
 integer :: ln,ierr
 logical, save :: switch_done = .false.

 if (checked) return
 checked = .true.
#ifdef GPU
 if (switch_done) return
 switch_done = .true.
 call get_environment_variable('COSMO_MIRROR',val,ln,ierr)
 gpu_mirror = (ierr == 0 .and. ln > 0 .and. ind_timesteps)
 if (gpu_mirror) write(iprint,'(a)') ' cosmoSPHere: COSMO_MIRROR, device twins of the step checked against the host'
 call get_environment_variable('COSMO_RESIDENT',val,ln,ierr)
 gpu_resident_on = (ierr == 0 .and. ln > 0 .and. ind_timesteps)
 if (gpu_resident_on) then
    gpu_mirror = .true.
    write(iprint,'(a)') ' cosmoSPHere: COSMO_RESIDENT, the device is the truth between dumps'
 endif
 call get_environment_variable('COSMO_SYNC_LEVELS',val,ln,ierr)
 if (ierr == 0 .and. ln > 0) read(val,*,iostat=ierr) sync_levels
 call get_environment_variable('COSMO_RESIDENT_FEED',val,ln,ierr)
 gpu_feed = gpu_mirror .and. ((ierr == 0 .and. ln > 0) .or. gpu_resident_on)
 if (gpu_feed) write(iprint,'(a)') ' cosmoSPHere: COSMO_RESIDENT_FEED, the GPU passes read the resident state'
#endif

end subroutine rs_init_switch

!--every particle is gas or dust (which also means no boundary particles); the
!  twins of get_force and accretion assume it
logical function only_gas_and_dust()
 use part, only:npartoftype,igas,idust,idustlast
 integer :: it
 only_gas_and_dust = .true.
 do it = 1,size(npartoftype)
    if (it == igas .or. (it >= idust .and. it <= idustlast)) cycle
    if (npartoftype(it) > 0) only_gas_and_dust = .false.
 enddo
end function only_gas_and_dust

!--number of velocity components: 3, or 4 with u
integer function nvu()
 use dim, only:maxvxyzu
 nvu = min(maxvxyzu,4)
end function nvu

subroutine grow(n)
#ifdef GPU
 use iso_c_binding, only:c_loc,c_size_t,c_sizeof
#endif
 integer, intent(in) :: n
 integer :: nalloc
 nalloc = n
 if (allocated(hbuf)) then
    if (size(hbuf) >= 4*n) return
#ifdef GPU
    call cosmo_unpin_host(c_loc(hbuf(1)))
    call cosmo_unpin_host(c_loc(dbuf(1)))
#endif
    deallocate(hbuf,dbuf)
    !--a regrow (MPI: the owned count drifts between syncs): headroom, as re-pinning two
    !  4n-double buffers costs ~250 ms on 4M particles
    nalloc = n + n/8
 endif
 allocate(hbuf(4*nalloc),dbuf(4*nalloc))
#ifdef GPU
 !--registered with the driver: the whole-state copies (.ev writes, dumps) go at pinned
 !  speed rather than through pageable staging (19.2M particles: ~17 ms per field)
 call cosmo_pin_host(c_loc(hbuf(1)),int(size(hbuf),kind=c_size_t)*c_sizeof(hbuf(1)))
 call cosmo_pin_host(c_loc(dbuf(1)),int(size(dbuf),kind=c_size_t)*c_sizeof(dbuf(1)))
#endif
end subroutine grow

!-----------------------------------------------------------------------
!+
!  Pack a host field into hbuf as ncomp runs of npart doubles
!+
!-----------------------------------------------------------------------
subroutine pack(field,npart,ncomp)
 use part, only:xyzh,vxyzu,vpred,fxyzu,twas,ibin,ibin_old,ibin_wake,iphase,iactive, &
                divcurlv,eos_vars,ics,alphaind,fext,iamtype,rho,gradh,igasP,dustfrac,ndustsmall,iorig
 use gpu_dens_iface, only:xi_gpu
 use dim,  only:nalpha
 integer, intent(in)  :: field,npart
 integer, intent(out) :: ncomp
 integer :: i,c

 call grow(npart)
 ncomp = 1
 select case(field)
 case(rs_pos)
    ncomp = 3
    do c = 1,3
       hbuf((c-1)*npart+1:c*npart) = xyzh(c,1:npart)
    enddo
 case(rs_h)
    hbuf(1:npart) = xyzh(4,1:npart)
 case(rs_v,rs_vpred,rs_f)
    ncomp = nvu()
    do c = 1,ncomp
       select case(field)
       case(rs_v);     hbuf((c-1)*npart+1:c*npart) = vxyzu(c,1:npart)
       case(rs_vpred); hbuf((c-1)*npart+1:c*npart) = vpred(c,1:npart)
       case(rs_f);     hbuf((c-1)*npart+1:c*npart) = fxyzu(c,1:npart)
       end select
    enddo
 case(rs_twas)
    hbuf(1:npart) = twas(1:npart)
 case(rs_ibin)
    hbuf(1:npart) = real(ibin(1:npart),kind=c_double)
 case(rs_ibinold)
    hbuf(1:npart) = real(ibin_old(1:npart),kind=c_double)
 case(rs_ibinwake)
    hbuf(1:npart) = real(ibin_wake(1:npart),kind=c_double)
 case(rs_active)
    do i = 1,npart
       hbuf(i) = merge(1._c_double,0._c_double,iactive(iphase(i)))
    enddo
 case(rs_divv)
    hbuf(1:npart) = real(divcurlv(1,1:npart),kind=c_double)
 case(rs_cs)
    hbuf(1:npart) = eos_vars(ics,1:npart)
 case(rs_alpha1)
    hbuf(1:npart) = real(alphaind(1,1:npart),kind=c_double)
 case(rs_alphaloc)
    hbuf(1:npart) = real(alphaind(min(2,size(alphaind,1)),1:npart),kind=c_double)
 case(rs_pres)
    hbuf(1:npart) = eos_vars(igasP,1:npart)
 case(rs_rho)
    hbuf(1:npart) = rho(1:npart)
 case(rs_gradh)
    ncomp = 2
    hbuf(1:npart)         = real(gradh(1,1:npart),kind=c_double)
    hbuf(npart+1:2*npart) = real(gradh(2,1:npart),kind=c_double)
 case(rs_xi)
    if (allocated(xi_gpu)) then
       if (size(xi_gpu) >= npart) then
          hbuf(1:npart) = xi_gpu(1:npart)
       else
          hbuf(1:npart) = 1.
       endif
    else
       hbuf(1:npart) = 1.
    endif
 case(rs_ddivvdt)
    if (nalpha >= 3) then
       hbuf(1:npart) = real(alphaind(3,1:npart),kind=c_double)
    else
       hbuf(1:npart) = 0.
    endif
 case(rs_dustgas)
    hbuf(1:npart) = dustfrac(ndustsmall+1,1:npart)
 case(rs_type)
    do i = 1,npart
       hbuf(i) = real(iamtype(iphase(i)),kind=c_double)
    enddo
 case(rs_iorig)
    hbuf(1:npart) = real(iorig(1:npart),kind=c_double)
 case(rs_fext)
    ncomp = 3
    do c = 1,3
       hbuf((c-1)*npart+1:c*npart) = fext(c,1:npart)
    enddo
 end select

end subroutine pack

!--send one host field to the device
subroutine put(field,npart,force)
 use io, only:fatal
 integer, intent(in) :: field,npart
 logical, intent(in), optional :: force   ! the host's copy is the truth for this field (priming)
 integer :: ncomp
 !--with the device the truth there is no host copy to send: a field reaching here has
 !  no twin in this configuration
 if (resident_mode .and. .not.present(force)) &
    call fatal('gpu_resident','COSMO_RESIDENT: this field has no device twin here',ival=field)
 call pack(field,npart,ncomp)
#ifdef GPU
 call cosmo_rs_put(int(field,c_int),hbuf,int(npart,c_int))
#endif
end subroutine put

!-----------------------------------------------------------------------
!+
!  Compare one device field with the host's, live particles only, bit
!  for bit.  Adds the mismatching particles to check icheck's count and
!  re-sends the field if any.
!+
!-----------------------------------------------------------------------
subroutine compare(icheck,field,npart,maxulp)
 use part, only:xyzh,isdead_or_accreted
 integer, intent(in) :: icheck,field,npart
 real,    intent(in), optional :: maxulp   ! allow this many ulps (of the host's value)
 integer :: ncomp,i,c
 integer(kind=8) :: nb
 real :: tol,ulps,ulpmax
 logical :: single   ! the field is real*4 in phantom: ulps at single precision

 if (resident_mode) then   ! nothing to compare with: the host copy is stale by design
    nrun(icheck) = nrun(icheck) + 1
    return
 endif

 call pack(field,npart,ncomp)
#ifdef GPU
 call cosmo_rs_get(int(field,c_int),dbuf,int(npart,c_int))
#endif
 nb = 0
 tol = 0.
 if (present(maxulp)) tol = maxulp
 ulpmax = 0.
 single = any(field == (/rs_divv,rs_alpha1,rs_alphaloc,rs_gradh,rs_ddivvdt/))
 !$omp parallel do default(none) private(i,c,ulps) shared(npart,ncomp,hbuf,dbuf,xyzh,tol,single) &
 !$omp reduction(+:nb) reduction(max:ulpmax)
 do i = 1,npart
    if (isdead_or_accreted(xyzh(4,i))) cycle
    do c = 1,ncomp
       if (hbuf((c-1)*npart+i) /= dbuf((c-1)*npart+i)) then
          if (single) then
             ulps = abs(hbuf((c-1)*npart+i) - dbuf((c-1)*npart+i))/spacing(real(hbuf((c-1)*npart+i),kind=4))
          else
             ulps = abs(hbuf((c-1)*npart+i) - dbuf((c-1)*npart+i))/spacing(hbuf((c-1)*npart+i))
          endif
          ulpmax = max(ulpmax,ulps)
          if (ulps > tol) then
             nb = nb + 1
             exit
          endif
       endif
    enddo
 enddo
 !$omp end parallel do
 if (present(maxulp)) maxulp_step = max(maxulp_step,ulpmax)
 nbad(icheck) = nbad(icheck) + nb
 nrun(icheck) = nrun(icheck) + 1
#ifdef GPU
 if (nb > 0) call cosmo_rs_put(int(field,c_int),hbuf,int(npart,c_int))
#endif

end subroutine compare

!-----------------------------------------------------------------------
!+
!  Start of a step, after set_active_particles: the first time, send the
!  whole state; then the activity twin, compared with iphase.
!+
!-----------------------------------------------------------------------
subroutine rs_start_step(npart,istepfrac,nbinmax,nactive,nalive,nactivetot,nalivetot)
 use timestep_ind, only:ibinnow
 use dim,          only:use_dust
 use part,         only:all_active
 use mpiutils,     only:reduceall_mpi
 integer,         intent(in)    :: npart,istepfrac
 integer(kind=1), intent(in)    :: nbinmax
 integer,         intent(inout) :: nactive,nalive
 integer(kind=8), intent(inout) :: nactivetot,nalivetot
 integer :: ncount,i
 integer(c_int) :: nalive_dev

 call rs_init_switch()
 if (.not.gpu_mirror .or. npart <= 0) return
 nbad  = 0
 nrun  = 0
 maxulp_step = 0.
 nstep = nstep + 1
#ifdef GPU
 if (.not.started) call rs_first_send(npart,.false.)
 if (priming) then
    !--primed in initial (MPI): the device has the density and force with halos, but the
    !  bins and twas are the host's, as the unprimed first send has them: at t = 0 phantom
    !  puts every particle on the finest bin (step_leapfrog init_step), ibin_old and
    !  ibin_wake are 0 and twas is untouched, whatever the new-bins twin did in initial
    call put(rs_twas,npart,force=.true.)
    call put(rs_ibin,npart,force=.true.)
    call put(rs_ibinold,npart,force=.true.)
    call put(rs_ibinwake,npart,force=.true.)
#ifdef MPI
    !--the puts reach owned slots only: the same fields to every halo copy (a full exchange)
    if (gpu_domain_on) call cosmo_dom_exchange_after_force(1_c_int)
#endif
    priming = .false.
 endif
 ncount = cosmo_rs_set_active(int(npart,c_int),int(istepfrac,c_int),int(nbinmax,c_int),nalive_dev)
 call compare(1,rs_active,npart)
 if (resident_mode) then
    !--the host skipped set_active_particles: its counts and ibinnow from here
    nactive = ncount; nalive = nalive_dev
    !--the totals over every rank, as set_active_particles does: evolve decides from them when
    !  to write the .ev file (collective), and rank-local counts let the ranks disagree and
    !  deadlock (1M disc, P = 2: one rank in compute_energies, the other in the next substep)
    nactivetot = reduceall_mpi('+',ncount); nalivetot = reduceall_mpi('+',int(nalive_dev))
    ibinnow = nbinmax
    i = 0
    do while (ibinnow == nbinmax .and. i < nbinmax)
       if (mod(istepfrac,2**(nbinmax-i))==0) ibinnow = int(i,kind=1)
       i = i + 1
    enddo
    all_active = (nactive == nalive)
 endif
 !--from the step after the one that sent the state, the device is the truth
 if (gpu_resident_on .and. started) resident_mode = .true.
#endif

end subroutine rs_start_step

!--after the predictor: v to the half step, ibin_old of the active
subroutine rs_after_predictor(npart,timei,ufloor,icooling)
 integer, intent(in) :: npart,icooling
 real,    intent(in) :: timei,ufloor
 if (.not.gpu_mirror .or. npart <= 0) return
#ifdef GPU
 call cosmo_rs_predictor(int(npart,c_int),real(timei,c_double),real(ufloor,c_double),int(icooling,c_int))
 call compare(2,rs_v,npart)
 call compare(2,rs_ibinold,npart)
#endif
end subroutine rs_after_predictor

!-----------------------------------------------------------------------
!+
!  After the substep.  The plain drift (substep_sph) has a twin; the
!  substep with sinks, external forces, damping or cooling does not yet,
!  so what it changes (positions, velocities, h of accreted particles)
!  is re-sent.
!+
!-----------------------------------------------------------------------
subroutine rs_after_substep(npart,dtsph,plain)
 integer, intent(in) :: npart
 real,    intent(in) :: dtsph
 logical, intent(in) :: plain
 if (.not.gpu_mirror .or. npart <= 0) return
#ifdef GPU
 !--the substep with sinks or external forces is twinned inside it (kick, drift) and
 !  re-sent where it is not (get_force, accretion)
 if (plain) then
    call cosmo_rs_drift(int(npart,c_int),real(dtsph,c_double))
    call compare(3,rs_pos,npart)
 endif
#endif
end subroutine rs_after_substep

!--inside the substep, after kick: v += dkdt fext
subroutine rs_after_kick(npart,dkdt)
 integer, intent(in) :: npart
 real,    intent(in) :: dkdt
 if (.not.gpu_mirror .or. npart <= 0) return
#ifdef GPU
 call cosmo_rs_kick(int(npart,c_int),real(dkdt,c_double))
 call compare(6,rs_v,npart)
#endif
end subroutine rs_after_kick

!--inside the substep, after drift: x += ckdt v
subroutine rs_after_subdrift(npart,ckdt)
 integer, intent(in) :: npart
 real,    intent(in) :: ckdt
 if (.not.gpu_mirror .or. npart <= 0) return
#ifdef GPU
 call cosmo_rs_drift(int(npart,c_int),real(ckdt,c_double))
 call compare(3,rs_pos,npart)
#endif
end subroutine rs_after_subdrift

!--get_force, before its gas loop: the sinks' acceleration so far (sink-sink), so that
!  the particles' share of it can be compared with the twin's
subroutine rs_before_gas_loop(nptmass,fxyz_ptmass)
 integer, intent(in) :: nptmass
 real,    intent(in) :: fxyz_ptmass(:,:)
 if (.not.gpu_mirror .or. nptmass <= 0) return
 if (allocated(fsink_before)) deallocate(fsink_before)
 allocate(fsink_before(3,nptmass))
 fsink_before = fxyz_ptmass(1:3,1:nptmass)
end subroutine rs_before_gas_loop

!-----------------------------------------------------------------------
!+
!  After get_force.  The sink-gas force has a twin in the plain case: one
!  particle type, sinks not in the tree, isink_potential 0, no J2, no
!  regularised n-body, no external force, damping or sink radiation, no
!  GR.  It is compared on fext and on the largest fonrmax (dtsinkgas).
!  Anything else, fext is re-sent.
!+
!-----------------------------------------------------------------------
subroutine rs_after_get_force(npart,ntypes,nptmass,xyzmh_ptmass,fxyz_ptmass,fonrmax,dtphi2,extrapfac,fsink_old)
 use dim,              only:use_sinktree,gr
 use options,          only:iexternalforce
 use damping,          only:idamp
 use ptmass,           only:isink_potential,use_regnbody,dtfacphi2
 use ptmass_radiation, only:isink_radiation
 use part,             only:ihsoft,iJ2,massoftype,igas,maxtypes
 integer, intent(in)    :: npart,ntypes,nptmass
 real,    intent(in)    :: xyzmh_ptmass(:,:)
 real,    intent(inout) :: fxyz_ptmass(:,:),fonrmax,dtphi2
 real,    intent(in), optional :: extrapfac,fsink_old(:,:)
 real(c_double), allocatable :: sink(:),sinkacc(:)
 real(c_double) :: fonrmax_dev,dtphi2_dev
 real    :: fhost(3),scale
 integer :: j
 logical :: twin

 if (.not.gpu_mirror .or. npart <= 0) return
 twin = (nptmass > 0 .and. only_gas_and_dust() .and. .not.use_sinktree .and. .not.gr .and. &
         iexternalforce == 0 .and. idamp == 0 .and. isink_radiation == 0 .and. &
         isink_potential == 0 .and. .not.use_regnbody)
 if (twin) twin = all(abs(xyzmh_ptmass(iJ2,1:nptmass)) <= 0.)
 if (.not.twin) then
    call put(rs_fext,npart)
    return
 endif
 allocate(sink(8*nptmass),sinkacc(6*nptmass))
 sink = 0.
 do j = 1,nptmass
    sink(8*j-7:8*j-3) = (/xyzmh_ptmass(1:4,j),xyzmh_ptmass(ihsoft,j)/)
    if (present(fsink_old)) sink(8*j-2:8*j) = fsink_old(1:3,j)
 enddo
#ifdef GPU
 if (present(extrapfac)) then
    fonrmax_dev = cosmo_rs_sink_gas(int(npart,c_int),sink,int(nptmass,c_int), &
                                    real(massoftype(1:maxtypes),c_double),int(maxtypes,c_int), &
                                    1_c_int,real(extrapfac,c_double),sinkacc,real(dtfacphi2,c_double),dtphi2_dev)
 else
    fonrmax_dev = cosmo_rs_sink_gas(int(npart,c_int),sink,int(nptmass,c_int), &
                                    real(massoftype(1:maxtypes),c_double),int(maxtypes,c_int), &
                                    0_c_int,0._c_double,sinkacc,real(dtfacphi2,c_double),dtphi2_dev)
 endif
 if (resident_mode) then
    !--the host skipped its gas loop: the particles' pull on the sinks, and the sink-gas
    !  timestep's inputs, from the device
    do j = 1,nptmass
       fxyz_ptmass(1:3,j) = fxyz_ptmass(1:3,j) + sinkacc(6*j-5:6*j-3)
    enddo
    fonrmax = max(fonrmax,fonrmax_dev)
    dtphi2  = min(dtphi2,dtphi2_dev)
    call compare(7,rs_fext,npart)
    return
 endif
 call compare(7,rs_fext,npart)
 if (fonrmax_dev /= fonrmax) nbad(7) = nbad(7) + 1
 if (dtphi2_dev /= dtphi2)   nbad(7) = nbad(7) + 1
 if (allocated(fsink_before)) then
    if (size(fsink_before,2) == nptmass) then
       do j = 1,nptmass
          fhost = fxyz_ptmass(1:3,j) - fsink_before(:,j)
          scale = max(maxval(sinkacc(6*j-2:6*j)),tiny(scale))
          if (maxval(abs(sinkacc(6*j-5:6*j-3) - fhost)) > 1.e-12*scale) nbad(7) = nbad(7) + 1
       enddo
    endif
 endif
#else
 fonrmax_dev = 0.
#endif

end subroutine rs_after_get_force

!-----------------------------------------------------------------------
!+
!  Accretion, after the host's particle loop (skipped with COSMO_RESIDENT)
!  and before it updates the sinks.  The particle's side has a twin (h =
!  -|h| for the accreted, the wake flags) and the device also sums what
!  each sink takes (dptmass): one particle type or gas and dust, at most
!  nfastacc sinks, no external-force accretion, no star formation, no APR.
!  Mirror: the count exactly, dptmass to 1e-12 relative.  Resident: the
!  host gets the device's count and sums.
!+
!-----------------------------------------------------------------------
subroutine rs_after_accretion(npart,ntypes,nptmass,xyzmh_ptmass,pxyz_ptmass,dptmass,naccreted,nlive,nbinmax)
 use dim,         only:use_apr
 use options,     only:iexternalforce
 use ptmass,      only:icreate_sinks,f_acc
 use ptmass_tree, only:nfastacc
 use part,        only:ihacc,massoftype,maxtypes
 use kernel,      only:radkern2
 integer,         intent(in)    :: npart,ntypes,nptmass
 real,            intent(in)    :: xyzmh_ptmass(:,:),pxyz_ptmass(:,:)
 real,            intent(inout) :: dptmass(:,:)
 integer,         intent(inout) :: naccreted,nlive
 integer(kind=1), intent(in), optional :: nbinmax
 real(c_double), allocatable :: sink(:),dd(:)
 integer :: j,nacc
 real    :: scale
 logical :: twin

 if (.not.gpu_mirror .or. npart <= 0) return
 twin = (nptmass > 0 .and. nptmass <= nfastacc .and. only_gas_and_dust() .and. iexternalforce == 0 &
         .and. icreate_sinks /= 2 .and. .not.use_apr .and. present(nbinmax))
 if (.not.twin) then
    call put(rs_h,npart)
    call put(rs_ibinwake,npart)
    return
 endif
 allocate(sink(8*nptmass),dd(13*nptmass))
 do j = 1,nptmass
    sink(8*j-7:8*j) = (/xyzmh_ptmass(1:4,j),xyzmh_ptmass(ihacc,j),pxyz_ptmass(1:3,j)/)
 enddo
#ifdef GPU
 nacc = cosmo_rs_accrete(int(npart,c_int),sink,int(nptmass,c_int),real(f_acc,c_double), &
                         int(nbinmax,c_int),real(radkern2,c_double), &
                         real(massoftype(1:maxtypes),c_double),int(maxtypes,c_int),dd)
 call compare(8,rs_h,npart)
 call compare(8,rs_ibinwake,npart)
 if (resident_mode) then
    naccreted = nacc
    nlive     = npart    ! only feeds the host's 'all particles accreted' check
    do j = 1,nptmass
       dptmass(1:13,j) = dd(13*j-12:13*j)
    enddo
    return
 endif
 if (nacc /= naccreted) nbad(8) = nbad(8) + 1
 do j = 1,nptmass
    scale = max(maxval(abs(dptmass(1:13,j))),tiny(scale))
    if (maxval(abs(dd(13*j-12:13*j) - dptmass(1:13,j))) > 1.e-12*scale) nbad(8) = nbad(8) + 1
 enddo
#else
 nacc = 0
#endif

end subroutine rs_after_accretion

!--after predict_sph: h, the predicted velocity, the viscosity switch
subroutine rs_after_predict_sph(npart,dtsph,timei,ufloor)
 use dim,             only:maxalpha,maxp,nalpha
 use shock_capturing, only:avdecayconst,alpha,alphamax
 use timestep_ind,    only:nbinmax,ibinnow
 integer, intent(inout) :: npart   ! MPI: the rank's owned count after the sync
 real,    intent(in) :: dtsph,timei,ufloor
 integer(kind=8) :: wc0,wc1,wcrate
 integer :: amode
 if (.not.gpu_mirror .or. npart <= 0) return
 !--derivs is next, and the force pass changes nbinmax; the new-bins twin needs it before
 nbinmax_before_derivs = nbinmax
 amode = 0
 if (maxalpha == maxp) amode = merge(2,1,nalpha >= 2)
#ifdef GPU
 call cosmo_rs_predict_sph(int(npart,c_int),real(dtsph,c_double),real(timei,c_double), &
                           real(ufloor,c_double),int(amode,c_int),real(avdecayconst,c_double), &
                           real(alpha,c_double),real(alphamax,c_double))
 call compare(4,rs_h,npart)
 call compare(4,rs_vpred,npart)
 if (amode > 0) call compare(4,rs_alpha1,npart)
#endif
#ifdef MPI
 !--the positions have drifted and derivs is next: re-decompose and find the halos when the
 !  coarse bins are active (the coarsest active bin is within sync_levels of the finest),
 !  or when the displacement guard says the halo margin may no longer hold
 if (gpu_domain_on .and. resident_mode) then
    call system_clock(wc0,wcrate)
    if (int(ibinnow) <= int(nbinmax) - sync_levels) then
       call dom_sync(npart,.false.)
    elseif (cosmo_dom_need_sync() /= 0) then
       call dom_sync(npart,.false.)
    else
       call cosmo_dom_check_halos(int(nstep,c_int))   ! COSMO_DOMAIN_DEBUG=3 only
    endif
    call system_clock(wc1)
    wall_syncstep = wall_syncstep + real(wc1-wc0,kind=8)/real(wcrate,kind=8)
 endif
#endif
end subroutine rs_after_predict_sph

!-----------------------------------------------------------------------
!+
!  In derivs, on the GPU path, straight after the force pass (derivs sets
!  dtnew from nbinmax next): the force writeback twin, unless the host
!  adds to the force (driving, cooling in the force pass), and the new
!  bins, wake flags and du/dt limit.  Resident: the host's nbinmax is the
!  device's.
!+
!-----------------------------------------------------------------------
subroutine rs_after_force(npart)
 use mpiutils, only:reduceall_mpi
 use dim,             only:use_dust,maxvxyzu,gr,driving
 use timestep,        only:dtmax,C_cour,C_force
 use timestep_ind,    only:nbinmax,istepfrac,ibinnow
 use shock_capturing, only:alpha
 use options,         only:icooling
 use cooling,         only:cooling_in_step
 use eos,             only:ieos
 integer, intent(in) :: npart
#ifdef MPI
 interface
  subroutine cosmo_dom_reverse_wake() bind(C)
  end subroutine cosmo_dom_reverse_wake
 end interface
#endif
 integer :: nbm
 logical :: force_twin,limit_u
 if (.not.gpu_mirror .or. npart <= 0 .or. .not.started) return
#ifdef GPU
 force_twin = .not.driving .and. .not.(maxvxyzu >= 4 .and. icooling > 0 .and. .not.cooling_in_step)
 if (force_twin) then
    call cosmo_rs_force_writeback(int(npart,c_int),merge(1_c_int,0_c_int,use_dust))
 else
    call put(rs_f,npart)
    call put(rs_divv,npart)
 endif
 limit_u = (maxvxyzu >= 4 .and. .not.gr .and. ieos /= 23 .and. force_twin)
 nbm = cosmo_rs_new_bins(int(npart,c_int),real(dtmax,c_double),real(C_cour,c_double), &
                         real(C_force,c_double),real(alpha,c_double),int(istepfrac,c_int), &
                         int(nbinmax_before_derivs,c_int),int(ibinnow,c_int), &
                         merge(1_c_int,0_c_int,use_dust),merge(1_c_int,0_c_int,limit_u))
 call compare(9,rs_ibin,npart)
 call compare(9,rs_ibinwake,npart)
#ifdef MPI
 !--the wake-up flags set on halo copies back to their owners, then the owners' new forces,
 !  bins and flags to the copies, before the corrector reads them
 !  (one call: the reverse wake and the forces share an MPI round, then the forward wake)
 if (gpu_domain_on) call cosmo_dom_after_force_all()
 if (gpu_domain_on) call cosmo_dom_check_halos(int(10*nstep+1,c_int))   ! COSMO_DOMAIN_DEBUG=3 only
#endif
 ! the bins are global: every rank must take the same substeps (MPI; no-op otherwise)
 nbm = int(reduceall_mpi('max',int(nbm)),kind=c_int)
 if (resident_mode) then
    nbinmax = int(nbm,kind=1)
 elseif (nbm /= int(nbinmax)) then
    nbad(9) = nbad(9) + 1
 endif
 if (force_twin) then
    call compare(11,rs_f,npart)
    call compare(11,rs_divv,npart)
 endif
#endif
end subroutine rs_after_force

!-----------------------------------------------------------------------
!+
!  After derivs.  On the GPU path the twins ran inside it (rs_after_cons2prim,
!  rs_after_force).  On the CPU path its passes have no twins: re-send what
!  they changed.
!+
!-----------------------------------------------------------------------
subroutine rs_after_derivs(npart)
 use dim,             only:maxalpha,maxp,nalpha
 use gpu_dens_iface,  only:use_gpu_dens
 integer, intent(in) :: npart
 if (.not.gpu_mirror .or. npart <= 0) return
 if (use_gpu_dens) return
 call put(rs_pos,npart)
 call put(rs_cs,npart)
 call put(rs_pres,npart)
 if (maxalpha == maxp .and. nalpha >= 2) call put(rs_alphaloc,npart)
 call put(rs_h,npart)
 call put(rs_rho,npart)
 call put(rs_gradh,npart)
 call put(rs_f,npart)
 call put(rs_divv,npart)
 if (nalpha >= 3) call put(rs_ddivvdt,npart)
 call put(rs_ibin,npart)
 call put(rs_ibinwake,npart)
end subroutine rs_after_derivs

!-----------------------------------------------------------------------
!+
!  COSMO_RESIDENT: copy the whole resident state to the host arrays, for
!  the .ev file and dumps.  The host's copy is then current; nothing is
!  sent back, as the device copy is the same.
!+
!-----------------------------------------------------------------------
subroutine rs_download_all(npart)
 use dim,  only:maxalpha,maxp,nalpha,use_dust
 use part, only:recount_npartoftype
 use io,   only:id
 integer, intent(in) :: npart
 integer :: f,nf
 integer(kind=8) :: ic0,ic1,ica,icb,icopy,crate
 character(len=8) :: statsenv
 if (.not.resident_mode .or. npart <= 0) return
#ifdef GPU
 call system_clock(ic0,crate)
 icopy = 0
 nf = 0
 do f = rs_pos,rs_last
    if ((f == rs_type .or. f == rs_iorig) .and. .not.gpu_domain_on) cycle
    if (f == rs_dustgas .and. .not.use_dust) cycle
    if (f == rs_alpha1 .and. maxalpha /= maxp) cycle
    if (f == rs_alphaloc .and. .not.(maxalpha == maxp .and. nalpha >= 2)) cycle
    if (f == rs_ddivvdt .and. nalpha < 3) cycle
    call system_clock(ica)
    call cosmo_rs_get(int(f,c_int),dbuf,int(npart,c_int))
    call system_clock(icb)
    icopy = icopy + (icb - ica)
    call unpack(f,npart)
    nf = nf + 1
 enddo
 call system_clock(ic1)
 !--particles have changed rank since the read: the per-type counts in the dump header
 if (gpu_domain_on) call recount_npartoftype
 call get_environment_variable('COSMO_DENS_STATS',statsenv)
 if (len_trim(statsenv) > 0) write(0,'(a,i3,a,f10.2,a,f10.2)') 'COSMO_DOWNLOAD_ALL fields=',nf, &
    ' ms=',1.e3*real(ic1-ic0)/real(crate),' of which copies=',1.e3*real(icopy)/real(crate)
 if (len_trim(statsenv) > 0 .and. gpu_domain_on) write(0,'(a,i8,a,f10.3)') 'COSMO_DOMAIN syncs=',nsync, &
    ' s=',tsync_tot
#ifdef MPI
 call get_environment_variable('COSMO_DOMTIME',statsenv)
 if (len_trim(statsenv) > 0 .and. gpu_domain_on) call cosmo_dom_report()
 if (len_trim(statsenv) > 0 .and. gpu_domain_on) write(0,'(a,i4,a,f9.3,a)') 'COSMO_SYNCWALL rank',id, &
    ' sync decision + syncs, host wall ',wall_syncstep,' s'
#endif
#endif
end subroutine rs_download_all

!--the inverse of pack: dbuf (ncomp runs of npart doubles) into the host field
subroutine unpack(field,npart)
 use part, only:xyzh,vxyzu,vpred,fxyzu,twas,ibin,ibin_old,ibin_wake,iphase,iamtype, &
                divcurlv,eos_vars,ics,alphaind,fext,rho,gradh,igasP,isetphase, &
                dustfrac,ndustsmall,iorig,iactive
 use gpu_dens_iface, only:xi_gpu
 integer, intent(in) :: field,npart
 integer :: i,c,ncomp
 logical :: hasxi

 !--every case is a parallel loop over particles: at 19.2M particles the whole-state
 !  refresh is ~20 fields of strided writes, and serial it cost ~2 s per .ev write
 call grow(npart)
 ncomp = nvu()
 hasxi = .false.
 if (allocated(xi_gpu)) hasxi = (size(xi_gpu) >= npart)
 !$omp parallel default(shared) private(i,c)
 select case(field)
 case(rs_pos)
    !$omp do schedule(static)
    do i = 1,npart
       xyzh(1,i) = dbuf(i); xyzh(2,i) = dbuf(npart+i); xyzh(3,i) = dbuf(2*npart+i)
    enddo
    !$omp end do
 case(rs_h)
    !$omp do schedule(static)
    do i = 1,npart
       xyzh(4,i) = dbuf(i)
    enddo
    !$omp end do
 case(rs_v)
    !$omp do schedule(static)
    do i = 1,npart
       do c = 1,ncomp
          vxyzu(c,i) = dbuf((c-1)*npart+i)
       enddo
    enddo
    !$omp end do
 case(rs_vpred)
    !$omp do schedule(static)
    do i = 1,npart
       do c = 1,ncomp
          vpred(c,i) = dbuf((c-1)*npart+i)
       enddo
    enddo
    !$omp end do
 case(rs_f)
    !$omp do schedule(static)
    do i = 1,npart
       do c = 1,ncomp
          fxyzu(c,i) = dbuf((c-1)*npart+i)
       enddo
    enddo
    !$omp end do
 case(rs_twas)
    !$omp do schedule(static)
    do i = 1,npart
       twas(i) = dbuf(i)
    enddo
    !$omp end do
 case(rs_ibin)
    !$omp do schedule(static)
    do i = 1,npart
       ibin(i) = int(nint(dbuf(i)),kind=1)
    enddo
    !$omp end do
 case(rs_ibinold)
    !$omp do schedule(static)
    do i = 1,npart
       ibin_old(i) = int(nint(dbuf(i)),kind=1)
    enddo
    !$omp end do
 case(rs_ibinwake)
    !$omp do schedule(static)
    do i = 1,npart
       ibin_wake(i) = int(nint(dbuf(i)),kind=1)
    enddo
    !$omp end do
 case(rs_active)
    !$omp do schedule(static)
    do i = 1,npart
       iphase(i) = isetphase(iamtype(iphase(i)),iactive=(dbuf(i) > 0.5_c_double))
    enddo
    !$omp end do
 case(rs_type)
    !$omp do schedule(static)
    do i = 1,npart
       iphase(i) = isetphase(int(nint(dbuf(i))),iactive=iactive(iphase(i)))
    enddo
    !$omp end do
 case(rs_iorig)
    !$omp do schedule(static)
    do i = 1,npart
       iorig(i) = nint(dbuf(i),kind=8)
    enddo
    !$omp end do
 case(rs_divv)
    !$omp do schedule(static)
    do i = 1,npart
       divcurlv(1,i) = real(dbuf(i),kind=4)
    enddo
    !$omp end do
 case(rs_cs)
    !$omp do schedule(static)
    do i = 1,npart
       eos_vars(ics,i) = dbuf(i)
    enddo
    !$omp end do
 case(rs_alpha1)
    !$omp do schedule(static)
    do i = 1,npart
       alphaind(1,i) = real(dbuf(i),kind=4)
    enddo
    !$omp end do
 case(rs_alphaloc)
    !$omp do schedule(static)
    do i = 1,npart
       alphaind(2,i) = real(dbuf(i),kind=4)
    enddo
    !$omp end do
 case(rs_pres)
    !$omp do schedule(static)
    do i = 1,npart
       eos_vars(igasP,i) = dbuf(i)
    enddo
    !$omp end do
 case(rs_rho)
    !$omp do schedule(static)
    do i = 1,npart
       rho(i) = dbuf(i)
    enddo
    !$omp end do
 case(rs_gradh)
    !$omp do schedule(static)
    do i = 1,npart
       gradh(1,i) = real(dbuf(i),kind=4)
       gradh(2,i) = real(dbuf(npart+i),kind=4)
    enddo
    !$omp end do
 case(rs_xi)
    if (hasxi) then
       !$omp do schedule(static)
       do i = 1,npart
          xi_gpu(i) = dbuf(i)
       enddo
       !$omp end do
    endif
 case(rs_ddivvdt)
    !$omp do schedule(static)
    do i = 1,npart
       alphaind(3,i) = real(dbuf(i),kind=4)
    enddo
    !$omp end do
 case(rs_fext)
    !$omp do schedule(static)
    do i = 1,npart
       fext(1,i) = dbuf(i); fext(2,i) = dbuf(npart+i); fext(3,i) = dbuf(2*npart+i)
    enddo
    !$omp end do
 case(rs_dustgas)
    !--the device's copy changed where densityiterate_gpu's writeback would have, and
    !  holds the host's value elsewhere (one large grain size on the GPU, so this is the
    !  only dustfrac row in use)
    !$omp do schedule(static)
    do i = 1,npart
       dustfrac(ndustsmall+1,i) = dbuf(i)
    enddo
    !$omp end do
 end select
 !$omp end parallel

end subroutine unpack

!-----------------------------------------------------------------------
!+
!  In derivs, on the GPU path, after the density pass and cons2prim and
!  before the force pass: the density writeback and cons2prim twins, so
!  that the resident state the force pass may read (COSMO_RESIDENT_FEED)
!  is this step's.  The positions were wrapped into a periodic box by the
!  density staging: by the device too with the feed (compared), else
!  re-sent.
!+
!-----------------------------------------------------------------------
subroutine rs_after_cons2prim(npart)
 use dim,  only:nalpha,use_dust
 use part, only:massoftype,maxtypes,hfact
 integer, intent(in) :: npart
 if (.not.gpu_mirror .or. npart <= 0 .or. .not.started) return
 if (resident_feed) then
    call compare(13,rs_pos,npart)
 else
    call put(rs_pos,npart)
 endif
#ifdef GPU
 call cosmo_rs_dens_writeback(int(npart,c_int),real(massoftype(1:maxtypes),c_double), &
                              int(maxtypes,c_int),real(hfact,c_double),merge(1_c_int,0_c_int,nalpha >= 3), &
                              merge(1_c_int,0_c_int,use_dust))
 call compare(10,rs_h,npart)
 call compare(10,rs_rho,npart)
 call compare(10,rs_gradh,npart)
 call compare(10,rs_xi,npart)
 if (nalpha >= 3) call compare(10,rs_ddivvdt,npart)
 if (use_dust) call compare(10,rs_dustgas,npart)
 call rs_cons2prim_twin(npart)
#endif
end subroutine rs_after_cons2prim

!-----------------------------------------------------------------------
!+
!  cons2prim's twin (GPU path, after the density writeback twin): gas
!  with ieos 1, 2 (with u) or 3, no dust fraction, no variable
!  composition, nucleation, chemistry or radiation.  Otherwise the
!  fields are re-sent.  ieos 3 is compared to 4 ulps (pow).
!+
!-----------------------------------------------------------------------
subroutine rs_cons2prim_twin(npart)
 use dim,             only:maxalpha,maxp,nalpha,maxvxyzu,do_radiation,do_nucleation,update_muGamma,use_krome
 use eos,             only:ieos,polyk,gamma,qfacdisc,cs_min,isink
 use options,         only:use_dustfrac,use_var_comp
 use shock_capturing, only:alpha,alphamax
 use part,            only:xyzmh_ptmass,nptmass
 integer, intent(in) :: npart
 logical :: twin,aloc
 real    :: tolulp
 real(c_double) :: sxyz(3)

 aloc = (maxalpha == maxp .and. nalpha >= 2)
 twin = (ieos == 1 .or. ieos == 3 .or. (ieos == 2 .and. maxvxyzu >= 4) .or. &
         (ieos == 6 .and. isink >= 1 .and. isink <= nptmass)) .and. .not.use_dustfrac &
        .and. .not.use_var_comp .and. .not.do_nucleation .and. .not.update_muGamma .and. .not.use_krome &
        .and. .not.do_radiation
 if (.not.twin) then
    call put(rs_cs,npart)
    call put(rs_pres,npart)
    if (aloc) call put(rs_alphaloc,npart)
    return
 endif
 sxyz = 0.
 if (ieos == 6) sxyz = xyzmh_ptmass(1:3,isink)
#ifdef GPU
 call cosmo_rs_cons2prim(int(npart,c_int),int(ieos,c_int),real(polyk,c_double),real(gamma,c_double), &
                         real(qfacdisc,c_double),real(cs_min,c_double),merge(1_c_int,0_c_int,aloc), &
                         real(alpha,c_double),real(alphamax,c_double),sxyz)
#endif
 tolulp = 0.
 if (ieos == 3 .or. ieos == 6) tolulp = 4.
 call compare(12,rs_cs,npart,tolulp)
 call compare(12,rs_pres,npart,tolulp)
 if (aloc) call compare(12,rs_alphaloc,npart,tolulp)

end subroutine rs_cons2prim_twin

!--after the corrector: v, twas, and the bins after waking
subroutine rs_after_corrector(npart,timei,ibin_dts,nbins,ithdt,ittwas,nbinmax,ufloor,icooling)
 use io, only:iprint
 integer,         intent(in) :: npart,nbins,ithdt,ittwas,icooling
 integer(kind=1), intent(in) :: nbinmax
 real,            intent(in) :: timei,ufloor
 real,            intent(in) :: ibin_dts(4,0:nbins)
 integer :: f
 if (.not.gpu_mirror .or. npart <= 0) return
#ifdef GPU
 call cosmo_rs_corrector(int(npart,c_int),real(timei,c_double),real(ibin_dts,c_double), &
                         int(nbins,c_int),int(ithdt,c_int),int(ittwas,c_int),int(nbinmax,c_int), &
                         real(ufloor,c_double),int(icooling,c_int))
 call compare(5,rs_v,npart)
 call compare(5,rs_twas,npart)
 call compare(5,rs_ibin,npart)
 call compare(5,rs_ibinwake,npart)
#ifdef MPI
 if (gpu_domain_on) call cosmo_dom_check_halos(int(10*nstep+2,c_int))   ! COSMO_DOMAIN_DEBUG=3 only
#endif
#endif
 !--one line per step: the mismatching particles of each check (resident mode compares
 !  nothing: every 1000th step only, as a sign of life; a formatted line per substep on
 !  every rank, through mpirun, sat in the step's critical path)
 if (.not.resident_mode .or. mod(nstep,1000) == 0) &
    write(iprint,'(a,i8,13(1x,a,i8,a,i3),a,f6.2)') ' COSMO_MIRROR step',nstep, &
    (trim(checkname(f))//'=',int(nbad(f)),'/',nrun(f),f=1,nchecks),' maxulp=',maxulp_step
 nbadtot = nbadtot + nbad
end subroutine rs_after_corrector

#ifdef MPI
!-----------------------------------------------------------------------
!+
!  The whole state to the device, once: the first step's send (rs_start_step)
!  or, under MPI, before the first derivs in initial (rs_prime).  With
!  everything = .true. the active flags and vpred go too (initial: every
!  particle active, vpred = v), else the step sets them.  Then, under MPI,
!  the device Domain (dom_start), whose first sync moves nothing.
!+
!-----------------------------------------------------------------------
subroutine rs_first_send(npart,everything)
 use dim,  only:use_dust
 use io,   only:nprocs
 use part, only:vpred,vxyzu
 integer, intent(in) :: npart
 logical, intent(in) :: everything
 integer :: f
#ifdef GPU
 call cosmo_rs_init(int(npart,c_int),int(nvu(),c_int))
 !--with the passes fed from the state, keep it in the tree's order (cosmoSPHere
 !  re-sorts it at each tree build; puts and gets translate)
 !--one rank: the state in the tree's order (slot mode).  MPI: the state stays in the
 !  Domain's layout and the solve has its own order (the type permutation of plan
 !  section 3): dead and dust particles then sort as they always did
 if (gpu_feed .and. nprocs == 1) call cosmo_rs_use_slots(int(npart,c_int))
 if (everything) vpred(:,1:npart) = vxyzu(:,1:npart)
 do f = rs_pos,rs_last
    if (f == rs_dustgas .and. .not.use_dust) cycle
    if (.not.everything .and. (f == rs_active .or. f == rs_vpred)) cycle
    call put(f,npart)
 enddo
 started = .true.
 resident_feed = gpu_feed
#ifdef MPI
 !--the device Domain over the state (GPU_MPI_PLAN.md Phase 2); its first sync moves nothing
 if (gpu_feed .and. nprocs > 1) call dom_start(npart)
#endif
#else
 f = npart; if (everything) f = 0
#endif
end subroutine rs_first_send

!-----------------------------------------------------------------------
!+
!  MPI: the resident state and the device Domain BEFORE the first derivs
!  in initial (get_derivs_initial), so that derivs already runs with the
!  halos: on its own, a rank's first density and force are wrong at the
!  seams, and the wrong f and h would be sent to the device as the state.
!  The device is the truth from here (resident_mode); initial then
!  downloads before the alpha update and the t = 0 dump.
!+
!-----------------------------------------------------------------------
subroutine rs_prime(npart)
 use io, only:iprint,id,master
 integer, intent(in) :: npart
 call rs_init_switch()
 if (.not.gpu_resident_on .or. started .or. npart <= 0) return
 !--every resident run, not only MPI ones: the first derivs then takes one path on any
 !  rank count (the host's own Cullen-Dehnen initialisation and the device twin's differ
 !  at the threshold on a few particles), and the P-rank gate compares like with like
 call rs_first_send(npart,.true.)
 resident_mode = .true.
 priming = .true.
 if (id==master) write(iprint,'(a)') ' gpu_resident: state primed before the first derivs (and the Domain, under MPI)'
end subroutine rs_prime

!--MPI: a sync between initial's two derivs calls, so the second sees the first's forces
!  and densities on the halos too (div_a for the Cullen-Dehnen switch reads a neighbour's f)
subroutine rs_resync(npart)
 integer, intent(inout) :: npart
#ifdef MPI
 if (gpu_domain_on .and. resident_mode) call dom_sync(npart,.false.)
#else
 if (npart < 0) npart = 0   ! nothing to do
#endif
end subroutine rs_resync

!--initial's alpha = max(alpha, alphaloc) after the first derivs, on the device's copy
subroutine rs_put_alpha1(npart)
 integer, intent(in) :: npart
 interface
  subroutine cosmo_rs_alpha_max(n) bind(C)
   import c_int
   integer(c_int), value :: n
  end subroutine cosmo_rs_alpha_max
 end interface
 if (.not.resident_mode .or. npart <= 0) return
#ifdef GPU
 call cosmo_rs_alpha_max(int(npart,c_int))
#endif
end subroutine rs_put_alpha1

!-----------------------------------------------------------------------
!+
!  MPI (GPU_MPI_PLAN.md Phase 2): the device Domain over the resident
!  state.  dom_start creates it with the startup split's parameters and
!  runs the first sync, which moves nothing (Phase 1's split, the same
!  curve, box and bucket size); dom_sync runs every later one, after the
!  drift and before derivs, and adopts the rank's new owned count as
!  npart.  Between dumps the host arrays are stale by design (resident
!  mode): the ids are refreshed here, the rest by rs_download_all.
!+
!-----------------------------------------------------------------------
subroutine dom_start(npart)
 use mpi
 use io,       only:nprocs
 use dim,      only:periodic,use_dust
 use boundary, only:xmin,xmax,ymin,ymax,zmin,zmax
 use mpiutils, only:reduceall_mpi
 integer, intent(in) :: npart
 interface
  subroutine cosmo_dom_init(bucket,bucketfocus,periodic,box,fcomm,withdvdx) bind(C)
   import c_int,c_double
   integer(c_int), value :: bucket,bucketfocus,periodic,fcomm,withdvdx
   real(c_double)        :: box(6)
  end subroutine cosmo_dom_init
 end interface
 integer(kind=8) :: ntot
 integer         :: bucket,np
 real(c_double)  :: box(6)

 ntot   = reduceall_mpi('+',npart)
 bucket = int(max(64_8, ntot/(100_8*int(nprocs,kind=8))))   ! as gpu_domain_assign
 box    = (/xmin,xmax,ymin,ymax,zmin,zmax/)
 !--focus bucket 64 (no larger than the global bucket, which Cornerstone requires).
 !  COSMO_BUCKET_FOCUS=256 makes the halo search ~40% cheaper for ~18% more halos (16M disc,
 !  P = 4: 9.49 -> 9.17 s), but with it the 1M disc at P = 4 aborted in UCX rocm_ipc after
 !  ~400-550 syncs, 4 times out of 4 (64: the whole dtmax cycle, 1660 syncs).  Dust: the record
 !  carries dvdx (a partner's drag reconstruction reads it)
 call cosmo_dom_init(int(bucket,c_int),int(min(64,bucket),c_int),merge(1_c_int,0_c_int,periodic),box,int(MPI_COMM_WORLD,c_int), &
                     merge(1_c_int,0_c_int,use_dust))
 gpu_domain_on = .true.
 np = npart
 call dom_sync(np,.true.)

end subroutine dom_start

subroutine dom_sync(npart,first)
 use io,       only:id,master,iprint,fatal
 use timestep,     only:dtmax
 use timestep_ind, only:nbinmax
 use mpiutils, only:reduceall_mpi
 integer, intent(inout) :: npart
 logical, intent(in)    :: first
 interface
  subroutine cosmo_dom_sync(first,nown,nhalo,tsync,tnext) bind(C)
   import c_int,c_double
   integer(c_int), value :: first
   integer(c_int)        :: nown,nhalo
   real(c_double)        :: tsync
   real(c_double), value :: tnext
  end subroutine cosmo_dom_sync
 end interface
 real(c_double)  :: tnext
 integer(c_int)  :: nown,nhalo
 real(c_double)  :: tsync
 integer(kind=8) :: nhalotot,ntot

 !--the time to the next planned sync (the coarse bins' step): every particle's search radius
 !  covers what it can travel in it (the guard catches the rest)
 tnext = dtmax/2.0_c_double**max(0,int(nbinmax) - sync_levels)
 if (first .or. nbinmax <= 0) tnext = 0.   ! in initial the bins are not set yet: no margin
 call cosmo_dom_sync(merge(1_c_int,0_c_int,first),nown,nhalo,tsync,real(tnext,c_double))
 if (first .and. nown /= npart) call fatal('gpu_resident','the first device sync changed the owned count', &
                                           ival=int(nown))
 npart     = int(nown)
 nsync     = nsync + 1
 tsync_tot = tsync_tot + real(tsync)
 !--the host copy (ids included) is stale between dumps by design: rs_download_all fetches it,
 !  through dbuf, which must hold the owned count's 4-component fields (no-op unless it grew)
 call grow(npart)
 if (first .or. mod(nsync,1000) == 0) then
    ntot     = reduceall_mpi('+',int(npart,kind=8))
    nhalotot = reduceall_mpi('+',int(nhalo,kind=8))
    tsync    = reduceall_mpi('max',tsync)
    if (id==master) write(iprint,"(a,i8,a,i11,a,i11,a,f7.2,a,f8.4,a)") ' gpu_domain: sync',nsync,': owned ',ntot, &
       ' halos ',nhalotot,' (',100.*real(nhalotot)/real(ntot),'%) ',tsync,' s'
 endif

end subroutine dom_sync
#endif


end module gpu_resident
