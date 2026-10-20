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
 use iso_c_binding, only:c_double,c_int
 use gpu_arrays,    only:resident_feed
 implicit none

 logical, public :: gpu_mirror = .false.
 logical, save   :: gpu_feed = .false.     ! COSMO_RESIDENT_FEED: the pair passes read the resident state
 public :: rs_init_switch,rs_start_step,rs_after_predictor,rs_after_substep
 public :: rs_after_predict_sph,rs_after_derivs,rs_after_corrector,rs_after_cons2prim
 public :: rs_after_kick,rs_after_subdrift,rs_after_get_force,rs_after_accretion,rs_before_gas_loop

 private

 !--field ids, as cosmoSPHere/src/resident.cu RsField
 integer, parameter :: rs_pos = 1, rs_h = 2, rs_v = 3, rs_vpred = 4, rs_f = 5, rs_twas = 6, &
                       rs_ibin = 7, rs_ibinold = 8, rs_ibinwake = 9, rs_active = 10, &
                       rs_divv = 11, rs_cs = 12, rs_alpha1 = 13, rs_alphaloc = 14, rs_fext = 15, &
                       rs_type = 16, rs_rho = 17, rs_gradh = 18, rs_xi = 19, rs_ddivvdt = 20, rs_pres = 21

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
 integer, save :: nrun(nchecks) = 0   ! comparisons made this step, per check
 integer, save :: nstep = 0
 logical, save :: started = .false., checked = .false.
 real(c_double), allocatable :: hbuf(:),dbuf(:)

#ifdef GPU
 interface
  subroutine cosmo_rs_init(n,nvu) bind(C)
   import :: c_int
   integer(c_int), value :: n,nvu
  end subroutine cosmo_rs_init
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
  integer(c_int) function cosmo_rs_set_active(n,istepfrac,nbinmax) bind(C)
   import :: c_int
   integer(c_int), value :: n,istepfrac,nbinmax
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
  real(c_double) function cosmo_rs_sink_gas(n,sink,nsink,massoftype,ntype,extrap,extrapfac,sinkacc) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n,nsink,ntype,extrap
   real(c_double), intent(in) :: sink(*),massoftype(*)
   real(c_double), value :: extrapfac
   real(c_double), intent(out) :: sinkacc(*)
  end function cosmo_rs_sink_gas
  integer(c_int) function cosmo_rs_accrete(n,sink,nsink,facc,nbinmax,radkern2) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n,nsink,nbinmax
   real(c_double), intent(in) :: sink(*)
   real(c_double), value :: facc,radkern2
  end function cosmo_rs_accrete
  integer(c_int) function cosmo_rs_new_bins(n,dtmax,ccour,cforce,alpha,istepfrac,nbinmaxold, &
                                            ibinnow,usedust,limitu) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n,istepfrac,nbinmaxold,ibinnow,usedust,limitu
   real(c_double), value :: dtmax,ccour,cforce,alpha
  end function cosmo_rs_new_bins
  subroutine cosmo_rs_dens_writeback(n,massoftype,ntype,hfact,nalpha3) bind(C)
   import :: c_int,c_double
   integer(c_int), value :: n,ntype,nalpha3
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

 if (checked) return
 checked = .true.
#ifdef GPU
 call get_environment_variable('COSMO_MIRROR',val,ln,ierr)
 gpu_mirror = (ierr == 0 .and. ln > 0 .and. ind_timesteps)
 if (gpu_mirror) write(iprint,'(a)') ' cosmoSPHere: COSMO_MIRROR, device twins of the step checked against the host'
 call get_environment_variable('COSMO_RESIDENT_FEED',val,ln,ierr)
 gpu_feed = gpu_mirror .and. (ierr == 0 .and. ln > 0)
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
 integer, intent(in) :: n
 if (allocated(hbuf)) then
    if (size(hbuf) >= 4*n) return
    deallocate(hbuf,dbuf)
 endif
 allocate(hbuf(4*n),dbuf(4*n))
end subroutine grow

!-----------------------------------------------------------------------
!+
!  Pack a host field into hbuf as ncomp runs of npart doubles
!+
!-----------------------------------------------------------------------
subroutine pack(field,npart,ncomp)
 use part, only:xyzh,vxyzu,vpred,fxyzu,twas,ibin,ibin_old,ibin_wake,iphase,iactive, &
                divcurlv,eos_vars,ics,alphaind,fext,iamtype,rho,gradh,igasP
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
 case(rs_type)
    do i = 1,npart
       hbuf(i) = real(iamtype(iphase(i)),kind=c_double)
    enddo
 case(rs_fext)
    ncomp = 3
    do c = 1,3
       hbuf((c-1)*npart+1:c*npart) = fext(c,1:npart)
    enddo
 end select

end subroutine pack

!--send one host field to the device
subroutine put(field,npart)
 integer, intent(in) :: field,npart
 integer :: ncomp
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
subroutine rs_start_step(npart,istepfrac,nbinmax)
 integer,         intent(in) :: npart,istepfrac
 integer(kind=1), intent(in) :: nbinmax
 integer :: f,ncount

 call rs_init_switch()
 if (.not.gpu_mirror .or. npart <= 0) return
 nbad  = 0
 nrun  = 0
 maxulp_step = 0.
 nstep = nstep + 1
#ifdef GPU
 if (.not.started) then
    call cosmo_rs_init(int(npart,c_int),int(nvu(),c_int))
    do f = rs_pos,rs_pres
       if (f /= rs_active .and. f /= rs_vpred) call put(f,npart)
    enddo
    started = .true.
    resident_feed = gpu_feed
 endif
 ncount = cosmo_rs_set_active(int(npart,c_int),int(istepfrac,c_int),int(nbinmax,c_int))
 call compare(1,rs_active,npart)
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
subroutine rs_after_get_force(npart,ntypes,nptmass,xyzmh_ptmass,fxyz_ptmass,fonrmax,extrapfac,fsink_old)
 use dim,              only:use_sinktree,gr
 use options,          only:iexternalforce
 use damping,          only:idamp
 use ptmass,           only:isink_potential,use_regnbody
 use ptmass_radiation, only:isink_radiation
 use part,             only:ihsoft,iJ2,massoftype,igas,maxtypes
 integer, intent(in) :: npart,ntypes,nptmass
 real,    intent(in) :: xyzmh_ptmass(:,:),fonrmax,fxyz_ptmass(:,:)
 real,    intent(in), optional :: extrapfac,fsink_old(:,:)
 real(c_double), allocatable :: sink(:),sinkacc(:)
 real(c_double) :: fonrmax_dev
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
                                    1_c_int,real(extrapfac,c_double),sinkacc)
 else
    fonrmax_dev = cosmo_rs_sink_gas(int(npart,c_int),sink,int(nptmass,c_int), &
                                    real(massoftype(1:maxtypes),c_double),int(maxtypes,c_int), &
                                    0_c_int,0._c_double,sinkacc)
 endif
 call compare(7,rs_fext,npart)
 if (fonrmax_dev /= fonrmax) nbad(7) = nbad(7) + 1
 !--the particles' pull on each sink: sums in another order, so to 1e-12 of the sum of
 !  the absolute terms (a disc's net pull on its star nearly cancels)
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
!  After accretion.  The particle's side has a twin (h = -|h| for the
!  accreted, the wake flags): one particle type, at most nfastacc sinks,
!  no external-force accretion, no star formation (icreate_sinks 2), no
!  APR.  Otherwise h and the wake flags are re-sent.
!+
!-----------------------------------------------------------------------
subroutine rs_after_accretion(npart,ntypes,nptmass,xyzmh_ptmass,pxyz_ptmass,nbinmax)
 use dim,         only:use_apr
 use options,     only:iexternalforce
 use ptmass,      only:icreate_sinks,f_acc
 use ptmass_tree, only:nfastacc
 use part,        only:ihacc
 use kernel,      only:radkern2
 integer,         intent(in) :: npart,ntypes,nptmass
 real,            intent(in) :: xyzmh_ptmass(:,:),pxyz_ptmass(:,:)
 integer(kind=1), intent(in), optional :: nbinmax
 real(c_double), allocatable :: sink(:)
 integer :: j,nacc
 logical :: twin

 if (.not.gpu_mirror .or. npart <= 0) return
 twin = (nptmass > 0 .and. nptmass <= nfastacc .and. only_gas_and_dust() .and. iexternalforce == 0 &
         .and. icreate_sinks /= 2 .and. .not.use_apr .and. present(nbinmax))
 if (.not.twin) then
    call put(rs_h,npart)
    call put(rs_ibinwake,npart)
    return
 endif
 allocate(sink(8*nptmass))
 do j = 1,nptmass
    sink(8*j-7:8*j) = (/xyzmh_ptmass(1:4,j),xyzmh_ptmass(ihacc,j),pxyz_ptmass(1:3,j)/)
 enddo
#ifdef GPU
 nacc = cosmo_rs_accrete(int(npart,c_int),sink,int(nptmass,c_int),real(f_acc,c_double), &
                         int(nbinmax,c_int),real(radkern2,c_double))
 call compare(8,rs_h,npart)
 call compare(8,rs_ibinwake,npart)
#else
 nacc = 0
#endif

end subroutine rs_after_accretion

!--after predict_sph: h, the predicted velocity, the viscosity switch
subroutine rs_after_predict_sph(npart,dtsph,timei,ufloor)
 use dim,             only:maxalpha,maxp,nalpha
 use shock_capturing, only:avdecayconst,alpha,alphamax
 use timestep_ind,    only:nbinmax
 integer, intent(in) :: npart
 real,    intent(in) :: dtsph,timei,ufloor
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
end subroutine rs_after_predict_sph

!-----------------------------------------------------------------------
!+
!  After derivs, whose passes have no twins yet: re-send what they
!  changed.  h (density, active particles), positions (wrapped into a
!  periodic box), the force, div v, the sound speed, the local alpha, the
!  new bins and the wake flags.
!+
!-----------------------------------------------------------------------
subroutine rs_after_derivs(npart)
 use dim,             only:maxalpha,maxp,nalpha,use_dust,maxvxyzu,gr,driving
 use gpu_dens_iface,  only:use_gpu_dens
 use timestep,        only:dtmax,C_cour,C_force
 use timestep_ind,    only:nbinmax,istepfrac,ibinnow
 use shock_capturing, only:alpha
 use part,            only:massoftype,maxtypes,hfact
 use options,         only:icooling
 use cooling,         only:cooling_in_step
 use eos,             only:ieos
 integer, intent(in) :: npart
 integer :: nbm
 logical :: force_twin,limit_u
 if (.not.gpu_mirror .or. npart <= 0) return
#ifdef GPU
 if (use_gpu_dens) then
    !--(the density writeback and cons2prim twins ran in rs_after_cons2prim)
    !--the force pass's writeback, unless the host adds to the force (driving, cooling)
    force_twin = .not.driving .and. .not.(maxvxyzu >= 4 .and. icooling > 0 .and. .not.cooling_in_step)
    if (force_twin) then
       call cosmo_rs_force_writeback(int(npart,c_int),merge(1_c_int,0_c_int,use_dust))
    else
       call put(rs_f,npart)
       call put(rs_divv,npart)
    endif
    !--the new bins and wake flags (and the du/dt limit) from the force pass's results
    limit_u = (maxvxyzu >= 4 .and. .not.gr .and. ieos /= 23 .and. force_twin)
    nbm = cosmo_rs_new_bins(int(npart,c_int),real(dtmax,c_double),real(C_cour,c_double), &
                            real(C_force,c_double),real(alpha,c_double),int(istepfrac,c_int), &
                            int(nbinmax_before_derivs,c_int),int(ibinnow,c_int), &
                            merge(1_c_int,0_c_int,use_dust),merge(1_c_int,0_c_int,limit_u))
    call compare(9,rs_ibin,npart)
    call compare(9,rs_ibinwake,npart)
    if (nbm /= int(nbinmax)) nbad(9) = nbad(9) + 1
    if (force_twin) then
       call compare(11,rs_f,npart)
       call compare(11,rs_divv,npart)
    endif
    return
 endif
#endif
 nbm = 0
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
!  In derivs, on the GPU path, after the density pass and cons2prim and
!  before the force pass: the density writeback and cons2prim twins, so
!  that the resident state the force pass may read (COSMO_RESIDENT_FEED)
!  is this step's.  The positions were wrapped into a periodic box by the
!  density staging: by the device too with the feed (compared), else
!  re-sent.
!+
!-----------------------------------------------------------------------
subroutine rs_after_cons2prim(npart)
 use dim,  only:nalpha
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
                              int(maxtypes,c_int),real(hfact,c_double),merge(1_c_int,0_c_int,nalpha >= 3))
 call compare(10,rs_h,npart)
 call compare(10,rs_rho,npart)
 call compare(10,rs_gradh,npart)
 call compare(10,rs_xi,npart)
 if (nalpha >= 3) call compare(10,rs_ddivvdt,npart)
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
#endif
 !--one line per step: the mismatching particles of each check
 write(iprint,'(a,i8,13(1x,a,i8,a,i3),a,f6.2)') ' COSMO_MIRROR step',nstep, &
    (trim(checkname(f))//'=',int(nbad(f)),'/',nrun(f),f=1,nchecks),' maxulp=',maxulp_step
 nbadtot = nbadtot + nbad
end subroutine rs_after_corrector

end module gpu_resident
