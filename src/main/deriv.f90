!--------------------------------------------------------------------------!
! The Phantom Smoothed Particle Hydrodynamics code, by Daniel Price et al. !
! Copyright (c) 2007-2026 The Authors (see AUTHORS)                        !
! See LICENCE file for usage and distribution conditions                   !
! http://phantomsph.github.io/                                             !
!--------------------------------------------------------------------------!
module deriv
!
! this module is a wrapper for the main derivative evaluation
!
! :References: None
!
! :Owner: Daniel Price
!
! :Runtime parameters: None
!
! :Dependencies: HIIRegion, cons2prim, densityforce, derivutils, dim,
!   externalforces, forces, forcing, growth, io, metric_tools, neighkdtree,
!   options, part, porosity, ptmass, ptmass_radiation, radiation_implicit,
!   timestep, timestep_ind, timing
!
 implicit none

 public :: derivs, get_derivs_global, get_density_global
 real, private :: stressmax

 private

contains

!-------------------------------------------------------------
!+
!  calculates derivatives of all particle quantities
!  (wrapper for call to density and rates, calls neighbours etc first)
!+
!-------------------------------------------------------------
subroutine derivs(icall,npart,nactive,xyzh,vxyzu,fxyzu,fext,divcurlv,divcurlB,&
                  Bevol,dBevol,rad,drad,radprop,dustprop,ddustprop,&
                  dustevol,ddustevol,filfac,dustfrac,eos_vars,time,dt,dtnew,pxyzu,&
                  dens,metrics,apr_level)
 use dim,            only:mhd,fast_divcurlB,gr,periodic,do_radiation,driving,&
                          sink_radiation,use_dustgrowth,ind_timesteps,isothermal
 use io,             only:iprint,fatal,error
 use neighkdtree,    only:build_tree
 use densityforce,   only:densityiterate
 use gpu_dens_iface,  only:densityiterate_gpu,use_gpu_dens
 use gpu_force_iface, only:force_gpu,finish_gpu_force_timesteps
 use ptmass,         only:ipart_rhomax,ptmass_calc_enclosed_mass,ptmass_boundary_crossing,get_pressure_on_sinks
 use externalforces, only:externalforce
 use part,           only:dustgasprop,Vrel_disp,dvdx,Bxyz,set_boundaries_to_active,&
                          nptmass,xyzmh_ptmass,sinks_have_heating,dust_temp,VrelVf,fxyz_drag
 use timestep_ind,   only:nbinmax
 use timestep,       only:dtmax,dtcourant,dtforce,dtrad
 use forcing,        only:forceit
 use growth,           only:get_growth_rate
 use porosity,         only:get_disruption,get_probastick
 use ptmass_radiation, only:get_dust_temperature
 use timing,         only:get_timings
 use forces,         only:force, prepare_pro2_gpu
 use part,           only:mhd,gradh,alphaind,igas,iradxi,ifluxx,ifluxy,ifluxz,ithick
 use derivutils,     only:do_timing
 use cons2prim,      only:cons2primall,cons2prim_everything
 use metric_tools,   only:init_metric
 use radiation_implicit, only:do_radiation_implicit,ierr_failed_to_converge
 use options, only:implicit_radiation,implicit_radiation_store_drad, &
                   use_porosity,need_pressure_on_sinks,beta,alphau
 use HIIRegion,      only:HIIupdateflag,iH2R,HII_feedback
 integer,         intent(in)    :: icall
 integer,         intent(inout) :: npart
 integer,         intent(in)    :: nactive
 real,            intent(inout) :: xyzh(:,:)
 real,            intent(inout) :: vxyzu(:,:)
 real,            intent(inout) :: fxyzu(:,:)
 real,            intent(in)    :: fext(:,:)
 real(kind=4),    intent(out)   :: divcurlv(:,:)
 real(kind=4),    intent(out)   :: divcurlB(:,:)
 real,            intent(in)    :: Bevol(:,:)
 real,            intent(out)   :: dBevol(:,:)
 real,            intent(inout) :: rad(:,:)
 real,            intent(out)   :: eos_vars(:,:)
 real,            intent(out)   :: drad(:,:)
 real,            intent(inout) :: radprop(:,:)
 real,            intent(in)    :: dustevol(:,:)
 real,            intent(inout) :: dustprop(:,:)
 real,            intent(out)   :: dustfrac(:,:)
 real,            intent(out)   :: ddustevol(:,:),ddustprop(:,:)
 real,            intent(inout) :: filfac(:)
 real,            intent(in)    :: time,dt
 real,            intent(out)   :: dtnew
 real,            intent(inout) :: pxyzu(:,:), dens(:)
 real,            intent(inout) :: metrics(:,:,:,:)
 integer(kind=1), intent(in)    :: apr_level(:)
 integer                     :: ierr,i
 real(kind=4)                :: t1,tcpu1,tlast,tcpulast

 integer :: ifxyzu_unit,ifxyzu_ios !--for checking outputting force arrays

 real, allocatable :: pro2_gpu(:)
 real, allocatable :: spsound_gpu(:)
 real, allocatable :: alphaAV_gpu(:)
 real, allocatable :: u_gpu(:)
 real, allocatable :: vsigmax_gpu(:) 

 t1    = 0.
 tcpu1 = 0.
 call get_timings(t1,tcpu1)
 tlast    = t1
 tcpulast = tcpu1
!
!--check for errors in input options
!
 if (icall < 0 .or. icall > 2) call fatal('deriv','invalid icall on input')
!
! icall is a flag to say whether or not positions have changed
! since the last call to derivs.
!
! icall = 1 is the "standard" call to derivs: calculates all derivatives
! icall = 2 does not remake the tree build and does not recalculate density
!           (ie. only re-evaluates the SPH force term using updated values
!            of the input variables)
!
! build tree to prepare neighbour finding
!
 if (icall==1 .or. icall==0) then
    call build_tree(npart,nactive,xyzh,vxyzu)

    if (gr) then
       ! update time-dependent metric (e.g. binary BH) and repack at particle positions
       call init_metric(npart,xyzh,metrics,time=time)
    endif

    if (nptmass > 0 .and. periodic) call ptmass_boundary_crossing(nptmass,xyzmh_ptmass)
 endif

 call do_timing('tree',tlast,tcpulast,start=.true.)

 !
 ! compute disruption of dust particles
 !
 if (use_dustgrowth .and. use_porosity) call get_disruption(npart,xyzh,filfac,dustprop,dustgasprop)
!
! calculate density by direct summation
!

 if (icall==1) then
    if (use_gpu_dens) then
       !--GPU path: cosmoSPHere solves h, gradh(1,i)=1/omega, divv, dvdx and
       !  ddivvdt in one pass, so no CPU sweep over the kd-tree is needed here
       call densityiterate_gpu(npart,xyzh,vxyzu,fxyzu,fext,gradh,divcurlv,dvdx,alphaind)
    else
       !--CPU path: original phantom behaviour, unchanged
       call densityiterate(1,npart,nactive,xyzh,vxyzu,divcurlv,divcurlB,Bevol,&
                           stressmax,fxyzu,fext,alphaind,gradh,rad,radprop,dvdx,apr_level)
       if (.not. fast_divcurlB) then
          ! Repeat to calculate non-density quantities requiring up-to-date rho.
          call densityiterate(3,npart,nactive,xyzh,vxyzu,divcurlv,divcurlB,Bevol,&
                              stressmax,fxyzu,fext,alphaind,gradh,rad,radprop,dvdx,apr_level)
          ! put a similar flag for pressure calculation from dens: call cons2primall/everyhting and densityiterate(3. Import pressure from eos_vars in dens and use it to calculate delta_v
       endif
    endif
    set_boundaries_to_active = .false.     ! boundary particles are no longer treated as active
    call do_timing('dens',tlast,tcpulast)
 endif
!
!-- update ionising state of the particle if HII regions are used in cluster formation simulations
!
 if (iH2R >0) then
    if (HIIupdateflag) then
       call HII_feedback(nptmass,npart,xyzh,xyzmh_ptmass,vxyzu,eos_vars)
       HIIupdateflag = .false.
    endif
    call do_timing('HII_region',tlast,tcpulast)
 endif

 if (gr) then
    call cons2primall(npart,xyzh,metrics,pxyzu,vxyzu,dens,eos_vars)
 else
    call cons2prim_everything(npart,xyzh,vxyzu,dvdx,rad,eos_vars,radprop,Bevol,Bxyz,dustevol,dustfrac,alphaind)
 endif
 call do_timing('cons2prim',tlast,tcpulast)

 !
 ! implicit radiation update
 !
 if (do_radiation .and. implicit_radiation .and. dt > 0.) then
    call do_radiation_implicit(dt,npart,rad,xyzh,vxyzu,radprop,drad,ierr)
    if (ierr /= 0 .and. ierr /= ierr_failed_to_converge) call fatal('radiation','Failed in radiation')
    call do_timing('radiation',tlast,tcpulast)
 endif

 !
 ! compute forces
 !
 if (driving) then
    ! forced turbulence -- call driving routine
    call forceit(time,npart,xyzh,vxyzu,fxyzu)
    call do_timing('driving',tlast,tcpulast)
 endif

 !
 ! compute SPH forces
 !
 stressmax = 0.

 if (sinks_have_heating(nptmass,xyzmh_ptmass)) call ptmass_calc_enclosed_mass(nptmass,npart,xyzh)

 !--Build the symmetric (gather+scatter) j-leaf list the GPU force sum will need.
 !  No forces computed and nothing written back yet — this is here to measure the
 !  walk.  Guarded on use_gpu_dens because it consumes the octree and hmax that
 !  densityiterate_gpu leaves behind; it becomes its own switch with the kernel.

 if (use_gpu_dens) then
    allocate(pro2_gpu(npart))
	allocate(spsound_gpu(npart))
	allocate(alphaAV_gpu(npart))
	allocate(u_gpu(npart))
	allocate(vsigmax_gpu(npart))

	call prepare_pro2_gpu(npart,xyzh,vxyzu,eos_vars,alphaind, &
                      pro2_gpu,spsound_gpu,alphaAV_gpu,u_gpu)

	call force_gpu(npart,xyzh,vxyzu,pro2_gpu,spsound_gpu,alphaAV_gpu,u_gpu, &
	               beta,alphau,fxyzu,vsigmax_gpu,divcurlv)


    call finish_gpu_force_timesteps( &
        	   npart,xyzh,fxyzu,spsound_gpu,vsigmax_gpu)

	deallocate(pro2_gpu)
	deallocate(spsound_gpu)
	deallocate(alphaAV_gpu)
	deallocate(u_gpu)
	deallocate(vsigmax_gpu)

	!open(newunit=ifxyzu_unit,                         &
    ! 	 file='gpu_fxyzu.dat',                        &
    ! 	 status='replace',                            &
    ! 	 action='write',                              &
    ! 	 form='formatted',                            &
    ! !	 iostat=ifxyzu_ios)

	!if (ifxyzu_ios /= 0) then
    !	call fatal('deriv','could not open gpu_fxyzu.dat')
    !endif

	!write(ifxyzu_unit,'(a)') &
    !'# particle_index  fxyzu(1)  fxyzu(2)  fxyzu(3)  fxyzu(4)'

	!--do i=1,npart
    !  write(ifxyzu_unit,'(i10,1x,4(es24.16e3,1x))') &
    !  i,                                         &
    !  fxyzu(1,i),                                &
    !  fxyzu(2,i),                                &
    !  fxyzu(3,i),                                &
    !  fxyzu(4,i)
    !--enddo

	!close(ifxyzu_unit,iostat=ifxyzu_ios)

	!if (ifxyzu_ios /= 0) then
   	!	call fatal('deriv','error closing gpu_fxyzu.dat')
	!endif

	!call fatal('deriv', &
    !'wrote fxyzu(1:4,:) to gpu_fxyzu.dat; terminating deliberately')


    !--call fatal('deriv', &
        !'GPU force and fxyzu(4,:) was computed but time-step size was not computed, time integration cannot proceed')
    !--stop
 else
    call force(icall,npart,xyzh,vxyzu,fxyzu,divcurlv,divcurlB,Bevol,dBevol,&
              rad,drad,radprop,dustprop,dustgasprop,Vrel_disp,dustfrac,ddustevol,fext,fxyz_drag,&
              ipart_rhomax,dt,stressmax,eos_vars,dens,metrics,apr_level)

 endif
 !
 ! compute growth rate of dust particles
 !
 if (use_dustgrowth) then
    call get_growth_rate(npart,xyzh,vxyzu,dustgasprop,VrelVf,dustprop,filfac,ddustprop(1,:),Vrel_disp)!--we only get dm/dt (i.e 1st dimension of ddustprop)
    ! compute growth rate and probability of sticking/bouncing of porous dust
    if (use_porosity) call get_probastick(npart,xyzh,ddustprop(1,:),dustprop,dustgasprop,filfac)
 endif
!
! compute density and pressure at location of sink particles
!
 if (need_pressure_on_sinks) call get_pressure_on_sinks(nptmass,xyzmh_ptmass)
!
! compute dust temperature
!
 if (sink_radiation .and. .not.isothermal) then
    call get_dust_temperature(npart,xyzh,eos_vars,nptmass,xyzmh_ptmass,dust_temp)
 endif

 if (do_radiation .and. implicit_radiation .and. .not.implicit_radiation_store_drad) then
    !$omp parallel do shared(drad,fxyzu,npart) private(i)
    do i=1,npart
       drad(:,i) = 0.
       fxyzu(4,i) = 0.
    enddo
    !$omp end parallel do
 endif
!
! set new timestep from Courant/forces condition
!
 if (ind_timesteps) then
    dtnew = dtmax/2.**nbinmax  ! minimum timestep over all particles
 else
    dtnew = min(dtforce,dtcourant,dtrad,dtmax)
 endif

 !if (use_gpu_dens) then
 !  call write_gpu_force_snapshot( &
 !       icall,npart,time,dt,dtnew,dtcourant,dtforce,fxyzu)
 !else
 !  call write_cpu_force_snapshot( &
 !       icall,npart,time,dt,dtnew,dtcourant,dtforce,fxyzu)
 !endif

 call do_timing('total',t1,tcpu1,lunit=iprint)

end subroutine derivs

!-----------------------------------------------------------------------
!+
!  Append one complete GPU force evaluation to a formatted text file.
!+
!-----------------------------------------------------------------------
subroutine write_gpu_force_snapshot(icall,npart,time,dt,dtnew, &
                                    dtcourant,dtforce,fxyzu)
 use io, only:fatal

 integer, intent(in) :: icall,npart
 real,    intent(in) :: time,dt,dtnew,dtcourant,dtforce
 real,    intent(in) :: fxyzu(:,:)

 integer, save :: force_call_count = 0
 integer, save :: clock_start = 0
 logical, save :: clock_started = .false.

 integer :: i,iunit,ios
 integer :: clock_now,clock_rate
 real    :: wall_elapsed

 call system_clock(count=clock_now,count_rate=clock_rate)

 if (.not.clock_started) then
    clock_start   = clock_now
    clock_started = .true.
 endif

 wall_elapsed = real(clock_now-clock_start) / real(clock_rate)

 force_call_count = force_call_count + 1

 ! Start a new file on the first force call of this program execution.
 if (force_call_count == 1) then
    open(newunit=iunit,                         &
         file='gpu_force_history.dat',          &
         status='replace',                      &
         action='write',                        &
         form='formatted',                      &
         iostat=ios)
 else
    open(newunit=iunit,                         &
         file='gpu_force_history.dat',          &
         status='old',                          &
         position='append',                     &
         action='write',                        &
         form='formatted',                      &
         iostat=ios)
 endif

 if (ios /= 0) then
    call fatal('write_gpu_force_snapshot', &
               'could not open gpu_force_history.dat')
 endif

 write(iunit,'(a)') '# BEGIN_GPU_FORCE_CALL'
 write(iunit,'(a,i0)') '# force_call = ',force_call_count
 write(iunit,'(a,i0)') '# icall = ',icall
 write(iunit,'(a,es24.16e3)') '# simulation_time = ',time
 write(iunit,'(a,es24.16e3)') '# input_dt = ',dt
 write(iunit,'(a,es24.16e3)') '# proposed_dtnew = ',dtnew
 write(iunit,'(a,es24.16e3)') '# dtcourant = ',dtcourant
 write(iunit,'(a,es24.16e3)') '# dtforce = ',dtforce
 write(iunit,'(a,es24.16e3)') '# wall_elapsed_seconds = ', &
                               wall_elapsed

 write(iunit,'(a)') &
      '# particle_index  fxyzu(1)  fxyzu(2)  fxyzu(3)  fxyzu(4)'

 do i = 1,npart
    write(iunit,'(i10,1x,4(es24.16e3,1x))') &
         i,                                  &
         fxyzu(1,i),                         &
         fxyzu(2,i),                         &
         fxyzu(3,i),                         &
         fxyzu(4,i)
 enddo

 write(iunit,'(a)') '# END_GPU_FORCE_CALL'
 write(iunit,'(a)') ''

 close(iunit,iostat=ios)

 if (ios /= 0) then
    call fatal('write_gpu_force_snapshot', &
               'could not close gpu_force_history.dat')
 endif

end subroutine write_gpu_force_snapshot

!-----------------------------------------------------------------------
!+
!  Append one complete CPU force evaluation to a formatted text file.
!+
!-----------------------------------------------------------------------
subroutine write_cpu_force_snapshot(icall,npart,time,dt,dtnew, &
                                    dtcourant,dtforce,fxyzu)
 use io, only:fatal

 integer, intent(in) :: icall,npart
 real,    intent(in) :: time,dt,dtnew,dtcourant,dtforce
 real,    intent(in) :: fxyzu(:,:)

 integer, save :: force_call_count = 0
 integer, save :: clock_start = 0
 logical, save :: clock_started = .false.

 integer :: i,iunit,ios
 integer :: clock_now,clock_rate
 real    :: wall_elapsed

 call system_clock(count=clock_now,count_rate=clock_rate)

 if (.not.clock_started) then
    clock_start   = clock_now
    clock_started = .true.
 endif

 wall_elapsed = real(clock_now-clock_start) / real(clock_rate)

 force_call_count = force_call_count + 1

 if (force_call_count == 1) then
    open(newunit=iunit,                         &
         file='cpu_force_history.dat',          &
         status='replace',                      &
         action='write',                        &
         form='formatted',                      &
         iostat=ios)
 else
    open(newunit=iunit,                         &
         file='cpu_force_history.dat',          &
         status='old',                          &
         position='append',                     &
         action='write',                        &
         form='formatted',                      &
         iostat=ios)
 endif

 if (ios /= 0) then
    call fatal('write_cpu_force_snapshot', &
               'could not open cpu_force_history.dat')
 endif

 write(iunit,'(a)') '# BEGIN_CPU_FORCE_CALL'
 write(iunit,'(a,i0)') '# force_call = ',force_call_count
 write(iunit,'(a,i0)') '# icall = ',icall
 write(iunit,'(a,es24.16e3)') '# simulation_time = ',time
 write(iunit,'(a,es24.16e3)') '# input_dt = ',dt
 write(iunit,'(a,es24.16e3)') '# proposed_dtnew = ',dtnew
 write(iunit,'(a,es24.16e3)') '# dtcourant = ',dtcourant
 write(iunit,'(a,es24.16e3)') '# dtforce = ',dtforce
 write(iunit,'(a,es24.16e3)') '# wall_elapsed_seconds = ', &
                               wall_elapsed

 write(iunit,'(a)') &
      '# particle_index  fxyzu(1)  fxyzu(2)  fxyzu(3)  fxyzu(4)'

 do i = 1,npart
    write(iunit,'(i10,1x,4(es24.16e3,1x))') &
         i,                                  &
         fxyzu(1,i),                         &
         fxyzu(2,i),                         &
         fxyzu(3,i),                         &
         fxyzu(4,i)
 enddo

 write(iunit,'(a)') '# END_CPU_FORCE_CALL'
 write(iunit,'(a)') ''

 close(iunit,iostat=ios)

 if (ios /= 0) then
    call fatal('write_cpu_force_snapshot', &
               'could not close cpu_force_history.dat')
 endif

end subroutine write_cpu_force_snapshot
!--------------------------------------
!+
!  wrapper for the call to derivs
!  so only one line needs changing
!  if interface changes
!
!  this should NOT be called during timestepping, it is useful
!  for when one requires just a single call to evaluate derivatives
!  and store them in the global shared arrays
!+
!--------------------------------------
subroutine get_derivs_global(tused,dt_new,dt,icall)
 use part,         only:npart,xyzh,vxyzu,fxyzu,fext,divcurlv,divcurlB,&
                        Bevol,dBevol,rad,drad,radprop,dustprop,ddustprop,filfac,&
                        dustfrac,ddustevol,eos_vars,pxyzu,dens,metrics,dustevol,gr,&
                        apr_level
 use timing,       only:printused,getused
 use io,           only:id,master
 use cons2prim,    only:prim2consall
 use metric_tools, only:init_metric
 real(kind=4), intent(out), optional :: tused
 real,         intent(out), optional :: dt_new
 real,         intent(in),  optional :: dt  ! optional argument needed to test implicit radiation routine
 integer,      intent(in),  optional :: icall
 real(kind=4) :: t1,t2
 real    :: dtnew,dti,time
 integer :: icalli

 time = 0.
 dti = 0.
 icalli = 1
 if (present(dt)) dti = dt
 if (present(icall)) icalli = icall
 call getused(t1)
 ! update conserved quantities in the GR code
 if (gr) then
    call init_metric(npart,xyzh,metrics,time=time)
    call prim2consall(npart,xyzh,metrics,vxyzu,pxyzu,use_dens=.false.,dens=dens)
 endif

 ! evaluate derivatives
 call derivs(icalli,npart,npart,xyzh,vxyzu,fxyzu,fext,divcurlv,divcurlB,Bevol,dBevol,&
             rad,drad,radprop,dustprop,ddustprop,dustevol,ddustevol,filfac,dustfrac,&
             eos_vars,time,dti,dtnew,pxyzu,dens,metrics,apr_level)

 call getused(t2)
 if (id==master .and. present(tused)) call printused(t1)
 if (present(tused)) tused = t2 - t1
 if (present(dt_new)) dt_new = dtnew

end subroutine get_derivs_global

!--------------------------------------
!+
!  wrapper for the call to densityiterate
!  so only one line needs changing
!  if interface changes
!
!  this should be used when one requires just a density calculation
!  and store results in the global shared arrays
!+
!--------------------------------------
subroutine get_density_global(icall,nactive,zero_fxyzu,make_tree)
 use part,         only:npart,xyzh,vxyzu,fxyzu,fext,divcurlv,divcurlB,&
                        Bevol,alphaind,gradh,rad,radprop,dvdx,apr_level
 use densityforce, only:densityiterate
 use neighkdtree,  only:build_tree
 integer, intent(in) :: icall
 integer, intent(in), optional :: nactive
 logical, intent(in), optional :: zero_fxyzu
 logical, intent(in), optional :: make_tree
 integer :: nactivei
 logical :: do_tree
 real    :: stressmax

 nactivei = npart
 if (present(nactive)) nactivei = nactive

 do_tree = .true.
 if (present(make_tree)) do_tree = make_tree

 ! build tree to prepare neighbour finding (if requested)
 if (do_tree) call build_tree(npart,nactivei,xyzh,vxyzu)

 ! optionally zero fxyzu (useful for initialization)
 if (present(zero_fxyzu)) then
    if (zero_fxyzu) fxyzu = 0.
 endif

 ! evaluate density
 stressmax = 0.
 call densityiterate(icall,npart,nactivei,xyzh,vxyzu,divcurlv,divcurlB,Bevol,stressmax,&
                     fxyzu,fext,alphaind,gradh,rad,radprop,dvdx,apr_level)

end subroutine get_density_global

end module deriv
