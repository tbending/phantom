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
! The staging buffers are module level and reused across calls, so they are
! allocated once and grown only if npart rises, rather than allocated and freed
! on every force evaluation.
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
! :Dependencies: dim, gpu_dens_iface, iso_c_binding, options, part, timestep
!
 use iso_c_binding, only:c_double,c_int
 implicit none

#ifdef GPU
 interface
    subroutine force_gpu_c(n,pmass,vx,vy,vz, &
                         pro2,spsound,alphaAV,u,beta,alphau, &
                         fx,fy,fz,f4,vsigmax,divv) bind(C)
    use iso_c_binding, only:c_double,c_int

    integer(c_int), value       :: n
    real(c_double), value       :: pmass
    real(c_double), intent(in)  :: vx(*),vy(*),vz(*)
    real(c_double), intent(in)  :: pro2(*)
    real(c_double), intent(in)  :: spsound(*)
    real(c_double), intent(in)  :: alphaAV(*)
    real(c_double), intent(in)  :: u(*)
    real(c_double), value       :: beta
    real(c_double), value       :: alphau
    real(c_double), intent(out) :: fx(*),fy(*),fz(*),f4(*)
    real(c_double), intent(out) :: vsigmax(*)
    real(c_double), intent(out) :: divv(*)
    end subroutine force_gpu_c

 end interface
#endif

 public :: force_gpu
 private

!
! Staging buffers shared by the routines below.  nbuf is their current capacity;
! ensure_buffers grows them when npart exceeds it and is otherwise a no-op.
! They are registered with the driver while allocated (gpu_dens_iface pin_buffer).
!
 integer :: nbuf = 0
 real(c_double), allocatable, target :: vx8(:),vy8(:),vz8(:)
 real(c_double), allocatable, target :: pro2_8(:),spsound_8(:),alphaAV_8(:),u_8(:)
 real(c_double), allocatable, target :: fx8(:),fy8(:),fz8(:),f48(:)
 real(c_double), allocatable, target :: vsigmax8(:),divv8(:)

contains

!-----------------------------------------------------------------------
!+
!  Evaluate the SPH force on the GPU and write the results back into
!  phantom's arrays.  Drop-in replacement for a call to force.
!+
!-----------------------------------------------------------------------
subroutine force_gpu(npart,xyzh,vxyzu,eos_vars,alphaind,fxyzu,divcurlv)
 use part,    only:massoftype,igas
 use options, only:beta,alphau
 use dim,     only:maxvxyzu,driving

 integer,      intent(in)    :: npart
 real,         intent(in)    :: xyzh(:,:),vxyzu(:,:)
 real,         intent(in)    :: eos_vars(:,:)
 real(kind=4), intent(in)    :: alphaind(:,:)
 real,         intent(inout) :: fxyzu(:,:)
 real(kind=4), intent(inout) :: divcurlv(:,:)

#ifdef GPU
 integer :: i

 if (npart <= 0) return

 call ensure_buffers(npart)
 call prepare_pro2_gpu(npart,xyzh,vxyzu,eos_vars,alphaind)

 !--positions and h are not sent: the GPU uses its copies from the density solve,
 !  and it reads these velocities only when they may differ from the solve's
 !$omp parallel do default(none) schedule(static) private(i) shared(npart,vxyzu,vx8,vy8,vz8)
 do i = 1,npart
    vx8(i) = real(vxyzu(1,i),kind=c_double)
    vy8(i) = real(vxyzu(2,i),kind=c_double)
    vz8(i) = real(vxyzu(3,i),kind=c_double)
 enddo
 !$omp end parallel do

 call force_gpu_c(int(npart,kind=c_int),                &
                  real(massoftype(igas),kind=c_double), &
                  vx8,vy8,vz8,                          &
                  pro2_8,spsound_8,alphaAV_8,u_8,       &
                  real(beta,kind=c_double),             &
                  real(alphau,kind=c_double),           &
                  fx8,fy8,fz8,f48,vsigmax8,divv8)

 !--as force.F90: with driving, fxyzu already holds the driving force (forceit
 !  runs first), so the SPH force is added to it.  Isothermal builds have no
 !  u, so fxyzu has no fourth row and du/dt is discarded.
 !$omp parallel do default(none) schedule(static) private(i) &
 !$omp shared(npart,fxyzu,divcurlv,fx8,fy8,fz8,f48,divv8)
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
    divcurlv(1,i) = real(divv8(i),kind=kind(divcurlv))
 enddo
 !$omp end parallel do

 call finish_gpu_force_timesteps(npart,xyzh)
#else
 print *, 'ERROR: force_gpu called but phantom not compiled with GPU=yes'
 stop
#endif

end subroutine force_gpu

!-----------------------------------------------------------------------
!+
!  Grow the staging buffers to hold at least n particles.  No-op once they
!  are big enough, so the allocation happens on the first force call only.
!+
!-----------------------------------------------------------------------
subroutine ensure_buffers(n)
 use gpu_dens_iface, only:pin_buffer,unpin_buffer
 integer, intent(in) :: n

 if (nbuf >= n) return

 if (allocated(vx8)) then
    call unpin_buffer(vx8);       call unpin_buffer(vy8)
    call unpin_buffer(vz8);       call unpin_buffer(pro2_8);     call unpin_buffer(spsound_8)
    call unpin_buffer(alphaAV_8); call unpin_buffer(u_8);        call unpin_buffer(fx8)
    call unpin_buffer(fy8);       call unpin_buffer(fz8);        call unpin_buffer(f48)
    call unpin_buffer(vsigmax8);  call unpin_buffer(divv8)
    deallocate(vx8,vy8,vz8, &
               pro2_8,spsound_8,alphaAV_8,u_8, &
               fx8,fy8,fz8,f48,vsigmax8,divv8)
 endif

 allocate(vx8(n),vy8(n),vz8(n), &
          pro2_8(n),spsound_8(n),alphaAV_8(n),u_8(n), &
          fx8(n),fy8(n),fz8(n),f48(n),vsigmax8(n),divv8(n))

 call pin_buffer(vx8);       call pin_buffer(vy8)
 call pin_buffer(vz8);       call pin_buffer(pro2_8);     call pin_buffer(spsound_8)
 call pin_buffer(alphaAV_8); call pin_buffer(u_8);        call pin_buffer(fx8)
 call pin_buffer(fy8);       call pin_buffer(fz8);        call pin_buffer(f48)
 call pin_buffer(vsigmax8);  call pin_buffer(divv8)

 nbuf = n

end subroutine ensure_buffers

!-----------------------------------------------------------------------
!+
!  Derive the per-particle inputs the force kernel needs, into the staging
!  buffers.  No MHD/radiation/physical viscosity branch of get_stress.
!+
!-----------------------------------------------------------------------
subroutine prepare_pro2_gpu(npart,xyzh,vxyzu,eos_vars,alphaind)
 use dim,     only:maxalpha,maxp,maxvxyzu
 use options, only:alpha
 use part,    only:igas,igasP,ics,massoftype,rhoh

 integer,      intent(in) :: npart
 real,         intent(in) :: xyzh(:,:)
 real,         intent(in) :: vxyzu(:,:)
 real,         intent(in) :: eos_vars(:,:)
 real(kind=4), intent(in) :: alphaind(:,:)

 integer :: i
 real    :: rhoi,rho1i

 !$omp parallel do default(none) schedule(static) private(i,rhoi,rho1i) &
 !$omp shared(npart,xyzh,vxyzu,eos_vars,alphaind,massoftype,alpha,maxalpha,maxp) &
 !$omp shared(pro2_8,spsound_8,u_8,alphaAV_8)
 do i = 1,npart
    rhoi         = rhoh(xyzh(4,i),massoftype(igas))
    rho1i        = 1.0/rhoi

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
subroutine finish_gpu_force_timesteps(npart,xyzh)
 use options,  only:alpha
 use timestep, only:C_cour,C_force,bignumber,dtmax, &
                    dtcourant,dtforce,dtrad
 use part,     only:isdead_or_accreted

 integer, intent(in) :: npart
 real,    intent(in) :: xyzh(:,:)

 integer :: i
 real    :: hi,vsigdtc,f2i,dtc,dtf,dtcmin,dtfmin

 dtcmin = bignumber
 dtfmin = bignumber
 dtrad  = bignumber

 !--min is exact under reduction, so this is bit-identical to the serial loop
 !$omp parallel do default(none) schedule(static) private(i,hi,vsigdtc,f2i,dtc,dtf) &
 !$omp shared(npart,xyzh,fx8,fy8,fz8,vsigmax8,spsound_8,dtmax,C_cour,C_force,alpha) &
 !$omp reduction(min:dtcmin,dtfmin)
 do i = 1,npart
    hi = xyzh(4,i)
    !--as force.F90: dead and accreted particles (h <= 0) set no constraint.
    !  Left in, one of them makes dtc negative and dtf the sqrt of a negative
    !  number, which poisons the global timestep.
    if (isdead_or_accreted(hi)) cycle

    vsigdtc = max(vsigmax8(i),spsound_8(i))

    !--as force.F90: no signal speed means no Courant constraint, not a
    !  division by zero
    dtc = dtmax
    if (vsigdtc > tiny(vsigdtc)) then
       dtc = C_cour*hi/(vsigdtc*max(alpha,1.0))
    endif

    !--as force.F90, from the SPH force alone (before any driving force is added)
    f2i = fx8(i)*fx8(i) + &
          fy8(i)*fy8(i) + &
          fz8(i)*fz8(i)

    dtf = bignumber

    if (f2i > 0.0) then
       dtf = C_force*sqrt(hi/sqrt(f2i))
    endif

    dtcmin = min(dtcmin,dtc)
    dtfmin = min(dtfmin,dtf)
 enddo
 !$omp end parallel do

 dtcourant = dtcmin
 dtforce   = dtfmin

end subroutine finish_gpu_force_timesteps

end module gpu_force_iface
