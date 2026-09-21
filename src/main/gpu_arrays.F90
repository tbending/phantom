!--------------------------------------------------------------------------!
! The Phantom Smoothed Particle Hydrodynamics code, by Daniel Price et al. !
! Copyright (c) 2007-2026 The Authors (see AUTHORS)                        !
! See LICENCE file for usage and distribution conditions                   !
! http://phantomsph.github.io/                                             !
!--------------------------------------------------------------------------!
module gpu_arrays
!
! Host-side staging for the GPU passes: one pinned arena, described by a table.
!
! WHY THIS EXISTS
! ---------------
! The density and force interfaces each used to declare their own staging buffers,
! pin them, and grow them, with the force interface borrowing pin_buffer from the
! density one.  The same array was therefore packed by both, the rule for what moves
! under which physics option was an `if` in whichever interface happened to need it,
! and the residency contract (what is already on the device) lived on the C++ side
! where Fortran could not see it.
!
! Here the host owns the staging and the table states, in one place, what exists and
! why.  Adding a physics option to the GPU path becomes a row rather than an edit to
! two interfaces.
!
! THE ARENA
! ---------
! One allocation, pinned once, grown only if npart rises.  A bundle occupies a
! contiguous slice; within a bundle each component is a contiguous n-element run.
! So a component is still a plain double array that a bind(C) call can take, while
! the bundle as a whole is one contiguous block that a single transfer can move.
!
! SCOPE OF THIS STEP
! ------------------
! This is a relocation: the buffers move here and the interfaces stop owning them.
! The transfers themselves are still issued inside the cosmoSPHere C entry points,
! so `idir` below is recorded but not yet acted on, and every bundle a call needs is
! live.  Splitting the C API into transfer and compute calls, at which point `idir`
! and `live` start driving what actually moves, is a later step.
!
! :Dependencies: dim, iso_c_binding
!
 use iso_c_binding, only:c_double
 implicit none

 private

!
! Directions.  idir_device_only is for quantities that are produced on the device and
! consumed there (the velocity gradient tensor is the coming case: the force pass needs
! a neighbour's tensor, so it must stay resident and never be staged).  They take no
! arena space; the row exists so the table still describes them.
!
 integer, parameter, public :: idir_up          = 1
 integer, parameter, public :: idir_down        = 2
 integer, parameter, public :: idir_updown      = 3
 integer, parameter, public :: idir_device_only = 4

!
! Bundles.  Kept as parameters so that a use of one is a compile-time constant and the
! compiler can fold the surrounding logic, per the general rule in this code that a
! runtime test on a parameter costs nothing.
!
 integer, parameter, public :: ibun_pos      = 1   ! x, y, z
 integer, parameter, public :: ibun_hsml     = 2   ! h, in and out
 integer, parameter, public :: ibun_vel      = 3   ! vx, vy, vz
 integer, parameter, public :: ibun_accel    = 4   ! ax, ay, az (fxyzu + fext)
 integer, parameter, public :: ibun_dens_out = 5   ! rho, d(rho)/d(h)
 integer, parameter, public :: ibun_grad_out = 6   ! div v, xi, d(div v)/dt
 integer, parameter, public :: nbundle       = 6

 type :: bundle_t
    character(len=10) :: name  = ''
    integer           :: ncomp = 0
    integer           :: idir  = idir_up
    logical           :: live  = .false.
    integer           :: ioff  = 0   ! first element in the arena, 1-based
 end type bundle_t

 type(bundle_t) :: bundle(nbundle)

!
! The arena.  target so that c_loc can pin it, and so component pointers can be
! associated with slices of it.
!
 real(c_double), allocatable, target :: arena(:)
 integer :: nbuf = 0   ! particles the arena is currently sized for

#ifdef GPU
 interface
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

 public :: gpu_arrays_init, gpu_arrays_comp, gpu_arrays_nbuf
 public :: pin_buffer, unpin_buffer

contains

!-------------------------------------------------------------
!+
!  Describe the bundles for this build and this run, then size and
!  pin the arena to hold at least n particles.  No-op once it is big
!  enough and the table has not changed, so the allocation happens on
!  the first solve only.
!
!  `live` is where the physics options enter.  Everything the density
!  pass needs is live in this step because the C entry point still
!  takes all of it; the predicates each row will take once the API
!  splits are named against them below.
!+
!-------------------------------------------------------------
subroutine gpu_arrays_init(n)
 integer, intent(in) :: n
 integer :: ib, off

 if (n <= nbuf .and. allocated(arena)) return

 !--positions and h: always, for any GPU pass
 bundle(ibun_pos)      = bundle_t('pos',      3, idir_up,     .true.)
 bundle(ibun_hsml)     = bundle_t('hsml',     1, idir_updown, .true.)
 !--velocity: the density sweep differentiates it, the force sum uses it
 bundle(ibun_vel)      = bundle_t('vel',      3, idir_up,     .true.)
 !--acceleration: only the Cullen & Dehnen source term needs it, so this
 !  becomes `nalpha >= 2` once the call can leave it out
 bundle(ibun_accel)    = bundle_t('accel',    3, idir_up,     .true.)
 bundle(ibun_dens_out) = bundle_t('dens_out', 2, idir_down,   .true.)
 !--div v is always wanted; xi and d(div v)/dt are the Cullen & Dehnen
 !  switch, so this row splits when the call can leave them out
 bundle(ibun_grad_out) = bundle_t('grad_out', 3, idir_down,   .true.)

 !--lay the live bundles out end to end.  device_only rows take no arena space.
 off = 1
 do ib = 1, nbundle
    if (bundle(ib)%live .and. bundle(ib)%idir /= idir_device_only) then
       bundle(ib)%ioff = off
       off = off + bundle(ib)%ncomp*n
    else
       bundle(ib)%ioff = 0
    endif
 enddo

 if (allocated(arena)) then
    call unpin_buffer(arena)
    deallocate(arena)
 endif
 allocate(arena(off-1))
 call pin_buffer(arena)
 nbuf = n

end subroutine gpu_arrays_init

!-------------------------------------------------------------
!+
!  Pointer to one component of one bundle, as a contiguous run of
!  nbuf doubles.  This is what gets handed to the bind(C) calls, and
!  what the pack and unpack loops write through.
!
!  contiguous is truthful here, not a hint: the target is a stride-1
!  section of a rank-1 array, so no copy is made.
!+
!-------------------------------------------------------------
function gpu_arrays_comp(ib, ic) result(p)
 integer, intent(in) :: ib, ic
 real(c_double), pointer, contiguous :: p(:)
 integer :: i0

 i0 = bundle(ib)%ioff + (ic-1)*nbuf
 p => arena(i0:i0+nbuf-1)

end function gpu_arrays_comp

!-------------------------------------------------------------
!+
!  Particles the arena is sized for.  Only for assertions and stats.
!+
!-------------------------------------------------------------
pure integer function gpu_arrays_nbuf()

 gpu_arrays_nbuf = nbuf

end function gpu_arrays_nbuf

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

end module gpu_arrays
