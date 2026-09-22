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
! ADDING A PHYSICS OPTION TO THE GPU PATH
! ---------------------------------------
! Worked example: suppose the GPU force pass is to read a new per-particle quantity,
! say the dust fraction, and return a new one, say the drag heating.  Five places,
! and the line counts are about what you should expect:
!
!  1. This file, with the other ibun_* parameters:
!
!        integer, parameter, public :: ibun_dust     = 9   ! dustfrac
!        integer, parameter, public :: ibun_dragheat = 10  ! du/dt from drag
!        integer, parameter, public :: nbundle       = 10
!
!  2. This file, in the table in gpu_arrays_init, one row each.  `live` is where the
!     physics option enters; a row that is not live costs no memory and moves nothing:
!
!        bundle(ibun_dust)     = bundle_t('dust',     1, idir_up,   use_dust)
!        bundle(ibun_dragheat) = bundle_t('dragheat', 1, idir_down, use_dust)
!
!     Prefer a NEW row over widening an existing one.  A row whose width depends on an
!     option makes the offsets conditional, and two such options interact.
!
!  3. cosmoSPHere/include/arrays.hpp, the same numbers again:
!
!        COSMO_DUST = 9, COSMO_DRAGHEAT = 10
!
!  4. cosmoSPHere/src/arrays_c_api.cu, slotComponents(), one case each naming the
!     device arrays, in the SAME component order as the comment on the row above.
!     (gpu_arrays_init checks the widths agree at start-up and stops if they do not;
!     it cannot check the order, so that one is on you.)
!
!  5. The interface that owns the quantity -- gpu_force_iface here -- packs it into
!     the slice, sends the bundle, and unpacks the result:
!
!        p => gpu_arrays_comp(ibun_dust,1)
!        do i = 1,npart
!           p(i) = real(dustfrac(1,i), kind=c_double)
!        enddo
!        call gpu_arrays_upload(ibun_dust,npart)
!
!     Put the pack in an existing loop over particles rather than adding a new one;
!     these are memory-bound sweeps and another pass over npart is not free.
!
! Then delete the `call fatal` in the interface that refuses the option, and write the
! kernel.  Everything else -- allocation on both sides, pinning, the ordering, the
! transfers -- follows from the table.
!
! WHAT THIS MODULE WILL NOT LET YOU DO
! ------------------------------------
! Data on the device is in the device's own particle order, not phantom's, and the two
! differ.  Sending a bundle at the wrong moment would therefore scramble it silently,
! which is the one mistake in here that produces plausible numbers rather than an
! error.  gpu_arrays_upload works out which is needed and refuses the cases that
! cannot be right, so there is no ordering decision to get wrong at the call site.
!
! :Dependencies: dim, iso_c_binding
!
 use iso_c_binding, only:c_double,c_int
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
 integer, parameter, public :: ibun_thermo   = 7   ! p/rho^2, c_s, alpha_AV, u
 integer, parameter, public :: ibun_force_out= 8   ! fx, fy, fz, du/dt, vsigmax, div v
 integer, parameter, public :: nbundle       = 8

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

!
! "This bundle already holds current values."  Set by whoever packs it, and taken
! (read and cleared) by a later pass that would otherwise pack the same thing again.
!
! Skipping a pack on the strength of this is safe in a way that skipping a transfer
! would not be: the slice is left holding the values the earlier pass put there, which
! are the ones this pass wants, so even if the C entry point does copy the slice it
! copies the right numbers.  The mark is cleared when taken, so only the pass
! immediately following the producer may skip -- a later one packs again, which is
! what the leapfrog corrector needs after it has changed the velocities.
!
 logical :: packed(nbundle) = .false.

!
! Does the ordering on the device match the data on it?
!
! It does not while the density pass is sending the positions for a new step, because
! the ordering still describes where the PREVIOUS step's particles were.  It does from
! the moment the solve has rebuilt the tree until the next such send.
!
! This is what decides which way a bundle goes up, so that a caller never has to.  Both
! wrong answers are refused rather than silently transferring in the wrong order, which
! is the one failure in here that would produce plausible numbers.
!
 logical :: order_matches = .false.

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

!--C interface to cosmoSPHere/src/arrays_c_api.cu: sizes the device's particle-length
!  arrays, so the footprint is decided here rather than by whichever kernel first
!  touched an array.  The entry points refuse to run on a different count.
  subroutine cosmo_arrays_init(n) bind(C)
   use iso_c_binding, only:c_int
   integer(c_int), value :: n
  end subroutine cosmo_arrays_init

!--Copy one bundle back from the device into its arena slice.  The slice is ncomp
!  contiguous runs of n, which is the order the device writes them in, so one call
!  covers the whole bundle.
  subroutine cosmo_download(slot, host, n) bind(C)
   use iso_c_binding, only:c_double,c_int
   integer(c_int), value      :: slot
   real(c_double), intent(out) :: host(*)
   integer(c_int), value      :: n
  end subroutine cosmo_download

!--Send a bundle to the device as it stands, in phantom's particle order.  For use
!  before the tree for this particle set exists: the tree build sorts it.
  subroutine cosmo_upload(slot, host, n) bind(C)
   use iso_c_binding, only:c_double,c_int
   integer(c_int), value     :: slot
   real(c_double), intent(in) :: host(*)
   integer(c_int), value     :: n
  end subroutine cosmo_upload

!--Send a bundle and put it in the device's own particle order.  For use once the
!  tree exists, which is everything after the density solve.
  subroutine cosmo_upload_sorted(slot, host, n) bind(C)
   use iso_c_binding, only:c_double,c_int
   integer(c_int), value     :: slot
   real(c_double), intent(in) :: host(*)
   integer(c_int), value     :: n
  end subroutine cosmo_upload_sorted

!--How many components cosmoSPHere thinks a slot has, so the table below can be
!  checked against it at start-up.
  integer(c_int) function cosmo_slot_ncomp(slot) bind(C)
   use iso_c_binding, only:c_int
   integer(c_int), value :: slot
  end function cosmo_slot_ncomp
 end interface
#endif

 public :: gpu_arrays_init, gpu_arrays_comp, gpu_arrays_nbuf
 public :: gpu_arrays_upload, gpu_arrays_download
 public :: gpu_arrays_positions_moved, gpu_arrays_order_rebuilt
 public :: gpu_arrays_mark_packed, gpu_arrays_claim_packed
 public :: pin_buffer, unpin_buffer

contains

!-------------------------------------------------------------
!+
!  Describe the bundles for this build and this run, then size and
!  pin the arena to hold at least n particles.  No-op once it is big
!  enough and the table has not changed, so the allocation happens on
!  the first solve only.
!
!  `live` is where the physics options enter.  Everything the calls
!  need is live in this step because the C entry points still take all
!  of it; the option each row will key off once they can be told to
!  leave something out is named in a comment against that row.
!+
!-------------------------------------------------------------
subroutine gpu_arrays_init(n)
#ifdef GPU
 use io, only:fatal
#endif
 integer, intent(in) :: n
 integer :: ib, off
#ifdef GPU
 integer :: ncomp_there
#endif

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
 !--force pass.  u and du/dt exist only when maxvxyzu >= 4, but the C entry point
 !  takes them either way, so the counts are fixed until it can be told to leave
 !  them out; that is the same change that makes a row's width depend on an option.
 bundle(ibun_thermo)   = bundle_t('thermo',   4, idir_up,     .true.)
 bundle(ibun_force_out)= bundle_t('force_out',6, idir_down,   .true.)

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
#ifdef GPU
 !--size the device side to match, in one call, from the same n
 call cosmo_arrays_init(int(n, kind=c_int))

 !--the bundle ids here and the slot ids in cosmoSPHere/include/arrays.hpp are the
 !  same numbers by convention, and nothing but this makes them agree.  A row added
 !  on one side only, or with the wrong width, would transfer the wrong array and
 !  produce plausible numbers, so check every live row against what that side thinks
 !  before any of it is used.
 do ib = 1, nbundle
    if (.not.bundle(ib)%live .or. bundle(ib)%idir == idir_device_only) cycle
    ncomp_there = int(cosmo_slot_ncomp(int(ib, kind=c_int)))
    if (ncomp_there /= bundle(ib)%ncomp) &
       call fatal('gpu_arrays_init','bundle '//trim(bundle(ib)%name)//' has a different '//&
                  'number of components in cosmoSPHere: keep gpu_arrays and arrays.hpp in step', &
                  ival=ncomp_there)
 enddo
#endif
 packed = .false.   ! nothing in a fresh arena holds anything
 nbuf = n

end subroutine gpu_arrays_init

!-------------------------------------------------------------
!+
!  Record that a bundle now holds current values (see `packed` above).
!+
!-------------------------------------------------------------
subroutine gpu_arrays_mark_packed(ib)
 integer, intent(in) :: ib

 packed(ib) = .true.

end subroutine gpu_arrays_mark_packed

!-------------------------------------------------------------
!+
!  Claim the mark: says whether the pass just before this one packed
!  the bundle, and clears it.  One call rather than a query and a
!  separate clear, because the two could drift apart and forgetting the
!  clear would leave the corrector running on the predictor's values.
!+
!-------------------------------------------------------------
logical function gpu_arrays_claim_packed(ib)
 integer, intent(in) :: ib

 gpu_arrays_claim_packed = packed(ib)
 packed(ib) = .false.

end function gpu_arrays_claim_packed

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
!  Fetch a bundle from the device into its arena slice.  The bundle
!  ids here and the slot ids in cosmoSPHere/include/arrays.hpp are the
!  same numbers and must stay in step.
!+
!-------------------------------------------------------------
subroutine gpu_arrays_download(ib, n)
 integer, intent(in) :: ib, n
#ifdef GPU
 real(c_double), pointer, contiguous :: p(:)

 p => gpu_arrays_comp(ib,1)      ! component 1 starts the bundle
 call cosmo_download(int(ib, kind=c_int), p, int(n, kind=c_int))
#endif

end subroutine gpu_arrays_download

!-------------------------------------------------------------
!+
!  Send a bundle to the device.  This is the one to use.
!
!  Whether it needs putting into the device's own particle order
!  depends on where in the step we are; this works that out, so a
!  caller adding a new quantity does not have to know, and cannot
!  get it wrong quietly.
!+
!-------------------------------------------------------------
subroutine gpu_arrays_upload(ib, n)
#ifdef GPU
 use io, only:fatal
#endif
 integer, intent(in) :: ib, n
#ifdef GPU
 real(c_double), pointer, contiguous :: p(:)

 p => gpu_arrays_comp(ib,1)
 if (order_matches) then
    call cosmo_upload_sorted(int(ib, kind=c_int), p, int(n, kind=c_int))
 else
    !--the ordering on the device is stale, which is only the case between new
    !  positions and the solve that sorts them; the only bundles that belong in
    !  that window are the ones the solve itself reads
    if (ib /= ibun_pos .and. ib /= ibun_hsml .and. &
        ib /= ibun_vel .and. ib /= ibun_accel) &
       call fatal('gpu_arrays_upload','bundle sent between new positions and the '//&
                  'solve that sorts them: move it after the density solve',ival=ib)
    call cosmo_upload(int(ib, kind=c_int), p, int(n, kind=c_int))
 endif
#endif

end subroutine gpu_arrays_upload

!-------------------------------------------------------------
!+
!  The positions about to be sent are for a new step, so the
!  ordering on the device still describes where the particles were.
!  Called by the density pass, the only thing that moves them.
!+
!-------------------------------------------------------------
subroutine gpu_arrays_positions_moved()

 order_matches = .false.

end subroutine gpu_arrays_positions_moved

!-------------------------------------------------------------
!+
!  The solve has rebuilt the tree, so the ordering describes the
!  data on the device again.
!+
!-------------------------------------------------------------
subroutine gpu_arrays_order_rebuilt()

 order_matches = .true.

end subroutine gpu_arrays_order_rebuilt

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
