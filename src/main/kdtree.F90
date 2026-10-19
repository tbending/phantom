!--------------------------------------------------------------------------!
! The Phantom Smoothed Particle Hydrodynamics code, by Daniel Price et al. !
! Copyright (c) 2007-2026 The Authors (see AUTHORS)                        !
! See LICENCE file for usage and distribution conditions                   !
! http://phantomsph.github.io/                                             !
!--------------------------------------------------------------------------!
module kdtree
!
! This module implements the k-d tree build
!    and associated tree walking routines
!
! :References:
!    Gafton & Rosswog (2011), MNRAS 418, 770-781
!    Benz, Bowers, Cameron & Press (1990), ApJ 348, 647-667
!    Dehnen (2000), ApJL 536, L39; Dehnen (2002), JCoPh 179, 27
!    Marcello (2017), AJ 154, 92 (angular momentum conservation)
!
! :Owner: Daniel Price
!
! :Runtime parameters: None
!
! :Dependencies: allocutils, boundary, dim, dtypekdtree, io, mpibalance,
!   mpidomain, mpitree, mpiutils, part, timing
!
 use, intrinsic :: ieee_arithmetic, only:ieee_is_nan
 use dim,         only:maxp,ncellsmax,minpart,use_apr,use_sinktree,maxptmass,maxpsph,gravity
 use io,          only:nprocs
 use dtypekdtree, only:kdnode,lenfgrav
 use part,        only:ll,iphase,treecache,maxphase, &
                       apr_level,aprmassoftype

 implicit none

 integer, public,  allocatable :: inoderange(:,:)
 integer, public,  allocatable :: inodeparts(:)
 type(kdnode),     allocatable :: refinementnode(:)
!
!--tree parameters
!
 integer,          parameter, public :: irootnode    = 1
 character(len=1), parameter, public :: labelax(3)   = (/'x','y','z'/)
 integer,          parameter, public :: maxdepth     = 128
!
!--runtime options for this module
!
 real,    public  :: tree_accuracy    = 0.5
 logical, public  :: use_geosplit     = gravity ! only debug flag / should be on with gravity
 logical, private :: done_init_kdtree = .false.
 logical, private :: already_warned   = .false.
 integer, private :: numthreads
 ! scratch space for the parallel partition in build_top_parallel
 real,    allocatable, private :: tcbuf(:,:)
 integer, allocatable, private :: ipbuf(:)
 ! what revise_tree needs from the last full build: the nodes in the tree, the list of its
 ! leaves, the node masses and the particle count when it was built
 logical, private :: can_revise = .false.
 integer, private :: nleaves_built = 0, npart_built = -1, maxlevel_built = -1
 integer,         allocatable, private :: leaves_built(:),leafk(:)   ! leaves, and each leaf's number
 ! for revise_all_leaves: the subtrees rooted at level sub_lsub, split into one run of
 ! consecutive subtrees per thread with about equal numbers of particles (thread it
 ! takes roots sub_first(it) to sub_first(it+1)-1), for a team of sub_nthreads
 integer,         allocatable, private :: sub_first(:)
 integer, private :: sub_lsub = -1, sub_ltop = -1, sub_nthreads = -1
 integer(kind=1), allocatable, private :: in_tree(:)
 real,            allocatable, private :: mass_node(:)
 ! phase 2: a leaf whose particles are all inactive (and none newly woken) is given, when
 ! visited, a sphere that holds its particles as they drift until t_valid (the first time one
 ! of them is due to become active), so it need not be touched again until then.
 ! Per particle: its leaf (as an index into leaves_built) and its timestep bin when last seen;
 ! per leaf: whether it held an active particle when last visited, and until when its sphere
 ! holds.  visit_list holds the leaves to visit; node_stamp marks the internal nodes above
 ! them (dirty_list), which are the only ones recomputed.
 integer,         allocatable, private :: leaf_of(:)
 integer(kind=1), allocatable, private :: ibin_seen(:)
 logical,         allocatable, private :: was_active(:)
 real,            allocatable, private :: t_valid(:)
 integer,         allocatable, private :: visit_mark(:),visit_list(:),node_stamp(:),dirty_list(:)
 logical, private :: bounds_valid = .false.
 integer(kind=8), private :: ntot_seen = -1   ! particles alive at the last build or revision
 ! treecache's positions and h are current only after a full build (all_fresh); revisions
 ! do not refresh them, and the neighbour search then reads xyzh.  Its masses stay
 ! aligned with inodeparts (moved when a revision drops a dead particle).
 logical, public  :: all_fresh = .true.
 integer, private :: tree_id = 0
 integer, public  :: nleaves_visited = 0   ! leaves visited by the last revision (diagnostic)

! Index of the last node in the local tree that has been copied to
! the global tree
 integer, public :: irefine

 public :: allocate_kdtree, deallocate_kdtree
 public :: maketree, revtree,kdnode
 public :: revise_tree
 public :: maketreeglobal
 public :: empty_tree
 integer, public :: maxlevel_indexed, maxlevel

 ! neighbour cache indices (xyzcache); imported with only: from dens/force
 integer, parameter, public :: ix=1, iy=2, iz=3, ih1=4, im=5, irho=6, izetaomega=7, isoftomega=8

 type kdbuildstack
    integer :: node
    integer :: parent
    integer :: level
    integer :: npnode
    real    :: xmin(3)
    real    :: xmax(3)
 end type kdbuildstack

 private

contains

subroutine allocate_kdtree
 use dim, only:mpi
 use allocutils, only:allocate_array

 call allocate_array('inoderange', inoderange, 2, ncellsmax+1)
 call allocate_array('inodeparts', inodeparts, maxp)
 if (mpi) call allocate_array('refinementnode', refinementnode, ncellsmax+1)

end subroutine allocate_kdtree

subroutine deallocate_kdtree
 use dim, only:mpi
 if (allocated(inoderange)) deallocate(inoderange)
 if (allocated(inodeparts)) deallocate(inodeparts)
 if (mpi .and. allocated(refinementnode)) deallocate(refinementnode)
 if (allocated(tcbuf)) deallocate(tcbuf,ipbuf)
 if (allocated(leaves_built)) deallocate(leaves_built,leafk,in_tree,mass_node)
 if (allocated(sub_first)) deallocate(sub_first)
 if (allocated(leaf_of)) deallocate(leaf_of,ibin_seen)
 if (allocated(was_active)) deallocate(was_active,t_valid,visit_mark,visit_list)
 if (allocated(node_stamp)) deallocate(node_stamp,dirty_list)
 bounds_valid = .false.
 all_fresh    = .true.
 can_revise = .false.

end subroutine deallocate_kdtree

!--------------------------------------------------------------------------------
!+
!  Routine to build the tree from scratch
!
!  Notes/To do:
!  -openMP parallelisation of maketree_stack (done - April 2013)
!  -test centre of mass vs. geometric centre based cell sizes (done - 2013)
!  -need analysis module that times (and checks) build_tree and tree walk
!   for a given dump
!  -test bottom-up vs. top-down neighbour search
!  -should we try to store tree structure with particle arrays?
!  -need to compute centre of mass and moments for each cell on the fly (c.f. revtree?)
!  -need to implement long-range gravitational interaction (done - May 2013)
!  -implement revtree routine to update tree w/out rebuilding (done - Sep 2015)
!+
!-------------------------------------------------------------------------------
subroutine maketree(node, xyzh, np, leaf_is_active, ncells, apr_tree, refinelevels,nptmass,xyzmh_ptmass)
 use io,   only:fatal,warning,iprint,iverbose
!$ use omp_lib
 type(kdnode),    intent(out)   :: node(:) !ncellsmax+1)
 integer,         intent(in)    :: np
 real,            intent(inout) :: xyzh(:,:)  ! inout because of boundary crossing
 integer,         intent(out)   :: leaf_is_active(:) !ncellsmax+1)
 integer(kind=8), intent(out)   :: ncells
 logical,         intent(in)    :: apr_tree
 integer,         intent(out),   optional :: refinelevels
 integer,         intent(in),    optional :: nptmass
 real,            intent(inout), optional :: xyzmh_ptmass(:,:)

 integer :: i,npnode,il,ir,istack,nl,nr,mymum
 integer :: nnode,minlevel,level,nqueue
 real :: xmini(3),xmaxi(3),xminl(3),xmaxl(3),xminr(3),xmaxr(3)
 integer, parameter :: istacksize = 512
 type(kdbuildstack), save :: stack(istacksize)
 !$omp threadprivate(stack)
 type(kdbuildstack) :: queue(istacksize)
!$ integer :: threadid
 integer :: npcounter
 logical :: wassplit,finished,sinktree
 character(len=10) :: string

 ! must initialise: otherwise a garbage .true. takes the sinktree branch
 ! with absent optionals (illegal access of nptmass/xyzmh_ptmass)
 sinktree = .false.
 if (present(nptmass) .and. present(xyzmh_ptmass)) then
    sinktree = .true.
 endif

 !$omp parallel do schedule(static) default(none) shared(leaf_is_active) private(i)
 do i=1,size(leaf_is_active)
    leaf_is_active(i) = 0
 enddo
 !$omp end parallel do

 ir = 0
 il = 0
 nl = 0
 nr = 0
 wassplit = .false.
 finished = .false.

 ! construct root node, i.e. find bounds of all particles
 if (sinktree) then
    call construct_root_node(np,npcounter,irootnode,xmini,xmaxi,leaf_is_active,xyzh,xyzmh_ptmass,nptmass)
 else
    call construct_root_node(np,npcounter,irootnode,xmini,xmaxi,leaf_is_active,xyzh)
 endif

 if (inoderange(1,irootnode)==0 .or. inoderange(2,irootnode)==0 ) then
    call fatal('maketree','no particles or all particles dead/accreted')
 endif

! Put root node on top of stack
 ncells = 1
 maxlevel = 0
 minlevel = maxdepth - 1
 istack = 1

 ! maximum level where 2^k indexing can be used (thus avoiding critical sections)
 ! deeper than this we access cells via a stack as usual
 if (use_geosplit) then
    maxlevel_indexed = int(log(real(ncellsmax+1))/log(2.)) - 2
 else
    maxlevel_indexed = int(log(real(ncellsmax+1))/log(2.)) - 1
 endif

 ! default number of cells is the size of the `indexed' part of the tree
 ! this can be *increased* by building tree beyond indexed levels
 ! and is decreased afterwards according to the maximum depth actually reached
 ncells = 2**(maxlevel_indexed+1) - 1

 ! need to number of particles in node during build
 ! this is counted above to remove dead/accreted particles
 call push_onto_stack(queue(istack),irootnode,0,0,npcounter,xmini,xmaxi)

 if (.not.done_init_kdtree) then
    ! 1 thread for serial, overwritten when using OpenMP
    numthreads = 1

    ! get number of OpenMP threads
    !$omp parallel default(none) shared(numthreads)
!$  numthreads = omp_get_num_threads()
    !$omp end parallel
    done_init_kdtree = .true.
 endif

 nqueue = numthreads
 ! build the first levels with all threads working on every level (not for the APR
 ! merge tree, whose partition has to leave an even number of particles in each child)
 if (.not.apr_tree .and. nqueue > 1 .and. npcounter > max(minpart,64)) then
    call build_top_parallel(node,queue,istack,nqueue,leaf_is_active)
 endif
 ! build using a queue to build level by level until number of nodes = number of threads
 over_queue: do while (istack  <  nqueue)
    ! if the tree finished while building the queue, then we should just return
    ! only happens for small particle numbers
    if (istack <= 0) then
       finished = .true.
       exit over_queue
    endif
    ! pop off front of queue
    call pop_off_stack(queue(1), istack, nnode, mymum, level, npnode, xmini, xmaxi)

    ! shuffle queue forward
    do i=1,istack
       queue(i) = queue(i+1)
    enddo

    ! construct node
    if (sinktree) then
       call construct_node(node(nnode), nnode, mymum, level, xmini, xmaxi, npnode, .true., &  ! construct in parallel
                           il, ir, nl, nr, xminl, xmaxl, xminr, xmaxr, ncells, leaf_is_active, &
                           minlevel, maxlevel, wassplit, .false., apr_tree, xyzmh_ptmass)
    else
       call construct_node(node(nnode), nnode, mymum, level, xmini, xmaxi, npnode, .true., &  ! construct in parallel
                           il, ir, nl, nr, xminl, xmaxl, xminr, xmaxr, ncells, leaf_is_active, &
                           minlevel, maxlevel, wassplit, .false., apr_tree)
    endif

    if (wassplit) then ! add children to back of queue
       if (istack+2 > istacksize) call fatal('maketree',&
                                       'queue size exceeded in tree build, increase istacksize and recompile')

       istack = istack + 1
       call push_onto_stack(queue(istack),il,nnode,level+1,nl,xminl,xmaxl)
       istack = istack + 1
       call push_onto_stack(queue(istack),ir,nnode,level+1,nr,xminr,xmaxr)
    endif

 enddo over_queue
 ! build_top_parallel completes whole levels, so the queue may hold more than nqueue nodes
 if (.not.finished) nqueue = istack

 ! fix the indices

 done: if (.not.finished) then

    ! build using a stack which builds depth first
    ! each thread grabs a node from the queue and builds its own subtree

    !$omp parallel default(none) &
    !$omp shared(queue) &
    !$omp shared(ll, leaf_is_active) &
    !$omp shared(xyzmh_ptmass) &
    !$omp shared(np) &
    !$omp shared(node, ncells) &
    !$omp shared(nqueue,apr_tree,sinktree) &
    !$omp private(istack) &
    !$omp private(nnode, mymum, level, npnode, xmini, xmaxi) &
    !$omp private(ir, il, nl, nr) &
    !$omp private(xminr, xmaxr, xminl, xmaxl) &
    !$omp private(threadid) &
    !$omp private(wassplit) &
    !$omp reduction(min:minlevel) &
    !$omp reduction(max:maxlevel)
    !$omp do schedule(dynamic,1)
    do i = 1, nqueue

       stack(1) = queue(i)
       istack = 1

       over_stack: do while(istack > 0)

          ! pop node off top of stack
          call pop_off_stack(stack(istack), istack, nnode, mymum, level, npnode, xmini, xmaxi)

          ! construct node
          if (sinktree) then
             call construct_node(node(nnode), nnode, mymum, level, xmini, xmaxi, npnode, .false., &  ! don't construct in parallel
                                 il, ir, nl, nr, xminl, xmaxl, xminr, xmaxr, ncells, leaf_is_active, &
                                 minlevel, maxlevel, wassplit, .false., apr_tree, xyzmh_ptmass)
          else
             call construct_node(node(nnode), nnode, mymum, level, xmini, xmaxi, npnode, .false., &  ! don't construct in parallel
                                 il, ir, nl, nr, xminl, xmaxl, xminr, xmaxr, ncells, leaf_is_active, &
                                 minlevel, maxlevel, wassplit, .false., apr_tree)
          endif

          if (wassplit) then ! add children to top of stack
             if (istack+2 > istacksize) call fatal('maketree',&
                                       'stack size exceeded in tree build, increase istacksize and recompile')

             istack = istack + 1
             call push_onto_stack(stack(istack),il,nnode,level+1,nl,xminl,xmaxl)
             istack = istack + 1
             call push_onto_stack(stack(istack),ir,nnode,level+1,nr,xminr,xmaxr)
          endif

       enddo over_stack
    enddo
    !$omp enddo
    !$omp end parallel

 endif done

 ! decrease number of cells if tree is entirely within 2^k indexing limit
 if (maxlevel < maxlevel_indexed) then
    ncells = 2**(maxlevel+1) - 1
 endif

 if (maxlevel > maxlevel_indexed .and. .not.already_warned) then
    write(string,"(i10)") 2**(maxlevel-maxlevel_indexed)
    if (iverbose > 0) call warning('maketree','maxlevel > max_indexed: will run faster if recompiled with '// &
               'NCELLSMAX='//trim(adjustl(string))//'*maxp,')
 endif

 if (present(refinelevels)) refinelevels = minlevel

 ! keep what revise_tree needs to update this tree in place (not for the APR merge tree,
 ! which is built on a subset of the particles and overwrites the main tree)
 call record_tree(node,np,ncells,leaf_is_active,apr_tree)
 all_fresh = .true.   ! treecache holds every particle of this tree

 if (iverbose >= 3) then
    write(iprint,"(a,i10,3(a,i2))") ' maketree: nodes = ',ncells,', max level = ',maxlevel,&
       ', min leaf level = ',minlevel,' max level indexed = ',maxlevel_indexed
 endif

end subroutine maketree

!--------------------------------------------------------------------------------
!+
!  After a full build: record what revise_tree needs to update the tree in place.
!  The nodes that belong to this tree are found top-down from the root (the node
!  array keeps stale entries from earlier builds), and the leaves are listed.
!  Revision is possible only within the indexed levels (children of n are 2n, 2n+1).
!+
!--------------------------------------------------------------------------------
subroutine record_tree(node,np,ncells,leaf_is_active,apr_tree)
!$ use omp_lib, only:omp_get_max_threads
 use part, only:gravity
 type(kdnode),    intent(in) :: node(:)
 integer,         intent(in) :: np
 integer(kind=8), intent(in) :: ncells
 integer,         intent(in) :: leaf_is_active(:)
 logical,         intent(in) :: apr_tree
 integer, allocatable :: nlist(:)
 integer :: n,l,nc,ic,nchunk,k,ltop,lsub,ir,d

 can_revise = .false.
 ! (not with self-gravity: revise_tree does not update the nodes' multipole moments)
 if (gravity .or. apr_tree .or. maxlevel > maxlevel_indexed) return
 nc = int(ncells)
 if (.not.allocated(leaves_built)) then
    allocate(leaves_built(size(node)),leafk(size(node)),in_tree(size(node)),mass_node(size(node)))
 endif
 nchunk = 1
!$ nchunk = omp_get_max_threads()
 allocate(nlist(0:nchunk))
 nlist = 0

 call subtree_levels(nchunk,maxlevel,ltop,lsub)

 !$omp parallel default(none) &
 !$omp shared(nc,node,leaf_is_active,in_tree,leaves_built,leafk,nlist,nchunk,maxlevel,ltop,lsub) &
 !$omp shared(inoderange) &
 !$omp private(n,l,ic,k,ir,d)
 !$omp do schedule(static)
 do n=1,nc
    in_tree(n) = 0
 enddo
 !$omp end do
 ! a node is in the tree if its parent is an internal node of the tree: top down, the
 ! first levels serially, then a loop per level, then whole subtrees (see subtree_levels)
 !$omp single
 in_tree(1) = 1
 do l=0,ltop-1
    do n=2**l,min(2**(l+1)-1,nc)
       call mark_children(n)
    enddo
 enddo
 !$omp end single
 do l=ltop,lsub-1
    !$omp do schedule(static)
    do n=2**l,min(2**(l+1)-1,nc)
       call mark_children(n)
    enddo
    !$omp end do
 enddo
 !$omp do schedule(static,1)
 do ir=2**lsub,min(2**(lsub+1)-1,nc)
    do d=0,maxlevel-1-lsub
       do n=ir*2**d,min((ir+1)*2**d-1,nc)
          call mark_children(n)
       enddo
    enddo
 enddo
 !$omp end do
 ! the leaves listed (counted per chunk, then placed)
 !$omp do schedule(static)
 do ic=1,nchunk
    k = 0
    do n=int((int(ic-1,8)*nc)/nchunk)+1,int((int(ic,8)*nc)/nchunk)
       if (in_tree(n) == 1 .and. leaf_is_active(n) /= 0) k = k + 1
    enddo
    nlist(ic) = k
 enddo
 !$omp end do
 !$omp single
 do ic=1,nchunk
    nlist(ic) = nlist(ic) + nlist(ic-1)
 enddo
 !$omp end single
 !$omp do schedule(static)
 do ic=1,nchunk
    k = nlist(ic-1)
    do n=int((int(ic-1,8)*nc)/nchunk)+1,int((int(ic,8)*nc)/nchunk)
       if (in_tree(n) == 1 .and. leaf_is_active(n) /= 0) then
          k = k + 1
          leaves_built(k) = n
          leafk(n) = k
       endif
    enddo
 enddo
 !$omp end do
 !$omp end parallel

 nleaves_built  = nlist(nchunk)
 npart_built    = np
 maxlevel_built = maxlevel
 can_revise     = .true.
 deallocate(nlist)
 call split_subtrees(nchunk,nc)
 call record_leaves_of_particles()

contains

subroutine mark_children(n)
 integer, intent(in) :: n

 if (in_tree(n) == 1 .and. leaf_is_active(n) == 0 .and. inoderange(1,n) > 0) then
    in_tree(node(n)%leftchild)  = 1
    in_tree(node(n)%rightchild) = 1
 endif

end subroutine mark_children

end subroutine record_tree

!--------------------------------------------------------------------------------
!+
!  For phase 2 of revise_tree: every leaf to be visited at the next revision, which
!  sets up its drift bounds and records each of its particles' leaf and timestep bin
!+
!--------------------------------------------------------------------------------
subroutine record_leaves_of_particles()
 use part, only:npartoftype
 integer :: k

 if (.not.allocated(leaf_of)) then
    allocate(leaf_of(size(inodeparts)),ibin_seen(size(inodeparts)))
    leaf_of = 0
 endif
 if (allocated(was_active)) then
    if (size(was_active) < nleaves_built) deallocate(was_active,t_valid,visit_mark,visit_list)
 endif
 if (.not.allocated(was_active)) then
    allocate(was_active(nleaves_built),t_valid(nleaves_built),visit_mark(nleaves_built),&
             visit_list(nleaves_built))
 endif
 if (.not.allocated(node_stamp)) then
    allocate(node_stamp(size(inoderange,2)),dirty_list(size(inoderange,2)))
    node_stamp = -1
 endif
 tree_id = tree_id + 1
 !$omp parallel do schedule(static) default(none) &
 !$omp shared(nleaves_built,was_active,visit_mark) &
 !$omp private(k)
 do k=1,nleaves_built
    was_active(k) = .true.   ! visit every leaf at the first revision
    visit_mark(k) = 0
 enddo
 !$omp end parallel do
 bounds_valid = .false.
 ntot_seen    = sum(int(npartoftype,kind=8))

end subroutine record_leaves_of_particles

!--------------------------------------------------------------------------------
!+
!  Update the tree in place instead of rebuilding it (individual timesteps).  The
!  particles in each node stay the same.
!
!  Phase 1 (use_bounds = .false.): every leaf is recomputed from its particles
!  (the activity sign in inodeparts and leaf_is_active from xyzh and iphase).
!  Particles that have died or been accreted since the build are dropped from their
!  leaf (its remaining entries are moved up and its range shortened), so that the
!  neighbour search no longer finds them.
!
!  Phase 2 (use_bounds = .true.: no sinks, external forces or damping, so that an
!  inactive particle drifts at a constant velocity between its own kicks): only the
!  leaves holding an active, newly woken or newly dead particle, active when last
!  visited, or whose window has run out are visited.  A visited leaf has all its
!  particles recomputed, and its dead particles dropped as in phase 1.  A visited
!  leaf with no active particle is given a sphere (and hmax) that holds its particles
!  as they drift, and their h grows with their div v, until the first of them is due
!  to become active; it is not touched again until then.
!
!  Internal nodes are recomputed from their two children, deepest first: centre of
!  mass exactly, and a size that encloses the children's spheres.  When every leaf is
!  visited (phase 1, and the first phase 2 revision after a build), each thread
!  updates whole subtrees, leaves and then the nodes above them, so that it reads
!  back what it has just written itself (see revise_all_leaves).  Otherwise the
!  visited leaves are updated first, then the nodes above them.
!
!  Nodes are never re-split, so their spheres loosen as particles move: the caller
!  rebuilds periodically (see build_tree).  Returns revised=.false. (the caller must
!  rebuild with maketree) if there is no tree to revise or particles have been created.
!+
!--------------------------------------------------------------------------------
subroutine revise_tree(node,xyzh,np,leaf_is_active,ncells,time,use_bounds,revised,reason)
!$ use omp_lib, only:omp_get_max_threads,omp_get_thread_num
 use part,         only:iactive,isdead_or_accreted,massoftype,igas,npartoftype,ibin
 use dim,          only:maxtypes
 type(kdnode),    intent(inout) :: node(:)
 real,            intent(in)    :: xyzh(:,:)
 integer,         intent(in)    :: np
 integer,         intent(inout) :: leaf_is_active(:)
 integer(kind=8), intent(in)    :: ncells
 real,            intent(in)    :: time
 logical,         intent(in)    :: use_bounds
 logical,         intent(out)   :: revised
 integer,         intent(out)   :: reason   ! 0 revised, 1 no tree, 2 particles created
 integer, allocatable :: nlist(:),bylevel(:),levstart(:),mydirty(:),tmp(:),nthread_dirty(:)
 integer :: k,kk,n,ip,i,l,nvisit,ndirty,nchunk,ic,p,old,j,nmy,nthreads,it,lsub,ir,d,ltop
 logical :: partial,up_all,deaths
 integer(kind=8) :: ntot
 real    :: pmassi,dfac

 revised = .false.
 nleaves_visited = 0
 if (.not.can_revise) then
    reason = 1
    return
 endif
 ! new particles: rebuild (outside MPI they always raise npart; the alive count also
 ! catches one taking a dead particle's slot)
 ntot = sum(int(npartoftype,kind=8))
 if (np /= npart_built .or. ntot > ntot_seen) then
    reason = 2
    return
 endif
 deaths = (ntot < ntot_seen)   ! some have died or been accreted since the last revision
 ! reference mass, as in construct_node, to keep the centre of mass sums well scaled
 pmassi = massoftype(igas)
 if (pmassi <= 0.) pmassi = massoftype(maxloc(npartoftype(2:maxtypes),1)+1)
 dfac = 1.
 if (pmassi > 0.) dfac = 1./pmassi

 tree_id = tree_id + 1
 partial = use_bounds .and. bounds_valid
 if (partial) then
    !
    ! phase 2: mark the leaves holding an active particle, or one whose timestep bin has
    ! changed since it was last seen (a particle woken while inactive changes its drift)
    !
    !$omp parallel do schedule(static) default(none) &
    !$omp shared(np,xyzh,iphase,ibin,ibin_seen,leaf_of,visit_mark,tree_id) &
    !$omp private(i)
    do i=1,np
       ! the one-byte tests first: xyzh is read only for the few particles that pass them
       if (iactive(iphase(i)) .or. ibin(i) /= ibin_seen(i)) then
          if (isdead_or_accreted(xyzh(4,i))) cycle   ! not in the tree
          !$omp atomic write
          visit_mark(leaf_of(i)) = tree_id
       endif
    enddo
    !$omp end parallel do
    if (deaths) then
       ! and the leaves of particles that have died, to drop them (only when the alive
       ! count has fallen: this reads every particle's h)
       ! (a particle dead since before the last build, or already dropped, is in no
       ! leaf's list: its leaf_of is stale or unset, so check the list before marking)
       !$omp parallel do schedule(static) default(none) &
       !$omp shared(np,xyzh,leaf_of,visit_mark,tree_id,nleaves_built,leaves_built) &
       !$omp shared(inoderange,inodeparts) &
       !$omp private(i,k,n,ip)
       do i=1,np
          if (isdead_or_accreted(xyzh(4,i))) then
             k = leaf_of(i)
             if (k < 1 .or. k > nleaves_built) cycle
             n = leaves_built(k)
             do ip=inoderange(1,n),inoderange(2,n)
                if (abs(inodeparts(ip)) == i) then
                   !$omp atomic write
                   visit_mark(k) = tree_id
                   exit
                endif
             enddo
          endif
       enddo
       !$omp end parallel do
    endif
    !
    ! list them, with those active when last visited and those whose window has run out,
    ! in leaf order (counted per chunk, then placed)
    !
    nchunk = 1
    !$ nchunk = omp_get_max_threads()
    allocate(nlist(0:nchunk))
    nlist = 0
    !$omp parallel default(none) &
    !$omp shared(nchunk,nleaves_built,visit_mark,tree_id,was_active,time,t_valid,nlist,visit_list) &
    !$omp private(ic,k,kk)
    !$omp do schedule(static)
    do ic=1,nchunk
       kk = 0
       do k=int((int(ic-1,8)*nleaves_built)/nchunk)+1,int((int(ic,8)*nleaves_built)/nchunk)
          if (visit_mark(k) == tree_id .or. was_active(k) .or. time > t_valid(k)) kk = kk + 1
       enddo
       nlist(ic) = kk
    enddo
    !$omp end do
    !$omp single
    do ic=1,nchunk
       nlist(ic) = nlist(ic) + nlist(ic-1)
    enddo
    !$omp end single
    !$omp do schedule(static)
    do ic=1,nchunk
       kk = nlist(ic-1)
       do k=int((int(ic-1,8)*nleaves_built)/nchunk)+1,int((int(ic,8)*nleaves_built)/nchunk)
          if (visit_mark(k) == tree_id .or. was_active(k) .or. time > t_valid(k)) then
             kk = kk + 1
             visit_list(kk) = k
          endif
       enddo
    enddo
    !$omp end do
    !$omp end parallel
    nvisit = nlist(nchunk)
    deallocate(nlist)
 else
    nvisit = nleaves_built
 endif
 ! with few leaves visited, only the internal nodes above them are recomputed
 up_all = (nvisit > nleaves_built/8)

 if (.not.partial) then
    ! every leaf, and every internal node, subtree by subtree
    call revise_all_leaves(node,xyzh,leaf_is_active,ncells,dfac,use_bounds,time)
 else
    ndirty = 0
    nthreads = 1
    !$ nthreads = omp_get_max_threads()
    allocate(nthread_dirty(0:nthreads))
    nthread_dirty = 0

    !$omp parallel default(none) &
    !$omp shared(nvisit,visit_list,leaves_built,node,xyzh,leaf_is_active,dfac,use_bounds,time) &
    !$omp shared(up_all,tree_id,node_stamp,dirty_list,ndirty,nthread_dirty,nthreads) &
    !$omp private(kk,k,n,p,old,mydirty,tmp,nmy,it,j)
    nmy = 0
    it  = 0
    !$ it = omp_get_thread_num()
    if (.not.up_all) allocate(mydirty(256))
    ! static: the leaves are small and alike, and with few of them a dynamic schedule's
    ! shared counter, contended by every thread, costs more than the work
    !$omp do schedule(static)
    do kk=1,nvisit
       k = visit_list(kk)
       n = leaves_built(k)
       call revise_leaf(k,n,node,xyzh,leaf_is_active,dfac,use_bounds,time)
       if (.not.up_all) then
          ! mark the internal nodes above it (to be recomputed), stopping at one already
          ! marked; each thread lists those it marked, so no shared counter is contended
          p = n/2
          do while (p >= 1)
             if (node_stamp(p) == tree_id) exit   ! (a plain read first: the top nodes are shared)
             !$omp atomic capture
             old = node_stamp(p)
             node_stamp(p) = tree_id
             !$omp end atomic
             if (old == tree_id) exit
             if (nmy == size(mydirty)) then
                allocate(tmp(2*nmy))
                tmp(1:nmy) = mydirty(1:nmy)
                call move_alloc(tmp,mydirty)
             endif
             nmy = nmy + 1
             mydirty(nmy) = p
             p = p/2
          enddo
       endif
    enddo
    !$omp end do nowait
    if (.not.up_all) then
       ! each thread's marked nodes placed in the list after those of the threads before it
       ! (a running total, rather than a critical section every thread would queue for)
       nthread_dirty(it+1) = nmy
       !$omp barrier
       !$omp single
       do j=1,nthreads
          nthread_dirty(j) = nthread_dirty(j) + nthread_dirty(j-1)
       enddo
       ndirty = nthread_dirty(nthreads)
       !$omp end single
       dirty_list(nthread_dirty(it)+1:nthread_dirty(it)+nmy) = mydirty(1:nmy)
       deallocate(mydirty)
    endif
    !$omp end parallel
    deallocate(nthread_dirty)
 endif
 nleaves_visited = nvisit
 bounds_valid = use_bounds
 all_fresh    = .false.   ! treecache positions are now stale (see cache_neighbours)
 ntot_seen    = ntot

 if (partial) then
    ! internal nodes from their children, deepest level first
    if (up_all) then
       call subtree_levels(nthreads,maxlevel_built,ltop,lsub)
       !$omp parallel default(none) &
       !$omp shared(ltop,lsub,maxlevel_built,ncells,in_tree,leaf_is_active,node) &
       !$omp private(ir,d,n,l)
       ! (static: each thread takes neighbouring subtrees, so that nodes sharing a cache
       ! line are mostly written by the same thread; no shared counter to contend on)
       !$omp do schedule(static)
       do ir=2**lsub,min(2**(lsub+1)-1,int(ncells))
          if (in_tree(ir) /= 1 .or. leaf_is_active(ir) /= 0) cycle
          do d=maxlevel_built-1-lsub,0,-1
             do n=ir*2**d,min((ir+1)*2**d-1,int(ncells))
                if (in_tree(n) /= 1 .or. leaf_is_active(n) /= 0) cycle
                call combine_children(node,n)
             enddo
          enddo
       enddo
       !$omp end do
       do l=lsub-1,ltop,-1
          !$omp do schedule(static)
          do n=2**l,min(2**(l+1)-1,int(ncells))
             if (in_tree(n) /= 1 .or. leaf_is_active(n) /= 0) cycle
             call combine_children(node,n)
          enddo
          !$omp end do
       enddo
       !$omp single
       do l=ltop-1,0,-1
          do n=2**l,min(2**(l+1)-1,int(ncells))
             if (in_tree(n) /= 1 .or. leaf_is_active(n) /= 0) cycle
             call combine_children(node,n)
          enddo
       enddo
       !$omp end single
       !$omp end parallel
    else
       ! the marked nodes sorted by level (that of node p is floor(log2 p))
       allocate(levstart(0:maxlevel_built+1),bylevel(ndirty))
       levstart = 0
       do j=1,ndirty
          l = bit_size(dirty_list(j)) - 1 - leadz(dirty_list(j))
          levstart(l+1) = levstart(l+1) + 1
       enddo
       do l=1,maxlevel_built+1
          levstart(l) = levstart(l) + levstart(l-1)
       enddo
       do j=1,ndirty
          l = bit_size(dirty_list(j)) - 1 - leadz(dirty_list(j))
          levstart(l) = levstart(l) + 1
          bylevel(levstart(l)) = dirty_list(j)
       enddo
       ! (levstart(l) is now the last entry of level l)
       !$omp parallel default(none) if (ndirty > 4096) &
       !$omp shared(maxlevel_built,levstart,bylevel,node) &
       !$omp private(l,j)
       do l=maxlevel_built-1,0,-1
          !$omp do schedule(static)
          do j=merge(1,levstart(max(l-1,0))+1,l==0),levstart(l)
             call combine_children(node,bylevel(j))
          enddo
          !$omp end do
       enddo
       !$omp end parallel
       deallocate(levstart,bylevel)
    endif
 endif
 revised = .true.
 reason  = 0

end subroutine revise_tree

!--------------------------------------------------------------------------------
!+
!  For revise_tree: every leaf recomputed from its particles and every internal node
!  from its children, each thread taking whole subtrees (see subtree_levels), deepest
!  level first within each, so that the nodes it combines are the ones it has just
!  written (updating all the leaves first, then the nodes above, has each thread read
!  mostly nodes written by others: slow cache-to-cache transfers on many cores)
!+
!--------------------------------------------------------------------------------
subroutine revise_all_leaves(node,xyzh,leaf_is_active,ncells,dfac,use_bounds,time)
!$ use omp_lib, only:omp_get_thread_num,omp_get_num_threads
 type(kdnode),    intent(inout) :: node(:)
 real,            intent(in)    :: xyzh(:,:)
 integer,         intent(inout) :: leaf_is_active(:)
 integer(kind=8), intent(in)    :: ncells
 real,            intent(in)    :: dfac,time
 logical,         intent(in)    :: use_bounds
 integer :: ltop,lsub,ir,d,n,l,nc,it,nt,ir0,ir1

 nc   = int(ncells)
 ltop = sub_ltop
 lsub = sub_lsub

 !$omp parallel default(none) &
 !$omp shared(ltop,lsub,maxlevel_built,nc,in_tree,leaf_is_active,node,xyzh,dfac,use_bounds,time) &
 !$omp shared(leafk,sub_first,sub_nthreads) &
 !$omp private(ir,d,n,l,it,nt,ir0,ir1)
 ! each thread its own run of neighbouring subtrees, split by particle count when the
 ! tree was built (see split_subtrees); no shared counter to contend on
 it = 0
 nt = 1
 !$ it = omp_get_thread_num()
 !$ nt = omp_get_num_threads()
 if (nt == sub_nthreads) then
    ir0 = sub_first(it)
    ir1 = sub_first(it+1) - 1
 else   ! (a team of another size: an even split)
    ir0 = 2**lsub + int((int(it,8)*2**lsub)/nt)
    ir1 = 2**lsub + int((int(it+1,8)*2**lsub)/nt) - 1
 endif
 do ir=ir0,min(ir1,nc)
    if (in_tree(ir) /= 1) cycle
    do d=maxlevel_built-lsub,0,-1
       do n=ir*2**d,min((ir+1)*2**d-1,nc)
          if (in_tree(n) /= 1) cycle
          if (leaf_is_active(n) /= 0) then
             call revise_leaf(leafk(n),n,node,xyzh,leaf_is_active,dfac,use_bounds,time)
          else
             call combine_children(node,n)
          endif
       enddo
    enddo
 enddo
 !$omp barrier
 ! the levels above the subtrees: a loop per level, then the top serially
 do l=lsub-1,ltop,-1
    !$omp do schedule(static)
    do n=2**l,min(2**(l+1)-1,nc)
       if (in_tree(n) /= 1) cycle
       if (leaf_is_active(n) /= 0) then
          call revise_leaf(leafk(n),n,node,xyzh,leaf_is_active,dfac,use_bounds,time)
       else
          call combine_children(node,n)
       endif
    enddo
    !$omp end do
 enddo
 !$omp single
 do l=ltop-1,0,-1
    do n=2**l,min(2**(l+1)-1,nc)
       if (in_tree(n) /= 1) cycle
       if (leaf_is_active(n) /= 0) then
          call revise_leaf(leafk(n),n,node,xyzh,leaf_is_active,dfac,use_bounds,time)
       else
          call combine_children(node,n)
       endif
    enddo
 enddo
 !$omp end single
 !$omp end parallel

end subroutine revise_all_leaves

!--------------------------------------------------------------------------------
!+
!  For revise_tree: leaf n (number k in leaves_built) recomputed from its particles
!  (see revise_tree for the phase 2 window)
!+
!--------------------------------------------------------------------------------
subroutine revise_leaf(k,n,node,xyzh,leaf_is_active,dfac,use_bounds,time)
 use part,         only:iactive,isdead_or_accreted,massoftype,igas,vxyzu,divcurlv,ibin,twas,&
                        iamtype
 use timestep,     only:dtmax
 use timestep_ind, only:get_dt
 integer,      intent(in)    :: k,n
 type(kdnode), intent(inout) :: node(:)
 real,         intent(in)    :: xyzh(:,:)
 integer,      intent(inout) :: leaf_is_active(:)
 real,         intent(in)    :: dfac,time
 logical,      intent(in)    :: use_bounds
 integer :: ip,iw,i
 logical :: active
 real    :: xc,yc,zc,totmass,vc(3),tend,pm,fac,x0(3),dtw,r2max,hmax,dx,dy,dz,hi

 xc = 0.; yc = 0.; zc = 0.; totmass = 0.
 vc = 0.
 tend = huge(tend)
 active = .false.
 iw = inoderange(1,n)   ! where the next surviving entry goes
 do ip=inoderange(1,n),inoderange(2,n)
    i  = abs(inodeparts(ip))
    if (isdead_or_accreted(xyzh(4,i))) cycle   ! dropped from the leaf
    if (iw /= ip) treecache(5,iw) = treecache(5,ip)   ! its mass moves up with it
    pm = massoftype(igas)
    if (maxphase==maxp) pm = massoftype(iamtype(iphase(i)))
    if (iactive(iphase(i))) then
       inodeparts(iw) = i    ! +ve if active
       active = .true.
    else
       inodeparts(iw) = -i   ! -ve if inactive
       ! when it is next due to be active
       if (use_bounds) tend = min(tend,twas(i) + 0.5*get_dt(dtmax,ibin(i)))
    endif
    fac = pm*dfac   ! as construct_node, to avoid round-off error
    xc = xc + fac*xyzh(1,i)
    yc = yc + fac*xyzh(2,i)
    zc = zc + fac*xyzh(3,i)
    if (use_bounds) then   ! (phase 2 only: phase 1 needs no drift, bins or leaf index)
       leaf_of(i)   = k
       ibin_seen(i) = ibin(i)
       vc = vc + fac*vxyzu(1:3,i)
    endif
    totmass = totmass + pm
    iw = iw + 1
 enddo
 inoderange(2,n) = iw - 1   ! empty (end < start) if every particle has gone
 if (totmass > 0.) then
    x0(1) = xc/(totmass*dfac)
    x0(2) = yc/(totmass*dfac)
    x0(3) = zc/(totmass*dfac)
    vc    = vc/(totmass*dfac)
 else
    x0 = node(n)%xcen
    vc = 0.
 endif
 ! the window the sphere must hold for: none if a particle is active (the leaf is
 ! visited again at the next revision), otherwise until the first is due to be active
 dtw = 0.
 if (use_bounds .and. .not.active .and. tend < huge(tend)) dtw = max(tend - time,0.)
 ! centred on the centre of mass halfway through, enclosing each particle's drift
 ! from now to the end (a segment, so its ends are furthest from the centre)
 x0 = x0 + 0.5*dtw*vc
 r2max = 0.
 hmax  = 0.
 do ip=inoderange(1,n),inoderange(2,n)
    i  = abs(inodeparts(ip))   ! (its xyzh is in cache from the loop above)
    dx = xyzh(1,i) - x0(1)
    dy = xyzh(2,i) - x0(2)
    dz = xyzh(3,i) - x0(3)
    r2max = max(r2max,dx*dx + dy*dy + dz*dz)
    hi = xyzh(4,i)
    if (dtw > 0.) then
       dx = dx + dtw*vxyzu(1,i)
       dy = dy + dtw*vxyzu(2,i)
       dz = dz + dtw*vxyzu(3,i)
       r2max = max(r2max,dx*dx + dy*dy + dz*dz)
       hi = hi*exp(max(real(divcurlv(1,i)),0.)*dtw/3.)   ! as h = h(1 + dt div v/3) per substep
    endif
    hmax = max(hmax,hi)
 enddo
 node(n)%xcen = x0
 node(n)%size = sqrt(r2max) + epsilon(r2max)
 node(n)%hmax = hmax
 mass_node(n) = totmass
 if (active) then
    leaf_is_active(n) = 1
 else
    leaf_is_active(n) = -1
 endif
 was_active(k) = active
 t_valid(k)    = time + dtw

end subroutine revise_leaf

!--------------------------------------------------------------------------------
!+
!  For loops over the tree's nodes level by level (record_tree, revise_tree): the
!  levels above ltop have fewer nodes than threads (done serially), levels ltop to
!  lsub-1 are done one parallel loop per level, and below lsub each thread takes
!  whole subtrees (at least 16 per thread), level by level within each, with no
!  barrier between levels.  The subtree of node r at depth d below it is the nodes
!  r*2**d to (r+1)*2**d-1.
!+
!--------------------------------------------------------------------------------
pure subroutine subtree_levels(nthreads,maxlev,ltop,lsub)
 integer, intent(in)  :: nthreads,maxlev
 integer, intent(out) :: ltop,lsub

 ltop = 0
 do while (2**ltop < nthreads)
    ltop = ltop + 1
 enddo
 lsub = ltop
 do while (2**lsub < 16*nthreads)
    lsub = lsub + 1
 enddo
 lsub = max(min(lsub,maxlev-1),0)
 ltop = min(ltop,lsub)

end subroutine subtree_levels

!--------------------------------------------------------------------------------
!+
!  For revise_all_leaves: the subtrees rooted at level sub_lsub split into one run of
!  consecutive roots per thread, with about equal numbers of particles in each run
!  (subtrees at one level can differ a lot in size, e.g. in a disc, where the tree
!  splits at the centre of mass rather than the median)
!+
!--------------------------------------------------------------------------------
subroutine split_subtrees(nthreads,nc)
 integer, intent(in) :: nthreads,nc
 integer :: r,r0,r1,it
 integer(kind=8) :: ntot,ncum

 call subtree_levels(nthreads,maxlevel_built+1,sub_ltop,sub_lsub)
 sub_nthreads = nthreads
 if (allocated(sub_first)) then
    if (size(sub_first) /= nthreads+1) deallocate(sub_first)
 endif
 if (.not.allocated(sub_first)) allocate(sub_first(0:nthreads))
 r0 = 2**sub_lsub
 r1 = min(2**(sub_lsub+1)-1,nc)
 ntot = 0
 do r=r0,r1
    if (in_tree(r) == 1) ntot = ntot + (inoderange(2,r) - inoderange(1,r) + 1)
 enddo
 ! thread it starts at the first root with at least it/nthreads of the particles before it
 it = 0
 sub_first(0) = r0
 ncum = 0
 do r=r0,r1
    do while (it < nthreads-1 .and. ncum*nthreads >= (it+1)*ntot)
       it = it + 1
       sub_first(it) = r
    enddo
    if (in_tree(r) == 1) ncum = ncum + (inoderange(2,r) - inoderange(1,r) + 1)
 enddo
 do while (it < nthreads-1)
    it = it + 1
    sub_first(it) = r1 + 1
 enddo
 sub_first(nthreads) = r1 + 1

end subroutine split_subtrees

!--------------------------------------------------------------------------------
!+
!  For revise_tree: an internal node from its two children, the centre of mass
!  exactly and a size that encloses the children's spheres
!+
!--------------------------------------------------------------------------------
subroutine combine_children(node,n)
 type(kdnode), intent(inout) :: node(:)
 integer,      intent(in)    :: n
 integer :: il,ir
 real    :: ml,mr,x0(3),sz

 il = node(n)%leftchild
 ir = node(n)%rightchild
 ml = mass_node(il)
 mr = mass_node(ir)
 if (ml + mr > 0.) then
    x0 = (ml*node(il)%xcen + mr*node(ir)%xcen)/(ml + mr)
 else
    x0 = 0.5*(node(il)%xcen + node(ir)%xcen)
 endif
 sz = max(norm2(node(il)%xcen - x0) + node(il)%size, &
          norm2(node(ir)%xcen - x0) + node(ir)%size)
 node(n)%xcen = x0
 node(n)%size = sz
 node(n)%hmax = max(node(il)%hmax,node(ir)%hmax)
 mass_node(n) = ml + mr

end subroutine combine_children

!--------------------------------------------------------------------------------
!+
!  Build the top levels of the tree a whole level at a time, with all threads
!  working on every level.
!
!  The queue loop in maketree constructs these nodes one at a time, and although
!  the centre of mass and size loops are parallel, the partition that splits each
!  node's particles runs on one thread.  That is log2(nthreads) serial passes over
!  every particle, which grows with the thread count.  Here every thread keeps its
!  own share of the particles through all the levels: at the root each takes one
!  piece of the particle list, and at each level it splits its piece of every node
!  into a left and a right piece (a stable partition into a scratch buffer, which
!  then becomes the particle list: the arrays are swapped rather than copied back),
!  which are its pieces of the two children at the next level.  So a thread only
!  ever reads particles it has written itself (reading what other cores have just
!  written is slow on many cores), and every thread has the same number of
!  particles at every level.  Per level: one pass for each node's size, hmax and
!  the count to the left of its pivot, and one for the partition, which also sums
!  the children's centres of mass and bounding boxes (only the root needs a pass
!  of its own for its centre of mass).  The nodes are those construct_node would
!  give, bar round-off in the sums, but the order of particles within each child
!  differs.
!
!  On entry queue(1:istack) holds one level of the tree (the root).  Returns once
!  istack >= nqueue, or earlier, leaving a consistent queue for the serial loop in
!  maketree to finish, if a level has small nodes or would not fit in the queue.
!+
!--------------------------------------------------------------------------------
subroutine build_top_parallel(node,queue,istack,nqueue,leaf_is_active)
!$ use omp_lib, only:omp_get_thread_num,omp_get_num_threads
 use part,      only:massoftype,igas,npartoftype
 use dim,       only:maxtypes
 use io,        only:fatal
 type(kdnode),       intent(inout) :: node(:)
 type(kdbuildstack), intent(inout) :: queue(:)
 integer,            intent(inout) :: istack
 integer,            intent(in)    :: nqueue
 integer,            intent(inout) :: leaf_is_active(:)
 ! piece c = (j-1)*nt + t + 1 is thread t's share of node j: chunkl(c) to chunkr(c) in
 ! the particle list (empty if chunkr < chunkl), and nextl/nextr the same for the next level
 integer, allocatable :: chunkl(:),chunkr(:),cnl(:),coffl(:),coffr(:),nextl(:),nextr(:)
 integer, allocatable :: jnl(:),nodeax(:)
 logical, allocatable :: jdegen(:)
 real,    allocatable :: psum(:,:),phm(:),pr2(:),pbox(:,:),nodecom(:,:),nodecog(:,:)
 real,    allocatable :: psl(:,:),psr(:,:),csum(:,:),nsum(:,:)
#ifdef GRAVITY
 real,    allocatable :: pmom(:,:)
 real    :: quads(6),octs(10)
#endif
 type(kdbuildstack), allocatable :: newq(:)
 real,    allocatable :: tcswap(:,:)
 integer, allocatable :: ipswap(:)
 integer :: kmax,pmax,k,j,c,i,i1,n,nl,pl,pr,iax,nnode,il,ir,isplit,nt,t,it,nteam,nr
 logical :: have_sums
 real    :: pmassi,dfac,fac,hm,r2,dx,dy,dz,xpiv,totmass,x0(3),bl(6),br(6),sl(4),sr(4),sm(4)

 if (allocated(tcbuf)) then
    if (size(tcbuf,2) /= size(treecache,2)) deallocate(tcbuf,ipbuf)
 endif
 if (.not.allocated(tcbuf)) allocate(tcbuf(size(treecache,1),size(treecache,2)),ipbuf(size(inodeparts)))

 ! reference mass, as in construct_node, to keep the centre of mass sums well scaled
 pmassi = massoftype(igas)
 if (pmassi <= 0.) pmassi = massoftype(maxloc(npartoftype(2:maxtypes),1)+1)
 dfac = 1.
 if (pmassi > 0.) dfac = 1./pmassi

 !
 ! work arrays for every level, at their largest: a level has at most size(queue)/2
 ! nodes (its children must fit in the queue), each with one piece per thread
 !
 nt   = numthreads
 kmax = size(queue)/2
 pmax = nt*size(queue)
 allocate(jnl(kmax),nodeax(kmax),jdegen(kmax),nodecom(3,kmax),nodecog(3,kmax),newq(2*kmax))
 allocate(chunkl(pmax),chunkr(pmax),nextl(pmax),nextr(pmax),cnl(pmax),coffl(pmax),coffr(pmax))
 allocate(psum(4,pmax),phm(pmax),pr2(pmax),pbox(12,pmax),psl(4,pmax),psr(4,pmax))
 allocate(csum(4,2*kmax),nsum(4,2*kmax))
#ifdef GRAVITY
 allocate(pmom(16,pmax))
#endif

 ! the root's particles, one piece per thread
 i1 = inoderange(1,queue(1)%node)
 n  = queue(1)%npnode
 do t = 0,nt-1
    chunkl(t+1) = i1 + int((int(t,8)*n)/nt)
    chunkr(t+1) = i1 + int((int(t+1,8)*n)/nt) - 1
 enddo
 have_sums = .false.

 levels: do while (istack < nqueue)
    k = istack
    !
    ! the queue holds one whole level (all nodes have the same level); split them all
    ! together, or hand over to the serial loop if the next level would not fit in the
    ! queue, would pass the levels numbered 2n/2n+1, or any node is small
    !
    if (k > kmax .or. queue(1)%level >= maxlevel_indexed) exit levels
    if (any(queue(1:k)%npnode <= max(minpart,64))) exit levels
    do j = 1,k
       nodeax(j) = maxloc(queue(j)%xmax - queue(j)%xmin,1)   ! split along the longest axis
       nodecog(:,j) = 0.5*(queue(j)%xmin + queue(j)%xmax)
    enddo

    !$omp parallel default(none) &
    !$omp shared(have_sums,nt,k,chunkl,chunkr,treecache,dfac,psum,csum,nodecom,nodecog,nodeax) &
    !$omp shared(node,queue,leaf_is_active,inoderange,inodeparts,tcbuf,ipbuf,use_geosplit) &
    !$omp shared(pr2,phm,cnl,coffl,coffr,jdegen,jnl,pbox,psl,psr,nextl,nextr,newq,nsum) &
#ifdef GRAVITY
    !$omp shared(pmom) &
    !$omp private(quads,octs) &
#endif
    !$omp private(c,j,i,i1,n,nl,pl,pr,iax,nnode,il,ir,isplit,t,it,nteam,nr) &
    !$omp private(pmassi,fac,hm,r2,dx,dy,dz,xpiv,totmass,x0,bl,br,sl,sr,sm)
    ! thread it works on the pieces of thread number t = it, it+nteam, ... (its own
    ! pieces, unless the team is not the usual size)
    it = 0
    nteam = 1
    !$ it = omp_get_thread_num()
    !$ nteam = omp_get_num_threads()
    !
    ! the root's mass and centre of mass (every other node's come from the partition
    ! at the level above) ...
    !
    if (.not.have_sums) then
       do t = it,nt-1,nteam
          do j = 1,k
             c = (j-1)*nt + t + 1
             sm = 0.
             do i = chunkl(c),chunkr(c)
                pmassi = treecache(5,i)
                fac = pmassi*dfac
                sm(1:3) = sm(1:3) + fac*treecache(1:3,i)
                sm(4)   = sm(4) + pmassi
             enddo
             psum(:,c) = sm
          enddo
       enddo
       !$omp barrier
    endif
    ! ... per node, which also fixes the pivot (the centre of mass, or with use_geosplit
    ! the geometric centre, the centre of mass being the node's centre either way)
    !$omp do schedule(static)
    do j = 1,k
       if (have_sums) then
          sm = csum(:,j)
       else
          sm = sum(psum(:,(j-1)*nt+1:j*nt),dim=2)
       endif
       totmass = sm(4)
       if (totmass <= 0.) call fatal('mtree','totmass_node==0',val=totmass)
       nodecom(:,j) = sm(1:3)/(totmass*dfac)
#ifdef GRAVITY
       node(queue(j)%node)%mass = totmass
#endif
    enddo
    !$omp enddo
    !
    ! size (and multipole moments) about the centre of mass, hmax, and the number of
    ! particles left of the pivot, of each piece ...
    !
    do t = it,nt-1,nteam
       do j = 1,k
          c    = (j-1)*nt + t + 1
          x0   = nodecom(:,j)
          iax  = nodeax(j)
          if (use_geosplit) then
             xpiv = nodecog(iax,j)
          else
             xpiv = x0(iax)
          endif
          r2 = 0.
          hm = 0.
          nl = 0
#ifdef GRAVITY
          quads = 0.
          octs  = 0.
#endif
          do i = chunkl(c),chunkr(c)
             dx = treecache(1,i) - x0(1)
             dy = treecache(2,i) - x0(2)
             dz = treecache(3,i) - x0(3)
             hm = max(hm,treecache(4,i))
             r2 = max(r2,dx*dx + dy*dy + dz*dz)
             if (treecache(iax,i) <= xpiv) nl = nl + 1
#ifdef GRAVITY
             pmassi = treecache(5,i)
             call add_node_moments(pmassi,dx,dy,dz,quads,octs)
#endif
          enddo
          phm(c) = hm
          pr2(c) = r2
          cnl(c) = nl
#ifdef GRAVITY
          pmom(1:6,c)  = quads
          pmom(7:16,c) = octs
#endif
       enddo
    enddo
    !$omp barrier
    ! ... combined per node into the node itself, and each piece's write positions
    !$omp do schedule(static)
    do j = 1,k
       nnode = queue(j)%node
       i1 = inoderange(1,nnode)
       n  = queue(j)%npnode
       r2 = maxval(pr2((j-1)*nt+1:j*nt))
       node(nnode)%xcen   = nodecom(:,j)
       node(nnode)%size   = sqrt(r2) + epsilon(r2)
       node(nnode)%hmax   = maxval(phm((j-1)*nt+1:j*nt))
       node(nnode)%parent = queue(j)%parent
       node(nnode)%level  = queue(j)%level
#ifdef GRAVITY
       node(nnode)%quads  = sum(pmom(1:6,(j-1)*nt+1:j*nt),dim=2)
       node(nnode)%octs   = sum(pmom(7:16,(j-1)*nt+1:j*nt),dim=2)
#endif
       il = 2*nnode   ! indexing as per Gafton & Rosswog (2011), as in construct_node
       ir = il + 1
       node(nnode)%leftchild  = il
       node(nnode)%rightchild = ir
       leaf_is_active(nnode)  = 0
       nl = sum(cnl((j-1)*nt+1:j*nt))
       ! all particles on one side: split in half without moving them, as construct_node
       ! does (each piece stays where it is, its part before the split going left)
       jdegen(j) = (nl == 0 .or. nl == n)
       if (jdegen(j)) nl = n/2
       jnl(j) = nl
       isplit = i1 + nl   ! first particle of the right child
       pl = i1
       pr = i1 + nl
       do c = (j-1)*nt+1,j*nt
          if (jdegen(j)) then
             cnl(c)   = max(0,min(chunkr(c),isplit-1) - chunkl(c) + 1)
             coffl(c) = chunkl(c)
             coffr(c) = chunkl(c) + cnl(c)
          else
             coffl(c) = pl
             coffr(c) = pr
             pl = pl + cnl(c)
             pr = pr + (chunkr(c) - chunkl(c) + 1 - cnl(c))
          endif
       enddo
    enddo
    !$omp enddo
    !
    ! partition: each piece is scattered into the scratch buffer, and in the same pass
    ! the bounding box and sums of each side are accumulated; its two parts are this
    ! thread's pieces of the children at the next level
    !
    do t = it,nt-1,nteam
       do j = 1,k
          c    = (j-1)*nt + t + 1
          iax  = nodeax(j)
          if (use_geosplit) then
             xpiv = nodecog(iax,j)
          else
             xpiv = nodecom(iax,j)
          endif
          bl(1:3) =  huge(1.); bl(4:6) = -huge(1.)
          br = bl
          sl = 0.; sr = 0.
          if (jdegen(j)) then
             ! nothing moves, but the buffer becomes the particle list: copy as it is
             isplit = inoderange(1,queue(j)%node) + jnl(j)
             tcbuf(:,chunkl(c):chunkr(c)) = treecache(:,chunkl(c):chunkr(c))
             ipbuf(chunkl(c):chunkr(c))   = inodeparts(chunkl(c):chunkr(c))
             do i = chunkl(c),chunkr(c)
                pmassi = treecache(5,i)
                fac = pmassi*dfac
                if (i < isplit) then
                   bl(1:3) = min(bl(1:3),treecache(1:3,i))
                   bl(4:6) = max(bl(4:6),treecache(1:3,i))
                   sl(1:3) = sl(1:3) + fac*treecache(1:3,i)
                   sl(4)   = sl(4) + pmassi
                else
                   br(1:3) = min(br(1:3),treecache(1:3,i))
                   br(4:6) = max(br(4:6),treecache(1:3,i))
                   sr(1:3) = sr(1:3) + fac*treecache(1:3,i)
                   sr(4)   = sr(4) + pmassi
                endif
             enddo
          else
             pl = coffl(c)
             pr = coffr(c)
             do i = chunkl(c),chunkr(c)
                pmassi = treecache(5,i)
                fac = pmassi*dfac
                if (treecache(iax,i) <= xpiv) then
                   tcbuf(:,pl) = treecache(:,i)
                   ipbuf(pl)   = inodeparts(i)
                   pl = pl + 1
                   bl(1:3) = min(bl(1:3),treecache(1:3,i))
                   bl(4:6) = max(bl(4:6),treecache(1:3,i))
                   sl(1:3) = sl(1:3) + fac*treecache(1:3,i)
                   sl(4)   = sl(4) + pmassi
                else
                   tcbuf(:,pr) = treecache(:,i)
                   ipbuf(pr)   = inodeparts(i)
                   pr = pr + 1
                   br(1:3) = min(br(1:3),treecache(1:3,i))
                   br(4:6) = max(br(4:6),treecache(1:3,i))
                   sr(1:3) = sr(1:3) + fac*treecache(1:3,i)
                   sr(4)   = sr(4) + pmassi
                endif
             enddo
          endif
          pbox(1:6,c)  = bl
          pbox(7:12,c) = br
          psl(:,c) = sl
          psr(:,c) = sr
          ! this thread's pieces of the children (numbers 2j-1 and 2j at the next level)
          nr = chunkr(c) - chunkl(c) + 1 - cnl(c)
          nextl((2*j-2)*nt + t + 1) = coffl(c)
          nextr((2*j-2)*nt + t + 1) = coffl(c) + cnl(c) - 1
          nextl((2*j-1)*nt + t + 1) = coffr(c)
          nextr((2*j-1)*nt + t + 1) = coffr(c) + nr - 1
       enddo
    enddo
    !$omp barrier
    ! the children, as the next level of the queue
    !$omp do schedule(static)
    do j = 1,k
       nnode = queue(j)%node
       i1 = inoderange(1,nnode)
       n  = queue(j)%npnode
       nl = jnl(j)
       il = 2*nnode
       ir = il + 1
       inoderange(1,il) = i1
       inoderange(2,il) = i1 + nl - 1
       inoderange(1,ir) = i1 + nl
       inoderange(2,ir) = i1 + n - 1
       ! children's boxes from the pieces' partial boxes (into bl/br first, so that no
       ! array temporaries are passed to push_onto_stack)
       bl(1:3) = minval(pbox(1:3,(j-1)*nt+1:j*nt),dim=2)
       bl(4:6) = maxval(pbox(4:6,(j-1)*nt+1:j*nt),dim=2)
       br(1:3) = minval(pbox(7:9,(j-1)*nt+1:j*nt),dim=2)
       br(4:6) = maxval(pbox(10:12,(j-1)*nt+1:j*nt),dim=2)
       call push_onto_stack(newq(2*j-1),il,nnode,queue(j)%level+1,nl,bl(1:3),bl(4:6))
       call push_onto_stack(newq(2*j),ir,nnode,queue(j)%level+1,n-nl,br(1:3),br(4:6))
       ! the children's sums, for the next level
       nsum(:,2*j-1) = sum(psl(:,(j-1)*nt+1:j*nt),dim=2)
       nsum(:,2*j)   = sum(psr(:,(j-1)*nt+1:j*nt),dim=2)
    enddo
    !$omp enddo
    !$omp end parallel

    ! the buffer now holds the particle list in the new order: swap rather than copy back
    call move_alloc(treecache,tcswap)
    call move_alloc(tcbuf,treecache)
    call move_alloc(tcswap,tcbuf)
    call move_alloc(inodeparts,ipswap)
    call move_alloc(ipbuf,inodeparts)
    call move_alloc(ipswap,ipbuf)

    queue(1:2*k) = newq(1:2*k)
    csum(:,1:2*k) = nsum(:,1:2*k)
    chunkl(1:2*k*nt) = nextl(1:2*k*nt)
    chunkr(1:2*k*nt) = nextr(1:2*k*nt)
    have_sums = .true.
    istack = 2*k
 enddo levels

end subroutine build_top_parallel

!----------------------------
!+
! routine to empty the tree
!+
!----------------------------
subroutine empty_tree(node)
 type(kdnode), intent(out) :: node(:)
 integer :: i

!$omp parallel do default(none) shared(node) private(i)
 do i=1,size(node)
    node(i)%xcen = 0.
    node(i)%size = 0.
    node(i)%hmax = 0.
    node(i)%leftchild = 0
    node(i)%rightchild = 0
    node(i)%parent = 0
    node(i)%level  = 0
#ifdef GRAVITY
    node(i)%mass  = 0.
    node(i)%quads = 0.
    node(i)%octs  = 0.
#endif
 enddo
!$omp end parallel do

end subroutine empty_tree

!---------------------------------
!+
! routine to construct root node
!+
!---------------------------------
subroutine construct_root_node(np,nproot,irootnode,xmini,xmaxi,leaf_is_active,xyzh,xyzmh_ptmass,nptmass)
 use boundary, only:cross_boundary
 use mpidomain,only:isperiodic
 use part, only:iphase,iactive
 use part, only:isdead_or_accreted,ibelong
 use io,   only:fatal,id
 use dim,  only:ind_timesteps,mpi,periodic
 use part, only:isink,massoftype,igas,iamtype,maxphase,maxp,aprmassoftype,apr_level,ihsoft
!$ use omp_lib, only:omp_get_max_threads
 integer, intent(in)    :: np,irootnode
 integer, intent(out)   :: nproot
 real,    intent(out)   :: xmini(3), xmaxi(3)
 integer, intent(inout) :: leaf_is_active(:)
 real,    intent(inout) :: xyzh(:,:)
 real,    intent(inout), optional :: xyzmh_ptmass(:,:)
 integer, intent(in),    optional :: nptmass
 integer :: i,ncross,ic,nchunk,nl
 integer, allocatable :: nlive(:)
 real    :: xminpart,yminpart,zminpart,xmaxpart,ymaxpart,zmaxpart
 real    :: xi, yi, zi

 ! seed with the identity of min/max: an empty task (or a dead first particle)
 ! must not change the box reduced across MPI tasks
 xminpart =  huge(xminpart)
 yminpart =  huge(yminpart)
 zminpart =  huge(zminpart)
 xmaxpart = -huge(xmaxpart)
 ymaxpart = -huge(ymaxpart)
 zmaxpart = -huge(zmaxpart)

 ncross = 0
 nproot = 0
 ! the live particles are also counted per chunk of the arrays in this pass, for the copy below
 nchunk = 1
!$ nchunk = omp_get_max_threads()
 allocate(nlive(0:nchunk))
 nlive = 0
 !$omp parallel default(none) &
 !$omp shared(np,xyzh,nptmass,xyzmh_ptmass) &
 !$omp shared(inodeparts,iphase,treecache,nproot) &
 !$omp shared(id,use_sinktree) &
 !$omp shared(isperiodic,nchunk,nlive) &
 !$omp private(i,ic,nl,xi,yi,zi) &
 !$omp reduction(min:xminpart,yminpart,zminpart) &
 !$omp reduction(max:xmaxpart,ymaxpart,zmaxpart) &
 !$omp reduction(+:ncross)
 !$omp do schedule(static)
 do ic=1,nchunk
    nl = 0
    do i=int((int(ic-1,8)*np)/nchunk)+1,int((int(ic,8)*np)/nchunk)
       if (.not.isdead_or_accreted(xyzh(4,i))) then
          nl = nl + 1
          if (periodic) call cross_boundary(isperiodic,xyzh(:,i),ncross)
          xi = xyzh(1,i)
          yi = xyzh(2,i)
          zi = xyzh(3,i)
          if (ieee_is_nan(xi) .or. ieee_is_nan(yi) .or. ieee_is_nan(zi)) then
             call fatal('maketree','NaN in particle position, likely caused by NaN in force',i,var='x',val=xi)
          endif
          xminpart = min(xminpart,xi)
          yminpart = min(yminpart,yi)
          zminpart = min(zminpart,zi)
          xmaxpart = max(xmaxpart,xi)
          ymaxpart = max(ymaxpart,yi)
          zmaxpart = max(zmaxpart,zi)
       endif
    enddo
    nlive(ic) = nl   ! once per chunk: neighbouring counters share a cache line
 enddo
 !$omp enddo
 !$omp barrier
 if (use_sinktree) then
    if (nptmass>0) then
       !$omp do schedule(guided,1)
       do i=1,nptmass
          if (xyzmh_ptmass(4,i)>0.) then
             if (periodic) call cross_boundary(isperiodic,xyzmh_ptmass(1:3,i),ncross)
             xi = xyzmh_ptmass(1,i)
             yi = xyzmh_ptmass(2,i)
             zi = xyzmh_ptmass(3,i)
             if (ieee_is_nan(xi) .or. ieee_is_nan(yi) .or. ieee_is_nan(zi)) then
                call fatal('maketree','NaN in ptmass position, likely caused by NaN in force',i,var='x',val=xi)
             endif
             xminpart = min(xminpart,xi)
             yminpart = min(yminpart,yi)
             zminpart = min(zminpart,zi)
             xmaxpart = max(xmaxpart,xi)
             ymaxpart = max(ymaxpart,yi)
             zmaxpart = max(zmaxpart,zi)
          endif
       enddo
       !$omp enddo
    endif
 endif
 !$omp end parallel

 !
 ! copy the live particles into the tree's list in parallel: from the counts above, each
 ! chunk of the particle arrays knows where its particles go, and the order is the same
 ! as a serial copy
 !
 do ic=1,nchunk
    nlive(ic) = nlive(ic) + nlive(ic-1)
 enddo
 !$omp parallel do schedule(static) default(none) &
 !$omp shared(np,nchunk,xyzh,nlive,inodeparts,treecache,iphase,massoftype,aprmassoftype,apr_level) &
 !$omp shared(maxp,maxphase) &
 !$omp private(ic,i,nproot)
 do ic=1,nchunk
    nproot = nlive(ic-1)
    do i=int((int(ic-1,8)*np)/nchunk)+1,int((int(ic,8)*np)/nchunk)
       isnotdead: if (.not.isdead_or_accreted(xyzh(4,i))) then
          nproot = nproot + 1

          if (ind_timesteps) then
             if (iactive(iphase(i))) then
                inodeparts(nproot) = i  ! +ve if active
             else
                inodeparts(nproot) = -i ! -ve if inactive
             endif
             if (use_apr) inodeparts(nproot) = abs(inodeparts(nproot))
          else
             inodeparts(nproot) = i
          endif
          treecache(1:4,nproot) = xyzh(1:4,i)
          if (maxphase==maxp) then
             if (use_apr) then
                treecache(5,nproot) = aprmassoftype(iamtype(iphase(i)),apr_level(i))
             else
                treecache(5,nproot) = massoftype(iamtype(iphase(i)))
             endif
          elseif (use_apr) then
             treecache(5,nproot) = aprmassoftype(igas,apr_level(i))
          else
             treecache(5,nproot) = massoftype(igas)
          endif
       endif isnotdead
    enddo
 enddo
 !$omp end parallel do
 nproot = nlive(nchunk)
 deallocate(nlive)

 if (use_sinktree) then
    if (nptmass > 0) then
       do i=1,nptmass
          if (mpi) then
             if (ibelong(maxpsph+i) /= id) cycle
          endif
          if (xyzmh_ptmass(4,i)<0.) cycle
          nproot = nproot + 1
          inodeparts(nproot) = (maxpsph) + i
          treecache(1:3,nproot) = xyzmh_ptmass(1:3,i)
          treecache(4,nproot)   = xyzmh_ptmass(ihsoft,i)
          treecache(5,nproot)   = xyzmh_ptmass(4,i)
       enddo
    endif
 endif

 if (nproot /= 0) then
    inoderange(1,irootnode) = 1
    inoderange(2,irootnode) = nproot
 else
    inoderange(:,irootnode) = 0
 endif

 xmini(1) = xminpart
 xmini(2) = yminpart
 xmini(3) = zminpart
 xmaxi(1) = xmaxpart
 xmaxi(2) = ymaxpart
 xmaxi(3) = zmaxpart

end subroutine construct_root_node

! also used for queue push
pure subroutine push_onto_stack(stackentry,node,parent,level,npnode,xmin,xmax)
 type(kdbuildstack), intent(out) :: stackentry
 integer,            intent(in)  :: node,parent,level
 integer,            intent(in)  :: npnode
 real,               intent(in)  :: xmin(3),xmax(3)

 stackentry%node   = node
 stackentry%parent = parent
 stackentry%level  = level
 stackentry%npnode = npnode
 stackentry%xmin   = xmin
 stackentry%xmax   = xmax

end subroutine push_onto_stack

! also used for queue pop
pure subroutine pop_off_stack(stackentry, istack, nnode, mymum, level, npnode, xmini, xmaxi)
 type(kdbuildstack), intent(in)    :: stackentry
 integer,            intent(inout) :: istack
 integer,            intent(out)   :: nnode, mymum, level, npnode
 real,               intent(out)   :: xmini(3), xmaxi(3)

 nnode  = stackentry%node
 mymum  = stackentry%parent
 level  = stackentry%level
 npnode = stackentry%npnode
 xmini  = stackentry%xmin
 xmaxi  = stackentry%xmax
 istack = istack - 1

end subroutine pop_off_stack

subroutine compute_nodes_cofm(npnode,nnode,xyzcofm,totmass_node,doparallel)
 use dim,       only:maxtypes
 use part,      only:massoftype,igas,npartoftype
 integer, intent(in)  :: npnode,nnode
 real,    intent(out) :: xyzcofm(3),totmass_node
 logical, intent(in)  :: doparallel
 real    :: pmassi,fac,dfac
 real    :: xi,yi,zi,xcofm,ycofm,zcofm
 integer :: i1,i
!
! to avoid round off error from repeated multiplication by pmassi (which is small)
! we compute the centre of mass with a factor relative to gas particles
! but only if gas particles are present
!
 pmassi = massoftype(igas)
 fac    = 1.
 totmass_node = 0.
 if (pmassi > 0.) then
    dfac = 1./pmassi
 else
    pmassi = massoftype(maxloc(npartoftype(2:maxtypes),1)+1)
    if (pmassi > 0.) then
       dfac = 1./pmassi
    else
       dfac = 1.
    endif
 endif

 ! note that dfac can be a constant value across all particles even if APR is used
 i1=inoderange(1,nnode)
 xcofm = 0.
 ycofm = 0.
 zcofm = 0.

 ! during initial queue build which is serial, we can parallelise this loop
 if (npnode > 1000 .and. doparallel) then
    !$omp parallel do schedule(static) default(none) &
    !$omp shared(npnode,dfac) &
    !$omp shared(treecache,i1) &
    !$omp private(i,xi,yi,zi) &
    !$omp firstprivate(pmassi,fac) &
    !$omp reduction(+:xcofm,ycofm,zcofm,totmass_node)
    do i=i1,i1+npnode-1
       xi = treecache(1,i)
       yi = treecache(2,i)
       zi = treecache(3,i)
       pmassi = treecache(5,i)
       fac    = pmassi*dfac ! to avoid round-off error
       totmass_node = totmass_node + pmassi
       xcofm = xcofm + fac*xi
       ycofm = ycofm + fac*yi
       zcofm = zcofm + fac*zi
    enddo
    !$omp end parallel do
 else
    do i=i1,i1+npnode-1
       xi = treecache(1,i)
       yi = treecache(2,i)
       zi = treecache(3,i)
       pmassi = treecache(5,i)
       fac    = pmassi*dfac ! to avoid round-off error
       totmass_node = totmass_node + pmassi
       xcofm = xcofm + fac*xi
       ycofm = ycofm + fac*yi
       zcofm = zcofm + fac*zi
    enddo
 endif

 xyzcofm = (/xcofm,ycofm,zcofm/)

 ! if there are no particles in this node, then the cofm will
 ! remain at zero
 if (totmass_node > 0.) then
    xyzcofm(:)   = xyzcofm(:)/(totmass_node*dfac)
 endif

end subroutine compute_nodes_cofm

subroutine set_nodes_properties(npnode,nnode,x0,totmass_node,mymum,nodeentry,xmini,xmaxi,&
                                level,global_build,doparallel)
 use mpitree,   only:reduce_group
 use dim,       only:mpi
 type(kdnode),    intent(out)   :: nodeentry
 integer,         intent(in)    :: npnode,mymum,level,nnode
 real,            intent(inout) :: xmini(3), xmaxi(3), totmass_node
 real,            intent(in)    :: x0(3)
 logical,         intent(in)    :: doparallel,global_build
 real    :: pmassi
 real    :: dx,dy,dz,dr2,xi,yi,zi,hi
 real    :: hmax,r2max,totmass
 integer :: i1,i
#ifdef GRAVITY
 real    :: quads(6)
 real    :: octs(10)
#endif

 pmassi  = 0.
 totmass = 0.
 r2max = 0.
 hmax  = 0.
#ifdef GRAVITY
 quads(:) = 0.
 octs(:) = 0.
#endif

 i1=inoderange(1,nnode)

 !--compute size of node ! Parallel obsolete when build top tree is on but APR don't have it
 ! parallelise this loop if node is large enough
 ! use !$omp parallel do when doparallel=.true. (not in parallel region)
 ! when doparallel=.false., we're already in a parallel region but can't use nested reductions
 ! so we'll use thread-local accumulators and combine at the end
 if (npnode > 1000 .and. doparallel) then
    !$omp parallel do schedule(static) default(none) &
    !$omp shared(npnode,treecache,x0,i1,use_geosplit) &
    !$omp private(i,xi,yi,zi,hi,dx,dy,dz,dr2) &
    !$omp firstprivate(pmassi) &
#ifdef GRAVITY
    !$omp reduction(+:totmass,quads,octs) &
#endif
    !$omp reduction(max:r2max,hmax)
    do i=i1,i1+npnode-1
       xi = treecache(1,i)
       yi = treecache(2,i)
       zi = treecache(3,i)
       hi = treecache(4,i)
       dx    = xi - x0(1)
       dy    = yi - x0(2)
       dz    = zi - x0(3)
       dr2   = dx*dx + dy*dy + dz*dz
       r2max = max(r2max,dr2)
       hmax  = max(hmax,hi)
#ifdef GRAVITY
       pmassi = treecache(5,i)
       totmass  = totmass  + pmassi
       call add_node_moments(pmassi,dx,dy,dz,quads,octs)
#endif
    enddo
    !$omp end parallel do
 else
    do i=i1,i1+npnode-1
       xi = treecache(1,i)
       yi = treecache(2,i)
       zi = treecache(3,i)
       hi = treecache(4,i)
       dx    = xi - x0(1)
       dy    = yi - x0(2)
       dz    = zi - x0(3)
       dr2   = dx*dx + dy*dy + dz*dz
       r2max = max(r2max,dr2)
       hmax = max(hmax,hi)
#ifdef GRAVITY
       pmassi = treecache(5,i)
       totmass  = totmass  + pmassi
       call add_node_moments(pmassi,dx,dy,dz,quads,octs)
#endif
    enddo
 endif

 ! reduce node limits and quads across MPI tasks belonging to this group
 if (mpi .and. global_build) then
    r2max     = reduce_group(r2max,'max',level)
    hmax      = reduce_group(hmax,'max',level)

    xmini(1)  = reduce_group(xmini(1),'min',level)
    xmini(2)  = reduce_group(xmini(2),'min',level)
    xmini(3)  = reduce_group(xmini(3),'min',level)

    xmaxi(1)  = reduce_group(xmaxi(1),'max',level)
    xmaxi(2)  = reduce_group(xmaxi(2),'max',level)
    xmaxi(3)  = reduce_group(xmaxi(3),'max',level)
#ifdef GRAVITY
    quads(1)  = reduce_group(quads(1),'+',level)
    quads(2)  = reduce_group(quads(2),'+',level)
    quads(3)  = reduce_group(quads(3),'+',level)
    quads(4)  = reduce_group(quads(4),'+',level)
    quads(5)  = reduce_group(quads(5),'+',level)
    quads(6)  = reduce_group(quads(6),'+',level)
    octs(1)   = reduce_group(octs(1),'+',level)
    octs(2)   = reduce_group(octs(2),'+',level)
    octs(3)   = reduce_group(octs(3),'+',level)
    octs(4)   = reduce_group(octs(4),'+',level)
    octs(5)   = reduce_group(octs(5),'+',level)
    octs(6)   = reduce_group(octs(6),'+',level)
    octs(7)   = reduce_group(octs(7),'+',level)
    octs(8)   = reduce_group(octs(8),'+',level)
    octs(9)   = reduce_group(octs(9),'+',level)
    octs(10)  = reduce_group(octs(10),'+',level)
#endif
 endif

 ! assign properties to node
 nodeentry%xcen       = x0(:)
 nodeentry%size       = sqrt(r2max) + epsilon(r2max)
 nodeentry%hmax       = hmax
 nodeentry%parent     = mymum
 nodeentry%level      = level
#ifdef GRAVITY
 nodeentry%mass       = totmass_node
 nodeentry%quads      = quads
 nodeentry%octs       = octs
#endif

end subroutine set_nodes_properties

#ifdef GRAVITY
!----------------------------------------------------------------
!+
!  Accumulate quadrupole and octupole moments of a particle
!  about the node centre of mass (extensive Cartesian form).
!+
!----------------------------------------------------------------
pure subroutine add_node_moments(pmassi,dx,dy,dz,quads,octs)
 real, intent(in)    :: pmassi,dx,dy,dz
 real, intent(inout) :: quads(6),octs(10)
 real :: dx2,dy2,dz2

 dx2 = dx*dx
 dy2 = dy*dy
 dz2 = dz*dz
 quads(1) = quads(1) + pmassi*dx2          ! Q_xx
 quads(2) = quads(2) + pmassi*dx*dy        ! Q_xy
 quads(3) = quads(3) + pmassi*dx*dz        ! Q_xz
 quads(4) = quads(4) + pmassi*dy2          ! Q_yy
 quads(5) = quads(5) + pmassi*dy*dz        ! Q_yz
 quads(6) = quads(6) + pmassi*dz2          ! Q_zz
 octs(1)  = octs(1)  + pmassi*dx2*dx       ! xxx
 octs(2)  = octs(2)  + pmassi*dx2*dy       ! xxy
 octs(3)  = octs(3)  + pmassi*dx2*dz       ! xxz
 octs(4)  = octs(4)  + pmassi*dx*dy2       ! xyy
 octs(5)  = octs(5)  + pmassi*dx*dy*dz     ! xyz
 octs(6)  = octs(6)  + pmassi*dx*dz2       ! xzz
 octs(7)  = octs(7)  + pmassi*dy2*dy       ! yyy
 octs(8)  = octs(8)  + pmassi*dy2*dz       ! yyz
 octs(9)  = octs(9)  + pmassi*dy*dz2       ! yzz
 octs(10) = octs(10) + pmassi*dz2*dz       ! zzz

end subroutine add_node_moments
#endif

!--------------------------------------------------------------------
!+
!  create all the properties for a given node such as centre of mass,
!  size, max smoothing length, etc
!  will also split the node if necessary, setting wassplit=true
!  returns the left and right child information if split
!+
!--------------------------------------------------------------------
subroutine construct_node(nodeentry, nnode, mymum, level, xmini, xmaxi, npnode, doparallel,&
                          il, ir, nl, nr, xminl, xmaxl, xminr, xmaxr,ncells, leaf_is_active, &
                          minlevel, maxlevel, wassplit, global_build,apr_tree, &
                          xyzmh_ptmass)
 use dim,       only:maxtypes,mpi,ind_timesteps
 use io,        only:fatal,error
 use mpitree,   only:get_group_cofm,reduce_group
 type(kdnode),    intent(out)   :: nodeentry
 integer,         intent(in)    :: nnode, mymum, level
 real,            intent(inout) :: xmini(3), xmaxi(3)
 integer,         intent(in)    :: npnode
 logical,         intent(in)    :: doparallel
 integer,         intent(out)   :: il, ir, nl, nr
 real,            intent(out)   :: xminl(3), xmaxl(3), xminr(3), xmaxr(3)
 integer(kind=8), intent(inout) :: ncells
 integer,         intent(out)   :: leaf_is_active(:)
 integer,         intent(inout) :: maxlevel, minlevel
 logical,         intent(out)   :: wassplit
 logical,         intent(in)    :: global_build
 logical,         intent(in)    :: apr_tree
 real,            intent(in), optional :: xyzmh_ptmass(:,:)

 integer(kind=8) :: myslot
 real    :: xyzcofm(3)
 real    :: totmass_node
 real    :: xyzcofmg(3)
 real    :: totmassg
 integer :: npnodetot
 logical :: nodeisactive
 integer :: i,npcounter,ipart
 real    :: x0(3)
 integer :: iaxis
 real    :: xpivot

 nodeisactive = .false.
 if (inoderange(1,nnode) > 0) then
    checkactive: do i = inoderange(1,nnode),inoderange(2,nnode)
       if (inodeparts(i) > 0) then
          nodeisactive = .true.
          exit checkactive
       endif
    enddo checkactive
    npcounter = inoderange(2,nnode) - inoderange(1,nnode) + 1
 else
    npcounter = 0
 endif

 if (npcounter /= npnode) then
    print*,'constructing node ',nnode,': found ',npcounter,' particles, expected:',npnode,' particles for this node'
    call fatal('maketree', 'expected number of particles in node differed from actual number')
 endif

 if (mpi .and. global_build) then
    npnodetot = reduce_group(npnode,'+',level)
 else
    npnodetot = npnode
 endif

 ! following lines to avoid compiler warnings on intent(out) variables
 ir = 0
 il = 0
 nl = 0
 nr = 0

 wassplit    = (npnodetot > minpart)

 xminl = 0.
 xmaxl = 0.
 xminr = 0.
 xmaxr = 0.

 if ((.not. global_build) .and. (npnode  <  1)) return ! node has no particles, just quit

 xyzcofm(:) = 0.

 call compute_nodes_cofm(npnode,nnode,xyzcofm,totmass_node,doparallel)
 ! if this is global node construction, get the cofm and total mass
 ! of all particles in this node (some on other MPI tasks)
 if (mpi .and. global_build) then
    call get_group_cofm(xyzcofm,totmass_node,level,xyzcofmg,totmassg)
    xyzcofm = xyzcofmg
    totmass_node = totmassg
 endif
 ! checks the reduced mass in the case of global maketree
 if (totmass_node<=0. .and. use_apr) call fatal('mtree + apr', &
    'totmass_node==0, something almost certainly wrong with aprmassoftype')
 if (totmass_node<=0.) call fatal('mtree','totmass_node==0',val=totmass_node)

 call set_nodes_properties(npnode,nnode,xyzcofm,totmass_node,mymum,nodeentry,xmini,xmaxi,&
                           level,global_build,doparallel)

 if (use_geosplit) then !--for gravity KDtree, we need the geo centre to split the node
    x0 = (xmaxi+xmini)*0.5
 else  !--for default KDtree, we need the split centre to be the centre of mass
    x0 = xyzcofm
 endif

 if (apr_tree)   wassplit = (npnode > 2)

 if (.not. wassplit) then
    nodeentry%leftchild  = 0
    nodeentry%rightchild = 0
    maxlevel = max(level,maxlevel)
    minlevel = min(level,minlevel)

    if (maxlevel > maxdepth) call fatal('maketree','maximum tree depth reached !!')
    ! individual timesteps where we mark leaf node as active/inactive
    if (ind_timesteps) then
       !
       !--mark leaf node as active (contains some active particles)
       !  or inactive by setting the firstincell to +ve (active) or -ve (inactive)
       !
       if (nodeisactive) then
          leaf_is_active(nnode) = 1
       else
          leaf_is_active(nnode) = -1
       endif
    else
       leaf_is_active(nnode) = 1
    endif
 else ! split this node and add children to stack
    iaxis  = maxloc(xmaxi - xmini,1) ! split along longest axis
    xpivot = x0(iaxis)

    maxlevel = max(level,maxlevel)
    if (maxlevel > maxdepth) call fatal('maketree','maximum tree depth reached !!')
    ! create two children nodes and point to them from current node
    ! always use G&R indexing for global tree
    if (((level < maxlevel_indexed) .or. global_build)) then !.and. (.not. use_geosplit)) then
       il = 2*nnode   ! indexing as per Gafton & Rosswog (2011)
       ir = il + 1
    else
       ! no need to lock, we could just atomic the update
       !$omp atomic capture
       ncells = ncells + 2
       myslot = ncells
       !$omp end atomic
       ir = int(myslot)
       il = int(myslot-1)
       if (ir > ncellsmax) call fatal('maketree',&
          'number of nodes exceeds array dimensions, increase ncellsmax and recompile',ival=int(ncellsmax))
    endif
    nodeentry%leftchild  = il
    nodeentry%rightchild = ir

    leaf_is_active(nnode) = 0

    if (npnode > 0) then
       if (apr_tree) then
          ! apr special sort - only used for merging particles
          call special_sort_particles_in_cell(iaxis,inoderange(1,nnode),inoderange(2,nnode),inoderange(1,il),inoderange(2,il),&
                                    inoderange(1,ir),inoderange(2,ir),nl,nr,xpivot,treecache,inodeparts,&
                                    npnode)
       else
          ! regular sort
          call sort_particles_in_cell(iaxis,inoderange(1,nnode),inoderange(2,nnode),inoderange(1,il),inoderange(2,il),&
                                  inoderange(1,ir),inoderange(2,ir),nl,nr,xpivot,treecache,inodeparts)
       endif

       if (nr + nl  /=  npnode) then
          call error('maketree','number of left + right != parent while splitting (likely cause: NaNs in position arrays)')
       endif

       ! see if all the particles ended up in one node, if so, arbitrarily build 2 cells. This should never happen
       if ((.not. global_build) .and. ((nl==npnode) .or. (nr==npnode))) then
          ! no need to move particles because if they all ended up in one node,
          ! then they are still in the original order
          nl = npnode / 2
          inoderange(1,il) = inoderange(1,nnode)
          inoderange(2,il) = inoderange(1,nnode) + nl - 1
          inoderange(1,ir) = inoderange(1,nnode) + nl
          inoderange(2,ir) = inoderange(2,nnode)
          nr = npnode - nl
       endif

       ! compute min/max with explicit loops for better cache behavior
       xminl(:) =  huge(xminl)
       xmaxl(:) = -huge(xmaxl)
       do ipart=inoderange(1,il),inoderange(2,il)
          xminl(1) = min(xminl(1),treecache(1,ipart))
          xminl(2) = min(xminl(2),treecache(2,ipart))
          xminl(3) = min(xminl(3),treecache(3,ipart))
          xmaxl(1) = max(xmaxl(1),treecache(1,ipart))
          xmaxl(2) = max(xmaxl(2),treecache(2,ipart))
          xmaxl(3) = max(xmaxl(3),treecache(3,ipart))
       enddo

       xminr(:) =  huge(xminr)
       xmaxr(:) = -huge(xmaxr)
       do ipart=inoderange(1,ir),inoderange(2,ir)
          xminr(1) = min(xminr(1),treecache(1,ipart))
          xminr(2) = min(xminr(2),treecache(2,ipart))
          xminr(3) = min(xminr(3),treecache(3,ipart))
          xmaxr(1) = max(xmaxr(1),treecache(1,ipart))
          xmaxr(2) = max(xmaxr(2),treecache(2,ipart))
          xmaxr(3) = max(xmaxr(3),treecache(3,ipart))
       enddo
    else
       nl = 0
       nr = 0
       xminl =  huge(xminl)
       xmaxl = -huge(xmaxl)
       xminr =  huge(xminr)
       xmaxr = -huge(xmaxr)
    endif

    ! Reduce node limits of children across MPI tasks belonging to this group.
    ! The synchronisation needs to happen here, not at the next level, because
    ! the groups will be independent by then.
    if (mpi .and. global_build) then
       xminl(1) = reduce_group(xminl(1),'min',level)
       xminl(2) = reduce_group(xminl(2),'min',level)
       xminl(3) = reduce_group(xminl(3),'min',level)

       xmaxl(1) = reduce_group(xmaxl(1),'max',level)
       xmaxl(2) = reduce_group(xmaxl(2),'max',level)
       xmaxl(3) = reduce_group(xmaxl(3),'max',level)

       xminr(1) = reduce_group(xminr(1),'min',level)
       xminr(2) = reduce_group(xminr(2),'min',level)
       xminr(3) = reduce_group(xminr(3),'min',level)

       xmaxr(1) = reduce_group(xmaxr(1),'max',level)
       xmaxr(2) = reduce_group(xmaxr(2),'max',level)
       xmaxr(3) = reduce_group(xmaxr(3),'max',level)
    endif

 endif

end subroutine construct_node

!----------------------------------------------------------------
!+
!  Categorise particles into daughter nodes by whether they
!  fall to the left or the right of the pivot axis
!+
!----------------------------------------------------------------
subroutine sort_particles_in_cell(iaxis,imin,imax,min_l,max_l,min_r,max_r,nl,nr,xpivot,&
                                   treecache,inodeparts)
 integer, intent(in)    :: iaxis,imin,imax
 integer, intent(out)   :: min_l,max_l,min_r,max_r,nl,nr
 real,    intent(inout) :: xpivot,treecache(:,:)
 integer, intent(inout) :: inodeparts(:)
 logical :: i_lt_pivot,j_lt_pivot
 integer :: inodeparts_swap,i,j
 real :: xyzh_swap(5)
 real :: xi_coord, xj_coord

 !print*,'nnode ',imin,imax,' pivot = ',iaxis,xpivot
 i = imin
 j = imax

 xi_coord = treecache(iaxis,i)
 xj_coord = treecache(iaxis,j)
 i_lt_pivot = xi_coord <= xpivot
 j_lt_pivot = xj_coord <= xpivot
 !  k = 0

 do while(i < j)
    if (i_lt_pivot) then
       i = i + 1
       xi_coord = treecache(iaxis,i)
       i_lt_pivot = xi_coord <= xpivot
    else
       if (.not.j_lt_pivot) then
          j = j - 1
          xj_coord = treecache(iaxis,j)
          j_lt_pivot = xj_coord <= xpivot
       else
          ! swap i and j positions in list
          inodeparts_swap = inodeparts(i)
          xyzh_swap(1:5)  = treecache(1:5,i)

          inodeparts(i)   = inodeparts(j)
          treecache(1:5,i) = treecache(1:5,j)

          inodeparts(j)   = inodeparts_swap
          treecache(1:5,j) = xyzh_swap(1:5)

          i = i + 1
          j = j - 1
          xi_coord = treecache(iaxis,i)
          xj_coord = treecache(iaxis,j)
          i_lt_pivot = xi_coord <= xpivot
          j_lt_pivot = xj_coord <= xpivot
          ! k = k + 1
       endif
    endif
 enddo
 if (.not.i_lt_pivot) i = i - 1
 if (j_lt_pivot)      j = j + 1

 min_l = imin
 max_l = i
 min_r = j
 max_r = imax

 if ( j /= i+1) print*,' ERROR ',i,j
 nl = max_l - min_l + 1
 nr = max_r - min_r + 1

end subroutine sort_particles_in_cell

!----------------------------------------------------------------
!+
!  Categorise particles into daughter nodes by whether they
!  fall to the left or the right of the pivot axis, but additionally
!  force the cells to have a certain minimum number of particles per cell
!+
!----------------------------------------------------------------
subroutine special_sort_particles_in_cell(iaxis,imin,imax,min_l,max_l,min_r,max_r,&
                                nl,nr,xpivot,treecache,inodeparts,npnode)
 use io, only:error
 integer, intent(in)    :: iaxis,imin,imax,npnode
 integer, intent(out)   :: min_l,max_l,min_r,max_r,nl,nr
 real,    intent(inout) :: xpivot,treecache(:,:)
 integer, intent(inout) :: inodeparts(:)
 logical :: i_lt_pivot,j_lt_pivot,slide_l,slide_r
 integer :: inodeparts_swap,i,j,nchild_in
 integer :: k,ii,rem_nr,rem_nl
 real :: xyzh_swap(5),dpivot(npnode)

 dpivot = 0.0
 nchild_in = 2

 if (modulo(npnode,nchild_in) > 0) then
    call error('apr sort','number of particles sent in to kdtree is not divisible by 2')
 endif

! print*,'nnode ',imin,imax,npnode,' pivot = ',iaxis,xpivot
 i = imin
 j = imax

 i_lt_pivot = treecache(iaxis,i) <= xpivot
 j_lt_pivot = treecache(iaxis,j) <= xpivot
 dpivot(i-imin+1) = xpivot - treecache(iaxis,i)
 dpivot(j-imin+1) = xpivot - treecache(iaxis,j)
 !k = 0
 do while(i < j)
    if (i_lt_pivot) then
       i = i + 1
       dpivot(i-imin+1) = xpivot - treecache(iaxis,i)
       i_lt_pivot = treecache(iaxis,i) <= xpivot
    else
       if (.not.j_lt_pivot) then
          j = j - 1
          dpivot(j-imin+1) = xpivot - treecache(iaxis,j)
          j_lt_pivot = treecache(iaxis,j) <= xpivot
       else
          ! swap i and j positions in list
          inodeparts_swap = inodeparts(i)
          xyzh_swap(1:5)  = treecache(1:5,i)

          inodeparts(i)   = inodeparts(j)
          treecache(1:5,i) = treecache(1:5,j)

          inodeparts(j)   = inodeparts_swap
          treecache(1:5,j) = xyzh_swap(1:5)

          i = i + 1
          j = j - 1

          dpivot(i-imin+1) = xpivot - treecache(iaxis,i)
          dpivot(j-imin+1) = xpivot - treecache(iaxis,j)

          i_lt_pivot = treecache(iaxis,i) <= xpivot
          j_lt_pivot = treecache(iaxis,j) <= xpivot
       endif
    endif
 enddo

 if (.not.i_lt_pivot) then
    i = i - 1
    dpivot(i-imin+1) = xpivot - treecache(iaxis,i)
 endif
 if (j_lt_pivot) then
    j = j + 1
    dpivot(j-imin+1) = xpivot - treecache(iaxis,j)
 endif

 min_l = imin
 max_l = i
 min_r = j
 max_r = imax

 if ( j /= i+1) print*,' ERROR ',i,j
 nl = max_l - min_l + 1
 nr = max_r - min_r + 1

 ! does the pivot need to be adjusted?
 rem_nl = modulo(nl,nchild_in)
 rem_nr = modulo(nr,nchild_in)
 if (rem_nl == 0 .and. rem_nr == 0) return

 ! Decide which direction the pivot needs to go
 if (rem_nl < rem_nr) then
    slide_l = .true.
    slide_r = .false.
 else
    slide_l = .false.
    slide_r = .true.
 endif
 ! Override this if there's less than nchild*2 in the cell
 if (nl < nchild_in) then
    slide_r = .true.
    slide_l = .false.
 elseif (nr < nchild_in) then
    slide_r = .false.
    slide_l = .true.
 endif

 ! Move across particles by distance from xpivot till we get
 ! the right number of particles in each cell
 if (slide_r) then
    do ii = 1,rem_nr
       ! next particle to shift across
       k = minloc(dpivot,dim=1,mask=dpivot > 0.) + imin - 1
       if (k-imin+1==0) k = maxloc(dpivot,dim=1,mask=dpivot < 0.) + imin - 1

       ! swap this with the first particle on the j side
       inodeparts_swap = inodeparts(k)
       xyzh_swap(1:5)  = treecache(1:5,k)

       inodeparts(k)   = inodeparts(j)
       treecache(1:5,k) = treecache(1:5,j)

       inodeparts(j)   = inodeparts_swap
       treecache(1:5,j) = xyzh_swap(1:5)

       ! and now shift to the right
       i = i + 1
       j = j + 1

       ! ditch it, go again
       dpivot(k-imin+1) = huge(k-imin+1)
    enddo
 else
    do ii = 1,rem_nl
       ! next particle to shift across
       k = maxloc(dpivot,dim=1,mask=dpivot < 0.) + imin - 1
       if (k-imin+1==0) k = minloc(dpivot,dim=1,mask=dpivot > 0.) + imin - 1

       ! swap this with the last particle on the i side
       inodeparts_swap = inodeparts(k)
       xyzh_swap(1:5)  = treecache(1:5,k)

       inodeparts(k)   = inodeparts(i)
       treecache(1:5,k) = treecache(1:5,i)

       inodeparts(i)   = inodeparts_swap
       treecache(1:5,i) = xyzh_swap(1:5)

       ! and now shift to the left
       i = i - 1
       j = j - 1

       ! ditch it, go again
       dpivot(k-imin+1) = huge(k-imin+1)

    enddo
 endif

 ! tidy up outputs
 max_l = i
 min_r = j
 nl = max_l - min_l + 1
 nr = max_r - min_r + 1

end subroutine special_sort_particles_in_cell

!-----------------------------------------------
!+
!  Routine to update a constructed tree
!  Note: current version ONLY works if
!  tree is built to < maxlevel_indexed
!  That is, it relies on the 2^n style tree
!  indexing to sweep out each level
!+
!-----------------------------------------------
subroutine revtree(node, xyzh, leaf_is_active, ncells)
 use dim,  only:maxp,use_apr,ind_timesteps
 use part, only:maxphase,iphase,igas,massoftype,iamtype,aprmassoftype,&
                apr_level,iactive,treecache,isdead_or_accreted
 use io,   only:fatal
 type(kdnode),    intent(inout) :: node(:) !ncellsmax+1)
 real,            intent(in)    :: xyzh(:,:)
 integer,         intent(inout) :: leaf_is_active(:) !ncellsmax+1)
 integer(kind=8), intent(in)    :: ncells
 real :: hmax, r2max
 real :: xi, yi, zi, hi
 real :: dx, dy, dz, dr2
#ifdef GRAVITY
 real :: quads(6)
 real :: octs(10)
#endif
 integer :: inode, ipart, ipartidx, i, nptot
 real :: pmassi, totmass
 real :: x0(3)
 real :: xcofm, ycofm, zcofm, fac, dfac
 logical :: nodeisactive

 pmassi = massoftype(igas)

 ! find maximum index in inodeparts that we need to update in treecache
 nptot = 0
 do i=1,int(ncells)
    if (i > 1 .and. node(i)%parent == 0) cycle
    if (inoderange(1,i) > 0 .and. inoderange(2,i) >= inoderange(1,i)) then
       nptot = max(nptot, inoderange(2,i))
    endif
 enddo

 ! update treecache for particles in the tree only
 ! mark dead/accreted particles by setting treecache(4,i) negative
 !$omp parallel default(none) &
 !$omp shared(nptot,inodeparts,xyzh,iphase,apr_level) &
 !$omp shared(massoftype,aprmassoftype,treecache) &
 !$omp shared(maxphase,maxp) &
 !$omp private(i,ipartidx)
 !$omp do schedule(static)
 do i=1,nptot
    if (inodeparts(i) == 0) cycle
    ipartidx = abs(inodeparts(i))
    treecache(1:4,i) = xyzh(1:4,ipartidx)
    ! compute and store mass
    if (maxphase==maxp) then
       if (use_apr) then
          treecache(5,i) = aprmassoftype(iamtype(iphase(ipartidx)),apr_level(ipartidx))
       else
          treecache(5,i) = massoftype(iamtype(iphase(ipartidx)))
       endif
    elseif (use_apr) then
       treecache(5,i) = aprmassoftype(igas,apr_level(ipartidx))
    else
       treecache(5,i) = massoftype(igas)
    endif
 enddo
 !$omp enddo
 !$omp end parallel

!$omp parallel default(none) &
!$omp shared(maxp,maxphase) &
!$omp shared(ncells) &
!$omp shared(node,inoderange,inodeparts,treecache,leaf_is_active) &
!$omp private(hmax,r2max,xi,yi,zi,hi) &
!$omp private(dx,dy,dz,dr2,inode,ipart,x0) &
!$omp private(xcofm,ycofm,zcofm,fac,dfac,nodeisactive) &
#ifdef GRAVITY
!$omp private(quads,octs) &
#endif
!$omp firstprivate(pmassi) &
!$omp private(totmass)
!$omp do schedule(guided)
 over_nodes: do inode=1,int(ncells)
    if (inode > 1 .and. node(inode)%parent == 0) cycle
    ! initialize node properties
    node(inode)%xcen(:) = 0.
    node(inode)%size    = 0.
    node(inode)%hmax    = 0.
#ifdef GRAVITY
    node(inode)%mass    = 0.
    node(inode)%quads(:)= 0.
    node(inode)%octs(:) = 0.
#endif
    ! initialize leaf_is_active (will be set for leaf nodes below)
    leaf_is_active(inode) = 0

    ! check if node has particles
    if (inoderange(1,inode) <= 0 .or. inoderange(2,inode) < inoderange(1,inode)) cycle over_nodes

    ! find centre of mass from particle list using same algorithm as maketree
    ! also check for active particles and compute hmax during this loop
    xcofm = 0.
    ycofm = 0.
    zcofm = 0.
    totmass = 0.0
    hmax = 0.
    dfac = 1.
    if (pmassi > 0.) then
       dfac = 1./pmassi
    endif
    nodeisactive = .false.
    do ipart = inoderange(1,inode), inoderange(2,inode)
       if (inodeparts(ipart) == 0) cycle
       xi = treecache(1,ipart)
       yi = treecache(2,ipart)
       zi = treecache(3,ipart)
       hi = treecache(4,ipart)
       ! check condition after loading (dead/accreted particles have hi <= 0)
       if (hi <= 0.) cycle
       hi = abs(hi)
       pmassi = treecache(5,ipart)
       ! check for active particles (for leaf_is_active flag)
       if (ind_timesteps .and. .not. nodeisactive) then
          if (inodeparts(ipart) > 0) nodeisactive = .true.
       endif
       fac = pmassi*dfac
       xcofm = xcofm + fac*xi
       ycofm = ycofm + fac*yi
       zcofm = zcofm + fac*zi
       totmass = totmass + pmassi
       hmax = max(hi, hmax)
    enddo
    if (.not. ind_timesteps) nodeisactive = .true.

    if (totmass <= 0.0) cycle over_nodes

    x0(1) = xcofm/(totmass*dfac)
    x0(2) = ycofm/(totmass*dfac)
    x0(3) = zcofm/(totmass*dfac)

    ! update cell size and quads
    r2max = 0.
#ifdef GRAVITY
    quads = 0.
    octs  = 0.
#endif
    do ipart = inoderange(1,inode), inoderange(2,inode)
       ! load all treecache values sequentially (1,2,3,4,5) for cache efficiency
       xi = treecache(1,ipart)
       yi = treecache(2,ipart)
       zi = treecache(3,ipart)
       hi = treecache(4,ipart)
       ! check condition after loading (dead/accreted particles have hi <= 0)
       if (hi <= 0.) cycle
       pmassi = treecache(5,ipart)
       dx = xi - x0(1)
       dy = yi - x0(2)
       dz = zi - x0(3)
       dr2 = dx*dx + dy*dy + dz*dz
       r2max = max(dr2, r2max)
#ifdef GRAVITY
       call add_node_moments(pmassi,dx,dy,dz,quads,octs)
#endif
    enddo

    node(inode)%xcen(1) = x0(1)
    node(inode)%xcen(2) = x0(2)
    node(inode)%xcen(3) = x0(3)
    node(inode)%size = sqrt(r2max) + epsilon(r2max)
    node(inode)%hmax = hmax
#ifdef GRAVITY
    node(inode)%mass = totmass
    node(inode)%quads = quads
    node(inode)%octs  = octs
#endif

    ! set leaf_is_active flag for leaf nodes (matching maketree behavior)
    if (node(inode)%leftchild == 0 .and. node(inode)%rightchild == 0) then
       if (ind_timesteps) then
          if (nodeisactive) then
             leaf_is_active(inode) = 1
          else
             leaf_is_active(inode) = -1
          endif
       else
          leaf_is_active(inode) = 1
       endif
    endif
 enddo over_nodes
!$omp enddo
!$omp end parallel

end subroutine revtree

!--------------------------------------------------------------------------------
!+
!  Routine to build the global level tree
!+
!-------------------------------------------------------------------------------
subroutine maketreeglobal(nodeglobal,node,nodemap,globallevel,refinelevels,xyzh,&
                          np,cellatid,leaf_is_active,ncells,apr_tree,nptmass,xyzmh_ptmass)
 use io,           only:fatal,warning,id,nprocs,master
 use mpiutils,     only:reduceall_mpi
 use mpibalance,   only:balancedomains
 use mpitree,      only:tree_sync,tree_bcast
 use part,         only:isdead_or_accreted,iactive,ibelong,isink,massoftype,igas,&
                        iamtype,maxphase,maxp,aprmassoftype,apr_level,ihsoft
 use timing,       only:increment_timer,get_timings,itimer_balance
 use dim,          only:ind_timesteps

 type(kdnode),    intent(out)   :: nodeglobal(:)    ! ncellsmax+1
 type(kdnode),    intent(out)   :: node(:)          ! ncellsmax+1
 integer,         intent(out)   :: nodemap(:)       ! ncellsmax+1
 integer,         intent(out)   :: globallevel
 integer,         intent(out)   :: refinelevels
 integer,         intent(inout) :: np
 real,            intent(inout) :: xyzh(:,:)
 integer,         intent(out)   :: cellatid(:)      ! ncellsmax+1
 integer,         intent(out)   :: leaf_is_active(:)  ! ncellsmax+1)
 integer(kind=8), intent(out)   :: ncells
 logical,         intent(in)    :: apr_tree
 integer,         intent(in),    optional :: nptmass
 real,            intent(inout), optional :: xyzmh_ptmass(:,:)
 real                              :: xmini(3),xmaxi(3)
 real                              :: xminl(3),xmaxl(3)
 real                              :: xminr(3),xmaxr(3)
 integer                           :: minlevel, maxlevel
 integer                           :: idleft, idright
 integer                           :: groupsize,ifirstingroup,groupsplit
 type(kdnode)                      :: mynode(1)
 integer                           :: nl, nr
 integer                           :: il, ir, iself, parent
 integer                           :: level
 integer                           :: nnodestart, nnodeend,locstart,locend
 integer                           :: npcounter
 integer                           :: i, k, offset, roffset, roffset_prev, coffset
 integer                           :: inode
 integer                           :: npnode
 logical                           :: wassplit,sinktree
 real(kind=4)                      :: t1,t2,tcpu1,tcpu2

 sinktree = .false.
 if (present(nptmass).and.present(xyzmh_ptmass)) sinktree=.true.
 parent = 0
 iself = irootnode
 leaf_is_active = 0

 ! root is level 0
 globallevel = int(ceiling(log(real(nprocs)) / log(2.0)))

 minlevel = maxdepth - 1
 maxlevel = 0

 levels: do level = 0, globallevel
    groupsize = 2**(globallevel - level)
    ifirstingroup = (id / groupsize) * groupsize
    if (level == 0) then
       if (sinktree) then
          call construct_root_node(np,npcounter,irootnode,xmini,xmaxi,leaf_is_active,xyzh,&
                                   xyzmh_ptmass,nptmass)
       else
          call construct_root_node(np,npcounter,irootnode,xmini,xmaxi,leaf_is_active,xyzh)
       endif
    else
       npcounter = npnode
    endif
    if (sinktree) then
       call construct_node(mynode(1), iself, parent, level, xmini, xmaxi, npcounter, .false., &
                           il, ir, nl, nr, xminl, xmaxl, xminr, xmaxr,ncells, leaf_is_active, &
                           minlevel, maxlevel, wassplit,.true.,apr_tree,xyzmh_ptmass)
    else
       call construct_node(mynode(1), iself, parent, level, xmini, xmaxi, npcounter, .false., &
                        il, ir, nl, nr, xminl, xmaxl, xminr, xmaxr,ncells, leaf_is_active, &
                        minlevel, maxlevel, wassplit,.true.,apr_tree)
    endif

    if (.not.wassplit) then
       call fatal('maketreeglobal','insufficient particles for splitting at the global level: '// &
            'use more particles or less MPI threads')
    endif

    ! set which tree child this proc will belong to next
    groupsplit = ifirstingroup + (groupsize / 2)

    ! record parent for next round
    parent = iself

    ! which half of the tree this task is on
    if (id < groupsplit) then
       ! i for the next node we construct
       iself = il
       ! the left and right task IDs
       idleft = id
       idright = id + 2**(globallevel - level - 1)
       xmini = xminl
       xmaxi = xmaxl
    else
       iself = ir
       idleft = id - 2**(globallevel - level - 1)
       idright = id
       xmini = xminr
       xmaxi = xmaxr
    endif
    if (sinktree) then
       if (nptmass>0) then
          ibelong(maxpsph+1:maxpsph+nptmass) = -1
       endif
    endif
    if (npcounter > 0) then
       do i = inoderange(1,il), inoderange(2,il)
          ibelong(abs(inodeparts(i))) = idleft
       enddo
       do i = inoderange(1,ir), inoderange(2,ir)
          ibelong(abs(inodeparts(i))) = idright
       enddo
    endif

    call get_timings(t1,tcpu1)
    ! move particles to where they belong
    call balancedomains(np)
    call get_timings(t2,tcpu2)
    if (sinktree) ibelong(maxpsph+1:maxpsph+nptmass) = int(reduceall_mpi("max", ibelong(maxpsph+1:maxpsph+nptmass)))
    call increment_timer(itimer_balance,t2-t1,tcpu2-tcpu1)
    ! move particles from old array
    ! this is a waste of time, but maintains compatibility
    npnode = 0
    do i=1,np
       npnode = npnode + 1
       !
       ! tag inactive particles with negative index
       ! in the particle list for the node
       !
       if (ind_timesteps) then
          if (iactive(iphase(i))) then
             inodeparts(npnode) = i
          else
             inodeparts(npnode) = -i
          endif
       else
          inodeparts(npnode) = i
       endif
       treecache(1:4,npnode) = xyzh(1:4,i)
       if (maxphase==maxp) then
          if (use_apr) then
             treecache(5,npnode) = aprmassoftype(iamtype(iphase(i)),apr_level(i))
          else
             treecache(5,npnode) = massoftype(iamtype(iphase(i)))
          endif
       elseif (use_apr) then
          treecache(5,npnode) = aprmassoftype(igas,apr_level(i))
       else
          treecache(5,npnode) = massoftype(igas)
       endif
    enddo
    if (sinktree) then
       if (nptmass > 0) then
          do i=1,nptmass
             if (ibelong(maxpsph + i) /= id) cycle
             if (xyzmh_ptmass(4,i) < 0.) cycle ! dead sink particle
             npnode = npnode + 1
             inodeparts(npnode) = maxpsph + i
             treecache(1:3,npnode) = xyzmh_ptmass(1:3,i)
             treecache(4,npnode)   = xyzmh_ptmass(ihsoft,i)
             treecache(5,npnode)   = xyzmh_ptmass(4,i)
          enddo
       endif
    endif

    ! set all particles to belong to this node
    inoderange(1,iself) = 1
    inoderange(2,iself) = npnode

    ! range of newly written tree
    nnodestart = 2**level
    nnodeend = 2**(level + 1) - 1

    ! synchronize tree with other owners if this proc is the first in group
    call tree_sync(mynode,1,nodeglobal(nnodestart:nnodeend),nprocs/groupsize,ifirstingroup,level)

    ! at level 0, tree_sync already 'broadcasts'
    if (level > 0) then
       ! tree broadcast to non-owners
       call tree_bcast(nodeglobal(nnodestart:nnodeend), nnodeend - nnodestart + 1, level)
    endif

 enddo levels

 ! local tree
 if (sinktree) then
    call maketree(node,xyzh,np,leaf_is_active,ncells,apr_tree,refinelevels,nptmass,xyzmh_ptmass)
 else
    call maketree(node,xyzh,np,leaf_is_active,ncells,apr_tree,refinelevels)
 endif

 ! tree refinement
 refinelevels = int(reduceall_mpi('min',refinelevels),kind=kind(refinelevels))
 roffset_prev = 1

 irefine = 0
 do i = 1,refinelevels
    offset = 2**(globallevel + i)
    roffset = 2**i

    nnodestart = offset
    nnodeend   = 2*nnodestart-1

    if (nnodeend > ncellsmax) call fatal('kdtree', 'global tree refinement has exceeded ncellsmax')

    locstart   = roffset
    locend     = 2*locstart-1

    ! index shift the node to the global level
    do k = roffset,2*roffset-1
       refinementnode(k) = node(k)
       coffset = refinementnode(k)%parent - roffset_prev

       refinementnode(k)%parent = 2**(globallevel + i - 1) + id * roffset_prev + coffset

       if (i /= refinelevels) then
          refinementnode(k)%leftchild  = 2**(globallevel + i + 1) + 2*id*roffset + 2*(k - roffset)
          refinementnode(k)%rightchild = refinementnode(k)%leftchild + 1
       else
          refinementnode(k)%leftchild = 0
          refinementnode(k)%rightchild = 0
       endif
    enddo

    roffset_prev = roffset
    ! sync, replacing level with globallevel, since all procs will get synced
    ! and deeper comms do not exist
    call tree_sync(refinementnode(locstart:locend),roffset, &
                   nodeglobal(nnodestart:nnodeend),nnodestart-nnodeend, &
                   id,globallevel)

    ! get the mapping from the local tree to the global tree, for future hmax updates
    do inode = locstart,locend
       nodemap(inode) = nnodestart + (id * roffset) + (inode - locstart)
    enddo
 enddo
!  The index up to which the local tree is copied to the global tree
 irefine = 2*roffset-1

 ! cellatid is zero by default
 cellatid = 0
 do i = 1,nprocs
    offset = 2**(globallevel+refinelevels)
    roffset = 2**refinelevels
    do k = 1,roffset
       cellatid(offset + (i - 1) * roffset + (k - 1)) = i
    enddo
 enddo

end subroutine maketreeglobal

end module kdtree
