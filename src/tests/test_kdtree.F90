!--------------------------------------------------------------------------!
! The Phantom Smoothed Particle Hydrodynamics code, by Daniel Price et al. !
! Copyright (c) 2007-2026 The Authors (see AUTHORS)                        !
! See LICENCE file for usage and distribution conditions                   !
! http://phantomsph.github.io/                                             !
!--------------------------------------------------------------------------!
module testkdtree
!
! This module performs unit tests of the kdtree module
!   The tests here are specific to the tree, some general
!   tests of neighbour finding are done in test_neigh
!
! :References: None
!
! :Owner: Daniel Price
!
! :Runtime parameters: None
!
! :Dependencies: dim, io, kdtree, kernel, mpidomain, neighkdtree, part,
!   random, testutils, timestep, timing, unifdis
!
 implicit none
 public :: test_kdtree

 private

contains
!-----------------------------------------------------------------------
!+
!   Unit tests of tree code
!+
!-----------------------------------------------------------------------
subroutine test_kdtree(ntests,npass)
 use dim,         only:maxp,periodic,ind_timesteps
 use io,          only:id,master,iverbose
 use neighkdtree, only:leaf_is_active,ncells,node
 use part,        only:npart,xyzh,hfact,massoftype,igas,maxphase,iphase,isetphase,iactive
 use kernel,      only:hfact_default
 use kdtree,      only:maketree,revtree,kdnode,empty_tree
 use unifdis,     only:set_unifdis
 use testutils,   only:checkvalbuf,checkvalbuf_end,update_test_scores,checkval
 use timing,      only:print_time,getused
 use mpidomain,   only:i_belong
 integer, intent(inout) :: ntests,npass
 logical :: test_revtree, test_revise, test_all
 integer :: i,nfailed(22),nchecked(22),nfailed_leaf(1),nchecked_leaf(1),ierrmax_leaf(1)
 real    :: psep,tol,tol_octs,errmax(22)
 real(4) :: t2,t1,tmaketree
 type(kdnode), allocatable :: old_tree(:)
 integer, allocatable :: leaf_is_active_saved(:)

 test_all = .true.
 test_revtree = .true.
 test_revise  = .true.
 iverbose = 2

 if (id==master) write(*,"(a,/)") '--> TESTING KDTREE'

 if (test_revtree .or. test_all) then
    if (id==master) write(*,"(/,a)") '--> testing revtree routine'
    !
    ! set up a random particle distribution
    !
    psep = 1./100.
    hfact = hfact_default
    npart = 0
    call set_unifdis('random',id,master,-0.5,0.5,-0.5,0.5,-0.5,0.5,&
                     psep,hfact,npart,xyzh,periodic,mask=i_belong)
    massoftype(igas) = 1000./npart
    if (maxphase==maxp) iphase(:) = isetphase(igas,iactive=.true.)

    !
    ! call maketree to build the tree
    !
    call empty_tree(node)
    call cpu_time(t1)
    call maketree(node,xyzh,npart,leaf_is_active,ncells,apr_tree=.false.)
    call cpu_time(t2)
    call print_time(t2-t1,'maketree completed in')
    !
    ! now save the tree structure and leaf_is_active
    !
    allocate(old_tree(int(ncells)))
    old_tree(1:ncells) = node(1:ncells)
    allocate(leaf_is_active_saved(int(ncells)))
    leaf_is_active_saved(1:int(ncells)) = leaf_is_active(1:int(ncells))

    !
    ! erase all information in the existing tree except the structure
    !
    do i=1,int(ncells)
       node(i)%xcen(:) = 0.
       node(i)%size    = 0.
       node(i)%hmax    = 0.
#ifdef GRAVITY
       node(i)%mass    = 0.
       node(i)%quads(:)= 0.
       node(i)%octs(:) = 0.
#endif
       leaf_is_active(i) = 0
    enddo

    !
    ! call revtree to rebuild
    !
    tmaketree = t2-t1
    call cpu_time(t1)
    call revtree(node,xyzh,leaf_is_active,ncells)
    call cpu_time(t2)
    call print_time(t2-t1,'revtree completed in')
    if (id==master) print*,' ratio of revtree/maketree: ',(t2-t1)/tmaketree

    !
    ! check that the revised tree matches the tree built
    !
    nfailed(:)  = 0
    nchecked(:) = 0
    errmax(:)   = 0.
    tol = 2.e-11
    ! use larger tolerance for some octupole moments due to variation from openMP loop ordering
    tol_octs = 1.e-9
    do i=1,int(ncells)
       if (i > 1 .and. node(i)%parent == 0) cycle
       ! if (leaf_is_active(i) /= 0) then
       call checkvalbuf(node(i)%xcen(1),old_tree(i)%xcen(1),tol,'x0',nfailed(1),nchecked(1),errmax(1))
       call checkvalbuf(node(i)%xcen(2),old_tree(i)%xcen(2),tol,'y0',nfailed(2),nchecked(2),errmax(2))
       call checkvalbuf(node(i)%xcen(3),old_tree(i)%xcen(3),tol,'z0',nfailed(3),nchecked(3),errmax(3))
!       call checkvalbuf(node(i)%size,old_tree(i)%size,tol,'size',nfailed(4),nchecked(4),errmax(4))
       call checkvalbuf((node(i)%size + tol >= old_tree(i)%size),.true.,'size',nfailed(4),nchecked(4))
       call checkvalbuf(node(i)%hmax,old_tree(i)%hmax,tol,'hmax',nfailed(5),nchecked(5),errmax(5))
#ifdef GRAVITY
       call checkvalbuf(node(i)%mass,old_tree(i)%mass,tol,'mass',nfailed(6),nchecked(6),errmax(6))
       call checkvalbuf(node(i)%quads(1),old_tree(i)%quads(1),tol,'qxx',nfailed(7),nchecked(7),errmax(7))
       call checkvalbuf(node(i)%quads(2),old_tree(i)%quads(2),2.*tol,'qxy',nfailed(8),nchecked(8),errmax(8))
       call checkvalbuf(node(i)%quads(3),old_tree(i)%quads(3),tol,'qxz',nfailed(9),nchecked(9),errmax(9))
       call checkvalbuf(node(i)%quads(4),old_tree(i)%quads(4),tol,'qyy',nfailed(10),nchecked(10),errmax(10))
       call checkvalbuf(node(i)%quads(5),old_tree(i)%quads(5),tol,'qyz',nfailed(11),nchecked(11),errmax(11))
       call checkvalbuf(node(i)%quads(6),old_tree(i)%quads(6),tol,'qzz',nfailed(12),nchecked(12),errmax(12))
       call checkvalbuf(node(i)%octs(1),old_tree(i)%octs(1),tol_octs,'oxxx',nfailed(13),nchecked(13),errmax(13))
       call checkvalbuf(node(i)%octs(2),old_tree(i)%octs(2),tol,'oxxy',nfailed(14),nchecked(14),errmax(14))
       call checkvalbuf(node(i)%octs(3),old_tree(i)%octs(3),tol,'oxxz',nfailed(15),nchecked(15),errmax(15))
       call checkvalbuf(node(i)%octs(4),old_tree(i)%octs(4),tol_octs,'oxyy',nfailed(16),nchecked(16),errmax(16))
       call checkvalbuf(node(i)%octs(5),old_tree(i)%octs(5),tol,'oxyz',nfailed(17),nchecked(17),errmax(17))
       call checkvalbuf(node(i)%octs(6),old_tree(i)%octs(6),tol_octs,'oxzz',nfailed(18),nchecked(18),errmax(18))
       call checkvalbuf(node(i)%octs(7),old_tree(i)%octs(7),tol,'oyyy',nfailed(19),nchecked(19),errmax(19))
       call checkvalbuf(node(i)%octs(8),old_tree(i)%octs(8),tol,'oyyz',nfailed(20),nchecked(20),errmax(20))
       call checkvalbuf(node(i)%octs(9),old_tree(i)%octs(9),tol,'oyzz',nfailed(21),nchecked(21),errmax(21))
       call checkvalbuf(node(i)%octs(10),old_tree(i)%octs(10),tol,'ozzz',nfailed(22),nchecked(22),errmax(22))
#endif
       ! endif
    enddo
    call checkvalbuf_end('x0',nchecked(1),nfailed(1),errmax(1),tol)
    call checkvalbuf_end('y0',nchecked(2),nfailed(2),errmax(2),tol)
    call checkvalbuf_end('z0',nchecked(3),nfailed(3),errmax(3),tol)
    call checkvalbuf_end('size',nchecked(4),nfailed(4),errmax(4),tol)
    call checkvalbuf_end('hmax',nchecked(5),nfailed(5),errmax(5),tol)
#ifdef GRAVITY
    call checkvalbuf_end('mass',nchecked(6),nfailed(6),errmax(6),tol)
    call checkvalbuf_end('qxx',nchecked(7),nfailed(7),errmax(7),tol)
    call checkvalbuf_end('qxy',nchecked(8),nfailed(8),errmax(8),tol)
    call checkvalbuf_end('qxz',nchecked(9),nfailed(9),errmax(9),tol)
    call checkvalbuf_end('qyy',nchecked(10),nfailed(10),errmax(10),tol)
    call checkvalbuf_end('qyz',nchecked(11),nfailed(11),errmax(11),tol)
    call checkvalbuf_end('qzz',nchecked(12),nfailed(12),errmax(12),tol)
    call checkvalbuf_end('oxxx',nchecked(13),nfailed(13),errmax(13),tol_octs)
    call checkvalbuf_end('oxxy',nchecked(14),nfailed(14),errmax(14),tol)
    call checkvalbuf_end('oxxz',nchecked(15),nfailed(15),errmax(15),tol)
    call checkvalbuf_end('oxyy',nchecked(16),nfailed(16),errmax(16),tol_octs)
    call checkvalbuf_end('oxyz',nchecked(17),nfailed(17),errmax(17),tol)
    call checkvalbuf_end('oxzz',nchecked(18),nfailed(18),errmax(18),tol_octs)
    call checkvalbuf_end('oyyy',nchecked(19),nfailed(19),errmax(19),tol)
    call checkvalbuf_end('oyyz',nchecked(20),nfailed(20),errmax(20),tol)
    call checkvalbuf_end('oyzz',nchecked(21),nfailed(21),errmax(21),tol)
    call checkvalbuf_end('ozzz',nchecked(22),nfailed(22),errmax(22),tol)
#endif
    call update_test_scores(ntests,nfailed,npass)

    !
    ! check that leaf_is_active matches what maketree set
    !
    nfailed_leaf(:) = 0
    nchecked_leaf(:) = 0
    ierrmax_leaf(:) = 0
    do i=1,int(ncells)
       ! only check leaf nodes (non-zero leaf_is_active)
       if (leaf_is_active_saved(i) /= 0) then
          call checkvalbuf(leaf_is_active(i),leaf_is_active_saved(i),0,'leaf_is_active', &
                          nfailed_leaf(1),nchecked_leaf(1),ierrmax_leaf(1))
       endif
    enddo
    if (nchecked_leaf(1) > 0) then
       call checkvalbuf_end('leaf_is_active',nchecked_leaf(1),nfailed_leaf(1),ierrmax_leaf(1),0)
    endif
    call update_test_scores(ntests,nfailed_leaf,npass)

    deallocate(old_tree)
    deallocate(leaf_is_active_saved)
 endif

 if ((test_revise .or. test_all) .and. ind_timesteps .and. maxphase==maxp) then
    call test_revise_tree(ntests,npass)
 endif

 if (id==master) write(*,"(/,a,/)") '<-- KDTREE TEST COMPLETE'

end subroutine test_kdtree

!-----------------------------------------------------------------------
!+
!   Unit tests of revise_tree, the update of the tree in place between
!   rebuilds with individual timesteps: after particles move, change h,
!   become active or inactive and die, every live particle must still be
!   in its leaf's sphere with its h within the leaf's hmax, every child's
!   sphere and hmax within its parent's, and the activity flags right.
!   Phase 1 recomputes every leaf; phase 2 only the leaves that need it,
!   so the check also covers leaves it did not visit (their windowed
!   bounds must hold while their particles drift and their h grows)
!+
!-----------------------------------------------------------------------
subroutine test_revise_tree(ntests,npass)
 use dim,         only:periodic
 use io,          only:id,master
 use neighkdtree, only:leaf_is_active,ncells,node
 use part,        only:npart,npartoftype,xyzh,vxyzu,divcurlv,iphase,iactive,igas,hfact,&
                        massoftype,ibin,twas,kill_particle,isdead_or_accreted,gravity
 use kernel,      only:hfact_default
 use kdtree,      only:maketree,revise_tree,empty_tree
 use timestep,    only:dtmax
 use unifdis,     only:set_unifdis
 use random,      only:ran2
 use testutils,   only:checkval,update_test_scores
 use mpidomain,   only:i_belong
 integer, intent(inout) :: ntests,npass
 integer :: i,iseed,nfailed(6)
 real    :: psep,t0,t1,t2
 logical :: revised

 if (gravity) return   ! revise_tree is not used with self-gravity
 if (id==master) write(*,"(/,a)") '--> testing revise_tree (updating the tree in place)'
 !
 ! a random particle distribution with random velocities and div v, all in the timestep
 ! bin of dtmax, halfway through their step at t=dtmax/2
 !
 psep  = 1./50.
 hfact = hfact_default
 npart = 0
 call set_unifdis('random',id,master,-0.5,0.5,-0.5,0.5,-0.5,0.5,&
                  psep,hfact,npart,xyzh,periodic,mask=i_belong)
 npartoftype(:)    = 0
 npartoftype(igas) = npart
 massoftype(igas)  = 1./npart
 dtmax = 1.
 iseed = -5
 do i=1,npart
    vxyzu(1:3,i)  = (/ran2(iseed),ran2(iseed),ran2(iseed)/) - 0.5
    divcurlv(1,i) = real(2.*ran2(iseed) - 1.,kind=kind(divcurlv))
    ibin(i) = 0
    twas(i) = 0.5*dtmax
 enddo
 !
 ! phase 1: particles moved at random, h changed, a tenth active, some killed
 !
 call set_active(1.1,iseed)
 call empty_tree(node)
 call maketree(node,xyzh,npart,leaf_is_active,ncells,apr_tree=.false.)
 call set_active(0.1,iseed)
 do i=1,npart
    xyzh(1:3,i) = xyzh(1:3,i) + 0.5*psep*((/ran2(iseed),ran2(iseed),ran2(iseed)/) - 0.5)
    xyzh(4,i)   = xyzh(4,i)*(0.9 + 0.2*ran2(iseed))
 enddo
 do i=1,npart,97
    call kill_particle(i,npartoftype)
 enddo
 call revise_tree(node,xyzh,npart,leaf_is_active,ncells,0.1,.false.,revised)
 call checkval(revised,.true.,nfailed(1),'phase 1: tree revised')
 call check_revised_tree('phase 1',nfailed(2:6))
 call update_test_scores(ntests,nfailed,npass)
 !
 ! phase 2: the first revision after a build visits every leaf and sets its window
 ! (to t=dtmax, when every particle is next due); later ones visit only the leaves
 ! with active or newly dead particles, or active when last visited.  Between them
 ! inactive particles drift in straight lines and their h grows with div v, while
 ! active ones (kicked) move anywhere.  Few particles are active, so that few leaves
 ! are visited (with more than 1/8 of them to visit, every leaf is recomputed)
 !
 call set_active(1.1,iseed)
 call empty_tree(node)
 call maketree(node,xyzh,npart,leaf_is_active,ncells,apr_tree=.false.)
 t0 = 0.25*dtmax
 call set_active(0.002,iseed)
 call revise_tree(node,xyzh,npart,leaf_is_active,ncells,t0,.true.,revised)
 call checkval(revised,.true.,nfailed(1),'phase 2, first revision: tree revised')
 call check_revised_tree('phase 2, first revision',nfailed(2:6))
 call update_test_scores(ntests,nfailed,npass)

 t1 = 0.6*dtmax
 call move_particles(t1-t0,psep,iseed)
 call set_active(0.002,iseed)
 do i=1,npart,101
    if (.not.iactive(iphase(i)) .and. .not.isdead_or_accreted(xyzh(4,i))) call kill_particle(i,npartoftype)
 enddo
 call revise_tree(node,xyzh,npart,leaf_is_active,ncells,t1,.true.,revised)
 call checkval(revised,.true.,nfailed(1),'phase 2, second revision: tree revised')
 call check_revised_tree('phase 2, second revision',nfailed(2:6))
 call update_test_scores(ntests,nfailed,npass)

 t2 = 0.9*dtmax
 call move_particles(t2-t1,psep,iseed)
 call set_active(0.001,iseed)
 call revise_tree(node,xyzh,npart,leaf_is_active,ncells,t2,.true.,revised)
 call checkval(revised,.true.,nfailed(1),'phase 2, third revision: tree revised')
 call check_revised_tree('phase 2, third revision',nfailed(2:6))
 call update_test_scores(ntests,nfailed,npass)

contains

subroutine set_active(frac,iseed)
 use part, only:isetphase
 real,    intent(in)    :: frac
 integer, intent(inout) :: iseed
 integer :: j

 do j=1,npart
    iphase(j) = isetphase(igas,iactive=(ran2(iseed) < frac))
 enddo

end subroutine set_active

subroutine move_particles(dt,psep,iseed)
 real,    intent(in)    :: dt,psep
 integer, intent(inout) :: iseed
 integer :: j

 do j=1,npart
    if (isdead_or_accreted(xyzh(4,j))) cycle
    if (iactive(iphase(j))) then
       xyzh(1:3,j) = xyzh(1:3,j) + 0.5*psep*((/ran2(iseed),ran2(iseed),ran2(iseed)/) - 0.5)
    else
       xyzh(1:3,j) = xyzh(1:3,j) + dt*vxyzu(1:3,j)
       xyzh(4,j)   = xyzh(4,j)*(1. + dt*divcurlv(1,j)/3.)
    endif
 enddo

end subroutine move_particles

end subroutine test_revise_tree

!-----------------------------------------------------------------------
!+
!   check a revised tree, walking it from the root: every live particle
!   in exactly one leaf, inside the leaf's sphere with h no more than its
!   hmax, no dead particle left in a leaf, the activity sign of each entry
!   and each leaf's flag right, and every child's sphere and hmax within
!   its parent's
!+
!-----------------------------------------------------------------------
subroutine check_revised_tree(label,nfailed)
 use part,        only:npart,xyzh,iphase,iactive,isdead_or_accreted
 use kdtree,      only:inodeparts,inoderange
 use neighkdtree, only:node,leaf_is_active,ncells
 use testutils,   only:checkval
 character(len=*), intent(in)  :: label
 integer,          intent(out) :: nfailed(5)
 integer, allocatable :: stack(:),nfound(:)
 integer :: nst,n,c,ip,i,k,nout,ndead,nflag,nnest,nlost
 logical :: anyactive
 real, parameter :: tol = 1.e-12

 allocate(stack(int(ncells)+1),nfound(npart))
 nfound = 0
 nout = 0; ndead = 0; nflag = 0; nnest = 0
 nst = 1
 stack(1) = 1
 do while (nst > 0)
    n = stack(nst)
    nst = nst - 1
    if (leaf_is_active(n) /= 0) then
       anyactive = .false.
       do ip=inoderange(1,n),inoderange(2,n)
          i = abs(inodeparts(ip))
          if (isdead_or_accreted(xyzh(4,i))) then
             ndead = ndead + 1
             cycle
          endif
          nfound(i) = nfound(i) + 1
          if (norm2(xyzh(1:3,i) - node(n)%xcen) > node(n)%size*(1. + tol)) nout = nout + 1
          if (xyzh(4,i) > node(n)%hmax*(1. + tol)) nout = nout + 1
          if ((inodeparts(ip) > 0) .neqv. iactive(iphase(i))) nflag = nflag + 1
          if (iactive(iphase(i))) anyactive = .true.
       enddo
       if (anyactive .neqv. (leaf_is_active(n) > 0)) nflag = nflag + 1
    else
       do k=1,2
          c = node(n)%leftchild
          if (k==2) c = node(n)%rightchild
          if (norm2(node(c)%xcen - node(n)%xcen) + node(c)%size > node(n)%size*(1. + tol)) nnest = nnest + 1
          if (node(c)%hmax > node(n)%hmax) nnest = nnest + 1
          nst = nst + 1
          stack(nst) = c
       enddo
    endif
 enddo
 nlost = 0
 do i=1,npart
    if (.not.isdead_or_accreted(xyzh(4,i)) .and. nfound(i) /= 1) nlost = nlost + 1
 enddo
 call checkval(nout,0,0,nfailed(1),label//': particles outside their leaf (or h > hmax)')
 call checkval(ndead,0,0,nfailed(2),label//': dead particles left in leaves')
 call checkval(nlost,0,0,nfailed(3),label//': live particles not in exactly one leaf')
 call checkval(nflag,0,0,nfailed(4),label//': wrong activity signs or leaf flags')
 call checkval(nnest,0,0,nfailed(5),label//': children not within their parents')
 deallocate(stack,nfound)

end subroutine check_revised_tree

end module testkdtree
