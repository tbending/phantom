!--------------------------------------------------------------------------!
! The Phantom Smoothed Particle Hydrodynamics code, by Daniel Price et al. !
! Copyright (c) 2007-2026 The Authors (see AUTHORS)                        !
! See LICENCE file for usage and distribution conditions                   !
! http://phantomsph.github.io/                                             !
!--------------------------------------------------------------------------!
module gpu_domain
!
! The GPU path's domain decomposition under MPI (GPU_MPI_PLAN.md, Phase 1): at startup,
! after the dump has been read in slabs, Cornerstone's space-filling-curve assignment is
! computed on the host and every particle is marked with the rank that owns its piece of
! the curve (ibelong); phantom's own balancedomains then moves the particles.  The split
! is the one the device domain uses later, so its first sync moves nothing.
!
! :References: GPU_MPI_PLAN.md
!
! :Owner: Thomas Bending
!
! :Runtime parameters: None
!
! :Dependencies: boundary, dim, io, iso_c_binding, mpi, mpiutils, part
!
 implicit none
 public :: gpu_domain_assign,gpu_domain_check

 private

contains

!-----------------------------------------------------------------------
!+
!  set ibelong(1:npart) from Cornerstone's assignment of the global
!  particle set; prints the per-rank counts
!+
!-----------------------------------------------------------------------
subroutine gpu_domain_assign(npart)
#if defined(MPI) && defined(GPU)
 use mpi
 use iso_c_binding, only:c_int,c_double,c_long_long
 use io,            only:id,master,nprocs,iprint,fatal
 use part,          only:xyzh,ibelong
 use dim,           only:periodic
 use boundary,      only:xmin,xmax,ymin,ymax,zmin,zmax
 use mpiutils,      only:reduceall_mpi
 integer, intent(in) :: npart
 interface
  subroutine cosmo_dd_assign(n,x,y,z,periodic,box,bucket,fcomm,rankout,countout) bind(C)
   import c_int,c_double,c_long_long
   integer(c_int),       value :: n,periodic,bucket,fcomm
   real(c_double)              :: x(*),y(*),z(*),box(6)
   integer(c_int)              :: rankout(*)
   integer(c_long_long)        :: countout(*)
  end subroutine cosmo_dd_assign
 end interface
 real(c_double),       allocatable :: x(:),y(:),z(:)
 integer(c_int),       allocatable :: rankout(:)
 integer(c_long_long), allocatable :: countout(:)
 real(c_double)  :: box(6)
 integer(kind=8) :: ntot
 integer         :: i,bucket

 if (nprocs <= 1) then
    ibelong(1:npart) = id
    return
 endif
 ntot = reduceall_mpi('+',npart)
 ! the global tree's leaf size, as SPH-EXA's early sync chooses it
 bucket = int(max(64_8, ntot/(100_8*int(nprocs,kind=8))))
 allocate(x(max(npart,1)),y(max(npart,1)),z(max(npart,1)),rankout(max(npart,1)),countout(nprocs))
 do i=1,npart
    x(i) = xyzh(1,i)
    y(i) = xyzh(2,i)
    z(i) = xyzh(3,i)
 enddo
 box = (/xmin,xmax,ymin,ymax,zmin,zmax/)
 call cosmo_dd_assign(int(npart,c_int),x,y,z,merge(1_c_int,0_c_int,periodic),box,int(bucket,c_int), &
                      int(MPI_COMM_WORLD,c_int),rankout,countout)
 do i=1,npart
    if (rankout(i) < 0 .or. rankout(i) >= nprocs) call fatal('gpu_domain','bad rank from cosmo_dd_assign',ival=int(rankout(i)))
    ibelong(i) = int(rankout(i))
 enddo
 if (id==master) then
    write(iprint,"(a,i4,a,i11,a)") ' gpu_domain: Cornerstone split of ',nprocs,' ranks over ',ntot,' particles:'
    write(iprint,"(a,8i11)") '   per rank ',countout(1:min(nprocs,8))
    if (nprocs > 8) write(iprint,"(a,i11,a,i11)") '   min ',minval(countout),' max ',maxval(countout)
 endif
 deallocate(x,y,z,rankout,countout)
#else
 integer, intent(in) :: npart
 integer :: i
 i = npart   ! no MPI: nothing to assign
#endif

end subroutine gpu_domain_assign

!-----------------------------------------------------------------------
!+
!  Phase 2's first milestone: the first Cornerstone Domain::sync on the
!  device, on copies of the host arrays, right after balancedomains.
!  With the split's own curve, box and bucket size it must move nothing:
!  every rank keeps its particle set, and only halos appear.  Fatal if
!  not: the device decomposition would disagree with the host's.
!+
!-----------------------------------------------------------------------
subroutine gpu_domain_check(npart)
#if defined(MPI) && defined(GPU)
 use mpi
 use iso_c_binding, only:c_int,c_double
 use io,            only:id,master,nprocs,iprint,fatal
 use part,          only:xyzh,iorig
 use dim,           only:periodic
 use boundary,      only:xmin,xmax,ymin,ymax,zmin,zmax
 use mpiutils,      only:reduceall_mpi
 integer, intent(in) :: npart
 interface
  subroutine cosmo_dom_check(n,x,y,z,h,pid,periodic,box,bucket,bucketfocus,fcomm,nown,nhalo,idok,tsync) bind(C)
   import c_int,c_double
   integer(c_int), value :: n,periodic,bucket,bucketfocus,fcomm
   real(c_double)        :: x(*),y(*),z(*),h(*),pid(*),box(6)
   integer(c_int)        :: nown,nhalo,idok
   real(c_double)        :: tsync
  end subroutine cosmo_dom_check
 end interface
 real(c_double), allocatable :: x(:),y(:),z(:),h(:),pid(:)
 real(c_double)  :: box(6),tsync,tmax
 integer(c_int)  :: nown,nhalo,idok
 integer(kind=8) :: ntot,nhalotot
 integer         :: i,bucket,nbad

 if (nprocs <= 1) return
 ntot   = reduceall_mpi('+',npart)
 bucket = int(max(64_8, ntot/(100_8*int(nprocs,kind=8))))   ! as gpu_domain_assign
 allocate(x(max(npart,1)),y(max(npart,1)),z(max(npart,1)),h(max(npart,1)),pid(max(npart,1)))
 do i=1,npart
    x(i)   = xyzh(1,i)
    y(i)   = xyzh(2,i)
    z(i)   = xyzh(3,i)
    h(i)   = xyzh(4,i)
    pid(i) = real(iorig(i),kind=c_double)
 enddo
 box = (/xmin,xmax,ymin,ymax,zmin,zmax/)
 call cosmo_dom_check(int(npart,c_int),x,y,z,h,pid,merge(1_c_int,0_c_int,periodic),box, &
                      int(bucket,c_int),64_c_int,int(MPI_COMM_WORLD,c_int),nown,nhalo,idok,tsync)
 deallocate(x,y,z,h,pid)
 write(*,"(a,i4,a,i10,a,i10,a,f6.2,a,l1)") ' gpu_domain_check: rank ',id,' owned ',nown,' halos ',nhalo, &
       ' (',100.*real(nhalo)/real(max(int(nown),1)),'%), id set unchanged ',idok==1
 nbad     = int(reduceall_mpi('+',merge(0,1,nown == npart .and. idok == 1)))
 nhalotot = reduceall_mpi('+',int(nhalo,kind=8))
 tmax     = reduceall_mpi('max',tsync)
 if (id==master) write(iprint,"(a,i11,a,f6.2,a,f8.3,a)") ' gpu_domain_check: first device Domain::sync: halos ', &
       nhalotot,' (',100.*real(nhalotot)/real(ntot),'% of the particles), ',tmax,' s'
 if (nbad > 0) call fatal('gpu_domain','the first device Domain::sync moved particles or changed ownership', &
                          ival=nbad)
#else
 integer, intent(in) :: npart
 integer :: i
 i = npart   ! no MPI: nothing to check
#endif

end subroutine gpu_domain_check

end module gpu_domain
