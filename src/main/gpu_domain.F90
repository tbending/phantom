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
 public :: gpu_domain_assign

 private

contains

!-----------------------------------------------------------------------
!+
!  set ibelong(1:npart) from Cornerstone's assignment of the global
!  particle set; prints the per-rank counts
!+
!-----------------------------------------------------------------------
subroutine gpu_domain_assign(npart)
#ifdef MPI
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

end module gpu_domain
