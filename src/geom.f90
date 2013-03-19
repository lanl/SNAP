!-----------------------------------------------------------------------
!
! MODULE: geom_module
!> @brief
!> This module contains the variables that relate to the geometry of the
!> problem and the subroutines necessary to allocate and deallocate
!> geometry related data as necessary.
!
!-----------------------------------------------------------------------

MODULE geom_module

  USE global_module, ONLY: i_knd, r_knd, zero, one, two

  IMPLICIT NONE

  PUBLIC

  SAVE
!_______________________________________________________________________
!
! Module Input Variables
!
! ndimen   - number of spatial dimensions 1/2/3
! nx       - number of x-dir spatial cells (global)
! ny       - number of y-dir spatial cells (global on input, reset to
!            per PE in setup)
! nz       - number of z-dir spatial cells (global on input, reset to
!            per PE in setup)
! lx       - total length of x domain
! ly       - total length of y domain
! lz       - total length of z domain
!_______________________________________________________________________

  INTEGER(i_knd) :: ndimen=1, nx=4, ny=1, nz=1

  REAL(r_knd) :: lx=one, ly=one, lz=one
!_______________________________________________________________________
!
! Run-time variables
!
! ny_gl    - global number of y-dir spatial cells
! nz_gl    - global number of z-dir spatial cells
! jlb      - global index of local lower y bound
! jub      - global index of local upper y bound
! klb      - global index of local lower z bound
! kub      - global index of local upper z bound
!
! dx       - x width of spatial cell
! dy       - y width of spatial cell
! dz       - z width of spatial cell
!
! nc       - number of i-chunks, nx/ichunk
!
! hi       - Spatial DD x-coefficient
! hj(nang) - Spatial DD y-coefficient
! hk(nang) - Spatial DD z-coefficient
!
! dinv(nang,nx,ny,nz,ng) - Sweep denominator, pre-computed/inverted
!_______________________________________________________________________

  INTEGER(i_knd) :: ny_gl, nz_gl, jlb, jub, klb, kub, nc

  REAL(r_knd) :: dx, dy, dz, hi

  REAL(r_knd), ALLOCATABLE, DIMENSION(:) :: hj, hk

  REAL(r_knd), ALLOCATABLE, DIMENSION(:,:,:,:,:) :: dinv


  CONTAINS


  SUBROUTINE geom_alloc ( nang, ng, ierr )

!-----------------------------------------------------------------------
!
! Allocate the geometry-related solution arrays. Called 
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: nang, ng

    INTEGER(i_knd), INTENT(OUT) :: ierr
!_______________________________________________________________________

    ierr = 0

    ALLOCATE( hj(nang), hk(nang), dinv(nang,nx,ny,nz,ng), STAT=ierr )
    IF ( ierr /= 0 ) RETURN

    hi = zero
    hj = zero
    hk = zero
    dinv = zero
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE geom_alloc


  SUBROUTINE geom_dealloc

!-----------------------------------------------------------------------
!
! Deallocate the geometry-related solution arrays.
!
!-----------------------------------------------------------------------
!_______________________________________________________________________

    DEALLOCATE( hj, hk, dinv )
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE geom_dealloc


  SUBROUTINE param_calc ( ichunk, nang, mu, eta, xi, cs, vd, d )

!-----------------------------------------------------------------------
!
! Calculate the DD spatial coefficients hi, hj, hk for all angles at
! the start of each time step. Compute the pre-computed/inverted dinv.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: ichunk, nang

    REAL(r_knd), INTENT(IN) :: vd

    REAL(r_knd), DIMENSION(nang), INTENT(IN) :: mu, eta, xi

    REAL(r_knd), DIMENSION(nx,ny,nz), INTENT(IN) :: cs

    REAL(r_knd), DIMENSION(nang,nx,ny,nz), INTENT(OUT) :: d
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: i, j, k, m
!_______________________________________________________________________
!
!   Set the number of i-chunks
!_______________________________________________________________________

    nc = nx/ichunk
!_______________________________________________________________________
!
!   Set the DD coefficients
!_______________________________________________________________________

    hi = two/dx
    IF ( ndimen > 1 ) THEN
      hj = (two/dy)*eta
      IF ( ndimen > 2 ) hk = (two/dz)*xi
    END IF
!_______________________________________________________________________
!
!   Compute the inverted denominator, saved for sweep
!_______________________________________________________________________

    DO k = 1, nz
     DO j = 1, ny
      DO i = 1, nx
       DO m = 1, nang
        d(m,i,j,k) = one / (cs(i,j,k) + vd + mu(m)*hi + hj(m) + hk(m))
       END DO
      END DO
     END DO
    END DO
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE param_calc


END MODULE geom_module
