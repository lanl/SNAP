MODULE geom_module

!-----------------------------------------------------------------------
!
! This module contains the variables that relate to the geometry of the
! problem and the subroutines necessary to allocate and deallocate
! geometry related data as necessary.
!
!-----------------------------------------------------------------------

  USE global_module, ONLY: i_knd, r_knd, zero, one, two, l_knd

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
!
! ndiag    - number of diagonals of mini-KBA sweeps in nested threading
!_______________________________________________________________________

  INTEGER(i_knd) :: ny_gl, nz_gl, jlb, jub, klb, kub, nc, ndiag

  REAL(r_knd) :: dx, dy, dz, hi

  REAL(r_knd), ALLOCATABLE, DIMENSION(:) :: hj, hk

  REAL(r_knd), ALLOCATABLE, DIMENSION(:,:,:,:,:) :: dinv
!_______________________________________________________________________
!
! Derived data types for mini-KBA diagonals
!
! cell_id_type      - type for holding ijk indices of a cell on a given
!                     diagonal
! ic, j, k          - ijk integer indices of cell_id_type
! diag_type         - type for holding diagonal information
! len               - number of cells on a diagonal line/plane
! cell_id(len)      - array of cell ijk indices
! diag(ndiag)       - array of diagonal lines/planes information
!_______________________________________________________________________

  TYPE cell_id_type
    INTEGER(i_knd) :: ic, j, k
  END TYPE cell_id_type

  TYPE diag_type
    INTEGER(i_knd) :: len
    TYPE(cell_id_type), ALLOCATABLE, DIMENSION(:) :: cell_id
  END TYPE diag_type

  TYPE(diag_type), ALLOCATABLE, DIMENSION(:) :: diag


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
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: i
!_______________________________________________________________________
!
!   Deallocate the sweep parameters
!_______________________________________________________________________

    DEALLOCATE( hj, hk, dinv )
!_______________________________________________________________________
!
!   Deallocate the diagonal related arrays
!_______________________________________________________________________

    DO i = 1, ndiag
      DEALLOCATE( diag(i)%cell_id )
    END DO

    DEALLOCATE( diag )
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


  SUBROUTINE diag_setup ( do_nested, ichunk, ierr )

!-----------------------------------------------------------------------
!
! Allocate and set up the values of the derived data type 'diag' which
! stores number of diagonals, each one's number of cells/length, and the
! ijk indices of the cells on the diagonal.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: ichunk

    INTEGER(i_knd), INTENT(OUT) :: ierr

    LOGICAL(l_knd), INTENT(IN) :: do_nested
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: i, j, k, nn, ing

    INTEGER(i_knd), ALLOCATABLE, DIMENSION(:) :: indx
!_______________________________________________________________________
!
!   Set up the diagonal indices according to do_nested. If 1, use
!   mini-KBA sweeps and thus allocate many diagonals.
!_______________________________________________________________________

    ierr = 0

    IF ( do_nested ) THEN

      ndiag = ichunk + ny + nz - 2

      ALLOCATE( diag(ndiag), indx(ndiag), STAT=ierr )
      IF ( ierr /= 0 ) RETURN

      diag%len = 0
      indx = 0
!_______________________________________________________________________
!
!     Cells of same diagonal all have same value according to i+j+k-2
!     formula. Use that to compute len for each diagonal. Use ichunk.
!_______________________________________________________________________

      DO k = 1, nz
      DO j = 1, ny
      DO i = 1, ichunk
        nn = i + j + k - 2
        diag(nn)%len = diag(nn)%len + 1
      END DO
      END DO
      END DO
!_______________________________________________________________________
!
!     Next allocate cell_id array within diag type according to len
!_______________________________________________________________________

      DO nn = 1, ndiag
        ing = diag(nn)%len
        ALLOCATE( diag(nn)%cell_id(ing), STAT=ierr )
        IF ( ierr /= 0 ) RETURN
      END DO
!_______________________________________________________________________
!
!     Lastly, set each cell's actual ijk indices in this diagonal map
!_______________________________________________________________________

      DO k = 1, nz
      DO j = 1, ny
      DO i = 1, ichunk
        nn = i + j + k - 2
        indx(nn) = indx(nn) + 1
        ing = indx(nn)
        diag(nn)%cell_id(ing)%ic = i
        diag(nn)%cell_id(ing)%j  = j
        diag(nn)%cell_id(ing)%k  = k
      END DO
      END DO
      END DO

      DEALLOCATE( indx )

    ELSE
!_______________________________________________________________________
!
!     Otherwise, use standard sweep map. No mini-KBA. One "diagonal",
!     which contains all the cells in typical i, then j, then k
!     lexographical order.
!_______________________________________________________________________

      ndiag = 1
      ALLOCATE( diag(1), STAT=ierr )
      IF ( ierr /= 0 ) RETURN
      ALLOCATE( diag(1)%cell_id(ichunk*ny*nz), STAT=ierr )
      IF ( ierr /= 0 ) RETURN

      diag(1)%len = ichunk*ny*nz
      ing = 0
      DO k = 1, nz
      DO j = 1, ny
      DO i = 1, ichunk
        ing = ing + 1
        diag(1)%cell_id(ing)%ic = i
        diag(1)%cell_id(ing)%j  = j
        diag(1)%cell_id(ing)%k  = k
      END DO
      END DO
      END DO

    END IF        
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE diag_setup


END MODULE geom_module
