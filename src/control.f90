!-----------------------------------------------------------------------
!
! MODULE: control_module
!> @brief
!> This module contains the variables that control SNAP's solver
!> routines. This includes the time-dependent variables.
!
!-----------------------------------------------------------------------

MODULE control_module

  USE global_module, ONLY: i_knd, r_knd, l_knd, zero, one

  IMPLICIT NONE

  PUBLIC

  SAVE
!_______________________________________________________________________
!
! Module Input Variables
!
! epsi     - convergence criterion
! iitm     - max inner iterations
! oitm     - max outer iterations
! timedep  - 0/1=no/yes perform a time-dependent calculation
! tf       - final time
! nsteps   - number of time steps to cover the ts -> tf range
!
! it_det   - 0/1=no/yes full iteration details
! fluxp    - 0/1/2=print none/scalar flux/all flux moments to file
!
! fixup    - 0/1=no/yes perform flux fixup
!_______________________________________________________________________

  INTEGER(i_knd) :: iitm=5, oitm=100, timedep=0, nsteps=1, it_det=0,   &
    fluxp=0, fixup=0

  REAL(r_knd) :: epsi=1.0E-4_r_knd, tf=zero
!_______________________________________________________________________
!
! Run-time variables
!
! dt       - time-step size
!
! tolr      - parameter, small number used for determining how to
!             compute flux error
! dfmxi(ng) - max error of inner iteration
! dfmxo     - max error of outer iteration
!
! inrdone(ng)  - logical for inners being complete
! otrdone      - logical for outers being complete
!_______________________________________________________________________

  LOGICAL(l_knd) :: otrdone

  LOGICAL(l_knd), ALLOCATABLE, DIMENSION(:) :: inrdone

  REAL(r_knd) :: dt, dfmxo

  REAL(r_knd), PARAMETER :: tolr=1.0E-12_r_knd

  REAL(r_knd), ALLOCATABLE, DIMENSION(:) :: dfmxi


  CONTAINS


  SUBROUTINE control_alloc ( ng, ierr )

!-----------------------------------------------------------------------
!
! Allocate control module variables.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: ng

    INTEGER(i_knd), INTENT(OUT) :: ierr
!_______________________________________________________________________

    ierr = 0
    ALLOCATE( dfmxi(ng), inrdone(ng), STAT=ierr )
    IF ( ierr /= 0 ) RETURN

    dfmxi = -one
    inrdone = .FALSE.
    dfmxo = -one
    otrdone = .FALSE.
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE control_alloc


  SUBROUTINE control_dealloc

!-----------------------------------------------------------------------
!
! Deallocate control module arrays.
!
!-----------------------------------------------------------------------
!_______________________________________________________________________

    DEALLOCATE( dfmxi, inrdone )
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE control_dealloc


END MODULE control_module
