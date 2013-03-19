!-----------------------------------------------------------------------
!
! MODULE: dealloc_module
!> @brief
!> This module controls the deallocation process of run-time arrays.
!
!-----------------------------------------------------------------------

MODULE dealloc_module

  USE global_module, ONLY: i_knd

  USE sn_module, ONLY: sn_deallocate

  USE data_module, ONLY: data_deallocate

  USE mms_module, ONLY: mms_deallocate

  USE geom_module, ONLY: geom_dealloc

  USE solvar_module, ONLY: solvar_dealloc

  USE control_module, ONLY: control_dealloc

  IMPLICIT NONE

  PUBLIC


  CONTAINS


  SUBROUTINE dealloc_input ( flg )

!-----------------------------------------------------------------------
!
! Call for the data deallocation from individual deallocation
! subroutines. Covers the allocations from input.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: flg
!_______________________________________________________________________

    CALL sn_deallocate

    IF ( flg > 1 ) CALL data_deallocate

    IF ( flg > 2 ) CALL mms_deallocate
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE dealloc_input


  SUBROUTINE dealloc_solve ( flg )

!-----------------------------------------------------------------------
!
! Call for the data deallocation from individual deallocation
! subroutines. Covers the allocations from input.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: flg
!_______________________________________________________________________

    CALL geom_dealloc

    IF ( flg > 1 ) CALL solvar_dealloc

    IF ( flg > 2 ) CALL control_dealloc
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE dealloc_solve


END MODULE dealloc_module
