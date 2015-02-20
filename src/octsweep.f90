!-----------------------------------------------------------------------
!
! MODULE: octsweep_module
!> @brief
!> This module controls the setup and calls for sweeping a single octant
!> pair. It calls for the actual sweep logic depending on the spatial
!> dimensionality of the problem.
!
!-----------------------------------------------------------------------

MODULE octsweep_module

  USE global_module, ONLY: i_knd, r_knd, zero

  USE geom_module, ONLY: nc, ndimen, dinv, nx, ny, nz

  USE sn_module, ONLY: ec, nang, wmu, weta, wxi, cmom

  USE data_module, ONLY: vdelt, ng

  USE control_module, ONLY: timedep, swp_typ

  USE solvar_module, ONLY: psii, qtot, ptr_in, ptr_out, psij, psik,    &
    flux0, fluxm, jb_in, jb_out, kb_in, kb_out, flkx, flky, flkz, t_xs

  USE thrd_comm_module, ONLY: no_op_lock_control

  USE dim1_sweep_module, ONLY: dim1_sweep

  USE dim3_sweep_module, ONLY: dim3_sweep

  USE mkba_sweep_module, ONLY: mkba_sweep

  IMPLICIT NONE

  PUBLIC :: octsweep


  CONTAINS


  SUBROUTINE octsweep ( g, jd, kd, iop, t, nthrd_used, reqs, szreq,    &
    fmin, fmax )

!-----------------------------------------------------------------------
!
! Call for the appropriate sweeper depending on the dimensionality. Let
! the actual sweep routine know the octant info to get the order right.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: g, jd, kd, iop, t, nthrd_used, szreq

    INTEGER(i_knd), DIMENSION(szreq), INTENT(INOUT) :: reqs

    REAL(r_knd), DIMENSION(ng), INTENT(INOUT) :: fmin, fmax
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: id, oct, ich, d1, d2, d3, d4, i1, i2
!_______________________________________________________________________
!
!   If g=0, this thread has no work to do, but it cannot exit sweeps
!   immediately, because it must properly set it's own lock and unset
!   the subsequent thread's lock. Call for both recv ops and send ops.
!_______________________________________________________________________

    IF ( g == 0 ) THEN
      CALL no_op_lock_control ( t, nthrd_used )
      CALL no_op_lock_control ( t, nthrd_used )
      RETURN
    END IF
!_______________________________________________________________________
!
!   Determine octant and chunk index.
!_______________________________________________________________________

    id = 1 + (iop-1)/nc
    oct = id + 2*(jd - 1) + 4*(kd - 1)

    IF ( id == 1 ) THEN
      ich = nc - iop + 1
    ELSE
      ich = iop - nc
    END IF
!_______________________________________________________________________
!
!   Send ptr_in and ptr_out dimensions dependent on timedep because they
!   are not allocated if static problem
!_______________________________________________________________________

    d1 = 0; d2 = 0; d3 = 0; d4 = 0
    i1 = 0; i2 = 0
    IF ( timedep == 1 ) THEN
      d1 = nang; d2 = nx; d3 = ny; d4 = nz
      i1 = oct; i2 = g
    END IF
!_______________________________________________________________________
!
!   Call for the actual sweeper. Ensure proper size/bounds of time-dep
!   arrays is given to avoid errors.
!_______________________________________________________________________

    IF ( ndimen == 1 ) THEN

      CALL dim1_sweep ( id, d1, d2, d3, d4, oct, g, psii(:,1,1,g),     &
        qtot(:,:,1,1,ich,g), ec(:,:,oct), vdelt(g),                    &
        ptr_in(:,:,:,:,i1,i2), ptr_out(:,:,:,:,i1,i2),                 &
        dinv(:,:,1,1,ich,g), flux0(:,1,1,g), fluxm(:,:,1,1,g), wmu,    &
        flkx(:,1,1,g), t_xs(:,1,1,g), fmin(g), fmax(g) )

    ELSE IF ( swp_typ == 0 ) THEN

      CALL dim3_sweep ( ich, id, d1, d2, d3, d4, jd, kd, oct, g, iop,  &
        t, nthrd_used, reqs, szreq, psii(:,:,:,g), psij(:,:,:,g),      &
        psik(:,:,:,g), qtot(:,:,:,:,ich,g), ec(:,:,oct), vdelt(g),     &
        ptr_in(:,:,:,:,i1,i2), ptr_out(:,:,:,:,i1,i2),                 &
        dinv(:,:,:,:,ich,g), flux0(:,:,:,g), fluxm(:,:,:,:,g),         &
        jb_in(:,:,:,g), jb_out(:,:,:,g), kb_in(:,:,:,g),               &
        kb_out(:,:,:,g), wmu, weta, wxi, flkx(:,:,:,g), flky(:,:,:,g), &
        flkz(:,:,:,g), t_xs(:,:,:,g), fmin(g), fmax(g) )

    ELSE

      CALL mkba_sweep ( ich, id, d1, d2, d3, d4, jd, kd, oct, g, iop,  &
        t, nthrd_used, reqs, szreq, psii(:,:,:,g), psij(:,:,:,g),      &
        psik(:,:,:,g), qtot(:,:,:,:,ich,g), ec(:,:,oct), vdelt(g),     &
        ptr_in(:,:,:,:,i1,i2), ptr_out(:,:,:,:,i1,i2),                 &
        dinv(:,:,:,:,ich,g), flux0(:,:,:,g), fluxm(:,:,:,:,g),         &
        jb_in(:,:,:,g), jb_out(:,:,:,g), kb_in(:,:,:,g),               &
        kb_out(:,:,:,g), wmu, weta, wxi, flkx(:,:,:,g), flky(:,:,:,g), &
        flkz(:,:,:,g), t_xs(:,:,:,g), fmin(g), fmax(g) )

    END IF
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE octsweep


END MODULE octsweep_module
