SUBROUTINE translv

!-----------------------------------------------------------------------
!
! Solution driver. Contains the time and outer loops. Calls for outer
! iteration work. Checks convergence and handles eventual output.
!
!-----------------------------------------------------------------------

  USE global_module, ONLY: i_knd, r_knd, ounit, zero, half, one, two

  USE plib_module, ONLY: glmax, comm_snap, iproc, root, thread_num,    &
    ichunk, do_nested

  USE geom_module, ONLY: geom_alloc, geom_dealloc, dinv, param_calc,   &
    nx, ny_gl, nz_gl, diag_setup

  USE sn_module, ONLY: nang, noct, mu, eta, xi

  USE data_module, ONLY: ng, v, vdelt, mat, sigt, siga, slgg, src_opt, &
    qim

  USE control_module, ONLY: nsteps, timedep, dt, oitm, otrdone,        &
    control_alloc, control_dealloc, dfmxo, it_det

  USE utils_module, ONLY: print_error, stop_run

  USE solvar_module, ONLY: solvar_alloc, ptr_in, ptr_out, t_xs, a_xs,  &
    s_xs, flux, fluxm

  USE expxs_module, ONLY: expxs_reg, expxs_slgg

  USE outer_module, ONLY: outer

  USE time_module, ONLY: tslv, wtime, tgrind, tparam

  IMPLICIT NONE
!_______________________________________________________________________
!
! Local variables
!_______________________________________________________________________

  CHARACTER(LEN=1) :: star='*'

  CHARACTER(LEN=64) :: error

  INTEGER(i_knd) :: cy, otno, ierr, g, i, tot_iits, cy_iits, out_iits

  REAL(r_knd) :: sf, time, t1, t2, t3, t4, t5, t6, t7, tmp

  REAL(r_knd), DIMENSION(:,:,:,:,:,:), POINTER :: ptr_tmp
!_______________________________________________________________________
!
! Call for data allocations. Some allocations depend on the problem
! type being requested.
!_______________________________________________________________________

  CALL wtime ( t1 )

  ierr = 0
  error = ' '

  CALL geom_alloc ( nang, ng, ierr )
  CALL glmax ( ierr, comm_snap )
  IF ( ierr /= 0 ) THEN
    error = '***ERROR: GEOM_ALLOC: Allocation error of sweep parameters'
    CALL print_error ( ounit, error )
    CALL stop_run ( 3, 0, 0 )
  END IF

  CALL solvar_alloc ( ierr )
  CALL glmax ( ierr, comm_snap )
  IF ( ierr /= 0 ) THEN
    error = '***ERROR: SOLVAR_ALLOC: Allocation error of solution ' // &
            'arrays'
    CALL print_error ( ounit, error )
    CALL stop_run ( 3, 1, 0 )
  END IF

  CALL control_alloc ( ng, ierr )
  CALL glmax ( ierr, comm_snap )
  IF ( ierr /= 0 ) THEN
    error = '***ERROR: CONTROL_ALLOC: Allocation error of control ' // &
      'arrays'
    CALL print_error ( ounit, error )
    CALL stop_run ( 3, 2, 0 )
  END IF
!_______________________________________________________________________
!
! Call for setup of the mini-KBA diagonal map
!_______________________________________________________________________

  CALL diag_setup ( do_nested, ichunk, ierr )
  CALL glmax ( ierr, comm_snap )
  IF ( ierr /= 0 ) THEN
    error = '***ERROR: DIAG_SETUP: Allocation error of diag type array'
    CALL print_error ( ounit, error )
    CALL stop_run ( 3, 3, 0 )
  END IF

  CALL wtime ( t2 )
  tparam = tparam + t2 - t1
!_______________________________________________________________________
!
! The time loop solves the problem for nsteps. If static, there is
! only one step, and it does not have any time-absorption or -source
! terms. Set the pointers to angular flux arrays. Set time to one for
! static for proper multiplication in octsweep.
!_______________________________________________________________________

  IF ( iproc == root ) WRITE( ounit, 201) ( star, i = 1, 80 )

  tot_iits = 0

  time_loop: DO cy = 1, nsteps

    CALL wtime ( t3 )

    vdelt = zero
    time = one
    IF ( timedep == 1 ) THEN
      IF ( iproc == root ) WRITE( ounit, 202 ) ( star, i = 1, 30 ), cy
      vdelt = two / ( dt * v )
      time = dt * ( REAL( cy, r_knd ) - half )
    END IF

    IF ( cy > 1 ) THEN
      ptr_tmp => ptr_out
      ptr_out => ptr_in
      ptr_in  => ptr_tmp
    END IF
!_______________________________________________________________________
!
!   Scale the manufactured source for time
!_______________________________________________________________________

    IF ( src_opt == 3 ) THEN
      IF ( cy == 1 ) THEN
        qim = time*qim
      ELSE
        sf = REAL( 2*cy - 1, r_knd ) / REAL( 2*cy-3, r_knd )
        qim = qim*sf
      END IF
    END IF
!_______________________________________________________________________
!
!   Zero out flux arrays. Use threads when available.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(DYNAMIC,1) DEFAULT(SHARED) PRIVATE(g)
    DO g = 1, ng
      flux(:,:,:,g)    = zero
      fluxm(:,:,:,:,g) = zero
    END DO
  !$OMP END PARALLEL DO
!_______________________________________________________________________
!
!   Using Jacobi iterations in energy, and the work in the outer loop
!   will be parallelized with threads.
!_______________________________________________________________________

    otrdone = .FALSE.

    cy_iits = 0

    IF ( iproc==root .AND. it_det==0 ) WRITE( ounit, 203 )

    CALL wtime ( t4 )
    tparam = tparam + t4 - t3

    outer_loop: DO otno = 1, oitm

      CALL wtime ( t5 )

      IF ( iproc==root .AND. it_det==1 ) THEN
        WRITE( ounit, 204 ) ( star, i = 1, 20 ), otno
      END IF
!_______________________________________________________________________
!
!   Prepare some cross sections: total, in-group scattering, absorption.
!   Keep in the time loop for better consistency with PARTISN. Set up
!   geometric sweep parameters. Parallelize group loop with threads.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(DYNAMIC,1) DEFAULT(SHARED) PRIVATE(g)
      DO g = 1, ng
        CALL expxs_reg ( sigt(:,g), mat, t_xs(:,:,:,g) )
        CALL expxs_reg ( siga(:,g), mat, a_xs(:,:,:,g) )
        CALL expxs_slgg ( slgg(:,:,g,g), mat, s_xs(:,:,:,:,g) )
        CALL param_calc ( ichunk, nang, mu, eta, xi, t_xs(:,:,:,g),    &
          vdelt(g), dinv(:,:,:,:,g) )
      END DO
  !$OMP END PARALLEL DO
!_______________________________________________________________________
!
!     Perform an outer iteration. Add up inners. Check convergence.
!_______________________________________________________________________

      CALL wtime ( t6 )
      tparam = tparam + t6 - t5

      CALL outer ( out_iits )

      cy_iits = cy_iits + out_iits

      IF ( iproc == root ) WRITE( ounit, 205 ) otno, dfmxo, out_iits

      IF ( otrdone ) EXIT outer_loop

    END DO outer_loop
!_______________________________________________________________________
!
!   Print the time cycle details. Add time cycle iterations.
!_______________________________________________________________________

    IF ( timedep == 1 ) THEN
      IF ( otrdone ) THEN
        IF ( iproc == root ) WRITE( ounit, 206 ) cy, time, otno, cy_iits
      ELSE
        IF ( iproc == root ) WRITE( ounit, 207 ) cy, time, otno, cy_iits
      END IF
    ELSE
      IF ( otrdone ) THEN
        IF ( iproc == root ) WRITE( ounit, 208 ) otno, cy_iits
      ELSE
        IF ( iproc == root ) WRITE( ounit, 209 ) otno, cy_iits
      END IF
    END IF

    tot_iits = tot_iits + cy_iits

    IF ( .NOT. otrdone ) EXIT time_loop

  END DO time_loop

  IF ( timedep==1 .AND. iproc == root ) THEN
    WRITE( ounit, 210 ) ( star, i = 1, 30 ), tot_iits
  END IF
  IF ( iproc == root ) WRITE( ounit, 211 ) ( star, i = 1, 80 )

  CALL wtime ( t7 )
  tslv = t7 - t1
  tmp = REAL( nx, r_knd ) * REAL( ny_gl, r_knd ) * REAL( nz_gl, r_knd )&
        * REAL( nang, r_knd ) * REAL( noct, r_knd )                    &
        * REAL( tot_iits, r_knd )
  tgrind = tslv*1.0E9_r_knd / tmp
!_______________________________________________________________________

  201 FORMAT( 10X, 'Iteration Monitor', /, 80A )
  202 FORMAT( /, 1X, 30A, /, 2X, 'Time Cycle ', I3 )
  203 FORMAT( 2X, 'Outer' )
  204 FORMAT( 1X, 20A, /, 2X, 'Outer ', I3 )
  205 FORMAT( 2X, I3, 4X, 'Dfmxo=', ES11.4, 4X, 'No. Inners=', I5 )
  206 FORMAT( /, 2X, 'Cycle=', I4, 4X, 'Time=', ES11.4, 4X, 'No. ',    &
              'Outers=', I4, 4X, 'No. Inners=', I5 )
  207 FORMAT( /, 2X, '***UNCONVERGED*** Stopping Iterations!!', /, 2X, &
             'Cycle=', I4, 4X, 'Time=', ES11.4, 4X, 'No. Outers=', I4, &
             4X, 'No. Inners=', I5, / )
  208 FORMAT( /, 2X, 'No. Outers=', I4, 4X, 'No. Inners=', I5 )
  209 FORMAT( /, 2X, '***UNCONVERGED*** Stopping Iterations!!', /, 2X, &
              'No. Outers=', I4, 4X, 'No. Inners=', I5, / )
  210 FORMAT( /, 1X, 30A, /, 2X, 'Total inners for all time steps, '   &
              'outers = ', I6 )
  211 FORMAT( /, 80A, / )
!_______________________________________________________________________
!_______________________________________________________________________

END SUBROUTINE translv
