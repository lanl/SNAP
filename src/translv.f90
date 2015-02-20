SUBROUTINE translv

!-----------------------------------------------------------------------
!
! Solution driver. Contains the time and outer loops. Calls for outer
! iteration work. Checks convergence and handles eventual output.
!
!-----------------------------------------------------------------------

  USE global_module, ONLY: i_knd, r_knd, ounit, zero, half, one, two

  USE plib_module, ONLY: glmax, comm_snap, iproc, root, thread_num,    &
    ichunk, nthreads, nnested

  USE geom_module, ONLY: geom_alloc, geom_dealloc, dinv, param_calc,   &
    nx, ny_gl, nz_gl

  USE sn_module, ONLY: nang, noct, mu, eta, xi, nmom

  USE data_module, ONLY: ng, v, vdelt, mat, sigt, siga, slgg, src_opt, &
    qim

  USE control_module, ONLY: nsteps, timedep, dt, oitm, otrdone,        &
    control_alloc, control_dealloc, dfmxo, it_det, popout, swp_typ

  USE utils_module, ONLY: print_error, stop_run

  USE solvar_module, ONLY: solvar_alloc, ptr_in, ptr_out, t_xs, a_xs,  &
    s_xs, flux0, fluxm

  USE expxs_module, ONLY: expxs_reg

  USE outer_module, ONLY: outer

  USE time_module, ONLY: tslv, wtime, tgrind, tparam

  USE analyze_module, ONLY: pop_calc

  IMPLICIT NONE
!_______________________________________________________________________
!
! Local variables
!_______________________________________________________________________

  CHARACTER(LEN=1) :: star='*'

  CHARACTER(LEN=64) :: error

  INTEGER(i_knd) :: cy, otno, ierr, g, l, i, tot_iits, cy_iits, out_iits

  REAL(r_knd) :: sf, time, t1, t2, t3, t4, t5, tmp

  REAL(r_knd), DIMENSION(:,:,:,:,:,:), POINTER :: ptr_tmp
!_______________________________________________________________________
!
! Call for data allocations. Some allocations depend on the problem
! type being requested.
!_______________________________________________________________________

  CALL wtime ( t1 )

  ierr = 0
  error = ' '

  CALL geom_alloc ( nang, ng, swp_typ, ichunk, ierr )
  CALL glmax ( ierr, comm_snap )
  IF ( ierr /= 0 ) THEN
    error = '***ERROR: GEOM_ALLOC: Allocation error of sweep parameters'
    CALL print_error ( ounit, error )
    CALL stop_run ( 1, 3, 0, 0 )
  END IF

  CALL solvar_alloc ( ierr )
  CALL glmax ( ierr, comm_snap )
  IF ( ierr /= 0 ) THEN
    error = '***ERROR: SOLVAR_ALLOC: Allocation error of solution ' // &
            'arrays'
    CALL print_error ( ounit, error )
    CALL stop_run ( 1, 3, 1, 0 )
  END IF

  CALL control_alloc ( ng, ierr )
  CALL glmax ( ierr, comm_snap )
  IF ( ierr /= 0 ) THEN
    error = '***ERROR: CONTROL_ALLOC: Allocation error of control ' // &
            'arrays'
    CALL print_error ( ounit, error )
    CALL stop_run ( 1, 3, 2, 0 )
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

  IF ( iproc == root ) THEN
    WRITE( *, 201)     ( star, i = 1, 80 )
    WRITE( ounit, 201) ( star, i = 1, 80 )
  END IF

  tot_iits = 0

  time_loop: DO cy = 1, nsteps

    CALL wtime ( t3 )

    vdelt = zero
    time = one
    IF ( timedep == 1 ) THEN
      IF ( iproc == root ) THEN
        WRITE( *, 202 )     ( star, i = 1, 30 ), cy
        WRITE( ounit, 202 ) ( star, i = 1, 30 ), cy
      END IF
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
!   Prepare some cross sections: total, in-group scattering, absorption.
!   Keep in the time loop for better consistency with PARTISN. Set up
!   geometric sweep parameters. Reset flux moments to zero at start of
!   each time step. Parallelize group loop with threads.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(STATIC,1) NUM_THREADS(nthreads*nnested)   &
  !$OMP& DEFAULT(SHARED) PRIVATE(g,l)
    DO g = 1, ng
      CALL expxs_reg ( siga(:,g), mat, a_xs(:,:,:,g) )
      CALL expxs_reg ( sigt(:,g), mat, t_xs(:,:,:,g) )
      CALL param_calc ( nang, ichunk, mu, eta, xi, t_xs(:,:,:,g),      &
        vdelt(g), dinv(:,:,:,:,:,g) )
      DO l = 1, nmom
        CALL expxs_reg ( slgg(:,l,g,g), mat, s_xs(:,:,:,l,g) )
      END DO
      flux0(:,:,:,g) = zero
      fluxm(:,:,:,:,g) = zero
    END DO
  !$OMP END PARALLEL DO
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
!   Using Jacobi iterations in energy, and the work in the outer loop
!   will be parallelized with threads.
!_______________________________________________________________________

    otrdone = .FALSE.

    cy_iits = 0

    IF ( iproc==root .AND. it_det==0 ) THEN
      WRITE( *, 203 )
      WRITE( ounit, 203 )
    END IF

    CALL wtime ( t4 )
    tparam = tparam + t4 - t3

    outer_loop: DO otno = 1, oitm

      IF ( iproc==root .AND. it_det==1 ) THEN
        WRITE( *, 204 )     ( star, i = 1, 20 ), otno
        WRITE( ounit, 204 ) ( star, i = 1, 20 ), otno
      END IF
!_______________________________________________________________________
!
!     Perform an outer iteration. Add up inners. Check convergence.
!_______________________________________________________________________

      CALL outer ( out_iits )

      cy_iits = cy_iits + out_iits

      IF ( iproc == root ) THEN
        WRITE( *, 205 )     otno, dfmxo, out_iits
        WRITE( ounit, 205 ) otno, dfmxo, out_iits
      END IF

      IF ( otrdone ) EXIT outer_loop

    END DO outer_loop
!_______________________________________________________________________
!
!   Compute and print the particle spectrum every cycle.
!_______________________________________________________________________

    IF ( popout == 2 ) CALL pop_calc ( cy, time )
!_______________________________________________________________________
!
!   Print the time cycle details. Add time cycle iterations.
!_______________________________________________________________________

    IF ( iproc == root ) THEN
      IF ( timedep == 1 ) THEN
        IF ( otrdone ) THEN
          WRITE( *, 206 )     cy, time, otno, cy_iits
          WRITE( ounit, 206 ) cy, time, otno, cy_iits
        ELSE
          WRITE( *, 207 )     cy, time, otno, cy_iits
          WRITE( ounit, 207 ) cy, time, otno, cy_iits
        END IF
      ELSE
        IF ( otrdone ) THEN
          WRITE( *, 208 )     otno, cy_iits
          WRITE( ounit, 208 ) otno, cy_iits
        ELSE
          WRITE( *, 209 )     otno, cy_iits
          WRITE( ounit, 209 ) otno, cy_iits
        END IF
      END IF
    END IF

    tot_iits = tot_iits + cy_iits

  END DO time_loop
!_______________________________________________________________________
!
!   Compute and print the particle spectrum only at end of calc.
!_______________________________________________________________________

    IF ( popout == 1 ) CALL pop_calc ( cy, time )
!_______________________________________________________________________
!
!   Final prints.
!_______________________________________________________________________

  IF ( iproc == root ) THEN
    IF ( timedep == 1 ) THEN
      WRITE( *, 210 )     ( star, i = 1, 30 ), tot_iits
      WRITE( ounit, 210 ) ( star, i = 1, 30 ), tot_iits
    END IF
    WRITE( *, 211 )     ( star, i = 1, 80 )
    WRITE( ounit, 211 ) ( star, i = 1, 80 )
  END IF

  CALL wtime ( t5 )
  tslv = t5 - t1
  tmp = REAL( nx, r_knd ) * REAL( ny_gl, r_knd ) * REAL( nz_gl, r_knd )&
        * REAL( nang, r_knd ) * REAL( noct, r_knd )                    &
        * REAL( tot_iits, r_knd )
  tgrind = tslv*1.0E9_r_knd / tmp
!_______________________________________________________________________

  201 FORMAT( 10X, 'keyword Iteration Monitor', /, 80A )
  202 FORMAT( /, 1X, 30A, /, 2X, 'Time Cycle ', I3 )
  203 FORMAT( 2X, 'Outer' )
  204 FORMAT( 1X, 20A, /, 2X, 'Outer ', I3 )
  205 FORMAT( 2X, I3, 4X, 'Dfmxo=', ES11.4, 4X, 'No. Inners=', I5 )
  206 FORMAT( /, 2X, 'Cycle=', I4, 4X, 'Time=', ES11.4, 4X, 'No. ',    &
              'Outers=', I4, 4X, 'No. Inners=', I5 )
  207 FORMAT( /, 2X, '*WARNING: Unconverged outer iterations', /, 2X,  &
             'Cycle=', I4, 4X, 'Time=', ES11.4, 4X, 'No. Outers=', I4, &
             4X, 'No. Inners=', I5, / )
  208 FORMAT( /, 2X, 'No. Outers=', I4, 4X, 'No. Inners=', I5 )
  209 FORMAT( /, 2X, '*WARNING: Unconverged outer iteration!', /, 2X,  &
              'No. Outers=', I4, 4X, 'No. Inners=', I5, / )
  210 FORMAT( /, 1X, 30A, /, 2X, 'Total inners for all time steps, '   &
              'outers = ', I6 )
  211 FORMAT( /, 80A, / )
!_______________________________________________________________________
!_______________________________________________________________________

END SUBROUTINE translv
