!-----------------------------------------------------------------------
!
! MODULE: sweep_module
!> @brief
!> This module contains all the subroutines related to the mesh sweep.
!
!-----------------------------------------------------------------------

MODULE sweep_module

  USE global_module, ONLY: i_knd, zero, l_knd

  USE geom_module, ONLY: ndimen, ny, nz, nc

  USE sn_module, ONLY: nang

  USE data_module, ONLY: ng

  USE control_module, ONLY: inrdone

  USE octsweep_module, ONLY: octsweep

  USE solvar_module, ONLY: jb_in, jb_out, kb_in, kb_out, flkx, flky,   &
    flkz

  USE plib_module, ONLY: ylop, yhip, zlop, zhip, firsty, lasty, firstz,&
    lastz, g_off, ichunk, ycomm, zcomm, psend, precv, yproc, zproc,    &
    nproc, nthreads, thread_level, thread_serialized, thread_multiple, &
    num_grth, plock_omp

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: sweep

  SAVE
!_______________________________________________________________________
!
! Module variables
!
! mtag        - message tag
! *p_snd      - rank of process sending to
! *p_rcv      - rank of process receiving from
! incoming*   - logical determining if proc will receive a message
! outgoing*   - logical determining if proc will send a message
!_______________________________________________________________________

  INTEGER(i_knd) :: mtag, yp_snd, yp_rcv, zp_snd, zp_rcv

  LOGICAL(l_knd) :: incomingy, incomingz, outgoingy, outgoingz


  CONTAINS


  SUBROUTINE sweep

!-----------------------------------------------------------------------
!
! Driver for the mesh sweeps. Manages the loops over octant pairs.
!
!-----------------------------------------------------------------------
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: jd, kd, jlo, jhi, jst, klo, khi, kst, g, iop, i

    INTEGER(i_knd), DIMENSION(1) :: gnext

    INTEGER(i_knd), DIMENSION(num_grth) :: dogrp

    INTEGER(i_knd), DIMENSION(ng) :: grp_act

    LOGICAL(l_knd) :: use_lock
!_______________________________________________________________________
!
!   Set up OpenMP lock if necessary.
!_______________________________________________________________________

    use_lock = nproc>1 .AND. nthreads>1 .AND.                          &
      thread_level/=thread_multiple

    IF ( use_lock ) CALL plock_omp ( 'init' )
!_______________________________________________________________________
!
!   Start OpenMP parallel region for entire sweep subroutine. Then clear
!   leakage arrays with threaded do loop.
!_______________________________________________________________________

  !$OMP PARALLEL DEFAULT(SHARED)

  !$OMP DO SCHEDULE(DYNAMIC,1) PRIVATE(g)
    DO g = 1, ng
      IF ( inrdone(g) ) CYCLE
      flkx(:,:,:,g) = zero
      flky(:,:,:,g) = zero
      flkz(:,:,:,g) = zero
    END DO
  !$OMP END DO
!_______________________________________________________________________
!
!   Loop over octant pairs, according to ndimen. Set up the sweep order.
!   Place a barrier at start of new octant to make sure all threads have
!   same kd and jd. Only one thread needs to set shared values of loop
!   bounds, strides, and communication parameters.
!_______________________________________________________________________

    kd_loop: DO kd = 1, MAX( ndimen-1, 1 )
    jd_loop: DO jd = 1, MIN( ndimen, 2 )

  !$OMP BARRIER

  !$OMP SINGLE

      IF ( jd == 1 ) THEN
        jlo = ny; jhi = 1; jst = -1
        yp_snd = ylop; yp_rcv = yhip
      ELSE
        jlo = 1; jhi = ny; jst = 1
        yp_snd = yhip; yp_rcv = ylop
      END IF

      IF ( kd == 1 ) THEN
        klo = nz; khi = 1; kst = -1
        zp_snd = zlop; zp_rcv = zhip
      ELSE
        klo = 1; khi = nz; kst = 1
        zp_snd = zhip; zp_rcv = zlop
      END IF

      incomingy = .NOT.( (jd==1 .AND. lasty) .OR. (jd==2 .AND. firsty) )
      incomingz = .NOT.( (kd==1 .AND. lastz) .OR. (kd==2 .AND. firstz) )
      outgoingy = .NOT.( (jd==1 .AND. firsty) .OR. (jd==2 .AND. lasty) )
      outgoingz = .NOT.( (kd==1 .AND. firstz) .OR. (kd==2 .AND. lastz) )

      mtag = 2*nc * (jd-1 + 2*(kd-1))
!_______________________________________________________________________
!
!     Set up groups to be swept. Only num_grth can be swept at a time
!     due to communications.
!_______________________________________________________________________

      grp_act = 1
      WHERE( inrdone ) grp_act = 0

      dogrp = 0

      DO i = 1, num_grth
        gnext = MAXLOC( grp_act )
        g = gnext(1)
        IF ( grp_act(g) == 0 ) EXIT
        grp_act(g) = 0
        dogrp(i) = g
      END DO

  !$OMP END SINGLE
!_______________________________________________________________________
!
!     Loop as long as groups that are not converged still need to be
!     done. Set an operation as looping over a chunk of i-cells.
!_______________________________________________________________________

      dogrp_loop: DO WHILE( ANY( dogrp > 0 ) )

        iop_loop: DO iop = 1, 2*nc
!_______________________________________________________________________
!
!         If supported, thread_multiple allows all threads over groups
!         to handle independent MPI communications. Loop over groups.
!_______________________________________________________________________

          IF ( thread_level == thread_multiple ) THEN

  !$OMP DO SCHEDULE(STATIC,1) PRIVATE(i)
            DO i = 1, num_grth
              IF ( dogrp(i) == 0 ) CYCLE
              CALL sweep_recv_bdry ( dogrp(i), iop )
              CALL octsweep ( dogrp(i), iop, jd, kd, jlo, jhi, jst,    &
                klo, khi, kst )
              CALL sweep_send_bdry ( dogrp(i), iop )
            END DO
  !$OMP END DO NOWAIT
!_______________________________________________________________________
!
!         Otherwise, thread_serialized makes sure only one thread calls
!         for MPI communications at a time. Use locks to make sure all
!         sends/receives are done before next receives/sends. Use
!         ordered loops to match threads' receives with sends.
!_______________________________________________________________________

          ELSE

            IF ( incomingy .OR. incomingz ) THEN
  !$OMP DO SCHEDULE(STATIC,1) ORDERED PRIVATE(i)
              DO i = 1, num_grth
                IF ( i==1 .AND. use_lock ) CALL plock_omp ( 'set' )
  !$OMP ORDERED
                IF ( dogrp(i) /= 0 )                                   &
                  CALL sweep_recv_bdry ( dogrp(i), iop )
  !$OMP END ORDERED
                IF ( i==num_grth .AND. use_lock )                      &
                  CALL plock_omp ( 'unset' )
              END DO
  !$OMP END DO NOWAIT
            END IF

  !$OMP DO SCHEDULE(STATIC,1) PRIVATE(i)
            DO i = 1, num_grth
              IF ( dogrp(i) == 0 ) CYCLE
              CALL octsweep( dogrp(i), iop, jd, kd, jlo, jhi, jst,     &
                klo, khi, kst )
            END DO
  !$OMP END DO NOWAIT

            IF ( outgoingy .OR. outgoingz ) THEN
  !$OMP DO SCHEDULE(STATIC,1) ORDERED PRIVATE(i)
              DO i = 1, num_grth
                IF ( i==1 .AND. use_lock ) CALL plock_omp ( 'set' )
  !$OMP ORDERED
                IF ( dogrp(i) /= 0 )                                   &
                  CALL sweep_send_bdry ( dogrp(i), iop )
  !$OMP END ORDERED
                IF ( i==num_grth .AND. use_lock )                      &
                  CALL plock_omp ( 'unset' )
              END DO
  !$OMP END DO NOWAIT
            END IF

          END IF
!_______________________________________________________________________
!
!       End sweep for octant pair. Use barrier statement to sync threads
!       and use single block to assign next set of groups, if any left
!_______________________________________________________________________

        END DO iop_loop

  !$OMP BARRIER

  !$OMP SINGLE
        dogrp = 0
        DO i = 1, num_grth
          gnext = MAXLOC( grp_act )
          g = gnext(1)
          IF ( grp_act(g) == 0 ) EXIT
          grp_act(g) = 0
          dogrp(i) = g
        END DO
  !$OMP END SINGLE

      END DO dogrp_loop

    END DO jd_loop
    END DO kd_loop

  !$OMP END PARALLEL
!_______________________________________________________________________
!
!   Destroy the lock
!_______________________________________________________________________

    IF ( use_lock) CALL plock_omp ( 'destroy' )
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE sweep


  SUBROUTINE sweep_recv_bdry ( g, iop )

!-----------------------------------------------------------------------
!
! Receive flux from upstream boundaries
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: g, iop
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: i
!_______________________________________________________________________

    i = iop + g*g_off + mtag

    CALL precv ( yp_rcv, yproc, nang, ichunk, nz, jb_in(:,:,:,g),      &
      ycomm, i )
    CALL precv ( zp_rcv, zproc, nang, ichunk, ny, kb_in(:,:,:,g),      &
      zcomm, i )
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE sweep_recv_bdry


  SUBROUTINE sweep_send_bdry ( g, iop )

!-----------------------------------------------------------------------
!
! Receive flux from upstream boundaries
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: g, iop
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: i
!_______________________________________________________________________

    i = iop + g*g_off + mtag

    CALL psend ( yp_snd, yproc, nang, ichunk, nz, jb_out(:,:,:,g),     &
      ycomm, i )
    CALL psend ( zp_snd, zproc, nang, ichunk, ny, kb_out(:,:,:,g),     &
      zcomm, i )
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE sweep_send_bdry


END MODULE sweep_module
