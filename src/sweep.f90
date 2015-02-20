!-----------------------------------------------------------------------
!
! MODULE: sweep_module
!> @brief
!> This module controls the mesh sweep scheduling. It directs the flow
!> of KBA stages for different octants, groups, spatial work chunks.
!
!-----------------------------------------------------------------------

MODULE sweep_module

  USE global_module, ONLY: i_knd, zero, l_knd, r_knd, one

  USE geom_module, ONLY: jdim, kdim, nc

  USE data_module, ONLY: ng

  USE control_module, ONLY: inrdone, swp_typ

  USE octsweep_module, ONLY: octsweep

  USE solvar_module, ONLY: flkx, flky, flkz

  USE plib_module, ONLY: nthreads, waitinit, iproc, root

  USE thrd_comm_module, ONLY: assign_thrd_set, destroy_task_set

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: sweep

  SAVE


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

    INTEGER(i_knd) :: jd, kd, g, iop, ng_per_thrd, nthrd_used,         &
      nnstd_used, t, n

    INTEGER(i_knd), DIMENSION(2) :: reqs

    INTEGER(i_knd), DIMENSION(ng) :: do_grp

    INTEGER(i_knd), DIMENSION(:,:), POINTER :: grp_act

    REAL(r_knd), DIMENSION(ng) :: fmin, fmax
!_______________________________________________________________________
!
!   Assign the work to threads. Main level threads always applied to
!   energy groups. Apply nested threads additionally to groups if
!   swp_typ is 0. Apply nested threads to mini-KBA if swp_typ is 1;
!   where spawning of said threads occurs just before sweep starts
!   (i.e., in dim3_sweep_mkba).
!_______________________________________________________________________

    do_grp = 1
    WHERE ( inrdone ) do_grp = 0

    IF ( swp_typ == 0 ) THEN
      CALL assign_thrd_set ( do_grp, ng, ng_per_thrd, nthrd_used, 0,   &
        nnstd_used, grp_act )
    ELSE
      CALL assign_thrd_set ( do_grp, ng, ng_per_thrd, nthrd_used, 1,   &
        nnstd_used, grp_act )
    END IF
!_______________________________________________________________________
!
!   Initialize the reqs (send request) array for asynchronous sends
!_______________________________________________________________________

    CALL waitinit ( reqs, SIZE( reqs ) )
!_______________________________________________________________________
!
!   Start the loop over main level threads. These threads will work on
!   different energy groups. If nested threads are to be applied to
!   groups according to swp_typ=0, which implies nnstd_used>1 from call
!   above, start that type of nested parallel section here. Nested by
!   mini-KBA handled in the dim3_sweep_mkba for now.
!_______________________________________________________________________

  !$OMP PARALLEL DO NUM_THREADS(nthrd_used) IF(nthrd_used>1)           &
  !$OMP& SCHEDULE(STATIC,1) DEFAULT(SHARED) PRIVATE(t,n,g,kd,jd,iop)   &
  !$OMP& FIRSTPRIVATE(reqs)
    main_thread_loop: DO t = 1, nthrd_used

  !$OMP PARALLEL NUM_THREADS(nnstd_used) IF(nnstd_used>1)
!_______________________________________________________________________
!
!     Use a SINGLE contruct so each main level thread clears the leakage
!     arrays and initializes fmin/fmax. Exit the loop if all work done.
!_______________________________________________________________________

  !$OMP SINGLE
      clean_loop: DO n = 1, ng_per_thrd

        g = grp_act(n,t)

        IF ( g == 0 ) EXIT clean_loop

        fmin(g) = HUGE( one )
        fmax(g) = zero

        flkx(:,:,:,g) = zero
        flky(:,:,:,g) = zero
        flkz(:,:,:,g) = zero

      END DO clean_loop
  !$OMP END SINGLE
!_______________________________________________________________________
!
!     Loop over octant pairs; set the starting corner, i.e., the
!     direction in y and z.
!_______________________________________________________________________

      kd_loop: DO kd = 1, kdim
      jd_loop: DO jd = 1, jdim
!_______________________________________________________________________
!
!       Loop over the groups assigned to each thread. If nested threads
!       exist here, they work concurrently on the groups in each main
!       level thread's full set.
!_______________________________________________________________________

  !$OMP DO SCHEDULE(STATIC,1) PRIVATE(n,g,iop)
        grp_loop: DO n = 1, ng_per_thrd

          g = grp_act(n,t)
!_______________________________________________________________________
!
!         Loop over a spatial work chunk. Loop contains twice the number
!         of chunks: half for negative x-dir sweep, half for positive
!         x-dir sweep.
!_______________________________________________________________________

          iop_loop: DO iop = 1, 2*nc
!_______________________________________________________________________
!
!           Now a single KBA stage or task is defined: know the octant
!           to sweep via kd, jd, iop; know the group, g; know the
!           spatial work chunk, iop. Call to sweep this task.
!_______________________________________________________________________

            CALL octsweep ( g, jd, kd, iop, t, nthrd_used, reqs,       &
                            SIZE( reqs ), fmin, fmax )
!_______________________________________________________________________
!
!         End the loops. Destroy the task set.
!_______________________________________________________________________

          END DO iop_loop

        END DO grp_loop
  !$OMP END DO

      END DO jd_loop
      END DO kd_loop

  !$OMP END PARALLEL

    END DO main_thread_loop
  !$OMP END PARALLEL DO

    CALL destroy_task_set ( grp_act )
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE sweep


END MODULE sweep_module
