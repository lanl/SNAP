!-----------------------------------------------------------------------
!
! MODULE: inner_module
!> @brief
!> This module controls the inner iterations. Inner iterations include
!> the KBA mesh sweep, which is parallelized via MPI and vectorized over
!> angles in a given octant. Inner source computed here and inner
!> convergence is checked.
!
!-----------------------------------------------------------------------

MODULE inner_module

  USE global_module, ONLY: i_knd, r_knd, one, ounit

  USE geom_module, ONLY: nx, ny, nz, nc

  USE sn_module, ONLY: nmom, cmom, lma

  USE data_module, ONLY: ng

  USE control_module, ONLY: epsi, tolr, dfmxi, inrdone, it_det

  USE solvar_module, ONLY: q2grp0, q2grpm, s_xs, flux0, flux0pi, fluxm,&
    qtot

  USE sweep_module, ONLY: sweep

  USE time_module, ONLY: tinrsrc, tsweeps, wtime

  USE plib_module, ONLY: glmax, comm_snap, iproc, root, ichunk

  USE thrd_comm_module, ONLY: assign_thrd_set, destroy_task_set

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: inner


  CONTAINS


  SUBROUTINE inner ( inno, iits )

!-----------------------------------------------------------------------
!
! Do a single inner iteration for all groups. Calculate the total source
! for each group and sweep the mesh over octants. Repeat for all groups
! unless the group is converged according to inrdone(g).
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: inno

    INTEGER(i_knd), DIMENSION(ng), INTENT(OUT) :: iits
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: g

    REAL(r_knd) :: t1, t2, t3, t4
!_______________________________________________________________________
!
!   Compute the inner source and add it to fixed + out-of-group sources
!_______________________________________________________________________

    CALL wtime ( t1 )

    CALL inr_src

    CALL wtime ( t2 )
    tinrsrc = tinrsrc + t2 - t1
!_______________________________________________________________________
!
!   With source computed, set previous copy of flux; new flux moments
!   iterates computed during sweep.
!_______________________________________________________________________

    DO g = 1, ng
      IF ( inrdone(g) ) CYCLE
      flux0pi(:,:,:,g) = flux0(:,:,:,g)
    END DO
!_______________________________________________________________________
!
!   Call for the transport sweep. Check convergence, using threads.
!_______________________________________________________________________

    CALL wtime ( t3 )

    CALL sweep

    CALL wtime ( t4 )
    tsweeps = tsweeps + t4 - t3

    CALL inr_conv ( inno, iits )
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE inner


  SUBROUTINE inr_src

!-----------------------------------------------------------------------
!
! Compute the inner source, i.e., the within-group scattering source.
! Thread over groups and use nested threads on i-lines when available.
!
!-----------------------------------------------------------------------
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: ng_per_thrd, nthrd_used, nnstd_used, t, n, g, k, j

    INTEGER(i_knd), DIMENSION(ng) :: do_grp

    INTEGER(i_knd), DIMENSION(:,:), POINTER :: grp_act
!_______________________________________________________________________
!
!   Will use threads to parallelize the source computation. Establish
!   the amount of work and assign to threads and nested threads if
!   available. Skip any group already done.
!_______________________________________________________________________

    do_grp = 1
    WHERE( inrdone ) do_grp = 0

    CALL assign_thrd_set ( do_grp, ng, ng_per_thrd, nthrd_used, ny*nz, &
      nnstd_used, grp_act )
!_______________________________________________________________________
!
!   Loop over the number of threads available. Use nested threads over
!   the k-j loops if available. Exit the loops if the thread has run out
!   of work, i.e., some groups were converged and had less to do than
!   full ng_per_thrd.
!_______________________________________________________________________

  !$OMP PARALLEL DO NUM_THREADS(nthrd_used) IF(nthrd_used>1)           &
  !$OMP& SCHEDULE(STATIC,1), DEFAULT(SHARED), PRIVATE(t,n,g,k,j)
    DO t = 1, nthrd_used

  !$OMP PARALLEL NUM_THREADS(nnstd_used) IF(nnstd_used>1)              &
  !$OMP& PRIVATE(n,g,k,j)
      DO n = 1, ng_per_thrd

        g = grp_act(n,t)
        IF ( g == 0 ) EXIT

  !$OMP DO SCHEDULE(STATIC,1) COLLAPSE(2)
        DO k = 1, nz
        DO j = 1, ny
          CALL inr_src_calc ( j, k, s_xs(:,:,:,:,g), flux0(:,j,k,g),   &
            fluxm(:,:,j,k,g), q2grp0(:,j,k,g), q2grpm(:,:,j,k,g),      &
            qtot(:,:,:,:,:,g) )
        END DO
        END DO
  !$OMP END DO NOWAIT

      END DO
  !$OMP END PARALLEL

    END DO
  !$OMP END PARALLEL DO
!_______________________________________________________________________
!
!   Deallocate the grp_act pointer
!_______________________________________________________________________

    CALL destroy_task_set ( grp_act )
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE inr_src


  SUBROUTINE inr_src_calc ( j, k, sxs_g, f0, fm, qo0, qom, q )

!-----------------------------------------------------------------------
!
! Compute the within-group scattering for a given group. Add it to fixed
! and out-of-group sources.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: j, k

    REAL(r_knd), DIMENSION(nx), INTENT(IN) :: f0, qo0

    REAL(r_knd), DIMENSION(cmom-1,nx), INTENT(IN) :: fm, qom

    REAL(r_knd), DIMENSION(nx,ny,nz,nmom), INTENT(IN) :: sxs_g

    REAL(r_knd), DIMENSION(cmom,ichunk,ny,nz,nc), INTENT(OUT) :: q
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: ic, ich, i, mom, l
!_______________________________________________________________________
!
!   Loop over i cells. Set the first source moment with flux0 (f0). Then
!   set remaining source moments with fluxm (fm) and combination of
!   higher scattering orders.
!_______________________________________________________________________

    ic = 0
    ich = 1

    DO i = 1, nx

      ic = ic + 1

      q(1,ic,j,k,ich) = qo0(i) + f0(i)*sxs_g(i,j,k,1)

      mom = 1
      DO l = 2, nmom
        q(mom+1:mom+lma(l),ic,j,k,ich) = qom(mom:mom+lma(l)-1,i) +     &
          fm(mom:mom+lma(l)-1,i)*sxs_g(i,j,k,l)
        mom = mom + lma(l)
      END DO

      IF ( ic == ichunk ) THEN
        ic = 0
        ich = ich + 1
      END IF

    END DO
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE inr_src_calc


  SUBROUTINE inr_conv ( inno, iits )

!-----------------------------------------------------------------------
!
! Check for inner iteration convergence using the flux array.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: inno

    INTEGER(i_knd), DIMENSION(ng), INTENT(OUT) :: iits
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: g

    REAL(r_knd), DIMENSION(nx,ny,nz,ng) :: df
!_______________________________________________________________________
!
!   Thread group loops for computing local difference (df) array.
!   compute max for that group.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(STATIC,1) DEFAULT(SHARED) PRIVATE(g)
    DO g = 1, ng
      IF ( inrdone(g) ) CYCLE
      iits(g) = inno
      WHERE( ABS( flux0pi(:,:,:,g) ) > tolr )
        df(:,:,:,g) = ABS( flux0(:,:,:,g)/flux0pi(:,:,:,g) - one )
      ELSEWHERE
        df(:,:,:,g) = ABS( flux0(:,:,:,g) - flux0pi(:,:,:,g) )
      END WHERE
      dfmxi(g) = MAXVAL( df(:,:,:,g) )
    END DO
  !$OMP END PARALLEL DO
!_______________________________________________________________________
!
!   All procs then reduce dfmxi for all groups, determine which groups
!   are converged and print requested info
!_______________________________________________________________________

    CALL glmax ( dfmxi, ng, comm_snap )
    WHERE( dfmxi <= epsi ) inrdone = .TRUE.

    IF ( iproc==root .AND. it_det==1 ) THEN
      DO g = 1, ng
        WRITE( ounit, 221 ) g, iits(g), dfmxi(g)
      END DO
    END IF
!_______________________________________________________________________

    221 FORMAT( 4X, 'Group ', I3, 4X, ' Inner ', I5, 4X, ' Dfmxi ',    &
                ES11.4 )
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE inr_conv


END MODULE inner_module
