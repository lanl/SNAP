!-----------------------------------------------------------------------
!
! MODULE: outer_module
!> @brief
!> This module controls the outer iterations. Outer iterations are
!> threaded over the energy dimension and represent a Jacobi iteration
!> strategy. Includes setting the outer source. Checking outer iteration
!> convergence.
!
!-----------------------------------------------------------------------

MODULE outer_module

  USE global_module, ONLY: i_knd, r_knd, one, zero

  USE geom_module, ONLY: nx, ny, nz

  USE sn_module, ONLY: nmom, cmom, lma

  USE data_module, ONLY: ng, mat, slgg, qi, nmat, src_opt

  USE solvar_module, ONLY: q2grp0, q2grpm, flux0, fluxm, flux0po

  USE control_module, ONLY: iitm, timedep, inrdone, tolr, dfmxo, epsi, &
    otrdone

  USE inner_module, ONLY: inner

  USE time_module, ONLY: totrsrc, tinners, tinrmisc, wtime

  USE plib_module, ONLY: glmax, comm_snap

  USE expxs_module, ONLY: expxs_reg, expxs_slgg

  USE thrd_comm_module, ONLY: assign_thrd_set, destroy_task_set

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: outer


  CONTAINS


  SUBROUTINE outer ( sum_iits )

!-----------------------------------------------------------------------
!
! Do a single outer iteration. Sets the out-of-group sources, performs
! inners for all groups.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(OUT) :: sum_iits
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: inno

    INTEGER(i_knd), DIMENSION(ng) :: iits

    REAL(r_knd) :: t1, t2, t3, t4
!_______________________________________________________________________
!
!   Compute the outer source: sum of fixed + out-of-group sources
!_______________________________________________________________________

    CALL wtime ( t1 )

    CALL otr_src

    CALL wtime ( t2 )
    totrsrc = totrsrc + t2 - t1
!_______________________________________________________________________
!
!   Zero out the inner iterations group count. Save the flux for
!   comparison.
!_______________________________________________________________________

    iits = 0
    flux0po = flux0
!_______________________________________________________________________
!
!   Start the inner iterations
!_______________________________________________________________________

    CALL wtime ( t3 )

    inrdone = .FALSE.

    inner_loop: DO inno = 1, iitm

      CALL inner ( inno, iits )

      IF ( ALL( inrdone ) ) EXIT inner_loop

    END DO inner_loop

    sum_iits = SUM( iits )

    CALL wtime ( t4 )
    tinners = tinners + t4 - t3
!_______________________________________________________________________
!
!   Check outer convergence
!_______________________________________________________________________

    CALL otr_conv
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE outer


  SUBROUTINE otr_src

!-----------------------------------------------------------------------
!
! Loop over groups to compute each one's outer loop source.
!
!-----------------------------------------------------------------------
!_______________________________________________________________________
!
!   Local Variables
!_______________________________________________________________________

    INTEGER(i_knd) :: ng_per_thrd, nthrd_used, nnstd_used, g, t, n, k, j

    INTEGER(i_knd), DIMENSION(ng) :: do_grp

    INTEGER(i_knd), DIMENSION(:,:), POINTER :: grp_act
!_______________________________________________________________________
!
!   Use assign_grp_set subroutine to make group sets and apply them to
!   the threads
!_______________________________________________________________________

    NULLIFY( grp_act )

    DO g = 1, ng
      do_grp(g) = ng - g + 1
    END DO

    CALL assign_thrd_set ( do_grp, ng, ng_per_thrd, nthrd_used, ny*nz, &
      nnstd_used, grp_act )
!_______________________________________________________________________
!
!   Parallelize outer source calculation with threads over group loop.
!   If nested threads are available, further parallelize over the k-j
!   loops (via OpenMP 'collapse'). Each thread loops over ng_per_thrd
!   groups and updates group sources according to grp_act.
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
          CALL otr_src_calc ( g, j, k, qi(:,j,k,g), slgg(:,:,:,g),     &
            mat(:,j,k), flux0, fluxm, q2grp0(:,j,k,g),                 &
            q2grpm(:,:,j,k,g) )
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

  END SUBROUTINE otr_src


  SUBROUTINE otr_src_calc ( g, j, k, qi0, sxs_g, map, f0, fm, qo0, qom )

!-----------------------------------------------------------------------
!
! Compute the scattering source for all cells and moments. This routine
! is called for each group. It computes source components from all the
! other groups, which are looped over here. It skips computing the
! source from itself, which is captured in the inner iterations.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: g, j, k

    INTEGER(i_knd), DIMENSION(nx), INTENT(IN) :: map

    REAL(r_knd), DIMENSION(nx), INTENT(IN) :: qi0

    REAL(r_knd), DIMENSION(nx), INTENT(OUT) :: qo0

    REAL(r_knd), DIMENSION(cmom-1,nx), INTENT(OUT) :: qom

    REAL(r_knd), DIMENSION(nmat,nmom,ng), INTENT(IN) :: sxs_g

    REAL(r_knd), DIMENSION(nx,ny,nz,ng), INTENT(IN) :: f0

    REAL(r_knd), DIMENSION(cmom-1,nx,ny,nz,ng), INTENT(IN) :: fm
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: gp

    REAL(r_knd), DIMENSION(nx) :: cs0

    REAL(r_knd), DIMENSION(cmom-1,nx) :: csm
!_______________________________________________________________________
!
!   Initialize sources. Include isotropic inhomogeneous source.
!_______________________________________________________________________

    qo0 = qi0
    qom = zero
!_______________________________________________________________________
!
!   Loop over originating groups, gp. Skip own group.
!_______________________________________________________________________

    DO gp = 1, ng

      IF ( gp == g ) CYCLE
!_______________________________________________________________________
!
!     Expand isotropic cross sections to fine mesh for current gp->g.
!     Add out of group scattering source.
!_______________________________________________________________________

      CALL expxs_reg ( sxs_g(:,1,gp), map, cs0 )

      qo0 = qo0 + cs0*f0(:,j,k,gp)
!_______________________________________________________________________
!
!     Expand anisotropic cross sections to fine mesh for gp->g. Add
!     components to group scattering source moments. Close the loop.
!_______________________________________________________________________

      IF ( nmom == 1 ) CYCLE

      CALL expxs_slgg ( sxs_g(:,:,gp), map, csm )

      qom = qom + csm*fm(:,:,j,k,gp)

    END DO
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE otr_src_calc


  SUBROUTINE otr_conv

!-----------------------------------------------------------------------
!
! Check for convergence of outer iterations. Use only the zeroth moment
! data (flux0/flux0po).
!
!-----------------------------------------------------------------------
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: g

    REAL(r_knd), DIMENSION(nx,ny,nz,ng) :: df
!_______________________________________________________________________
!
!   Thread to speed up computation of df by looping over groups. Rejoin
!   threads and then determine max error.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(STATIC,1) DEFAULT(SHARED) PRIVATE(g)
    DO g = 1, ng
      WHERE( ABS( flux0po(:,:,:,g) ) > tolr )
        df(:,:,:,g) = ABS( flux0(:,:,:,g)/flux0po(:,:,:,g) - one )
      ELSEWHERE
        df(:,:,:,g) = ABS( flux0(:,:,:,g) - flux0po(:,:,:,g) )
      END WHERE
    END DO
  !$OMP END PARALLEL DO

    dfmxo = MAXVAL( df )
    CALL glmax ( dfmxo, comm_snap )

    IF ( dfmxo<=100.0_r_knd*epsi .AND. ALL( inrdone ) ) otrdone = .TRUE.
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE otr_conv


END MODULE outer_module
