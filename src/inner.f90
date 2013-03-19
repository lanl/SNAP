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

  USE global_module, ONLY: i_knd, r_knd, l_knd, zero, one, ounit

  USE geom_module, ONLY: nx, ny, nz

  USE sn_module, ONLY: nmom, cmom, lma

  USE data_module, ONLY: ng

  USE control_module, ONLY: epsi, tolr, dfmxi, inrdone, it_det

  USE solvar_module, ONLY: q2grp, s_xs, flux, fluxpi, fluxm, qtot

  USE sweep_module, ONLY: sweep

  USE time_module, ONLY: tinrsrc, tsweeps, wtime

  USE plib_module, ONLY: glmax, comm_snap, iproc, root

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: inner


  CONTAINS


  SUBROUTINE inner ( inno, iits )

!-----------------------------------------------------------------------
!
! Do a single inner iteration for all groups. Calculate the total source
! for each group and sweep the mesh.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: inno

    INTEGER(i_knd), DIMENSION(ng), INTENT(OUT) :: iits

    REAL(r_knd) :: t1, t2, t3, t4
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: g
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
!   With source computed, set previous copy of flux and zero out current
!   copies--new flux moments iterates computed during sweep. Thread
!   over groups.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(DYNAMIC,1) DEFAULT(SHARED) PRIVATE(g)
    DO g = 1, ng
      IF ( inrdone(g) ) CYCLE
      fluxpi(:,:,:,g)   = flux(:,:,:,g)
    END DO
  !$OMP END PARALLEL DO
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
!
!-----------------------------------------------------------------------
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: g
!_______________________________________________________________________
!
!   Compute the within-group scattering source. Thread over groups.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(DYNAMIC,1) DEFAULT(SHARED) PRIVATE(g)
    DO g = 1, ng
      IF ( inrdone(g) ) CYCLE
      CALL inr_src_scat ( q2grp(:,:,:,:,g), s_xs(:,:,:,:,g),           &
        flux(:,:,:,g), fluxm(:,:,:,:,g), qtot(:,:,:,:,g) )
    END DO
  !$OMP END PARALLEL DO
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE inr_src


  SUBROUTINE inr_src_scat ( qo, cs, f, fm, q )

!-----------------------------------------------------------------------
!
! Compute the within-group scattering for a given group. Add it to fixed
! and out-of-group sources.
!
!-----------------------------------------------------------------------

    REAL(r_knd), DIMENSION(nx,ny,nz), INTENT(IN) :: f

    REAL(r_knd), DIMENSION(cmom-1,nx,ny,nz), INTENT(IN) :: fm

    REAL(r_knd), DIMENSION(cmom,nx,ny,nz), INTENT(IN) :: qo

    REAL(r_knd), DIMENSION(nmom,nx,ny,nz), INTENT(IN) :: cs

    REAL(r_knd), DIMENSION(cmom,nx,ny,nz), INTENT(OUT) :: q
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: i, j, k, l, m, mom
!_______________________________________________________________________
!
!   Loop over all cells. Set the first source moment with flux (f). Then
!   set remaining source moments with fluxm (fm) and combination of
!   higher scattering orders.
!_______________________________________________________________________

    DO k = 1, nz
    DO j = 1, ny
    DO i = 1, nx

      q(1,i,j,k) = qo(1,i,j,k) + cs(1,i,j,k)*f(i,j,k)
!_______________________________________________________________________
!
!     Work on other moments with fluxm array
!_______________________________________________________________________

      mom = 2
      DO l = 2, nmom
        DO m = 1, lma(l)
          q(mom,i,j,k) = qo(mom,i,j,k) + cs(l,i,j,k)*fm(mom-1,i,j,k)
          mom = mom + 1
        END DO
      END DO

    END DO
    END DO
    END DO
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE inr_src_scat


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

  !$OMP PARALLEL DO SCHEDULE(DYNAMIC,1) DEFAULT(SHARED) PRIVATE(g)
    DO g = 1, ng
      IF ( inrdone(g) ) CYCLE
      iits(g) = inno
      WHERE( ABS( fluxpi(:,:,:,g) ) > tolr )
        df(:,:,:,g) = ABS( flux(:,:,:,g)/fluxpi(:,:,:,g) - one )
      ELSEWHERE
        df(:,:,:,g) = ABS( flux(:,:,:,g) - fluxpi(:,:,:,g) )
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
