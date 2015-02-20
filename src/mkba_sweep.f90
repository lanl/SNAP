!-----------------------------------------------------------------------
!
! MODULE: mkba_sweep_module
!> @brief
!> This module contains the 2D and 3D mesh sweep logic using mini-KBA.
!
!-----------------------------------------------------------------------

MODULE mkba_sweep_module

  USE global_module, ONLY: i_knd, r_knd, zero, two, one, half

  USE plib_module, ONLY: ichunk, firsty, lasty, firstz, lastz,         &
    nnested

  USE geom_module, ONLY: nx, hi, hj, hk, ndimen, ny, nz, ndiag, diag, nc

  USE sn_module, ONLY: cmom, nang, mu, eta, xi, w, noct

  USE data_module, ONLY: src_opt, qim

  USE control_module, ONLY: fixup, tolr

  USE thrd_comm_module, ONLY: sweep_recv_bdry, sweep_send_bdry

  IMPLICIT NONE

  PUBLIC :: mkba_sweep


  CONTAINS


  SUBROUTINE mkba_sweep ( ich, id, d1, d2, d3, d4, jd, kd, oct, g, iop,&
    t, nthrd, reqs, szreq, psii, psij, psik, qtot, ec, vdelt, ptr_in,  &
    ptr_out, dinv, flux0, fluxm, jb_in, jb_out, kb_in, kb_out, wmu,    &
    weta, wxi, flkx, flky, flkz, t_xs, fmin, fmax )

!-----------------------------------------------------------------------
!
! 3-D slab mesh sweeper.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: ich, id, d1, d2, d3, d4, jd, kd, oct,&
      g, iop, t, nthrd, szreq

    INTEGER(i_knd), DIMENSION(szreq), INTENT(INOUT) :: reqs

    REAL(r_knd), INTENT(IN) :: vdelt

    REAL(r_knd), INTENT(INOUT) :: fmin, fmax

    REAL(r_knd), DIMENSION(nang), INTENT(IN) :: wmu, weta, wxi

    REAL(r_knd), DIMENSION(nang,cmom), INTENT(IN) :: ec

    REAL(r_knd), DIMENSION(nang,ny,nz), INTENT(INOUT) :: psii

    REAL(r_knd), DIMENSION(nang,ichunk,nz), INTENT(INOUT) :: psij,     &
      jb_in, jb_out

    REAL(r_knd), DIMENSION(nang,ichunk,ny), INTENT(INOUT) :: psik,     &
      kb_in, kb_out

    REAL(r_knd), DIMENSION(nx,ny,nz), INTENT(IN) :: t_xs

    REAL(r_knd), DIMENSION(nx,ny,nz), INTENT(INOUT) :: flux0

    REAL(r_knd), DIMENSION(nx+1,ny,nz), INTENT(INOUT) :: flkx

    REAL(r_knd), DIMENSION(nx,ny+1,nz), INTENT(INOUT) :: flky

    REAL(r_knd), DIMENSION(nx,ny,nz+1), INTENT(INOUT) :: flkz

    REAL(r_knd), DIMENSION(nang,ichunk,ny,nz), INTENT(IN) :: dinv

    REAL(r_knd), DIMENSION(cmom-1,nx,ny,nz), INTENT(INOUT) :: fluxm

    REAL(r_knd), DIMENSION(cmom,ichunk,ny,nz), INTENT(IN) :: qtot

    REAL(r_knd), DIMENSION(d1,d2,d3,d4), INTENT(IN) :: ptr_in

    REAL(r_knd), DIMENSION(d1,d2,d3,d4), INTENT(OUT) :: ptr_out
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: ist, jst, jlo, jhi, kst, klo, khi, d, n, ic, i,  &
      j, k, l, ibl, ibr, ibb, ibt, ibf, ibk

    REAL(r_knd) :: sum_hv

    REAL(r_knd), DIMENSION(nang) :: psi, pc, den

    REAL(r_knd), DIMENSION(nang,4) :: hv, fxhv
!_______________________________________________________________________
!
!   Set up the sweep order given octant info.
!_______________________________________________________________________

    IF ( id == 1 ) THEN
      ist = -1
    ELSE
      ist = +1
    END IF

    IF ( jd == 1 ) THEN
      jst = -1; jlo = ny; jhi = 1
    ELSE
      jst = +1; jlo = 1; jhi = ny
    END IF

    IF ( kd == 1 ) THEN
      kst = -1; klo = nz; khi = 1
    ELSE
      kst = +1; klo = 1; khi = nz
    END IF
!_______________________________________________________________________
!
!   Initialize the fixup counter
!_______________________________________________________________________

    fxhv = zero
!_______________________________________________________________________
!
!   Process boundaries in y- and z-directions come from preset boundary
!   conditions (vacuum) or from upstream neighbors. Must call to receive
!   any data. If upstream is boundary, no communication.
!_______________________________________________________________________

    CALL sweep_recv_bdry ( g, jd, kd, iop, t, nthrd, reqs, szreq, nc,  &
      nang, ichunk, ny, nz, jb_in, kb_in )
!_______________________________________________________________________
!
!   Loop over cells along the diagonals. When only 1 diagonal, it's
!   normal sweep order. Otherwise, nested threading performs mini-KBA.
!_______________________________________________________________________

  !$OMP PARALLEL NUM_THREADS(nnested) IF(nnested>1) DEFAULT(SHARED)    &
  !$OMP& FIRSTPRIVATE(fxhv)
    diagonal_loop: DO d = 1, ndiag

  !$OMP DO SCHEDULE(STATIC,1) PRIVATE(n,ic,i,j,k,l,psi,pc,sum_hv,hv,den)
      line_loop: DO n = 1, diag(d)%len
!_______________________________________________________________________
!
!       Set the indices chunk indices and the global i index
!_______________________________________________________________________

        ic = diag(d)%cell_id(n)%ic
        IF ( ist < 0 ) ic = ichunk - ic + 1

        i = (ich-1)*ichunk + ic

        j = diag(d)%cell_id(n)%j
        IF ( jst < 0 ) j = ny - j + 1

        k = diag(d)%cell_id(n)%k
        IF ( kst < 0 ) k = nz - k + 1
!_______________________________________________________________________
!
!       Dummy operation to match real transport code where nx is not
!       always even divided by ichunk
!_______________________________________________________________________

        IF ( i > nx ) THEN
          jb_out(:,ic,k) = zero
          kb_out(:,ic,j) = zero
          CYCLE line_loop
        END IF
!_______________________________________________________________________
!
!       Compute the angular source. Add the MMS contribution if
!       necessary.
!_______________________________________________________________________

        psi = qtot(1,ic,j,k)

        DO l = 2, cmom
          psi = psi + ec(:,l)*qtot(l,ic,j,k)
        END DO

        IF ( src_opt == 3 ) psi = psi + qim(:,i,j,k,oct,g)
!_______________________________________________________________________
!
!       Left/right boundary conditions, always vacuum. Dummy operations
!       mayb be ignored.
!_______________________________________________________________________

        ibl = 0
        ibr = 0
        IF ( i==nx .AND. ist==-1 ) THEN
          psii(:,j,k) = zero
        ELSE IF ( i==1 .AND. ist==1 ) THEN
          SELECT CASE ( ibl )
            CASE ( 0 )
              psii(:,j,k) = zero
            CASE ( 1 )
              psii(:,j,k) = zero
          END SELECT
        END IF
!_______________________________________________________________________
!
!       Top/bottom boundary condtions for chunk. Either received from
!       upstream neighbor or set to zero (always vacuum) if no upstream
!       neighbor.
!_______________________________________________________________________

        ibb = 0
        ibt = 0
        IF ( j == jlo ) THEN
          IF ( jd==1 .AND. lasty ) THEN
            psij(:,ic,k) = zero
          ELSE IF ( jd==2 .AND. firsty ) THEN
            SELECT CASE ( ibb )
              CASE ( 0 )
                psij(:,ic,k) = zero
              CASE ( 1 )
                psij(:,ic,k) = zero
            END SELECT
          ELSE
            psij(:,ic,k) = jb_in(:,ic,k)
          END IF
        END IF
!_______________________________________________________________________
!
!       Front/back boundary condtions. Vacuum at global boundaries, but
!       set to some incoming flux from neighboring proc.
!_______________________________________________________________________

        ibf = 0
        ibk = 0
        IF ( k == klo ) THEN
          IF ( (kd==1 .AND. lastz) .OR. ndimen<3 ) THEN
            psik(:,ic,j) = zero
          ELSE IF ( kd==2 .AND. firstz ) THEN
            SELECT CASE ( ibf )
              CASE ( 0 )
                psik(:,ic,j) = zero
              CASE ( 1 )
                psik(:,ic,j) = zero
            END SELECT
          ELSE
            psik(:,ic,j) = kb_in(:,ic,j)
          END IF
        END IF
!_______________________________________________________________________
!
!       Compute the initial solution
!_______________________________________________________________________

        IF ( vdelt /= zero ) THEN
          pc = ( psi + psii(:,j,k)*mu*hi + psij(:,ic,k)*eta*hj +       &
                 psik(:,ic,j)*xi*hk + ptr_in(:,i,j,k)*vdelt ) *        &
               dinv(:,ic,j,k)
        ELSE
          pc = ( psi + psii(:,j,k)*mu*hi + psij(:,ic,k)*eta*hj +       &
            psik(:,ic,j)*xi*hk ) * dinv(:,ic,j,k)
        END IF
!_______________________________________________________________________
!
!       Compute outgoing edges with diamond difference, no negative flux
!       fixup
!_______________________________________________________________________

        IF ( fixup == 0 ) THEN
 
          psi = pc

          psii(:,j,k) = two*psi - psii(:,j,k)
          psij(:,ic,k) = two*psi - psij(:,ic,k)
          IF ( ndimen == 3 ) psik(:,ic,j) = two*psi - psik(:,ic,j)
          IF ( vdelt /= zero )                                         &
            ptr_out(:,i,j,k) = two*psi - ptr_in(:,i,j,k)

        ELSE
!_______________________________________________________________________
!
!         Use negative flux fixup. Compute outgoing edges. If negative,
!         set to zero and rebalance. Multi-pass until all negativities
!         eliminated. Initialize counters and determine angles that will
!         need fixup first.
!_______________________________________________________________________

          hv = one
          sum_hv = SUM( hv )

          fixup_loop: DO

            fxhv(:,1) = two*pc - psii(:,j,k)
            fxhv(:,2) = two*pc - psij(:,ic,k)
            IF ( ndimen == 3 ) fxhv(:,3) = two*pc - psik(:,ic,j)
            IF ( vdelt /= zero ) fxhv(:,4) = two*pc - ptr_in(:,i,j,k)

            WHERE ( fxhv < zero ) hv = zero
!_______________________________________________________________________
!
!           Exit loop when all angles are fixed up, i.e., no hv change
!_______________________________________________________________________

            IF ( sum_hv == SUM( hv ) ) EXIT fixup_loop
            sum_hv = SUM( hv )
!_______________________________________________________________________
!
!           Recompute balance equation numerator and denominator and get
!           new cell average flux
!_______________________________________________________________________

            IF ( vdelt /= zero ) THEN
              pc = psi + half * ( psii(:,j,k)*mu*hi*(one+hv(:,1)) +    &
                                  psij(:,ic,k)*eta*hj*(one+hv(:,2)) +  &
                                  psik(:,ic,j)*xi*hk*(one+hv(:,3)) +   &
                                  ptr_in(:,i,j,k)*vdelt*(one+hv(:,4)) )
              den = t_xs(i,j,k) + mu*hi*hv(:,1) + eta*hj*hv(:,2) +     &
                xi*hk*hv(:,3) + vdelt*hv(:,4)
            ELSE
              pc = psi + half * ( psii(:,j,k)*mu*hi*(one+hv(:,1)) +    &
                                  psij(:,ic,k)*eta*hj*(one+hv(:,2)) +  &
                                  psik(:,ic,j)*xi*hk*(one+hv(:,3)) )
              den = t_xs(i,j,k) + mu*hi*hv(:,1) + eta*hj*hv(:,2) +     &
                xi*hk*hv(:,3)
            END IF

            WHERE( den > tolr )
              pc = pc/den
            ELSEWHERE
              pc = zero
            END WHERE

          END DO fixup_loop
!_______________________________________________________________________
!
!         Fixup done, compute edges with resolved center
!_______________________________________________________________________

          psi = pc

          psii(:,j,k) = fxhv(:,1) * hv(:,1)
          psij(:,ic,k) = fxhv(:,2) * hv(:,2)
          IF ( ndimen == 3 ) psik(:,ic,j) = fxhv(:,3) * hv(:,3)
          IF ( vdelt /= zero ) ptr_out(:,i,j,k) = fxhv(:,4) * hv(:,4)

        END IF
!_______________________________________________________________________
!
!       Save edge fluxes (dummy if checks for unused non-vacuum BCs)
!_______________________________________________________________________

        IF ( j == jhi ) THEN
          IF ( jd==2 .AND. lasty ) THEN
            CONTINUE
          ELSE IF ( jd==1 .AND. firsty ) THEN
            IF ( ibb == 1 ) CONTINUE
          ELSE
            jb_out(:,ic,k) = psij(:,ic,k)
          END IF
        END IF

        IF ( k == khi ) THEN
          IF ( kd==2 .AND. lastz ) THEN
            CONTINUE
          ELSE IF ( kd==1 .AND. firstz ) THEN
            IF ( ibf == 1 ) CONTINUE
          ELSE
            kb_out(:,ic,j) = psik(:,ic,j)
          END IF
        END IF
!_______________________________________________________________________
!
!       Compute dummy leakages (not used elsewhere currently)
!_______________________________________________________________________

        flkx(i+id-1,j,k) = flkx(i+id-1,j,k) + ist*SUM( wmu*psii(:,j,k) )
        flky(i,j+jd-1,k) = flky(i,j+jd-1,k) +                          &
                           jst*SUM( weta*psij(:,ic,k) )
        flkz(i,j,k+kd-1) = flkz(i,j,k+kd-1) +                          &
                           kst*SUM( wxi*psik(:,ic,j) )
!_______________________________________________________________________
!
!       Compute the flux moments
!_______________________________________________________________________

        psi = w*psi

        IF ( oct == 1 ) THEN
          flux0(i,j,k) = SUM( psi )
          DO l = 1, cmom-1
            fluxm(l,i,j,k) = SUM( ec(:,l+1)*psi )
          END DO
        ELSE
          flux0(i,j,k) = flux0(i,j,k) + SUM( psi )
          DO l = 1, cmom-1
            fluxm(l,i,j,k) = fluxm(l,i,j,k) + SUM( ec(:,l+1)*psi )
          END DO
        END IF
!_______________________________________________________________________
!
!       Calculate dummy min and max scalar fluxes (not used elsewhere
!       currently)
!_______________________________________________________________________

        IF ( oct == noct ) THEN
          fmin = MIN( fmin, flux0(i,j,k) )
          fmax = MAX( fmax, flux0(i,j,k) )
        END IF
!_______________________________________________________________________
!
!       Finish the loops
!_______________________________________________________________________

      END DO line_loop
  !$OMP END DO

    END DO diagonal_loop

  !$OMP END PARALLEL
!_______________________________________________________________________
!
!   Send data to downstream neighbors
!_______________________________________________________________________

    CALL sweep_send_bdry ( g, jd, kd, iop, t, nthrd, reqs, szreq, nc,  &
      nang, ichunk, ny, nz, jb_out, kb_out )
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE mkba_sweep


END MODULE mkba_sweep_module
