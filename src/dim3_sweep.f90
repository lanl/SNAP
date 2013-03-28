!-----------------------------------------------------------------------
!
! MODULE: dim3_sweep_module
!> @brief
!> This module contains the 2D and 3D mesh sweep logic.
!
!-----------------------------------------------------------------------

MODULE dim3_sweep_module

  USE global_module, ONLY: i_knd, r_knd, zero, two, one, half

  USE plib_module, ONLY: ichunk, firsty, lasty, firstz, lastz,         &
    nnested

  USE geom_module, ONLY: nx, hi, hj, hk, ndimen, ny, nz, ndiag, diag

  USE sn_module, ONLY: cmom, nang, mu, w, noct

  USE data_module, ONLY: src_opt, qim

  USE control_module, ONLY: fixup, tolr

  IMPLICIT NONE

  PUBLIC :: dim3_sweep

  SAVE
!_______________________________________________________________________
!
! Module variable
!
! fmin        - min scalar flux. Dummy for now, not used elsewhere.
! fmax        - max scalar flux. Dummy for now, not used elsewhere.
!
!_______________________________________________________________________

  REAL(r_knd) :: fmin=zero, fmax=zero


  CONTAINS


  SUBROUTINE dim3_sweep ( ich, id, d1, d2, d3, d4, jd, kd, jlo, klo,   &
    oct, g, jhi, khi, jst, kst, psii, psij, psik, qtot, ec, vdelt,     &
    ptr_in, ptr_out, dinv, flux, fluxm, jb_in, jb_out, kb_in, kb_out,  &
    wmu, weta, wxi, flkx, flky, flkz, t_xs )

!-----------------------------------------------------------------------
!
! 3-D slab mesh sweeper.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: ich, id, d1, d2, d3, d4, jd, kd,     &
      jlo, klo, oct, g, jhi, khi, jst, kst

    REAL(r_knd), INTENT(IN) :: vdelt

    REAL(r_knd), DIMENSION(nang), INTENT(IN) :: wmu, weta, wxi

    REAL(r_knd), DIMENSION(nang,cmom), INTENT(IN) :: ec

    REAL(r_knd), DIMENSION(nang,ny,nz), INTENT(INOUT) :: psii

    REAL(r_knd), DIMENSION(nang,ichunk,nz), INTENT(INOUT) :: psij

    REAL(r_knd), DIMENSION(nang,ichunk,ny), INTENT(INOUT) :: psik

    REAL(r_knd), DIMENSION(nx,ny,nz), INTENT(IN) :: t_xs

    REAL(r_knd), DIMENSION(nx,ny,nz), INTENT(INOUT) :: flux

    REAL(r_knd), DIMENSION(nang,ichunk,nz), INTENT(IN) :: jb_in

    REAL(r_knd), DIMENSION(nang,ichunk,nz), INTENT(OUT) :: jb_out

    REAL(r_knd), DIMENSION(nang,ichunk,ny), INTENT(IN) :: kb_in

    REAL(r_knd), DIMENSION(nang,ichunk,ny), INTENT(OUT) :: kb_out

    REAL(r_knd), DIMENSION(nx+1,ny,nz), INTENT(INOUT) :: flkx

    REAL(r_knd), DIMENSION(nx,ny+1,nz), INTENT(INOUT) :: flky

    REAL(r_knd), DIMENSION(nx,ny,nz+1), INTENT(INOUT) :: flkz

    REAL(r_knd), DIMENSION(nang,nx,ny,nz), INTENT(IN) :: dinv

    REAL(r_knd), DIMENSION(cmom,nx,ny,nz), INTENT(IN) :: qtot

    REAL(r_knd), DIMENSION(cmom-1,nx,ny,nz), INTENT(INOUT) :: fluxm

    REAL(r_knd), DIMENSION(d1,d2,d3,d4), INTENT(IN) :: ptr_in

    REAL(r_knd), DIMENSION(d1,d2,d3,d4), INTENT(OUT) :: ptr_out
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: ist, d, n, ic, i, j, k, l, ibl, ibr, ibb, ibt,   &
      ibf, ibk

    REAL(r_knd) :: sum_hv

    REAL(r_knd), DIMENSION(nang) :: psi, pc, den

    REAL(r_knd), DIMENSION(nang,4) :: hv, fxhv
!_______________________________________________________________________
!
!   Set up the sweep order in the i-direction.
!_______________________________________________________________________

    ist = -1
    IF ( id == 2 ) ist = 1
!_______________________________________________________________________
!
!   Zero out the outgoing boundary arrays and fixup array
!_______________________________________________________________________

    jb_out = zero
    kb_out = zero

    fxhv = zero
!_______________________________________________________________________
!
!   Loop over cells along the diagonals. When only 1 diagonal, it's
!   normal sweep order. Otherwise, nested threading performs mini-KBA.
!_______________________________________________________________________

  !$OMP PARALLEL NUM_THREADS(nnested) DEFAULT(SHARED) FIRSTPRIVATE(fxhv)

    diagonal_loop: DO d = 1, ndiag

  !$OMP DO SCHEDULE(STATIC,1) PRIVATE(n,ic,i,j,k,l,psi,pc,sum_hv,hv,den)
      line_loop: DO n = 1, diag(d)%len

        ic = diag(d)%cell_id(n)%ic

        IF ( ist < 0 ) THEN
          i = ich*ichunk - ic + 1
        ELSE
          i = (ich-1)*ichunk + ic
        END IF

        IF ( i > nx ) CYCLE line_loop

        j = diag(d)%cell_id(n)%j
        IF ( jst < 0 ) j = ny - j + 1

        k = diag(d)%cell_id(n)%k
        IF ( kst < 0 ) k = nz - k + 1
!_______________________________________________________________________
!
!       Left/right boundary conditions, always vacuum.
!_______________________________________________________________________

        ibl = 0; ibr = 0
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
!       Top/bottom boundary condtions. Vacuum at global boundaries, but
!       set to some incoming flux from neighboring proc.
!_______________________________________________________________________

        ibb = 0; ibt = 0
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

        ibf = 0; ibk = 0
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
!       Compute the angular source
!_______________________________________________________________________

        psi = qtot(1,i,j,k)
        IF ( src_opt == 3 ) psi = psi + qim(:,i,j,k,oct,g)

        DO l = 2, cmom
          psi = psi + ec(:,l)*qtot(l,i,j,k)
        END DO
!_______________________________________________________________________
!
!       Compute the numerator for the update formula
!_______________________________________________________________________

        pc = psi + psii(:,j,k)*mu*hi + psij(:,ic,k)*hj + psik(:,ic,j)*hk
        IF ( vdelt /= zero ) pc = pc + vdelt*ptr_in(:,i,j,k)
!_______________________________________________________________________
!
!       Compute the solution of the center. Use DD for edges. Use fixup
!       if requested.
!_______________________________________________________________________

        IF ( fixup == 0 ) THEN
 
          psi = pc*dinv(:,i,j,k)

          psii(:,j,k) = two*psi - psii(:,j,k)
          psij(:,ic,k) = two*psi - psij(:,ic,k)
          IF ( ndimen == 3 ) psik(:,ic,j) = two*psi - psik(:,ic,j)
          IF ( vdelt /= zero )                                         &
            ptr_out(:,i,j,k) = two*psi - ptr_in(:,i,j,k)

        ELSE
!_______________________________________________________________________
!
!         Multi-pass set to zero + rebalance fixup. Determine angles
!         that will need fixup first.
!_______________________________________________________________________

          hv = one; sum_hv = SUM( hv )

          pc = pc * dinv(:,i,j,k)

          fixup_loop: DO

            fxhv(:,1) = two*pc - psii(:,j,k)
            fxhv(:,2) = two*pc - psij(:,ic,k)
            IF ( ndimen == 3 ) fxhv(:,3) = two*pc - psik(:,ic,j)
            IF ( vdelt /= zero ) fxhv(:,4) = two*pc - ptr_in(:,i,j,k)

            WHERE ( fxhv < zero ) hv = zero
!_______________________________________________________________________
!
!           Exit loop when all angles are fixed up
!_______________________________________________________________________

            IF ( sum_hv == SUM( hv ) ) EXIT fixup_loop
            sum_hv = SUM( hv )
!_______________________________________________________________________
!
!           Recompute balance equation numerator and denominator and get
!           new cell average flux
!_______________________________________________________________________

            pc = psii(:,j,k)*mu*hi*(one+hv(:,1)) +                     &
              psij(:,ic,k)*hj*(one+hv(:,2)) +                          &
              psik(:,ic,j)*hk*(one+hv(:,3))
            IF ( vdelt /= zero )                                       &
              pc = pc + vdelt*ptr_in(:,i,j,k)*(one+hv(:,4))
            pc = psi + half*pc

            den = t_xs(i,j,k) + mu*hi*hv(:,1) + hj*hv(:,2) +           &
              hk*hv(:,3) + vdelt*hv(:,4)

            WHERE( den > tolr )
              pc = pc/den
            ELSEWHERE
              pc = zero
            END WHERE

          END DO fixup_loop
!_______________________________________________________________________
!
!         Fixup done, compute edges
!_______________________________________________________________________

          psi = pc

          psii(:,j,k) = fxhv(:,1) * hv(:,1)
          psij(:,ic,k) = fxhv(:,2) * hv(:,2)
          IF ( ndimen == 3 ) psik(:,ic,j) = fxhv(:,3) * hv(:,3)
          IF ( vdelt /= zero ) ptr_out(:,i,j,k) = fxhv(:,4) * hv(:,4)

        END IF
!_______________________________________________________________________
!
!       Clear the flux arrays
!_______________________________________________________________________

        IF ( oct == 1 ) THEN
          flux(i,j,k) = zero
          fluxm(:,i,j,k) = zero
        END IF
!_______________________________________________________________________
!
!       Compute the flux moments
!_______________________________________________________________________

        flux(i,j,k) = flux(i,j,k) + SUM( w*psi )
        DO l = 1, cmom-1
          fluxm(l,i,j,k) = fluxm(l,i,j,k) + SUM( ec(:,l+1)*w*psi )
        END DO
!_______________________________________________________________________
!
!       Calculate min and max scalar fluxes (not used elsewhere
!       currently)
!_______________________________________________________________________

        IF ( oct == noct ) THEN
          fmin = MIN( fmin, flux(i,j,k) )
          fmax = MAX( fmax, flux(i,j,k) )
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
!       Compute leakages (not used elsewhere currently)
!_______________________________________________________________________

        IF ( i+id-1==1 .OR. i+id-1==nx+1 ) THEN
          flkx(i+id-1,j,k) = flkx(i+id-1,j,k) +                        &
            ist*SUM( wmu*psii(:,j,k) )
        END IF

        IF ( (jd==1 .AND. firsty) .OR. (jd==2 .AND. lasty) ) THEN
          flky(i,j+jd-1,k) = flky(i,j+jd-1,k) +                        &
            jst*SUM( weta*psij(:,ic,k) )
        END IF

        IF ( ((kd==1 .AND. firstz) .OR. (kd==2 .AND. lastz)) .AND.     &
             ndimen==3 ) THEN
          flkz(i,j,k+kd-1) = flkz(i,j,k+kd-1) +                        &
            kst*SUM( wxi*psik(:,ic,j) )
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
!_______________________________________________________________________

  END SUBROUTINE dim3_sweep


END MODULE dim3_sweep_module
