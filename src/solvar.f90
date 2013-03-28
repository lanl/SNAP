!-----------------------------------------------------------------------
!
! MODULE: solvar_module
!> @brief
!> This module contains several variables that are used in the solution
!> process, including their allocation and deallocation. Also includes
!> initialization of sweep parameters.
! 
!-----------------------------------------------------------------------

MODULE solvar_module

  USE global_module, ONLY: i_knd, r_knd, zero

  USE plib_module, ONLY: ichunk

  USE geom_module, ONLY: nx, ny, nz

  USE sn_module, ONLY: nang, noct, nmom, cmom

  USE data_module, ONLY: ng

  USE control_module, ONLY: timedep

  IMPLICIT NONE

  PUBLIC

  SAVE
!_______________________________________________________________________
!
! Module variables
!
! ptr_in(nang,nx,ny,nz,noct,ng)   - Incoming time-edge flux pointer
! ptr_out(nang,nx,ny,nz,noct,ng)  - Outgoing time-edge flux pointer
!
! flux(nx,ny,nz,ng)          - Scalar flux moments array
! fluxpo(nx,ny,nz,ng)        - Previous outer copy of scalar flux array
! fluxpi(nx,ny,nz,ng)        - Previous inner copy of scalar flux array
! fluxm(cmom-1,nx,ny,nz,ng)  - Flux moments array
!
! q2grp(cmom,nx,ny,nz,ng)  - Out-of-group scattering + fixed sources
! qtot(cmom,nx,ny,nz,ng)   - Total source: q2grp + within-group source
!
! t_xs(nx,ny,nz,ng)       - Total cross section on mesh
! a_xs(nx,ny,nz,ng)       - Absorption cross section on mesh
! s_xs(nmom,nx,ny,nz,ng)  - In-group scattering cross section on mesh
!
! psii(nang,ny,nz,ng)     - Working psi_x array
! psij(nang,ichunk,nz,ng) - Working psi_y array
! psik(nang,ichunk,ny,ng) - Working psi_z array
!
! jb_in(nang,ichunk,nz,ng)  - y-dir boundary flux in from comm
! jb_out(nang,ichunk,nz,ng) - y-dir boundary flux out to comm
! kb_in(nang,ichunk,ny,ng)  - z-dir boundary flux in from comm
! kb_out(nang,ichunk,ny,ng) - z-dir boundary flux out to comm
!
! flkx(nx+1,ny,nz,ng)     - x-dir leakage array
! flky(nx,ny+1,nz,ng)     - y-dir leakage array
! flkz(nx,ny,nz+1,ng)     - z-dir leakage array
!
!_______________________________________________________________________

  REAL(r_knd), ALLOCATABLE, DIMENSION(:,:,:,:) :: flux, fluxpo, fluxpi,&
    t_xs, a_xs, psii, psij, psik, jb_in, jb_out, kb_in, kb_out, flkx,  &
    flky, flkz

  REAL(r_knd), ALLOCATABLE, DIMENSION(:,:,:,:,:) :: qtot, q2grp, fluxm,&
    s_xs

  REAL(r_knd), DIMENSION(:,:,:,:,:,:), POINTER :: ptr_in, ptr_out


  CONTAINS


  SUBROUTINE solvar_alloc ( ierr )

!-----------------------------------------------------------------------
!
! Allocate solution arrays.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(OUT) :: ierr
!_______________________________________________________________________
!
!   Allocate ptr_in/out if needed. Provide an initial condition of zero
!   This may be changed in the future if necessary.
!_______________________________________________________________________

    ierr = 0

    IF ( timedep == 1 ) THEN
      ALLOCATE( ptr_in(nang,nx,ny,nz,noct,ng),                         &
        ptr_out(nang,nx,ny,nz,noct,ng), STAT=ierr )
    ELSE
      ALLOCATE( ptr_in(0,0,0,0,0,0), ptr_out(0,0,0,0,0,0), STAT=ierr )
    END IF
    IF ( ierr /= 0 ) RETURN

    IF ( timedep == 1 ) THEN
      ptr_in = zero
      ptr_out = zero
    END IF
!_______________________________________________________________________
!
!   Allocate the flux moments arrays. Keep an old copy.
!_______________________________________________________________________

    ALLOCATE( flux(nx,ny,nz,ng), fluxpo(nx,ny,nz,ng),                  &
      fluxpi(nx,ny,nz,ng), fluxm(cmom-1,nx,ny,nz,ng), STAT=ierr )
    IF ( ierr /= 0 ) RETURN

    flux   = zero
    fluxpo = zero
    fluxpi = zero
    fluxm  = zero
!_______________________________________________________________________
!
!   Allocate the source arrays.
!_______________________________________________________________________

    ALLOCATE( q2grp(cmom,nx,ny,nz,ng), qtot(cmom,nx,ny,nz,ng),         &
      STAT=ierr )
    IF ( ierr /= 0 ) RETURN

    q2grp = zero
    qtot = zero
!_______________________________________________________________________
!
!   Allocate the cross section expanded to spatial mesh arrays
!_______________________________________________________________________

    ALLOCATE( t_xs(nx,ny,nz,ng), a_xs(nx,ny,nz,ng),                    &
      s_xs(nmom,nx,ny,nz,ng), STAT=ierr )
    IF ( ierr /= 0 ) RETURN

    t_xs = zero
    a_xs = zero
    s_xs = zero
!_______________________________________________________________________
!
!   Working arrays
!_______________________________________________________________________

    ALLOCATE( psii(nang,ny,nz,ng), psij(nang,ichunk,nz,ng),            &
      psik(nang,ichunk,ny,ng), STAT=ierr )
    IF ( ierr /= 0 ) RETURN

    psii = zero
    psij = zero
    psik = zero
!_______________________________________________________________________
!
!   PE boundary flux arrays
!_______________________________________________________________________

    ALLOCATE( jb_in(nang,ichunk,nz,ng), jb_out(nang,ichunk,nz,ng),     &
      kb_in(nang,ichunk,ny,ng), kb_out(nang,ichunk,ny,ng), STAT=ierr )
    IF ( ierr /= 0 ) RETURN

    jb_in  = zero
    jb_out = zero
    kb_in  = zero
    kb_out = zero
!_______________________________________________________________________
!
!   Leakage arrays
!_______________________________________________________________________

    ALLOCATE( flkx(nx+1,ny,nz,ng), flky(nx,ny+1,nz,ng),                &
      flkz(nx,ny,nz+1,ng), STAT=ierr )
    IF ( ierr /= 0 ) RETURN

    flkx = zero
    flky = zero
    flkz = zero
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE solvar_alloc


  SUBROUTINE solvar_dealloc

!-----------------------------------------------------------------------
!
! Deallocate solve_module arrays.
!
!-----------------------------------------------------------------------
!_______________________________________________________________________

    DEALLOCATE( ptr_in, ptr_out )
    DEALLOCATE( flux, fluxpo, fluxpi, fluxm )
    DEALLOCATE( q2grp, qtot )
    DEALLOCATE( t_xs, a_xs, s_xs )
    DEALLOCATE( psii, psij, psik )
    DEALLOCATE( jb_in, jb_out, kb_in, kb_out )
    DEALLOCATE( flkx, flky, flkz )
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE solvar_dealloc


END MODULE solvar_module
