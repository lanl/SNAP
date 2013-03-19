!-----------------------------------------------------------------------
!
! MODULE: expxs_module
!> @brief
!> This module contains the subroutines for expanding a cross section to
!> a full spatial map.
!
!-----------------------------------------------------------------------

MODULE expxs_module

  USE global_module, ONLY: i_knd, r_knd, zero

  USE geom_module, ONLY: nx, ny, nz

  USE sn_module, ONLY: nmom

  USE data_module, ONLY: nmat

  IMPLICIT NONE

  PUBLIC

  SAVE


  CONTAINS


  SUBROUTINE expxs_reg ( xs, map, cs )

!-----------------------------------------------------------------------
!
! Expand one of the sig*(nmat,ng) arrays to a spatial mapping. xs is the
! a generic cross section array, map is the material map, and cs is the
! cross section expanded to the mesh.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), DIMENSION(nx,ny,nz), INTENT(IN) :: map

    REAL(r_knd), DIMENSION(nmat), INTENT(IN) :: xs

    REAL(r_knd), DIMENSION(nx,ny,nz), INTENT(OUT) :: cs
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: i, j, k
!_______________________________________________________________________

    cs = zero

    DO k = 1, nz
      DO j = 1, ny
        DO i = 1, nx
          cs(i,j,k) = xs(map(i,j,k))
        END DO
      END DO
    END DO
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE expxs_reg


  SUBROUTINE expxs_slgg ( scat, map, cs )

!-----------------------------------------------------------------------
!
! Expand the slgg(nmat,nmom,ng,ng) array to a spatial mapping. scat
! is the slgg matrix for a single h->g group coupling, map is the
! material map, and cs is the scattering matrix expanded to the mesh.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), DIMENSION(nx,ny,nz), INTENT(IN) :: map

    REAL(r_knd), DIMENSION(nmat,nmom), INTENT(IN) :: scat

    REAL(r_knd), DIMENSION(nmom,nx,ny,nz), INTENT(OUT) :: cs
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: l, i, j, k
!_______________________________________________________________________

    cs = zero

    DO k = 1, nz
      DO j = 1, ny
        DO i = 1, nx
          DO l = 1, nmom
            cs(l,i,j,k) = scat(map(i,j,k),l)
          END DO
        END DO
      END DO
    END DO
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE expxs_slgg


END MODULE expxs_module
