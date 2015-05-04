/***********************************************************************
 * Module: expxs.c
 *
 * This module contains the subroutines for expanding a cross section to
 * a full spatial map.
 ***********************************************************************/
#include "snap.h"

#ifdef ROWORDER
#define CS_3D(X, Y, Z)                          \
    cs[ X   * NY * NZ                           \
        + Y * NZ                                \
        + Z ]
#else
#define  CS_3D(X, Y, Z)                         \
    cs[ Z   * NY * NX                           \
        + Y * NX                                \
        + X ]
#endif

#define MAP_3D(X, Y, Z) MAT_3D(X, Y, Z)

#define XS_1D(MAT1) SLGG_4D(MAT1, (l_indx-1), (gp-1), (g-1))

/***********************************************************************
 * Expand one of the sig*(nmat,ng) arrays to a spatial mapping. xs is the
 * a generic cross section array, map is the material map, and cs is the
 * cross section expanded to the mesh.
 ***********************************************************************/
void expxs_reg ( input_data *input_vars, data_data *data_vars, solvar_data *solvar_vars,
                 double *cs, int g, int gp, int l_indx )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int i, j, k;

    if ( !cs )
    {
        for ( k = 1; k <= NZ; k++ )
        {
            for ( j = 1; j <= NY; j++ )
            {
                for ( i = 1; i <= NX; i++ )
                {
                    T_XS_4D((i-1),(j-1),(k-1),(g-1))
                        = SIGT_2D(MAT_3D((i-1),(j-1),(k-1)) - 1, (g-1));
                    A_XS_4D((i-1),(j-1),(k-1),(g-1))
                        = SIGA_2D(MAT_3D((i-1),(j-1),(k-1)) - 1, (g-1));
                }
            }
        }
    }
    else
    {
        for ( k = 1; k <= NZ; k++ )
        {
            for ( j = 1; j <= NY; j++ )
            {
                for ( i = 1; i <= NX; i++ )
                {
                    CS_3D((i-1),(j-1),(k-1))
                        = XS_1D(MAP_3D((i-1),(j-1),(k-1)) - 1);
                }
            }
        }
    }
}

/***********************************************************************
 * Expand the slgg(nmat,nmom,ng,ng) array to a spatial mapping. scat
 * is the slgg matrix for a single h->g group coupling, map is the
 * material map, and cs is the scattering matrix expanded to the mesh.
 ***********************************************************************/
void expxs_slgg ( input_data *input_vars, data_data *data_vars,
                  solvar_data *solvar_vars, int g )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int l, i, j, k;

    for ( k = 1; k <= NZ; k++ )
    {
        for ( j = 1; j <= NY; j++ )
        {
            for ( i = 1; i <= NX; i++ )
            {
                for ( l = 1; l <= NMOM; l++ )
                {
                    S_XS_5D((l-1),(i-1),(j-1),(k-1),(g-1))
                        = SLGG_4D(MAT_3D((i-1),(j-1),(k-1))-1, (l-1), (g-1), (g-1));
                }
            }
        }
    }
}
