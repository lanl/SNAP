/***********************************************************************
 * Module: geom.c
 * This module contains the variables that relate to the geometry of the
 * problem and the subroutines necessary to allocate and deallocate
 * geometry related data as necessary.
 ***********************************************************************/
#include "snap.h"

void geom_data_init ( geom_data *geom_vars )
{
    NY_GL = 0;
    NZ_GL = 0;
    JLB   = 0;
    JUB   = 0;
    KLB   = 0;
    KUB   = 0;
    NC    = 0;
    NDIAG = 0;

    DX    = 0;
    DY    = 0;
    DZ    = 0;
    HI    = 0;

    HJ    = NULL;
    HK    = NULL;
    DINV  = NULL;
}

/***********************************************************************
 * Allocate the geometry-related solution arrays.
 ***********************************************************************/
void geom_alloc ( input_data *input_vars, geom_data *geom_vars, int *ierr  )
{
    ALLOC_1D(HJ, NANG, double, ierr);
    ALLOC_1D(HK, NANG, double, ierr);
    ALLOC_5D(DINV, NANG, NX, NY, NZ, NG, double, ierr);
}

/***********************************************************************
 * Dellocate the geometry-related solution arrays.
 ***********************************************************************/
void geom_dealloc ( geom_data *geom_vars )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int i;

/***********************************************************************
 * Deallocate the sweep parameters
 ***********************************************************************/
    FREE(HJ);
    FREE(HK);
    FREE(DINV);

/***********************************************************************
 * Deallocate the diagonal related arrays
 ***********************************************************************/
    for ( i = 1; i <= NDIAG; i++ )
    {
        FREE(DIAG_1D(i-1).cell_id_vars);
    }

    FREE(DIAG);
}

/***********************************************************************
 * Calculate the DD spatial coefficients hi, hj, hk for all angles at
 * the start of each time step. Compute the pre-computed/inverted dinv.
 ***********************************************************************/
void param_calc ( input_data *input_vars, sn_data *sn_vars,
                  solvar_data *solvar_vars, data_data *data_vars,
                  geom_data *geom_vars, int ng_indx )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int i, j, k, m;

/***********************************************************************
 * Set the number of i-chunks
 ***********************************************************************/
    NC = NX / ICHUNK;

/***********************************************************************
 * Set the DD coefficients
 ***********************************************************************/
    HI = 2.0 / DX;

    if ( NDIMEN > 1 )
    {
        for ( i = 0; i < NANG; i++ )
        {
            HJ_1D(i) = (2.0/DY) * ETA_1D(i);
        }

        if ( NDIMEN > 2)
        {
            for ( i = 0; i < NANG; i++ )
            {
                HK_1D(i) = (2.0/DZ) * XI_1D(i);
            }
        }
    }

/***********************************************************************
 * Compute the inverted denominator, saved for sweep
 ***********************************************************************/
// TODO: impliment USEMKL
    for ( k = 1; k <= NZ; k++ )
    {
        for ( j = 1; j <= NY; j++ )
        {
            for ( i = 1; i <= NX; i++ )
            {
                for ( m = 1; m <= NANG; m++ )
                {
                    DINV_5D((m-1),(i-1),(j-1),(k-1), (ng_indx-1)) =
                        1.0 / (T_XS_4D((i-1),(j-1),(k-1),(ng_indx-1))
                               + VDELT_1D(ng_indx-1)
                               + MU_1D(m-1) * HI
                               + HJ_1D(m-1)
                               + HK_1D(m-1));
                }
            }
        }
    }
}

void diag_setup ( input_data *input_vars, para_data *para_vars,
                  geom_data *geom_vars, int *ierr, char **error )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int i, j, k, nn, ing;
    int *indx;

    *ierr = 0;
/***********************************************************************
 * Set up the diagonal indices according to do_nested. If 1, use
 * mini-KBA sweeps and thus allocate many diagonals.
 ***********************************************************************/
    if ( DO_NESTED )
    {
        NDIAG = ICHUNK + NY + NZ - 2;

        ALLOC_1D(DIAG, NDIAG, diag_type, ierr);
        ALLOC_1D(indx, NDIAG, int, ierr);

        if ( *ierr != 0 ) return;

        for ( i = 0; i < NDIAG; i++ )
        {
            DIAG_1D(i).lenc = 0;
            indx[i] = 0;
        }

/***********************************************************************
 * Cells of same diagonal all have same value according to i+j+k-2
 * formula. Use that to compute len for each diagonal. Use ichunk.
 ***********************************************************************/
        for ( k = 1; k <= NZ; k++ )
        {
            for ( j = 1; j <= NY; j++ )
            {
                for ( i = 1; i <= ICHUNK; i++ )
                {
                    nn = i + j + k - 2;
                    DIAG_1D(nn-1).lenc += 1;
                }
            }
        }

/***********************************************************************
 * Next allocate cell_id array within diag type according to len
 ***********************************************************************/
        for ( nn = 1; nn <= NDIAG; nn++ )
        {
            ing = DIAG_1D(nn-1).lenc;
            ALLOC_1D(DIAG_1D(nn-1).cell_id_vars, ing, cell_id_type, ierr);

            if ( *ierr != 0 ) return;
        }

/***********************************************************************
 * Lastly, set each cell's actual ijk indices in this diagonal map
 ***********************************************************************/
        for ( k = 1; k <= NZ; k++ )
        {
            for ( j = 1; j <= NY; j++ )
            {
                for ( i = 1; i <= ICHUNK; i++ )
                {
                    nn = i + j + k - 2;
                    indx[nn-1] += 1;
                    ing = indx[nn-1];
                    DIAG_1D(nn-1).cell_id_vars[ing-1].ic = i;
                    DIAG_1D(nn-1).cell_id_vars[ing-1].jc = j;
                    DIAG_1D(nn-1).cell_id_vars[ing-1].kc = k;
                }
            }
        }

        FREE(indx);
    }

    else
    {
/***********************************************************************
 * Otherwise, use standard sweep map. No mini-KBA. One "diagonal",
 * which contains all the cells in typical i, then j, then k
 * lexographical order.
 ***********************************************************************/
        NDIAG = 1;
        ALLOC_1D(DIAG, 1, diag_type, ierr);

        if ( *ierr != 0) return;

        ALLOC_1D(DIAG_1D(0).cell_id_vars, (ICHUNK*NY*NZ), cell_id_type, ierr);

        if ( *ierr != 0) return;

        DIAG_1D(0).lenc = ICHUNK*NY*NZ;

        ing = 0;

        for ( k = 1; k <= NZ; k++ )
        {
            for ( j = 1; j <= NY; j++ )
            {
                for ( i = 1; i <= ICHUNK; i++ )
                {
                    ing += 1;
                    DIAG_1D(0).cell_id_vars[ing-1].ic = i;
                    DIAG_1D(0).cell_id_vars[ing-1].jc = j;
                    DIAG_1D(0).cell_id_vars[ing-1].kc = k;
                }
            }
        }
    }
}
