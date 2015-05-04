
/*******************************************************************************
 * Module: data.c
 * This module contains the variables and setup subroutines for the mock
 * cross section data. It establishes the number of groups and constructs
 * the cross section arrays.
 *******************************************************************************/
#include "snap.h"

/*******************************************************************************
 * Constructor for data_data type
 *******************************************************************************/
void data_data_init (data_data *data_vars )
{
    V     = NULL;
    NMAT  = 1;
    MAT   = NULL;
    QI    = NULL;
    QIM   = NULL;
    SIGT  = NULL;
    SIGA  = NULL;
    SIGS  = NULL;
    SLGG  = NULL;
    VDELT = NULL;
}

/*******************************************************************************
 * Allocate data_module arrays.
 *******************************************************************************/
void data_allocate ( data_data *data_vars, input_data *input_vars,
                     sn_data *sn_vars, int *ierr )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int i,j,k;

/*******************************************************************************
 * Establish number of materials according to mat_opt
 *******************************************************************************/
    if ( MAT_OPT > 0 )
    {
        NMAT = 2;
    }

/*******************************************************************************
 * Allocate velocities
 *******************************************************************************/
    if ( TIMEDEP == 1 )
    {
        ALLOC_1D(V, NG, double, ierr);
    }

    if ( *ierr != 0 ) return;

/*******************************************************************************
 * Allocate the material identifier array. ny and nz are 1 if not
 * 2-D/3-D.
 *******************************************************************************/
    ALLOC_3D(MAT, NX, NY, NZ, int, ierr);

    if ( *ierr != 0 ) return;

    for ( k=0; k < NZ; k++ )
    {
        for ( j=0; j < NY; j++ )
        {
            for ( i=0; i < NX; i++ )
            {
                MAT_3D(i, j, k) = 1;
            }
        }
    }

/*******************************************************************************
 * Allocate the fixed source array. If src_opt < 3, allocate the qi
 * array, not the qim. Do the opposite (store the full angular copy) of
 * the source, qim, if src_opt>=3 (MMS). Allocate array not used to 0.
 * ny and nz are 1 if not 2-D/3-D.
 *******************************************************************************/
    if ( SRC_OPT < 3 )
    {
        ALLOC_4D(QI, NX, NY, NZ, NG, double, ierr);

        if ( *ierr != 0 ) return;
    }
    else
    {
        ALLOC_4D(QI, NX, NY, NZ, NG, double, ierr);
        ALLOC_6D(QIM, NANG, NX, NY, NZ, NOCT, NG, double, ierr);

        if ( *ierr != 0 ) return;
    }


/*******************************************************************************
 * Allocate mock cross sections
 *******************************************************************************/
    if (NMAT != 0 )
    {
        ALLOC_2D(SIGT, NMAT, NG, double, ierr);
        ALLOC_2D(SIGA, NMAT, NG, double, ierr);
        ALLOC_2D(SIGS, NMAT, NG, double, ierr);
        ALLOC_4D(SLGG, NMAT, NMOM, NG, NG, double, ierr);

        if ( *ierr != 0 ) return;
    }
    else
    {
        ALLOC_1D(SIGT, NG, double, ierr);
        ALLOC_1D(SIGA, NG, double, ierr);
        ALLOC_1D(SIGS, NG, double, ierr);
        ALLOC_3D(SLGG, NMOM, NG, NG, double, ierr);

        if ( *ierr != 0 ) return;
    }


/*******************************************************************************
 * Allocate the vdelt array
 *******************************************************************************/
    ALLOC_1D(VDELT, NG, double, ierr);

    if ( *ierr != 0 ) return;
}

/*******************************************************************************
 * Deallocate the data module arrays
 *******************************************************************************/
void data_deallocate ( data_data *data_vars )
{
    FREE(V);
    FREE(MAT);
    FREE(QI);
    FREE(QIM);
    FREE(SIGT);
    FREE(SIGA);
    FREE(SIGS);
    FREE(SLGG);
    FREE(VDELT);
}

