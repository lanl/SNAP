
/***********************************************************************
 * Module: sn.c
 * This module contains the variables and setup subroutines related to
 * the mock discrete ordinates treament in SNAP.
 ***********************************************************************/
#include "snap.h"

void sn_data_init( sn_data *sn_vars )
{
    CMOM = 0;
    NOCT = 0;
}

/***********************************************************************
 * Allocate sn_module arrays.
 ***********************************************************************/
void sn_allocate ( sn_data *sn_vars, input_data *input_vars, int *ierr )
{
/***********************************************************************
 * Allocate cosines and weights.
 ***********************************************************************/
    CMOM = NMOM;
    NOCT = 2;

    // Allocate size nang for 1D mu array, w array, and wmu
    ALLOC_1D(MU,  NANG, double, ierr);
    ALLOC_1D(W,   NANG, double, ierr);
    ALLOC_1D(WMU, NANG, double, ierr);

    if ( *ierr != 0 ) return;

    if ( NDIMEN > 1 )
    {
        CMOM = NMOM * (NMOM+1) / 2;
        NOCT = 4;

        ALLOC_1D(ETA,  NANG, double, ierr);
        ALLOC_1D(WETA, NANG, double, ierr);
    }

    if ( *ierr != 0 ) return;

    if ( NDIMEN > 2 )
    {
        CMOM = NMOM * NMOM;
        NOCT = 8;
        ALLOC_1D(XI,  NANG, double, ierr);
        ALLOC_1D(WXI, NANG, double, ierr);
    }

    if ( *ierr != 0 ) return;

    ALLOC_3D(EC, NANG, CMOM, NOCT, double, ierr);
    ALLOC_1D(LMA, NMOM, int, ierr);

    if ( *ierr != 0 ) return;
}

void sn_deallocate ( sn_data *sn_vars )
{
    FREE(MU);
    FREE(W);
    FREE(WMU);
    FREE(ETA);
    FREE(WETA);
    FREE(XI);
    FREE(WXI);
    FREE(EC);
    FREE(LMA);
}

/***********************************************************************
 * Compute and store the scattering expansion coefficients. Coefficient
 * polynomial is (mu*eta*xi)^l, where l is the moment, starting at 0.
 ***********************************************************************/
void expcoeff ( input_data *input_vars, sn_data *sn_vars, int *ndimen )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int id, jd, kd, is, js, ks, l, oct, m, mom;
    int pp;

/***********************************************************************
 * Set the coefficient as simply the angle raised to a power equal to
 * the moment-1. Set octant and loop according to dimension.
 ***********************************************************************/
    if ( NDIMEN == 1 )
    {
        for ( pp = 0; pp < NANG; pp++)
        {
            WMU_1D(pp) = W_1D(pp)*MU_1D(pp);
        }

        for ( pp = 0; pp < NMOM; pp++)
        {
            LMA_1D(pp) = 1;
        }

        for ( id = 1; id <= 2; id++ )
        {
            is = -1;
            if ( id == 2 ) is = 1;

            for ( pp = 0; pp < NANG; pp++)
            {
                EC_3D(pp,0,(id-1)) = 1;
            }

            for ( l = 2; l <= NMOM; l++ )
            {
                for ( pp = 0; pp < NANG; pp++)
                    EC_3D(pp,(l-1),(id-1)) = pow((is*MU_1D(pp)),(2*l-3));
            }
        }
    }

    else if ( NDIMEN == 2 )
    {
        for ( pp = 0; pp < NANG; pp++)
        {
            WMU_1D(pp) = W_1D(pp)*MU_1D(pp);
            WETA_1D(pp) = W_1D(pp)*ETA_1D(pp);
        }

        for ( l = 1; l <= NMOM; l++ )
            LMA_1D(l-1) = l;

        for ( jd = 1; jd <= 2; jd++ )
        {
            js = -1;
            if ( jd == 2 ) js = 1;

            for ( id = 1; id <= 2; id++ )
            {
                is = -1;
                if ( id == 2 ) is = 1;
                oct = 2*(jd-1) + id;

                for ( pp = 0; pp < NANG; pp++)
                    EC_3D(pp,0,(oct-1)) = 1;

                mom = 2;

                for ( l = 2; l <= NMOM; l++ )
                {
                    for ( m = 1; m <= l; m++ )
                    {
                        for ( pp = 0; pp < NANG; pp++)
                        {
                            EC_3D(pp,(mom-1),(oct-1)) =
                                pow(is*MU_1D(pp), (2*l - 3))
                                * pow(js*ETA_1D(pp), (m-1));
                        }
                        mom = mom + 1;
                    }
                }
            }

        }
    }

    else if ( NDIMEN == 3 )
    {
        for ( pp = 0; pp < NANG; pp++)
        {
            WMU_1D(pp)  = W_1D(pp)*MU_1D(pp);
            WETA_1D(pp) = W_1D(pp)*ETA_1D(pp);
            WXI_1D(pp)  = W_1D(pp)*XI_1D(pp);
        }

        for ( l = 1; l <= NMOM; l++ )
        {
            LMA_1D(l-1) = 2*l - 1;
        }

        for ( kd = 1; kd <= 2; kd ++ )
        {
            ks = -1;
            if ( kd == 2 ) ks = 1;

            for ( jd = 1; jd <= 2; jd++)
            {
                js = -1;

                if ( jd == 2 ) js = 1;

                for ( id = 1; id <= 2; id++ )
                {
                    is = -1;

                    if ( id == 2 ) is = 1;

                    oct = 4*(kd-1) + 2*(jd-1) + id;

                    for ( pp = 0; pp < NANG; pp++)
                        EC_3D(pp,0,(oct-1)) = 1;

                    mom = 2;

                    for ( l = 2; l <= NMOM; l++ )
                    {
                        for ( m = 1; m <= (2*l-1); m++)
                        {
                            for ( pp = 0; pp < NANG; pp++)
                            {
                                EC_3D(pp,(mom-1),(oct-1)) =
                                    pow(is*MU_1D(pp),(2*l-3))
                                    * pow(ks*XI_1D(pp)*js*ETA_1D(pp),(m-1));
                            }

                            mom = mom + 1;
                        }
                    }
                }
            }
        }
    }
}
