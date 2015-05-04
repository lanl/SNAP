/***********************************************************************
 * Module: solvar.c
 *
 * This module contains several variables that are used in the solution
 * process, including their allocation and deallocation. Also includes
 * initialization of sweep parameters.
 ***********************************************************************/
#include "snap.h"

void solvar_data_init ( solvar_data *solvar_vars )
{
    FLUX    = NULL;
    FLUXPO  = NULL;
    FLUXPI  = NULL;
    T_XS    = NULL;
    A_XS    = NULL;
    PSII    = NULL;
    PSIJ    = NULL;
    PSIK    = NULL;
    JB_IN   = NULL;
    JB_OUT  = NULL;
    KB_IN   = NULL;
    KB_OUT  = NULL;
    FLKX    = NULL;
    FLKY    = NULL;
    FLKZ    = NULL;
    QTOT    = NULL;
    Q2GRP   = NULL;
    FLUXM   = NULL;
    S_XS    = NULL;
    PTR_IN  = NULL;
    PTR_OUT = NULL;
}

/***********************************************************************
 * Allocate solution arrays.
 ***********************************************************************/
void solvar_alloc ( input_data *input_vars, sn_data* sn_vars,
                    solvar_data *solvar_vars, int *ierr )
{
/***********************************************************************
 * Allocate ptr_in/out if needed. Provide an initial condition of zero
 * This may be changed in the future if necessary.
 ***********************************************************************/
    if ( TIMEDEP == 1 )
    {
        ALLOC_6D(PTR_IN,  NANG, NX, NY, NZ, NOCT, NG, double, ierr);
        ALLOC_6D(PTR_OUT, NANG, NX, NY, NZ, NOCT, NG, double, ierr);
    }

/***********************************************************************
 * Allocate the flux moments arrays. Keep an old copy.
 ***********************************************************************/
    ALLOC_4D(FLUX,   NX,      NY, NZ, NG, double, ierr);
    ALLOC_4D(FLUXPO, NX,      NY, NZ, NG, double, ierr);
    ALLOC_4D(FLUXPI, NX,      NY, NZ, NG, double, ierr);

    if ( (CMOM-1) != 0 )
    {
        ALLOC_5D(FLUXM, (CMOM-1), NX, NY, NZ, NG, double, ierr);
    }

/***********************************************************************
 * Allocate the source arrays.
 ***********************************************************************/
    ALLOC_5D(Q2GRP, CMOM, NX, NY, NZ, NG, double, ierr);
    ALLOC_5D(QTOT,  CMOM, NX, NY, NZ, NG, double, ierr);

/***********************************************************************
 * Allocate the cross section expanded to spatial mesh arrays
 ***********************************************************************/
    ALLOC_4D(T_XS, NX,   NY, NZ, NG, double, ierr);
    ALLOC_4D(A_XS, NX,   NY, NZ, NG, double, ierr);
    ALLOC_5D(S_XS, NMOM, NX, NY, NZ, NG, double, ierr);

/***********************************************************************
 * Working arrays
 ***********************************************************************/
    ALLOC_4D(PSII, NANG, NY,     NZ, NG, double, ierr);
    ALLOC_4D(PSIJ, NANG, ICHUNK, NZ, NG, double, ierr);
    ALLOC_4D(PSIK, NANG, ICHUNK, NY, NG, double, ierr);

/***********************************************************************
 * PE boundary flux arrays
 ***********************************************************************/
    ALLOC_4D(JB_IN,  NANG, ICHUNK, NZ, NG, double, ierr);
    ALLOC_4D(JB_OUT, NANG, ICHUNK, NZ, NG, double, ierr);
    ALLOC_4D(KB_IN,  NANG, ICHUNK, NY, NG, double, ierr);
    ALLOC_4D(KB_OUT, NANG, ICHUNK, NY, NG, double, ierr);

/***********************************************************************
 * Leakage arrays
 ***********************************************************************/
    ALLOC_4D(FLKX, (NX+1),     NY,     NZ, NG, double, ierr);
    ALLOC_4D(FLKY,     NX, (NY+1),     NZ, NG, double, ierr);
    ALLOC_4D(FLKZ,     NX,     NY, (NZ+1), NG, double, ierr);
}

void solvar_dealloc ( solvar_data *solvar_vars )
{
    FREE(FLUX);
    FREE(FLUXPO);
    FREE(FLUXPI);
    FREE(T_XS);
    FREE(A_XS);
    FREE(PSII);
    FREE(PSIJ);
    FREE(PSIK);
    FREE(JB_IN);
    FREE(JB_OUT);
    FREE(KB_IN);
    FREE(KB_OUT);
    FREE(FLKX);
    FREE(FLKY);
    FREE(FLKZ);
    FREE(QTOT);
    FREE(Q2GRP);
    FREE(FLUXM);
    FREE(S_XS);
    FREE(PTR_IN);
    FREE(PTR_OUT);
}
