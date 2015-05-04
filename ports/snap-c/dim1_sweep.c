/***********************************************************************
 * Module: dim1_sweep.c
 *
 * This module contains the 1D mesh sweep logic.
 ***********************************************************************/
#include "snap.h"

// Local variable array macro
#define PSI_1D(ANG)   psi[ANG]
#define PC_1D(ANG)    pc[ANG]
#define DEN_1D(ANG)   den[ANG]

#ifdef ROWORDER
#define HV_2D(ANG, X) hv[ ANG*2                 \
                          + X ]
#else
#define HV_2D(ANG, X) hv[ X*NANG                \
                          + ANG ]
#endif

#ifdef ROWORDER
#define FXHV_2D(ANG, X) fxhv[ ANG*2             \
                              + X ]
#else
#define FXHV_2D(ANG, X) fxhv[ X*NANG            \
                              + ANG ]
#endif

#ifdef ROWORDER
#define QM_2D(ANG, X) qm[ ANG*NX                \
                          + X ]
#else
#define QM_2D(ANG, X) qm[ X*NANG                \
                          + ANG ]
#endif

// Simplify array indexing when certain values constant throughout module
#define PSII_1D(ANG)             PSII_4D(ANG, 0, 0, (g-1))
#define QTOT_2D(MOM1, X)         QTOT_5D(MOM1, X, 0, 0, (g-1))
#define EC_2D(ANG, MOM1)         EC_3D(ANG, MOM1, (oct-1))
#define VDELT_CONST              VDELT_1D(g-1)
#define PTR_IN_4D(ANG, X, Y, Z)  PTR_IN_6D(ANG, X, Y, Z, (i1-1), (i2-1))
#define PTR_OUT_4D(ANG, X, Y, Z) PTR_OUT_6D(ANG, X, Y, Z, (i1-1), (i2-1))
#define DINV_2D(ANG, X)          DINV_5D(ANG, X, 0, 0, (g-1))
#define FLUX_1D(X)               FLUX_4D(X, 0, 0, (g-1))
#define FLUXM_2D(MOM1, X)        FLUXM_5D(MOM1, X, 0, 0, (g-1))
#define FLKX_1D(X)               FLKX_4D(X, 0, 0, (g-1))
#define T_XS_1D(X)               T_XS_4D(X, 0, 0, (g-1))

void dim1_sweep_data_init ( dim_sweep_data *dim_sweep_vars )
{
    FMIN = 0;
    FMAX = 0;
}

/***********************************************************************
 * 1-D slab mesh sweeper
 ***********************************************************************/
void dim1_sweep ( input_data *input_vars, geom_data *geom_vars,
                  sn_data *sn_vars, data_data *data_vars,
                  control_data *control_vars, solvar_data *solvar_vars,
                  dim_sweep_data *dim_sweep_vars, int id, int oct, int d1,
                  int d2, int d3, int d4, int i1, int i2, int g, int *ierr )
{
// TODO: add mkl/cblas support
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int ilo, ihi, ist, i, l, ii, ang;

    double sum_hv = 0, sum_hv_tmp = 0, sum_wpsi = 0, sum_ecwpsi = 0,
           sum_wmupsii = 0;

    double psi[NANG], pc[NANG], den[NANG];

    for ( ang = 0; ang < NANG; ang++ )
    {
        psi[NANG] = 0;
        pc[NANG]  = 0;
        den[NANG] = 0;
    }

    double hv[NANG*2], fxhv[NANG*2], qm[NANG*NX];

    for ( ang = 0; ang < (NANG*2); ang++ )
    {
        hv[ang]   = 0;
        fxhv[ang] = 0;
    }

    for ( ang = 0; ang < (NANG*NX); ang++ )
    {
        den[ang] = 0;
    }

//    double *psi, *pc, *den; // 1d
//    double *hv, *fxhv; // 2d
//    double *qm; // 2d

//    ALLOC_1D(psi, NANG, double, ierr);
//    ALLOC_1D(pc,  NANG, double, ierr);
//    ALLOC_1D(den, NANG, double, ierr);

//    ALLOC_2D(hv,   NANG, 2, double, ierr);
//    ALLOC_2D(fxhv, NANG, 2, double, ierr);

//    ALLOC_2D(qm, NANG, NX, double, ierr);

/***********************************************************************
 * Set up the mms source if necessary
 ***********************************************************************/
    if ( SRC_OPT == 3)
    {
        for ( ii = 0; ii < NX; ii++ )
        {
            for ( ang = 0; ang < NANG; ang++ )
            {
                QM_2D(ang,ii) = QIM_6D(ang,ii,0,0,(oct-1),(g-1));
            }
        }
    }

/***********************************************************************
 * Set up the sweep order in the i-direction. ilo here set according
 * to direction--different than ilo in octsweep. Setup the fixup
 * counter
 ***********************************************************************/
    if ( id == 1 )
    {
        ilo = NX;
        ihi = 1;
        ist = -1;
    }
    else
    {
        ilo = 1;
        ihi = NX;
        ist = 1;
    }

/***********************************************************************
 * Sweep the i cells. Set the boundary condition.
 ***********************************************************************/
    // i_loop
    for ( i = ilo; i <= ihi; i+=ist )
    {
        if ( i == ilo )
        {
            // PSII passed in as psii(nang)
            for ( ang = 0; ang < NANG; ang++ )
            {
                PSII_1D(ang) = 0;
            }
        }

/***********************************************************************
 * Compute the angular source. MMS source scales linearly with time.
 ***********************************************************************/
        for ( ang = 0; ang < NANG; ang++ )
        {
            PSI_1D(ang) = QTOT_2D(0,(i-1)) + QM_2D(ang,(i-1));
        }

        for ( l = 2; l <= CMOM; l++ )
        {
            for ( ang = 0; ang < NANG; ang++ )
            {
                PSI_1D(ang) += EC_2D(ang,(l-1))*QTOT_2D((l-1),(i-1));
            }
        }

/***********************************************************************
 * Compute the numerator for the update formula
 ***********************************************************************/
        for ( ang = 0; ang < NANG; ang++ )
        {
            PC_1D(ang) = PSI_1D(ang) + PSII_1D(ang)*MU_1D(ang)*HI;

            if ( VDELT_CONST != 0 )
            {
                PC_1D(ang) += VDELT_CONST*PTR_IN_4D(ang,(i-1),0,0);
            }
        }

/***********************************************************************
 * Compute the solution of the center. Use DD for edges. Use fixup
 * if requested.
 ***********************************************************************/
        if ( FIXUP == 0 )
        {
            for ( ang = 0; ang < NANG; ang++ )
            {
                PSI_1D(ang) = PC_1D(ang)*DINV_2D(ang,(i-1));

                PSII_1D(ang) = 2*PSI_1D(ang) - PSII_1D(ang);

                if ( VDELT_CONST != 0 )
                {
                    PTR_OUT_4D(ang,(i-1),0,0)
                        = 2*PSI_1D(ang) - PTR_IN_4D(ang,(i-1),0,0);
                }
            }
        }

        else
        {
/***********************************************************************
 * Multi-pass set to zero + rebalance fixup. Determine angles that
 * will need fixup first.
 ***********************************************************************/
            for ( ang = 0; ang < NANG; ang++ )
            {
                HV_2D(ang, 0) = 1;
                HV_2D(ang, 1) = 1;
                sum_hv += HV_2D(ang, 0) + HV_2D(ang,1);

                PC_1D(ang) *= DINV_2D(ang,(i-1));
            }

            // fixup_loop
            while (true)
            {
                sum_hv_tmp = 0;

                for ( ang = 0; ang < NANG; ang++ )
                {
                    FXHV_2D(ang, 0) = 2*PC_1D(ang) - PSII_1D(ang);

                    if ( VDELT_CONST != 0 )
                    {
                        FXHV_2D(ang, 1)
                            = 2*PC_1D(ang) - PTR_IN_4D(ang,(i-1),0,0);
                    }

                    if ( FXHV_2D(ang, 0) < 0 )
                    {
                        HV_2D(ang, 0) = 0;
                    }
                    if ( FXHV_2D(ang, 1) < 0 )
                    {
                        HV_2D(ang, 1) = 0;
                    }

                    sum_hv_tmp += HV_2D(ang, 0) + HV_2D(ang,1);
                }


/***********************************************************************
 * Exit loop when all angles are fixed up
 ***********************************************************************/
                if ( sum_hv == sum_hv_tmp ) break;

                sum_hv = sum_hv_tmp;

/***********************************************************************
A
 * Recompute balance equation numerator and denominator and get
 * new cell average flux
 ***********************************************************************/
                for ( ang = 0; ang < NANG; ang++ )
                {
                    PC_1D(ang) = PSII_1D(ang)*MU_1D(ang)*HI*(1+HV_2D(ang,0));

                    if ( VDELT_CONST != 0 )
                    {
                        PC_1D(ang) += VDELT_CONST
                                * PTR_IN_4D(ang,(i-1),0,0)
                                * (1+HV_2D(ang,1));
                    }

                    PC_1D(ang) = PSI_1D(ang) + 0.5*PC_1D(ang);

                    DEN_1D(ang) = T_XS_1D(i-1)
                        + MU_1D(ang)*HI*HV_2D(ang,0)
                        + VDELT_CONST*HV_2D(ang,1);

                    if ( DEN_1D(ang) > TOLR )
                    {
                        PC_1D(ang) /= DEN_1D(ang);
                    }
                    else
                    {
                        PC_1D(ang) = 0;
                    }
                }
            } // end fixup_loop

/***********************************************************************
 * Fixup done, compute edges
 ***********************************************************************/
            for ( ang = 0; ang < NANG; ang++ )
            {
                PSI_1D(ang) = PC_1D(ang);

                PSII_1D(ang) = FXHV_2D(ang, 0)*HV_2D(ang, 0);

                if ( VDELT_CONST != 0 )
                {
                    PTR_OUT_4D(ang,(i-1),0,0)
                        = FXHV_2D(ang, 1)*HV_2D(ang, 1);
                }
            }
        }

/***********************************************************************
 * Clear the flux arrays
 ***********************************************************************/
        if ( id == 1 )
        {
            FLUX_1D(i-1) = 0;

            for ( ii = 0; ii < (CMOM-1); ii++ )
            {
                FLUXM_2D(ii,(i-1)) = 0;
            }
        }

/***********************************************************************
 * Compute the flux moments
 ***********************************************************************/
        sum_wpsi = 0;
        for ( ang = 0; ang < NANG; ang++ )
        {
            sum_wpsi += W_1D(ang)*PSI_1D(ang);
        }

        FLUX_1D(i-1) += sum_wpsi;

        for ( l = 1; l <= (CMOM-1); l++ )
        {
            sum_ecwpsi = 0;
            for ( ang = 0; ang < NANG; ang++ )
            {
                sum_ecwpsi += EC_2D(ang,l)*W_1D(ang)*PSI_1D(ang);
            }

            FLUXM_2D((l-1),(i-1)) += sum_ecwpsi;

        }

/***********************************************************************
 * Calculate min and max scalar fluxes (not used elsewhere currently)
 ***********************************************************************/
        if ( id == 2 )
        {
            FMIN = MIN( FMIN, FLUX_1D(i-1) );
            FMAX = MIN( FMAX, FLUX_1D(i-1) );
        }

/***********************************************************************
 * Compute leakages (not used elsewhere currently)
 ***********************************************************************/
        if ( ((i+id-1) == 1) || ((i+id-1) == (NX+1)) )
        {
            sum_wmupsii = 0;
            for ( ang = 0; ang < NANG; ang++ )
            {
                sum_wmupsii += WMU_1D(ang)*PSII_1D(ang);
            }

            FLKX_1D(i-1) += ist*sum_wmupsii;
        }
    } // end i_loop

//    FREE(psi);
//    FREE(pc);
//    FREE(den);
//    FREE(hv);
//    FREE(fxhv);
//    FREE(qm);
}

