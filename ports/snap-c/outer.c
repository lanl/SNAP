/***********************************************************************
 * Module: outer.c
 *
 * This module controls the outer iterations. Outer iterations are
 * threaded over the energy dimension and represent a Jacobi iteration
 * strategy. Includes setting the outer source. Checking outer iteration
 * convergence.
 ***********************************************************************/
#include "snap.h"

// Define macro to index local variable tc(nx,ny,nz)
/* tc(nx,ny,nz) */
#ifdef ROWORDER
#define TC_3D(X, Y, Z)                          \
    tc[ X   * NY * NZ                           \
        + Y * NZ                                \
        + Z ]
#else
#define TC_3D(X, Y, Z)                          \
    tc[ Z   * NY * NX                           \
        + Y * NX                                \
        + X ]
#endif

// Define macro to index local variable df(nx,ny,nz,ng)
/* df(nx,ny,nz) */
#ifdef ROWORDER
#define DF_4D(X, Y, Z, G)                       \
    df[ X   * NY * NZ * NG                      \
        + Y * NZ * NG                           \
        + Z * NG                                \
        + G]
#else
#define DF_4D(X, Y, Z, G)                       \
    df[ G   * NZ * NY * NX                      \
        + Z * NY * NX                           \
        + Y * NX                                \
        + X ]
#endif

#define Q_4D(CMOM1, X, Y, Z)  Q2GRP_5D(CMOM1, X, Y, Z, (g-1))
#define CS_2D(MAT1, MOM1)     SLGG_4D(MAT1, MOM1, (gp-1), (g-1))
#define MAP_3D(X, Y, Z)       MAT_3D(X, Y, Z)
#define F_3D(X, Y, Z)         FLUX_4D(X, Y, Z, (gp-1))
#define FM_4D(CMOM1, X, Y, Z) FLUXM_5D(CMOM1, X, Y, Z, (gp-1))


/***********************************************************************
 * Do a single outer iteration. Sets the out-of-group sources, performs
 * inners for all groups.
 ***********************************************************************/
int outer ( input_data *input_vars, para_data *para_vars, geom_data *geom_vars,
            time_data *time_vars, data_data *data_vars, sn_data *sn_vars,
            control_data *control_vars, solvar_data *solvar_vars,
            sweep_data *sweep_vars, dim_sweep_data *dim_sweep_vars,
            FILE *fp_out, int *ierr )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int g, inno, sum_iits = 0;
    int i,j,k;

    int iits[NG];

    double t1, t2, t3, t4;

    bool allinrdone = false;

/***********************************************************************
 * Compute the outer source: sum of fixed + out-of-group sources
 ***********************************************************************/
    t1 = wtime();

    otr_src( input_vars, data_vars, sn_vars, solvar_vars, ierr );

    t2 = wtime();

    TOTRSRC += t2 - t1;

/***********************************************************************
 * Zero out the inner iterations group count. Save the flux for
 * comparison. Parallelize group loop with threads.
 ***********************************************************************/
#pragma omp parallel for schedule(dynamic, 1)
    for ( g = 1; g <= NG; g++ )
    {
        iits[g-1] = 0;
        for ( k = 0; k < NZ; k++ )
        {
            for ( j = 0; j < NY; j++ )
            {
                for ( i = 0; i < NX; i++ )
                {
                    FLUXPO_4D(i,j,k,(g-1)) = FLUX_4D(i,j,k,(g-1));
                }
            }
        }
    }

/***********************************************************************
 * Start the inner iterations
 ***********************************************************************/
    t3 = wtime();

    for ( g = 0; g < NG; g++ )
    {
        INRDONE_1D(g) = false;
    }

    // inner_loop
    for ( inno = 1; inno <= IITM; inno++ )
    {
        inner (input_vars, para_vars, geom_vars, time_vars, data_vars,
               sn_vars, control_vars, solvar_vars, sweep_vars,
               dim_sweep_vars, inno, iits, fp_out, ierr );

        for ( g = 0; g < NG; g++ )
        {
            if ( !INRDONE_1D(g) )
            {
                allinrdone = false;
                break;
            }
            else allinrdone = true;
        }

        if (allinrdone) break;
    } // end inner_loop

// TODO add USEMKL
    sum_iits = 0;
    for ( g = 0; g < NG; g++ )
        sum_iits += iits[g];

    t4 = wtime();

    TINNERS += t4 - t3;

/***********************************************************************
 * Check outer convergence
 ***********************************************************************/
    otr_conv ( input_vars, para_vars, control_vars, solvar_vars, ierr );

    return sum_iits;
}

/***********************************************************************
 * Loop over groups to compute each one's outer loop source.
 ***********************************************************************/
void otr_src ( input_data *input_vars, data_data *data_vars,
               sn_data *sn_vars, solvar_data *solvar_vars, int *ierr )
{
/***********************************************************************
 * Local variable
 ***********************************************************************/
    int g, gp, i, j, k;

/***********************************************************************
 * Initialize the source to fixed. Parallelize outer group loop with
 * threads.
 ***********************************************************************/
#pragma omp parallel for schedule(dynamic, 1) default(shared)   \
    private(g, gp)
    for ( g = 1; g <= NG; g++ )
    {
        for ( k = 0; k < NZ; k++ )
        {
            for ( j = 0; j < NY; j++ )
            {
                for ( i = 0; i < NX; i++ )
                {
                    Q2GRP_5D(0,i,j,k,(g-1)) = QI_4D(i,j,k,(g-1));
                }
            }
        }

/***********************************************************************
 * Loop over cells and moments to compute out-of-group scattering
 ***********************************************************************/
        for ( gp = 1; gp <= NG; gp++ )
        {
            if ( gp != g )
            {
                otr_src_scat(input_vars, data_vars, sn_vars,
                             solvar_vars, g, gp, ierr);
            }
        }

    }
}

/***********************************************************************
 * Compute the scattering source for all cells and moments
 ***********************************************************************/
void otr_src_scat ( input_data *input_vars, data_data *data_vars, sn_data *sn_vars,
                    solvar_data *solvar_vars, int g, int gp, int *ierr )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int l = 1, m, mom;
    int i, j, k;

    double tc[NX*NY*NZ];

/***********************************************************************
 * Use expxs_reg to expand current gp->g cross sections to the fine
 * mesh one scattering order at a time. Start with first. Then compute
 * source moment with flux.
 ***********************************************************************/
    expxs_reg( input_vars, data_vars, solvar_vars, tc, g, gp, l );

    for ( k = 0; k < NZ; k++ )
    {
        for ( j = 0; j < NY; j++ )
        {
            for ( i = 0; i < NX; i++ )
            {
                Q_4D(0,i,j,k) += TC_3D(i,j,k)*F_3D(i,j,k);
            }
        }
    }

/***********************************************************************
 * Repeat the process for higher scattering orders, source moments.
 * Use loop for multiple source moments of same scattering order.
 ***********************************************************************/
    mom = 2;

    for ( l = 2; l <= NMOM; l++ )
    {
        expxs_reg( input_vars, data_vars, solvar_vars, tc, g, gp, l);

        for ( m = 1; m <= LMA_1D(l-1); m++ )
        {
            for ( k = 0; k < NZ; k++ )
            {
                for ( j = 0; j < NY; j++ )
                {
                    for ( i = 0; i < NX; i++ )
                    {
                        Q_4D((mom-1),i,j,k)
                            += TC_3D(i,j,k)*FM_4D((mom-2),i,j,k);
                    }
                }
            }
            mom += 1;
        }
    }
}

/***********************************************************************
 * Check for convergence of outer iterations.
 ***********************************************************************/
void otr_conv ( input_data *input_vars, para_data *para_vars, control_data *control_vars,
                solvar_data *solvar_vars, int *ierr)
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int g, k, j, i;
    double dfmxo_tmp = -1;

    bool allinrdone = false;

    double df[NX*NY*NZ*NG];

/***********************************************************************
 * Thread to speed up computation of df by looping over groups. Rejoin
 * threads and then determine max error.
 ***********************************************************************/
// TODO: Add USEMKL
#pragma omp parallel for schedule(dynamic, 1) default(shared) \
    private(g)
    for ( g = 1; g <= NG; g++ )
    {
        for ( k = 0; k < NZ; k++ )
        {
            for ( j = 0; j < NY; j++ )
            {
                for ( i = 0; i < NX; i++ )
                {
                    if ( fabs(FLUXPO_4D(i,j,k,(g-1))) > TOLR )
                    {
                        DF_4D(i,j,k,(g-1))
                            = fabs( (FLUX_4D(i,j,k,(g-1)) / FLUXPO_4D(i,j,k,(g-1)) - 1) );
                    }
                    else
                    {
                        DF_4D(i,j,k,(g-1))
                            = fabs( (FLUX_4D(i,j,k,(g-1)) - FLUXPO_4D(i,j,k,(g-1))) );
                    }

                    dfmxo_tmp = MAX(dfmxo_tmp, DF_4D(i,j,k,(g-1)));
                }
            }
        }
    }

    DFMXO = dfmxo_tmp;

    glmax_d(&DFMXO, COMM_SNAP);

    for ( g = 1; g <= NG; g++ )
    {
        if ( !INRDONE_1D((g-1)) )
        {
            allinrdone = false;
            break;
        }
        else allinrdone = true;
    }

    if ( (DFMXO <= (100.0 * EPSI)) && allinrdone )
    {
        OTRDONE = true;
    }
}
