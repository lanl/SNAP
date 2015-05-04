/***********************************************************************
 * Module: inner.c
 *
 * This module controls the inner iterations. Inner iterations include
 * the KBA mesh sweep, which is parallelized via MPI and vectorized over
 * angles in a given octant. Inner source computed here and inner
 * convergence is checked.
 ***********************************************************************/
#include "snap.h"

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

#define F_3D(X, Y, Z)         FLUX_4D(X, Y, Z, (g-1))
#define FM_4D(CMOM1, X, Y, Z) FLUXM_5D(CMOM1, X, Y, Z, (g-1))
#define QO_4D(CMOM1, X, Y, Z) Q2GRP_5D(CMOM1, X, Y, Z, (g-1))
#define CS_4D(NMOM1, X, Y, Z) S_XS_5D(NMOM1, X, Y, Z, (g-1))
#define Q_4D(CMOM1, X, Y, Z)  QTOT_5D(CMOM1, X, Y, Z, (g-1))


/***********************************************************************
 * Do a single inner iteration for all groups. Calculate the total source
 * for each group and sweep the mesh.
 ***********************************************************************/
void inner ( input_data *input_vars, para_data *para_vars, geom_data *geom_vars,
             time_data *time_vars, data_data *data_vars, sn_data *sn_vars,
             control_data *control_vars, solvar_data *solvar_vars,
             sweep_data *sweep_vars, dim_sweep_data *dim_sweep_vars,
             int inno, int *iits, FILE *fp_out, int *ierr )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int g, k, j, i;

    double t1, t2, t3, t4;

/***********************************************************************
 * Compute the inner source and add it to fixed + out-of-group sources
 ***********************************************************************/
    t1 = wtime();

    inr_src ( input_vars, sn_vars, control_vars, solvar_vars, para_vars );

    t2 = wtime();

    TINRSRC += t2 - t1;

/***********************************************************************
 * With source computed, set previous copy of flux and zero out current
 * copies--new flux moments iterates computed during sweep. Thread
 * over groups.
 ***********************************************************************/
#pragma omp parallel for schedule(dynamic, 1) default(shared)   \
    private(g, k, j, i)
    for ( g = 0; g < NG; g++ )
    {
        if ( !INRDONE_1D(g) )
        {
            for ( k = 0; k < NZ; k++ )
            {
                for ( j = 0; j < NY; j++ )
                {
                    for ( i = 0; i < NX; i++ )
                    {
                        FLUXPI_4D(i,j,k,g) = FLUX_4D(i,j,k,g);
                    }
                }
            }
        }
    }

/***********************************************************************
 * Call for the transport sweep. Check convergence, using threads.
 ***********************************************************************/
    t3 = wtime();

    sweep ( input_vars, para_vars, geom_vars, sn_vars, data_vars,
            control_vars, solvar_vars, sweep_vars, dim_sweep_vars, ierr);

    t4 = wtime();

    TSWEEPS += t4 - t3;

    inr_conv ( input_vars, para_vars, control_vars, solvar_vars,
               inno, iits, fp_out, ierr );
}

/***********************************************************************
 * Compute the inner source, i.e., the within-group scattering source.
 ***********************************************************************/
void inr_src ( input_data *input_vars, sn_data *sn_vars,
               control_data *control_vars, solvar_data *solvar_vars, para_data *para_vars )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int g;

/***********************************************************************
 * Compute the within-group scattering source. Thread over groups.
 ***********************************************************************/
#pragma omp parallel for schedule(dynamic, 1) default(shared) private(g)
    for ( g = 1; g <= NG; g++ )
    {
        if ( !(INRDONE_1D(g-1)) )
        {
            inr_src_scat ( input_vars, sn_vars, solvar_vars, g, para_vars );
        }
    }
}

/***********************************************************************
 * Compute the within-group scattering for a given group. Add it to fixed
 * and out-of-group sources.
 ***********************************************************************/
void inr_src_scat ( input_data *input_vars, sn_data *sn_vars,
                    solvar_data *solvar_vars, int g, para_data *para_vars )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int i, j, k, l, m, mom;

    double F_CS_temp[NX*NY*NZ];

/***********************************************************************
 * Loop over all cells. Set the first source moment with flux (f). Then
 * set remaining source moments with fluxm (fm) and combination of
 * higher scattering orders.
 ***********************************************************************/
#ifdef USEVML_unchecked
    if (NX*NY*NZ > VECLEN_MIN)
    {
        vmdMul(NX*NY*NZ, &F_3D(0,0,0), &CS_4D(0,0,0,0), F_CS_temp,
               VML_ACCURACY | VML_HANDLING | VML_ERROR );

        vmdAdd(NX*NY*NZ, F_CS_temp, &QO_4D(0,0,0,0), &Q_4D(0,0,0,0),
               VML_ACCURACY | VML_HANDLING | VML_ERROR );

        mom = 2;

        for ( l = 2; l <= NMOM; l++ )
        {
            for ( m = 1; m <= LMA_1D(l-1); m++ )
            {
                vmdMul(NX*NY*NZ, &FM_4D((mom-2),0,0,0),
                       &CS_4D((l-1),0,0,0), &Q_4D((mom-1),0,0,0),
                       VML_ACCURACY | VML_HANDLING | VML_ERROR );

                vmdAdd(NX*NY*NZ, &QO_4D((mom-1),0,0,0),
                       &Q_4D((mom-1),0,0,0), &Q_4D((mom-1),0,0,0),
                       VML_ACCURACY | VML_HANDLING | VML_ERROR );

                mom += 1;
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
                    Q_4D(0,(i-1),(j-1),(k-1))
                        = QO_4D(0,(i-1),(j-1),(k-1))
                        + CS_4D(0,(i-1),(j-1),(k-1))
                        *F_3D((i-1),(j-1),(k-1));
                    mom = 2;

                    for ( l = 2; l <= NMOM; l++ )
                    {
                        for ( m = 1; m <= LMA_1D(l-1); m++ )
                        {
                            Q_4D((mom-1),(i-1),(j-1),(k-1))
                                = QO_4D((mom-1),(i-1),(j-1),(k-1))
                                + CS_4D((l-1),(i-1),(j-1),(k-1))
                                *FM_4D((mom-2),(i-1),(j-1),(k-1));

                            mom += 1;
                        }
                    }
                }
            }
        }
    }

#else
    for ( k = 1; k <= NZ; k++ )
    {
        for ( j = 1; j <= NY; j++ )
        {
            for ( i = 1; i <= NX; i++ )
            {
                Q_4D(0,(i-1),(j-1),(k-1))
                    = QO_4D(0,(i-1),(j-1),(k-1))
                    + CS_4D(0,(i-1),(j-1),(k-1))
                    *F_3D((i-1),(j-1),(k-1));

/***********************************************************************
 * Work on other moments with fluxm array
 ***********************************************************************/
                mom = 2;

                for ( l = 2; l <= NMOM; l++ )
                {
                    for ( m = 1; m <= LMA_1D(l-1); m++ )
                    {
                        Q_4D((mom-1),(i-1),(j-1),(k-1))
                            = QO_4D((mom-1),(i-1),(j-1),(k-1))
                            + CS_4D((l-1),(i-1),(j-1),(k-1))
                            *FM_4D((mom-2),(i-1),(j-1),(k-1));

                        mom += 1;
                    }
                }
            }
        }
    }
#endif
}

/***********************************************************************
 * Check for inner iteration convergence using the flux array.
 ***********************************************************************/
void inr_conv ( input_data *input_vars, para_data *para_vars,
                control_data *control_vars, solvar_data *solvar_vars,
                int inno, int *iits, FILE *fp_out, int *ierr )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int g, k, j, i;
    double dfmxi_tmp = -1;

    double df[NX*NY*NZ*NG];

/***********************************************************************
 * Thread group loops for computing local difference (df) array.
 * compute max for that group.
 ***********************************************************************/
// TODO: Add USEMKL
#pragma omp parallel for schedule(dynamic, 1) default(shared) \
    private(g)
    for ( g = 1; g <= NG; g++ )
    {
        if ( !INRDONE_1D(g-1) )
        {
            iits[g-1] = inno;

            for ( k = 0; k < NZ; k++ )
            {
                for ( j = 0; j < NY; j++ )
                {
                    for ( i = 0; i < NX; i++ )
                    {
                        if ( fabs(FLUXPI_4D(i,j,k,(g-1))) > TOLR )
                        {
                            DF_4D(i,j,k,(g-1))
                                = fabs( FLUX_4D(i,j,k,(g-1)) / FLUXPI_4D(i,j,k,(g-1)) - 1 );
                        }
                        else
                        {
                            DF_4D(i,j,k,(g-1))
                                = fabs( FLUX_4D(i,j,k,(g-1)) - FLUXPI_4D(i,j,k,(g-1)) );
                        }

                        dfmxi_tmp = MAX(dfmxi_tmp, DF_4D(i,j,k,(g-1)));
                    }
                }
            }

            DFMXI_1D(g-1) = dfmxi_tmp;
        }
    }

/***********************************************************************
 * All procs then reduce dfmxi for all groups, determine which groups
 * are converged and print requested info
 ***********************************************************************/
    glmax_d_1d(DFMXI, NG, COMM_SNAP);

    for ( g = 0; g < NG; g++ )
    {
        if ( DFMXI_1D(g) <= EPSI )
        {
            INRDONE_1D(g) = true;
        }
    }

    if ( (IPROC == ROOT) && (IT_DET == 1) )
    {
        for ( g = 1; g <= NG; g++ )
        {
            fprintf(fp_out, "    Group %i     Inner %i     Dfmxi %.4E\n",
                    g, iits[g-1], DFMXI_1D(g-1));
        }
    }
}
