/*******************************************************************************
 * Module: mms_module
 * This module contains all the setup and verification subroutines for
 * code verification with MMS.
 *******************************************************************************/
#include "snap.h"

/* df(nx,ny,nz,ng) */
/* mms_verify_1 local function */
#ifdef ROWORDER
#define DF_4D_MMS(IN, X, Y, Z, G)               \
    df[ X   * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN) \
        + Y * NZ_IN(IN) * NG_IN(IN)             \
        + Z * NG_IN(IN)                         \
        + G ]
#else
#define DF_4D_MMS(IN, X, Y, Z, G)               \
    df[ G   * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN) \
        + Z * NY_IN(IN) * NX_IN(IN)             \
        + Y * NX_IN(IN)                         \
        + X ]
#endif
// Assuming 'IN=input_vars'
#define DF_4D(X, Y, Z, G) DF_4D_MMS(input_vars, X, Y, Z, G)


void mms_data_init (mms_data *mms_vars)
{
    REF_FLUX  = NULL;
    REF_FLUXM = NULL;

    A_CONST = 0;
    B_CONST = 0;
    C_CONST = 0;

    IB = NULL;
    JB = NULL;
    KB = NULL;
}


/*******************************************************************************
 * This subroutine controls the MMS setup, including computing the MMS
 * source and the reference solution for comparison with/verification of
 * the SNAP-computed source.
 *******************************************************************************/
void mms_setup ( input_data *input_vars, para_data *para_vars, geom_data *geom_vars,
                 data_data *data_vars, sn_data *sn_vars, control_data *control_vars,
                 mms_data *mms_vars, int *ierr, char **error )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int tmpStrLen;

/*******************************************************************************
 * Start by allocating the reference, manufactured solution
 *******************************************************************************/
    mms_allocate ( input_vars, sn_vars, mms_vars, ierr, error );

    glmax_i ( ierr, COMM_SNAP );

    if ( *ierr != 0 )
    {
        tmpStrLen = strlen ( "***ERROR: MMS_ALLOCATE: "
                             "Error allocating MMS arrays\n" );

        ALLOC_STR(*error, tmpStrLen + 1, ierr);

        snprintf ( (char *) *error, tmpStrLen + 1,
                   "***ERROR: MMS_ALLOCATE: "
                   "Error allocating MMS arrays\n" );

        return;
    }

/*******************************************************************************
 * Set up cell boundaries
 *******************************************************************************/
    IB_1D(0) = 0;
    JB_1D(0) = YPROC * NY * DY;
    KB_1D(0) = ZPROC * NZ * DZ;

    A_CONST = 0;
    B_CONST = 0;
    C_CONST = 0;

    mms_cells( input_vars, geom_vars, mms_vars );

/*******************************************************************************
 * Compute the cell-average manufactured solution according to src_opt.
 * Compute for tfinal if timedep==1
 *
 * src_opt = 3 --> f = t*g * sin(pi*x/lx) * sin(pi*y/ly) * sin(pi*z/lz)
 *
 * Then compute the static MMS source. Will apply time dependence
 * directly in dim#_sweep.
 *******************************************************************************/
    if ( SRC_OPT == 3 )
    {
        mms_flux_1 ( input_vars, geom_vars, sn_vars, mms_vars );
        mms_src_1 ( input_vars, geom_vars, data_vars, sn_vars, mms_vars );
        if ( TIMEDEP == 1 ) mms_flux_1_2 ( input_vars, control_vars, mms_vars );
    }
}

/*******************************************************************************
 * Allocate MMS arrays.
 *******************************************************************************/
void mms_allocate ( input_data *input_vars, sn_data *sn_vars, mms_data *mms_vars,
                    int *ierr, char **error )
{
    ALLOC_4D(REF_FLUX, NX, NY, NZ, NG, double, ierr);
    ALLOC_5D(REF_FLUXM, (CMOM-1), NX, NY, NZ, NG, double, ierr);

    if ( *ierr != 0 ) return;

    ALLOC_1D(IB, (NX+1), double, ierr);
    ALLOC_1D(JB, (NY+1), double, ierr);
    ALLOC_1D(KB, (NZ+1), double, ierr);
}

/*******************************************************************************
 * Deallocate MMS arrays.
 *******************************************************************************/
void mms_deallocate (  mms_data *mms_vars )
{
    if (REF_FLUX)  FREE(REF_FLUX);
    if (REF_FLUXM) FREE(REF_FLUXM);
    if (IB)        FREE(IB);
    if (JB)        FREE(JB);
    if (KB)        FREE(KB);
}

/*******************************************************************************
 * Compute and store the cell boundaries arrays
 *******************************************************************************/
void mms_cells ( input_data *input_vars, geom_data *geom_vars, mms_data *mms_vars )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int i, j, k;

    for ( i = 1; i <= NX; i++ )
    {
        IB_1D((i+1)-1) = IB_1D(i-1) + DX;
    }

    if ( NDIMEN > 1 )
    {
        for ( j = 1; j <= NY; j++ )
        {
            JB_1D((j+1)-1) = JB_1D(j-1) + DY;
        }

        if ( NDIMEN > 2 )
        {
            for ( k = 1; k <= NZ; k++ )
            {
                KB_1D((k+1)-1) = KB_1D(k-1) + DZ;
            }
        }
    }

    if ( SRC_OPT == 3 )
    {
        A_CONST = M_PI / LX;
        if ( NDIMEN > 1 ) B_CONST = M_PI / LY;
        if ( NDIMEN > 2 ) C_CONST = M_PI / LZ;
    }
}

/*******************************************************************************
 * Manufactured solution is
 * t*g * sin(pi*x/lx) * sin(pi*y/ly) * sin(pi*z/lz).
 *
 * Where t = 1 for static, g = 1 for one group, y and z terms dropped for
 * 1-D. Compute the cell-average value for the static solution over all
 * cells.
 *******************************************************************************/
// TODO: add USEMKL functions
void mms_flux_1 ( input_data *input_vars, geom_data *geom_vars,
                  sn_data *sn_vars, mms_data *mms_vars )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int i, j, k, g, l, m, n, id, is, jd, js, kd, ks;
    int ierr = 0;

    int sum_wec = 0;

    double *p, *tx, *ty, *tz;

    ALLOC_1D(p, (CMOM-1), double, &ierr);
    ALLOC_1D(tx, NX,      double, &ierr);
    ALLOC_1D(ty, NY,      double, &ierr);
    ALLOC_1D(tz, NZ,      double, &ierr);

/*******************************************************************************
 * Get all the integrations done by dimension separately
 *******************************************************************************/
    mms_trigint ( "COS", NX, A_CONST, DX, IB, tx );

    if ( NDIMEN > 1 )
    {
        mms_trigint ( "COS", NY, B_CONST, DY, JB, ty );

        if ( NDIMEN > 2 )
        {
            mms_trigint ( "COS", NZ, C_CONST, DZ, KB, tz );
        }
        else
        {
            for ( k = 0; k < NZ; k++ )
                tz[k] = 1;
        }
    }
    else
    {
        for ( j = 0; j < NY; j++ )
            ty[j] = 1;
        for ( k = 0; k < NZ; k++ )
            tz[k] = 1;
    }

/*******************************************************************************
 * Combine all dimensions
 *******************************************************************************/
    for ( g = 1; g <= NG; g++ )
    {
        for ( k = 1; k <= NZ; k++)
        {
            for ( j = 1; j <= NY; j++ )
            {
                for ( i = 1; i <= NX; i++ )
                {
                    REF_FLUX_4D((i-1),(j-1),(k-1),(g-1))
                        = ((double) g) * tx[i-1] * ty[j-1] * tz[k-1];
                }
            }
        }
    }

/*******************************************************************************
 * Compute the angular coefficients for the moments
 *******************************************************************************/
    for ( l = 0; l < (CMOM-1); l++ )
    {
        p[l] = 0;
    }

    for ( n = 1; n <= NOCT; n++ )
    {
        for ( l = 1; l <= (CMOM-1); l++ )
        {
            sum_wec = 0;
            for ( i = 0; i < NANG; i++ )
            {
                sum_wec += W_1D(i)*EC_3D(i,(l+1)-1,(n-1));
            }

            p[l-1] += sum_wec;
        }
    }

/*******************************************************************************
 * Apply these coefficients to the angularly independent manufactured
 * solution to get the moments
 *******************************************************************************/
    for ( l = 1; l <= (CMOM-1); l++ )
    {
        for ( n = 0; n < NG; n++ )
        {
            for ( k = 0; k < NZ; k++ )
            {
                for ( j = 0; j < NY; j++ )
                {
                    for ( i = 0; i < NX; i++ )
                    {
                        REF_FLUXM_5D((l-1), i, j, k, n) = p[l-1]*REF_FLUX_4D(i,j,k,n);
                    }
                }
            }
        }
    }

    FREE(p);
    FREE(tx);
    FREE(ty);
    FREE(tz);
}

/*******************************************************************************
 * Perform the loop to do trig function integration --> sin/cos terms
 *******************************************************************************/
// TODO: add USEMKL functions for trig
void mms_trigint ( char *trig, int lc, double d, double del,
                   double *cb, double *fn )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int i;

/*******************************************************************************
 * Do the integration, divide by delta and constant. In case of sine
 * the constant isn't needed because these are streaming terms.
 *******************************************************************************/
    for ( i = 0; i < lc; i++ )
    {
        fn[i] = 0;
    }

    if ( strncmp(trig, "COS", 3) == 0 )
    {
        for ( i = 1; i <= lc; i++ )
        {
            fn[i-1] = cos(d*cb[i-1]) - cos(d*cb[(i+1)-1]);
        }

        for ( i = 0; i < lc; i++ )
        {
            fn[i] /= (d*del);
        }
    }

    else if ( strncmp(trig, "SIN", 3) == 0 )
    {
        for ( i = 1; i <= lc; i++ )
        {
            fn[i-1] = sin(d*cb[(i+1)-1]) - sin(d*cb[i-1]);
        }

        for ( i = 0; i < lc; i++ )
        {
            fn[i] /= del;
        }
    }

    else
    {
        return;
    }
}

/*******************************************************************************
 * Compute the MMS source for the manufactured solution above. Compute
 * the source up to the number of moments specified by the user. The
 * source must be the cell-average. Does not need to include the time
 * coefficient, which is done in octsweep.
 *******************************************************************************/
void mms_src_1( input_data *input_vars, geom_data *geom_vars,
                data_data *data_vars, sn_data *sn_vars, mms_data *mms_vars )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int i, j, k, g, m, gp, l, ll, lm, n, is, js, ks, id, jd, kd;
    int ierr = 0;

    double *cx, *sx, *cy, *sy, *cz, *sz;

    ALLOC_1D(cx, NX, double, &ierr);
    ALLOC_1D(sx, NX, double, &ierr);
    ALLOC_1D(cy, NY, double, &ierr);
    ALLOC_1D(sy, NY, double, &ierr);
    ALLOC_1D(cz, NZ, double, &ierr);
    ALLOC_1D(sz, NZ, double, &ierr);

/*******************************************************************************
 * Get the needed integrations. Need both sine and cosine for each
 * dimension.
 *******************************************************************************/
    mms_trigint ( "COS", NX, A_CONST, DX, IB, cx );
    mms_trigint ( "SIN", NX, A_CONST, DX, IB, sx );

    if ( NDIMEN > 1 )
    {
        mms_trigint ( "COS", NY, B_CONST, DY, JB, cy );
        mms_trigint ( "SIN", NY, B_CONST, DY, JB, sy );

        if ( NDIMEN > 2 )
        {
            mms_trigint ( "COS", NZ, C_CONST, DZ, KB, cz );
            mms_trigint ( "SIN", NZ, C_CONST, DZ, KB, sz );
        }
        else
        {
            for ( i = 0; i < NZ; i++ )
            {
                cz[i] = 1;
                sz[i] = 0;
            }
        }
    }

    else
    {
        for ( i = 0; i < NY; i++ )
        {
            cy[i] = 1;
            sy[i] = 0;
        }
        for ( i = 0; i < NZ; i++ )
        {
            cz[i] = 1;
            sz[i] = 0;
        }
    }

/*******************************************************************************
 * Start computing angular MMS source. Loop over dimensions according
 * to allocation. Start with total interaction and streaming terms.
 * Then loop over scattering terms. If threads are available, use them
 * on this larger group loop to help speed up job.
 *******************************************************************************/
#pragma omp parallel for schedule(dynamic, 1) default(shared)   \
    private(g,kd,ks,jd,js,id,is,n,k,j,i,m,gp,lm,l,ll)
    for ( g = 1; g <= NG; g++ )
    {
        for ( kd = 1; kd <= MAX(NDIMEN-1, 1); kd++ )
        {
            ks = -1;

            if ( kd == 2 ) ks = 1;

            for ( jd = 1; jd <= MIN(NDIMEN, 2); jd++ )
            {
                js = -1;

                if ( jd == 2 ) js = 1;

                for ( id = 1; id <= 2; id++ )
                {
                    is = -1;

                    if (id == 2 ) is = 1;

                    n = 4*(kd - 1) + 2*(jd - 1) + id;

                    for ( k = 1; k <= NZ; k++ )
                    {
                        for ( j = 1; j <= NY; j++ )
                        {
                            for ( i = 1; i <= NX; i++ )
                            {
                                for ( m = 1; m <= NANG; m++ )
                                {
                                    QIM_6D((m-1),(i-1),(j-1),(k-1),(n-1),(g-1))
                                        += ((double) g)*is*MU_1D(m-1)*sx[i-1]*cy[j-1]*cz[k-1]
                                        + SIGT_2D((MAT_3D((i-1),(j-1),(k-1))-1), (g-1))
                                        *REF_FLUX_4D((i-1),(j-1),(k-1),(g-1));

                                    if ( NDIMEN > 1 )
                                    {
                                        QIM_6D((m-1),(i-1),(j-1),(k-1),(n-1),(g-1))
                                            += ((double) g)*js*ETA_1D(m-1)*cx[i-1]*sy[j-1]*cz[k-1];
                                    }

                                    if ( NDIMEN > 2 )
                                    {
                                        QIM_6D((m-1),(i-1),(j-1),(k-1),(n-1),(g-1))
                                            += ((double) g)*ks*XI_1D(m-1)*cx[i-1]*cy[j-1]*sz[k-1];
                                    }

                                    for ( gp = 1; gp <= NG; gp++ )
                                    {
                                        QIM_6D((m-1),(i-1),(j-1),(k-1),(n-1),(g-1))
                                            -= SLGG_4D((MAT_3D((i-1),(j-1),(k-1))-1),0,(gp-1),(g-1))
                                            * REF_FLUX_4D((i-1),(j-1),(k-1),(gp-1));

                                        lm = 2;

                                        for ( l = 2; l <= NMOM; l++ )
                                        {
                                            for ( ll = 1; ll <= LMA_1D(l-1); ll++ )
                                            {
                                                QIM_6D((m-1),(i-1),(j-1),(k-1),(n-1),(g-1))
                                                    -= EC_3D((m-1),(lm-1),(n-1))
                                                    * SLGG_4D((MAT_3D((i-1),(j-1),(k-1))-1),(l-1),(gp-1),(g-1))
                                                    * REF_FLUXM_5D((lm-1)-1,(i-1),(j-1),(k-1),(gp-1));

                                                lm += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

/*******************************************************************************
 * Time-dependent problems have a time-independent source term that
 * can be stored in qi.
 *******************************************************************************/
    if ( TIMEDEP == 1 )
    {
        for ( g = 1; g <= NG; g++ )
        {
            for ( k = 0; k < NZ; k++ )
            {
                for ( j = 0; j < NY; j++ )
                {
                    for ( i = 0; i < NX; i++ )
                    {
                        QI_4D(i,j,k,(g-1)) = REF_FLUX_4D(i,j,k,(g-1)) / V_1D(g-1);
                    }
                }
            }
        }
    }

    FREE(cx);
    FREE(sx);
    FREE(cy);
    FREE(sy);
    FREE(cz);
    FREE(sz);
}

/*******************************************************************************
 * Now that static source is computed, can scale reference solution to
 * that of final time step. (Source is linearly scaled in time in
 * octsweep.)
 *******************************************************************************/
// TODO: add USEMKL functions
void mms_flux_1_2 ( input_data *input_vars, control_data *control_vars,
                    mms_data *mms_vars )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    double t;
    int x,y,z,g;

/*******************************************************************************
 * Compute the time at final time step center. Multiply with ref_flux.
 * No need for ref_fluxm since only ref_flux is checked in mms_verify.
 *******************************************************************************/
    t = TF - 0.5 * DT;

    for ( g = 0; g < NG; g++ )
    {
        for ( z = 0; z < NZ; z++ )
        {
            for ( y = 0; y < NY; y++ )
            {
                for ( x = 0; x < NX; x++ )
                {
                    REF_FLUX_4D(x,y,z,g) *= t;
                }
            }
        }
    }
}

/*******************************************************************************
 * Verify the final solution is near the reference solution.
 *******************************************************************************/
// TODO: Add USEMKL functions
void mms_verify_1 ( input_data *input_vars, para_data *para_vars, control_data *control_vars,
                    mms_data *mms_vars, solvar_data *solvar_vars, FILE *fp_out )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int i, j, k, g, ierr = 0;
    double dfmx, dfmn;
    double *df;

    // Use macro DF_4D(NX,NY,NZ,NG) to index df
    ALLOC_4D(df, NX, NY, NZ, NG, double, &ierr);

    for ( g = 0; g < NG; g++ )
    {
        for ( k = 0; k < NZ; k++ )
        {
            for ( j = 0; j < NY; j++ )
            {
                for ( i = 0; i < NX; i++ )
                {
                    if ( fabs(REF_FLUX_4D(i,j,k,g)) > TOLR )
                    {
                        DF_4D(i,j,k,g) = fabs( (FLUX_4D(i,j,k,g) / REF_FLUX_4D(i,j,k,g)) - 1 );
                    }
                    else
                    {
                        DF_4D(i,j,k,g) = fabs( FLUX_4D(i,j,k,g) - REF_FLUX_4D(i,j,k,g) );
                    }

                    if ( i==0 && j==0 && k==0 && g==0 )
                    {
                        dfmx = DF_4D(i,j,k,g);
                        dfmn = DF_4D(i,j,k,g);
                    }

                    if ( DF_4D(i,j,k,g) > dfmx )
                    {
                        dfmx = DF_4D(i,j,k,g);
                    }
                    else if ( DF_4D(i,j,k,g) < dfmn )
                    {
                        dfmn = DF_4D(i,j,k,g);
                    }
                }
            }
        }
    }

    glmax_d ( &dfmx, COMM_SNAP );
    glmin_d ( &dfmn, COMM_SNAP );


    if ( IPROC == ROOT )
    {
        fprintf ( fp_out, "\n          MMS Verification\n"
                  "****************************************"
                  "****************************************\n\n"
                  "    Manufactured/Computed Solutions Max Diff=%.6E\n"
                  "    Manufactured/Computed Solutions MIN Diff=%.6E\n\n"
                  "****************************************"
                  "****************************************\n\n", dfmx, dfmn );
    }

    FREE(df);
}
