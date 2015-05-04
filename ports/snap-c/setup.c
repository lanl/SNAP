/*******************************************************************************
 * Module: setup.c
 * Controls the problem setup, including geomety setup, angular domain setup,
 * data setup, material and source layouts. Calls for data allocation
 *******************************************************************************/
#include "snap.h"

/*******************************************************************************
 * Control the setup process
 *******************************************************************************/
void setup ( input_data *input_vars, para_data *para_vars, time_data *time_vars,
             geom_data *geom_vars, sn_data *sn_vars, data_data *data_vars,
             solvar_data *solvar_vars, control_data *control_vars,
             mms_data *mms_vars, FILE *fp_out, int *ierr, char **error )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int flg, mis, mie, mjs, mje, mks, mke, qis, qie, qjs, qje, qks, qke;

    double t1, t2;

 /*******************************************************************************
 *  First put input ny and nz into ny_gl and nz_gl respectively. Use ny
 *  and nz for local sizes. Determine global indices of local bounds.
 *  Establish min of nthreads and ng for threaded MPI calls in sweep.
 *******************************************************************************/
    t1 = wtime();

    NY_GL = NY;
    NZ_GL = NZ;

    NY = NY_GL / NPEY;
    NZ = NZ_GL / NPEZ;

    JLB =  YPROC      * NY + 1;
    JUB = (YPROC + 1) * NY;
    KLB =  ZPROC      * NZ + 1;
    KUB = (ZPROC + 1) * NZ;

    NUM_GRTH = MIN( NTHREADS, NG );

/*******************************************************************************
 *  Allocate needed arrays
 *******************************************************************************/
    setup_alloc( input_vars, para_vars, sn_vars, data_vars, &flg, ierr, error );

    if ( *ierr != 0 )
    {
        print_error ( fp_out, *error, IPROC, ROOT );

        stop_run( flg, 0, 0, para_vars, sn_vars, data_vars, mms_vars,
                   geom_vars, solvar_vars, control_vars );
    }

/*******************************************************************************
 *  Progress through setups. _delta sets cell and step sizes, _vel sets
 *  velocity array, _angle sets the ordinates/weights, _mat sets the
 *  material identifiers, _src sets fixed source, _data sets the
 *  mock cross section arrays, and expcoeff sets up the scattering
 *  expansion basis function array.
 *******************************************************************************/
    setup_delta( input_vars, geom_vars, control_vars );

    setup_vel( input_vars, data_vars );

    setup_angle( input_vars, sn_vars );

    setup_mat( input_vars, geom_vars, data_vars,
               &mis, &mie, &mjs, &mje, &mks, &mke );

    setup_data( input_vars, data_vars );

    expcoeff( input_vars, sn_vars, &NDIMEN );

    setup_src( input_vars, para_vars, geom_vars, sn_vars, data_vars, control_vars,
               mms_vars, &qis, &qie, &qjs, &qje, &qks, &qke, ierr, error );

    if ( *ierr != 0 )
    {
        print_error ( fp_out, *error, IPROC, ROOT );

        stop_run( 2, 0, 0, para_vars, sn_vars, data_vars, mms_vars,
                   geom_vars, solvar_vars, control_vars );
    }

/*******************************************************************************
 *  Echo the data from this module to the output file. If requested via
 *  scatp, print the full scattering matrix to file.
 *******************************************************************************/
    if ( IPROC == ROOT )
    {
        setup_echo ( fp_out, input_vars, para_vars, geom_vars, data_vars, sn_vars,
                     control_vars, mis, mie, mjs, mje, mks, mke,
                     qis, qie, qjs, qje, qks, qke );

        if ( SCATP == 1 ) setup_scatp( input_vars, para_vars, data_vars, ierr, error );
    }

    glmax_i ( ierr, COMM_SNAP );

    if ( *ierr != 0 )
    {
        print_error ( fp_out, *error, IPROC, ROOT );

        FREE ( error );

        stop_run ( 3, 0, 0, para_vars, sn_vars, data_vars, mms_vars,
                   geom_vars, solvar_vars, control_vars );
    }

    t2 = wtime();
    TSET = t2 - t1;
}

/*******************************************************************************
 * Call for individual allocation routines to size the run-time arrays.
 *******************************************************************************/
void setup_alloc( input_data *input_vars, para_data *para_vars, sn_data *sn_vars,
                  data_data *data_vars, int *flg, int *ierr, char **error )
{
    int tmpStrLen = 0;
    *flg = 0;

    sn_allocate ( sn_vars, input_vars, ierr );

    glmax_i ( ierr, COMM_SNAP );

    if ( *ierr != 0 )
    {
        tmpStrLen = strlen ( "***ERROR: SETUP_ALLOC: "
                             "Allocation error in SN_ALLOCATE\n" );

        ALLOC_STR(*error, tmpStrLen + 1, ierr);

        snprintf ( (char *) *error, tmpStrLen + 1,
                   "***ERROR: SETUP_ALLOC: "
                   "Allocation error in SN_ALLOCATE\n" );

        return;
    }

    data_allocate( data_vars, input_vars, sn_vars, ierr );

    glmax_i ( ierr, COMM_SNAP );

    if ( *ierr != 0 )
    {
        *flg = 1;

        tmpStrLen = strlen ( "***ERROR: SETUP_ALLOC: "
                             "Allocation error in DATA_ALLOCATE\n" );

        ALLOC_STR(*error, tmpStrLen + 1, ierr);

        snprintf ( (char *) *error, tmpStrLen + 1,
                   "***ERROR: SETUP_ALLOC: "
                   "Allocation error in DATA_ALLOCATE\n" );

        return;
    }

}

/*******************************************************************************
 * Call for individual allocation routines to size the run-time arrays.
 *******************************************************************************/

void setup_delta( input_data *input_vars, geom_data *geom_vars,
                  control_data *control_vars )
{
    DX = LX / ((double) NX);

    if ( NDIMEN > 1 )
    {
        DY = LY / ((double) NY_GL);
    }

    if ( NDIMEN > 2 )
    {
        DZ = LZ / ((double) NZ_GL);
    }

    if ( TIMEDEP == 1 )
    {
        DT = TF / ((double) NSTEPS);
    }
}

/*******************************************************************************
 * Setup a simple mock velocity array for time-dependent calcs
 *******************************************************************************/
void setup_vel ( input_data *input_vars, data_data *data_vars )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
   int g, t;

/*******************************************************************************
 * Loop over groups. Set velocities simply according to ng.
 *******************************************************************************/
    if ( TIMEDEP == 0 ) return;

    for ( g = 1; g <= NG; g++ )
    {
        t = NG - g + 1;
        V_1D(g-1) = (double) t;
    }
}

/*******************************************************************************
 * Create the mock quadrature sets for 1-D, 2-D, and 3-D problems.
 *******************************************************************************/
void setup_angle ( input_data *input_vars, sn_data *sn_vars)
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int i, m, ierr = 0;
    double dm;
    double t[NANG];

    //double *t;
    //ALLOC_1D(t, NANG, double, &ierr);

/*******************************************************************************
 * Arrays allocated according to dimensionality. So do loops by ndimen.
 *******************************************************************************/

    dm = 1.0 / ((double) NANG);

    for ( i = 0; i < NANG; i++ )
    {
        t[i] = 0;
    }

    MU_1D(0) = 0.5*dm;

    for ( m = 2; m <= NANG; m++ )
    {
        MU_1D(m-1) = MU_1D(m-2) + dm;
    }

    if ( NDIMEN > 1 )
    {
        ETA_1D(0) = 1 - 0.5*dm;

        for ( m = 2; m <= NANG; m++ )
        {
            ETA_1D(m-1) = ETA_1D(m-2) - dm;
        }

        if ( NDIMEN > 2 )
        {
// TODO: Find better mkl function to initialize vector
            for ( i = 0; i < NANG; i++)
            {
                t[i] = (MU_1D(i)*MU_1D(i)) + (ETA_1D(i)*ETA_1D(i));
            }

            for (m = 1; m <= NANG; m++)
            {
                XI_1D(m-1) = sqrt( 1.0 - t[m-1] );
            }
        }
    }

/*******************************************************************************
 *   Give all weights same value.
 *******************************************************************************/
    if ( NDIMEN == 1 )
    {
        for (m = 0; m < NANG; m++)
        {
            W_1D(m) = 0.5 / ((double) NANG);
        }
    }
    else if ( NDIMEN == 2 )
    {
        for (m = 0; m < NANG; m++)
        {
            W_1D(m) = 0.25 / ((double) NANG);
        }
    }
    else
    {
        for (m = 0; m < NANG; m++)
        {
            W_1D(m) = 0.125 / ((double) NANG);
        }
    }

//    FREE ( t );
}

/*******************************************************************************
 * Setup the material according to mat_opt.
 *
 * There are only two materials max; one per cell. mat_opt defines the
 * material layout and has a similar meaning for 1-D, 2-D, 3-D problems.
 *
 * 0 = Homogeneous (mat1) problem, regardless of dimension
 * 1 = Center. mat1 is the base. 1-D: 25%mat1/50%mat2/25%mat1
 *     2/D: same as 1-D but in two dimensions, so mat2 region is a square
 *     3/D: same as 1-D but in three dimensions, so mat2 is a cube
 * 2 = Corner. Same concept as 1, but move slab/square/cube to origin
 *
 * Return starting/ending indices for printint an echo.
 *******************************************************************************/
void setup_mat ( input_data *input_vars, geom_data *geom_vars, data_data *data_vars,
                 int *i1, int *i2, int *j1, int *j2, int *k1, int *k2 )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int i, j, k, jj, kk;

/*******************************************************************************
 * Form the base with mat1. Use dimension and mat_opt to determine
 * the rest of the layout.
 *******************************************************************************/
    for (k = 0; k < NZ; k++)
    {
        for (j = 0; j < NY; j++)
        {
            for (i=0; i < NX; i++)
            {
                MAT_3D(i,j,k) = 1;
            }
        }
    }

    *i1 = 1;
    *i2 = 1;

    *j1 = 1;
    *j2 = 1;

    *k1 = 1;
    *k2 = 1;

    if ( MAT_OPT == 1 )
    {
        *i1 = NX / 4 + 1;
        *i2 = 3 * NX / 4;

        if ( NDIMEN > 1 )
        {
            *j1 = NY_GL / 4 + 1;
            *j2 = 3 * NY_GL / 4;

            if ( NDIMEN > 2 )
            {
                *k1 = NZ_GL / 4 + 1;
                *k2 = 3 * NZ_GL / 4;
            }
        }
    }

    else if ( MAT_OPT == 2 )
    {
        *i2 = NX / 2;
        if ( NDIMEN > 1 )
        {
            *j2 = NY_GL / 2;
            if ( NDIMEN > 2 )
            {
                *k2 = NZ_GL / 2;
            }
        }
    }

    if ( MAT_OPT > 0 )
    {
        for ( k = *k1; k <= *k2; k++ )
        {
            if ( (KLB <= k) && (k <= KUB) )
            {
                kk = fmod( (k-1), NZ ) + 1;

                for ( j = *j1; j <= *j2; j++ )
                {
                    if ( (JLB <= j) && (j <= JUB) )
                    {
                        jj = fmod( (j-1), NY ) + 1;

                        for ( i = *i1; i <= *i2; i++ )
                        {
                            MAT_3D((i-1), (jj-1), (kk-1)) = 2;
                        }
                    }
                }
            }
        }
    }
}

/*******************************************************************************
* Setup the material according to src_opt.
*
* Source is either on at strength 1.0 or off, per cell. src_opt defines
* the layout and has a similar meaning for 1-D, 2-D, 3-D problems.
*
* 0 = Source everywhere, regardless of dimension
* 1 = Source occupies the slab/square/cube center of the problem
*     1/D: 25%/50%/25% = source off/on/off
*     2/D: same as 1-D but in two dimensions
*     3/D: same as 1-D but in three dimensions
* 2 = Corner. Same concept as 1, but move slab/square/cube to origin
* 3 = MMS option, f = sin(pi*x/lx)
* All options isotropic source
*
* Return starting/ending indices for printing an echo.
 *******************************************************************************/
void setup_src ( input_data *input_vars, para_data *para_vars, geom_data *geom_vars,
                 sn_data *sn_vars, data_data *data_vars, control_data *control_vars,
                 mms_data *mms_vars, int *i1, int *i2, int *j1, int *j2, int *k1,
                 int *k2, int *ierr, char **error )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int i, j, k, jj, kk, pp;

/*******************************************************************************
 * Form the base by setting the src to 0 to start. Then assign src to
 * spatial cells according to ndimen and src_opt.
 *******************************************************************************/
    if ( SRC_OPT == 3 )
    {
        mms_setup ( input_vars, para_vars, geom_vars, data_vars,
                    sn_vars, control_vars, mms_vars, ierr, error );
        return;
    }

/*******************************************************************************
 * If src_opt is 0, source is everywhere
 *******************************************************************************/
    *i1 = 1;
    *i2 = NX;

    *j1 = 1;
    *j2 = NY_GL;

    *k1 = 1;
    *k2 = NZ_GL;

/*******************************************************************************
 * if src_opt is not 0, reset indices for source's spatial range
 *******************************************************************************/
    if ( SRC_OPT == 1 )
    {
        *i1 = NX / 4 + 1;
        *i2 = 3 * NX / 4;

        if ( NDIMEN > 1 )
        {
            *j1 = NY_GL / 4 + 1;
            *j2 = 3 * NY_GL / 4;

            if ( NDIMEN > 2 )
            {
                *k1 = NZ_GL / 4 + 1;
                *k2 = 3 * NZ_GL / 4;
            }
        }
    }

    else if ( SRC_OPT == 2 )
    {
        *i2 = NX / 2;
        if ( NDIMEN > 1 )
        {
            *j2 = NY_GL / 2;

            if ( NDIMEN > 2 )
            {
                *k2 = NZ_GL / 2;
            }
        }
    }

/*******************************************************************************
 * Indices are all known, so set the source to unity for that range
 *******************************************************************************/
    for ( k = *k1; k <= *k2; k++ )
    {
        if ( (KLB <= k) && (k <= KUB) )
        {
            kk = fmod( (k-1), NZ ) + 1;
            for ( j = *j1; j <= *j2; j++ )
            {
                if ( (JLB <= j) && (j <= JUB) )
                {
                    jj = fmod( (j-1), NY ) + 1;
                    for ( i = *i1; i <= *i2; i++ )
                    {
                        for ( pp = 0; pp < NG; pp++ )
                        {
                            QI_4D((i-1),(jj-1),(kk-1),pp) = 1;
                        }
                    }
                }
            }
        }
    }
}

/*******************************************************************************
 * Setup the data arrays according to simple rules/functions to scale
 * to arbitrary number of groups, moments. Fixed at a library for two
 * materials only. Fixed group-to-group coupling properties.
 *******************************************************************************/
// TODO: Need to set optimized USEMKL loop options
void setup_data ( input_data *input_vars, data_data *data_vars)
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int i, g, n, pp, qq;
    double t;

/*******************************************************************************
 * Set the group 1 data for material 1 and material 2 (if present).
 *******************************************************************************/

    SIGT_2D(0,0) = 1.0;
    SIGA_2D(0,0) = 0.5;
    SIGS_2D(0,0) = 0.5;

    if ( NMAT == 2 )
    {
        SIGT_2D(1,0) = 2.0;
        SIGA_2D(1,0) = 0.8;
        SIGS_2D(1,0) = 1.2;
    }

/*******************************************************************************
 * Scale these values for increasing number of groups by adding 0.01
 * to sigt for each additional group. Split that 0.01 evenly among
 * siga and sigs.
 *******************************************************************************/
    for ( g = 2; g <= NG; g++ )
    {
        for ( i = 0; i < NMAT; i++ )
        {
            SIGT_2D(i,(g-1)) = SIGT_2D(i,(g-2)) + 0.01;
            SIGA_2D(i,(g-1)) = SIGA_2D(i,(g-2)) + 0.005;
            SIGS_2D(i,(g-1)) = SIGS_2D(i,(g-2)) + 0.005;
        }
    }

/*******************************************************************************
 * For material 1, upscattering from a given group to all above groups
 * is 10% of total scattering. 20% is in group. 70% is down-scattering.
 * For group 1/ng, no up/down-scattering, so self-scattering takes the
 * remaining fraction.
 *******************************************************************************/
    for ( g = 1; g <= NG; g++ )
    {
        if ( NG == 1 )
        {
            SLGG_4D(0,0,0,0) = SIGS_2D(0,(g-1));
            break;
        }

        SLGG_4D(0,0,(g-1),(g-1)) = 0.2 * SIGS_2D(0,(g-1));

        if ( g > 1 )
        {
            t = 1.0 / ((double) (g-1));

            for (pp = 1; pp <= (g-1); pp++)
            {
                SLGG_4D(0,0,(g-1),(pp-1)) = 0.1 * SIGS_2D(0,(g-1)) * t;
            }
        }
        else
        {
            SLGG_4D(0,0,0,0) = 0.3 * SIGS_2D(0,0);
        }

        if ( g < NG )
        {
            t = 1.0 / ((double) (NG - g));
            for (pp = g+1; pp <= NG; pp++)
            {
                SLGG_4D(0,0,(g-1),(pp-1)) = 0.7 * SIGS_2D(0,(g-1)) * t;
            }
        }
        else
        {
            SLGG_4D(0,0,(NG-1),(NG-1)) = 0.9 * SIGS_2D(0,(NG-1));
        }
    }

    if ( NMAT == 2 )
    {
/*******************************************************************************
 * Repeat for material 2. Up-scattering is 10%. In-scattering is 50%.
 * Down-scattering is 40%.
 *******************************************************************************/
        for ( g = 1; g <= NG; g++ )
        {
            if ( NG == 1 )
            {
                SLGG_4D(1,0,0,0) = SIGS_2D(1,(g-1));
                break;
            }

            SLGG_4D(1,0,(g-1),(g-1)) = 0.5 * SIGS_2D(1,(g-1));

            if ( g > 1 )
            {
                t = 1.0 / ((double) (g-1));
                for (pp = 1; pp <= (g-1); pp++)
                {
                    SLGG_4D(1,0,(g-1),(pp-1)) = 0.1 * SIGS_2D(1,(g-1)) * t;
                }
            }
            else
            {
                SLGG_4D(1,0,0,0) = 0.6 * SIGS_2D(1,0);
            }

            if ( g < NG )
            {
                t = 1.0 / ((double) (NG - g));

                for (pp = g+1; pp <= NG; pp++)
                {
                    SLGG_4D(1,0,(g-1),(pp-1)) = 0.4 * SIGS_2D(1,(g-1)) * t;
                }
            }
            else
            {
                SLGG_4D(1,0,(NG-1),(NG-1)) = 0.9 * SIGS_2D(1,(NG-1));
            }
        }
    }

/*******************************************************************************
 * Set group-to-group scattering moments. Allowed up 4 moments. For
 * material 1, start with the base above. Divide by 10 to get next
 * moment's data. Repeat that by 2 continuously, up to nmom or 4.
 *******************************************************************************/
    if ( NMOM > 1 )
    {
        for ( pp = 0; pp < NG; pp++ )
        {
            for ( qq = 0; qq < NG; qq++ )
            {
                SLGG_4D(0,1,pp,qq) = 0.1 * SLGG_4D(0,0,pp,qq);
            }
        }

        for ( n = 3; n <= NMOM; n++ )
        {
            for ( pp = 0; pp < NG; pp++ )
            {
                for ( qq = 0; qq < NG; qq++ )
                {
                    SLGG_4D(0,(n-1),pp,qq) = 0.5 * SLGG_4D(0,(n-2),pp,qq);
                }
            }
        }
    }

/*******************************************************************************
 * Similar procedure for material 2, but first multiply by 0.8. Then
 * reduce the data magnitudes by 0.6 for each successive moment.
 *******************************************************************************/
   if ( (NMAT == 2) && (NMOM > 1) )
    {
        for ( pp = 0; pp < NG; pp++ )
            {
                for ( qq = 0; qq < NG; qq++ )
                {
                    SLGG_4D(1,1,pp,qq) = 0.8 * SLGG_4D(1,0,pp,qq);
                }
            }

        for ( n = 3; n <= NMOM; n++ )
        {
            for ( pp = 0; pp < NG; pp++ )
            {
                for ( qq = 0; qq < NG; qq++ )
                {
                    SLGG_4D(1,(n-1),pp,qq) = 0.6 * SLGG_4D(1,(n-2),pp,qq);
                }
            }
        }
    }
}

/*******************************************************************************
 * Print run-time variables setup in this module to the output file.
 *******************************************************************************/
void setup_echo ( FILE *fp_out, input_data *input_vars, para_data *para_vars,
                  geom_data *geom_vars, data_data *data_vars,
                  sn_data *sn_vars, control_data *control_vars,
                  int mis, int mie, int mjs, int mje, int mks,
                  int mke, int qis, int qie, int qjs, int qje, int qks, int qke )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int i, j;

/*******************************************************************************/
    fprintf ( fp_out,
              "          Calculation Run-time Parameters Echo\n"
              "****************************************"
              "****************************************\n\n" );

    fprintf ( fp_out, "  Geometry\n" );
    fprintf ( fp_out, "    ndimen = %i\n",     NDIMEN );
    fprintf ( fp_out, "    nx     = %i\n",     NX );
    fprintf ( fp_out, "    ny     = %i\n",     NY_GL );
    fprintf ( fp_out, "    nz     = %i\n",     NZ_GL );
    fprintf ( fp_out, "    lx     = %.4E\n",   LX );
    fprintf ( fp_out, "    ly     = %.4E\n",   LY );
    fprintf ( fp_out, "    lz     = %.4E\n",   LZ );
    fprintf ( fp_out, "    dx     = %.4E\n",   DX );
    fprintf ( fp_out, "    dy     = %.4E\n",   DY );
    fprintf ( fp_out, "    dz     = %.4E\n\n", DZ );

    fprintf ( fp_out, "  Sn\n" );
    fprintf ( fp_out, "    nmom   = %i\n",   NMOM );
    fprintf ( fp_out, "    nang   = %i\n",   NANG );
    fprintf ( fp_out, "    noct   = %i\n\n", NOCT );

    fprintf ( fp_out, "    w      = %.4E", W_1D(0));
    fprintf ( fp_out, "   ...  uniform weights\n\n");

    if ( NDIMEN == 1 )
    {
        fprintf ( fp_out, "          mu\n");

        for ( i = 0; i < NANG; i++ )
        {
            fprintf ( fp_out, "     %.8E\n", MU_1D(i) );
        }

        fprintf ( fp_out, "\n");
    }
    else if ( NDIMEN == 2 )
    {
        fprintf ( fp_out, "          mu              eta\n");

        for ( i = 0; i < NANG; i++ )
        {
            fprintf ( fp_out, "     %.8E   %.8E\n", MU_1D(i), ETA_1D(i) );
        }

        fprintf ( fp_out, "\n");
    }
    else if ( NDIMEN == 3 )
    {
        fprintf ( fp_out, "           mu              eta              xi\n");

        for ( i = 0; i < NANG; i++ )
        {
            fprintf ( fp_out, "     %.8E   %.8E   %.8E\n", MU_1D(i), ETA_1D(i), XI_1D(i) );
        }

        fprintf ( fp_out, "\n");
    }

    fprintf ( fp_out, "  Material Map\n" );
    fprintf ( fp_out, "    mat_opt = %i   -->   nmat = %i\n",   MAT_OPT, NMAT );
    fprintf ( fp_out, "    Base material (default for every cell) = 1\n" );

    if ( MAT_OPT > 0 )
    {
        fprintf ( fp_out, "    Material 2 present:\n");
        fprintf ( fp_out, "        Starting cell: ( %i, %i, %i )\n", mis, mjs, mks);
        fprintf ( fp_out, "        Ending cell:   ( %i, %i, %i )\n", mie, mje, mke);
    }
    fprintf ( fp_out, "\n");

    fprintf ( fp_out, "  Source Map\n" );
    fprintf ( fp_out, "    src_opt = %i\n",   SRC_OPT );

    if ( SRC_OPT < 3 )
    {
        fprintf ( fp_out, "    Source strength per cell (where applied) = 1.0\n" );
        fprintf ( fp_out, "    Source map:\n");
        fprintf ( fp_out, "        Starting cell: ( %i, %i, %i )\n", qis, qjs, qks);
        fprintf ( fp_out, "        Ending cell:   ( %i, %i, %i )\n", qie, qje, qke);
    }
    else
    {
        fprintf ( fp_out, "    MMS-generated source\n");
    }
    fprintf ( fp_out, "\n");

    fprintf ( fp_out, "  Pseudo Cross Sections Data\n");
    fprintf ( fp_out, "    ng = %i\n\n", NG );

    for ( j = 1; j <= NMAT; j++ )
    {
        fprintf ( fp_out, "    Material %i\n", j );
        fprintf ( fp_out, "    Group         Total         Absorption      Scattering\n" );

        for ( i = 1; i <= NG; i++ )
        {
            fprintf ( fp_out, "       %i       %.6E    %.6E    %.6E\n",
                      i, SIGT_2D((j-1),(i-1)), SIGA_2D((j-1),(i-1)), SIGS_2D((j-1),(i-1)) );
        }
    }
    fprintf ( fp_out, "\n");

    if ( TIMEDEP == 1 )
    {
        fprintf ( fp_out, "  Time-Dependent Calculation Data\n" );
        fprintf ( fp_out, "    tf     = %.4E\n", TF );
        fprintf ( fp_out, "    nsteps = %i\n",   NSTEPS );
        fprintf ( fp_out, "    dt     = %.4E\n", DT );
        fprintf ( fp_out, "    Group        Speed\n" );

        for ( i = 1; i <= NG; i++ )
        {
            if ( i < 10 )
                fprintf ( fp_out, "       %i       %.4E\n", (i), V_1D(i-1) );
            else if ( i < 100 )
                fprintf ( fp_out, "      %i       %.4E\n", (i), V_1D(i-1) );
            else if ( i < 1000 )
                fprintf ( fp_out, "     %i       %.4E\n", (i), V_1D(i-1) );
            else
                fprintf ( fp_out, "    %i       %.4E\n", (i), V_1D(i-1) );
        }
    }
    fprintf ( fp_out, "\n");

    fprintf ( fp_out, "  Solution Control Parameters\n" );
    fprintf ( fp_out, "    epsi    = %.4E\n", EPSI );
    fprintf ( fp_out, "    iitm    = %i\n",   IITM );
    fprintf ( fp_out, "    oitm    = %i\n",   OITM );
    fprintf ( fp_out, "    timedep = %i\n",   TIMEDEP );
    fprintf ( fp_out, "    it_det  = %i\n",   IT_DET );
    fprintf ( fp_out, "    fluxp   = %i\n",   FLUXP );
    fprintf ( fp_out, "    fixup   = %i\n",   FIXUP );
    fprintf ( fp_out, "\n");

    fprintf ( fp_out, "  Parallelization Parameters\n" );
    fprintf ( fp_out, "    npey     = %i\n",   NPEY);
    fprintf ( fp_out, "    npez     = %i\n",   NPEZ );
    fprintf ( fp_out, "    nthreads = %i\n\n", NTHREADS );
    fprintf ( fp_out, "          Thread Support Level\n" );
    fprintf ( fp_out, "          %i - MPI_THREAD_SINGLE\n",     THREAD_SINGLE );
    fprintf ( fp_out, "          %i - MPI_THREAD_FUNNELED\n",   THREAD_FUNNELED );
    fprintf ( fp_out, "          %i - MPI_THREAD_SERIALIZED\n", THREAD_SERIALIZED );
    fprintf ( fp_out, "          %i - MPI_THREAD_MULTIPLE\n",   THREAD_MULTIPLE );
    fprintf ( fp_out, "     thread_level = %i\n\n",             THREAD_LEVEL );

    if ( DO_NESTED )
    {
        fprintf ( fp_out, "     .TRUE. nested threading\n" );
        fprintf ( fp_out, "          nnested = %i\n", NNESTED );
    }
    else
    {
        fprintf ( fp_out, "     .FALSE. nested threading\n" );
        fprintf ( fp_out, "          nnested = %i\n", NNESTED );
    }

    fprintf ( fp_out, "\n");

    fprintf ( fp_out, "****************************************"
              "****************************************\n\n");

}

/*******************************************************************************
 * Print the slgg (scattering matrix) array to special file 'slgg'.
 *******************************************************************************/
void setup_scatp( input_data *input_vars, para_data *para_vars,
                  data_data *data_vars, int *ierr, char **error )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int l, g1, g2, n;
    FILE *fp_slgg = NULL;

/*******************************************************************************
 * Open the 'slgg' file
 *******************************************************************************/
    *ierr = open_file ( &fp_slgg, "slgg", "w", error, IPROC, ROOT );

    if ( *ierr != 0 ) return;

/*******************************************************************************
 * Write out the matrix in standard Fortran column-ordering
 *******************************************************************************/
    fprintf ( fp_slgg,
              "slgg(nmat,nmom,ng,ng) echo\n"
              "Column-order loops: Mats (fastest), Moments, Groups, Groups (slowest)\n");


    for ( g2 = 1; g2 <= NG; g2++ )
    {
        for ( g1 = 1; g1 <= NG; g1++ )
        {
            for ( l = 1; l <= NMOM; l++ )
            {
                for ( n = 1; n <= NMAT; n++ )
                {
                    fprintf ( fp_slgg, "  %.8E", SLGG_4D((n-1),(l-1),(g1-1),(g2-1)) );
                }
            }
        }
        fprintf ( fp_slgg, "\n" );
    }

/*******************************************************************************
 * Close file
 *******************************************************************************/

    *ierr = close_file ( fp_slgg, "slgg", error, ROOT, IPROC );
}
