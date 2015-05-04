/***********************************************************************
 * Module: sweep.c
 *
 * This module controls final solution output.
 ***********************************************************************/
#include "snap.h"

// Define macro to index local variable fprnt(nx,ny_gl)
/* fprnt(nx,ny_gl) */
#ifdef ROWORDER
#define FPRNT_2D(X, Y_GL)                       \
    fprnt[ X   * NY_GL                          \
           + Y_GL ]
#else
#define FPRNT_2D(X, Y_GL)                       \
    fprnt[ Y_GL * NX                            \
           + X ]
#endif

/* fprnt_tmp(nx,lb:ub) */
#ifdef ROWORDER
#define FPRNT_TMP_2D(X, Y_L_U)                  \
    fprnt_tmp[ X * YL_GL                        \
               + Y_L_U ]
#else
#define FPRNT_TMP_2D(X, Y_L_U)                  \
    fprnt_tmp[ Y_L_U * NX                       \
               + X ]
#endif

/* flux(nx,ny) */
#ifdef ROWORDER
#define FLUX_TMP_2D(IN, X, Y)                   \
    flux_tmp[ X * NY                            \
              + Y ]
#else
#define FLUX_TMP_2D(X, Y)                       \
    flux_tmp[ Y * NX                            \
              + X ]
#endif

/***********************************************************************
 * Print the scalar flux output to the output file.
 ***********************************************************************/
void output ( input_data *input_vars, para_data *para_vars, time_data *time_vars,
              geom_data *geom_vars, data_data *data_vars, sn_data *sn_vars,
              control_data *control_vars, mms_data *mms_vars, solvar_data *solvar_vars,
              sweep_data *sweep_vars, FILE *fp_out, int *ierr, char **error )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int i, j, k, g, is, ii, klb, kub, jp, kloc, rank, jlb, jub;

    int tmpStrLen;

    int co[2];

    double t1, t2;

    double *fprnt;

    t1 = wtime();

/***********************************************************************
 * Return immediately if the flux didn't converge.
 ***********************************************************************/
    if ( !OTRDONE )
    {
        t2 = wtime();
        TOUT = t2 - t1;
        return;
    }

/***********************************************************************
 * If did converge, set up error check variables and allocate the
 * printing array.
 ***********************************************************************/
    if ( IPROC==ROOT )
    {
        fprintf ( fp_out,
                  "          Calculation Final Scalar Flux Solution\n"
                  "****************************************"
                  "****************************************\n");

        //       ALLOC_1D(co, 2, int, ierr);
        ALLOC_2D(fprnt, NX, NY_GL, double, ierr);
    }

    bcast_i_scalar ( ierr, COMM_SNAP, ROOT, NPROC );

    if ( *ierr != 0 )
    {
        tmpStrLen = strlen ( "***ERROR: OUTPUT: "
                             "Allocation error\n" );

        ALLOC_STR(*error, tmpStrLen + 1, ierr);

        snprintf ( (char *) *error, tmpStrLen + 1,
                   "***ERROR: OUTPUT: "
                   "Allocation error\n" );

        print_error ( fp_out, *error, IPROC, ROOT );

        FREE ( error );

        stop_run ( 3, 3, 0, para_vars, sn_vars, data_vars, mms_vars,
                   geom_vars, solvar_vars, control_vars );
    }

/***********************************************************************
 * Get global indices of local PE bounds
 ***********************************************************************/
    klb = ZPROC*NZ + 1;
    kub = (ZPROC+1) * NZ;

/***********************************************************************
 * Choose the mid-plane. Determine k local index.
 ***********************************************************************/
    k = NZ_GL / 2 + 1;
    kloc = (k-1)%NZ + 1;

/***********************************************************************
 * Loops over groups. Send/Recv message. Print flux.
 ***********************************************************************/
    // g_loop
    for ( g = 1; g <= NG; g++ )
    {

/***********************************************************************
 * If global k index is within proc's bounds, send message to root of
 * flux on that plane
 ***********************************************************************/
        if ( (klb <= k) && (k <= kub) )
        {
            MTAG = SPROC*NG + kloc + g;

            output_send ( NX, NY, COMM_SPACE, ROOT, SPROC,
                          MTAG, &FLUX_4D(0,0,(kloc-1),(g-1)), ierr );
        }

/***********************************************************************
 * Presets the printed flux to its own value if there is only one
 * proc or npez=1. Receives messages in order for proper printing.
 ***********************************************************************/
        if ( IPROC == ROOT )
        {
            co[0] = (k-1)/NZ;

            for ( j = 0; j < NY; j++ )
            {
                for (i = 0; i < NX; i++)
                {
                    FPRNT_2D(i,j) = FLUX_4D(i,j,(kloc-1),(g-1));
                }
            }

            fprintf ( fp_out, "\n ***********************************\n"
                        "  Group=   %i   Z Mid-Plane=    %i\n"
                        " ***********************************\n", g, k );

            for ( jp = 0; jp <= NPEY-1; jp++ )
            {
                jlb = jp*NY + 1;
                jub = (jp+1) * NY;
                co[1] = jp;

                cartrank ( co, &rank, COMM_SPACE );

                MTAG = rank*NG + kloc + g;

                output_recv ( NX, NY, COMM_SPACE, rank, SPROC,
                              MTAG, &FPRNT_2D(0,(jlb-1)), ierr );
            }

            for ( i = 1; i <= NX; i+=6)
            {
                is = i + 6 - 1;

                if ( is > NX ) is = NX;

                fprintf ( fp_out, "\n     y    ");

                for ( ii = i; ii <= is; ii++ )
                    fprintf( fp_out, "x    %i      ", ii);

                for ( j = NY_GL; j >= 1; j-- )
                {
                    fprintf( fp_out, "\n     %i", j);
                    for ( ii = i; ii <= is; ii++ )
                    {
                        fprintf( fp_out, "  %.4E", FPRNT_2D((ii-1),(j-1)) );
                    }
                }
            }

            fprintf ( fp_out, "\n\n****************************************"
                                  "****************************************\n");
        }
    }

/***********************************************************************
 * Cleanup
 ***********************************************************************/
    if ( IPROC == ROOT )
    {
        FREE(fprnt);
    }

/***********************************************************************
 * Print flux to file if requested
 ***********************************************************************/
    if ( FLUXP > 0)
    {
        output_flux_file ( input_vars, para_vars, geom_vars, data_vars,
                           sn_vars, control_vars, mms_vars, solvar_vars,
                           sweep_vars, klb, kub, ierr, error, fp_out );
    }

/***********************************************************************
 * If MMS solution, verify compared to ref_flux
 ***********************************************************************/
    if ( SRC_OPT == 3 )
    {
        mms_verify_1 ( input_vars, para_vars, control_vars,
                       mms_vars, solvar_vars, fp_out );
    }

    t2 = wtime();
    TOUT = t2 - t1;
}

/***********************************************************************
 * Send root chunk of flux data for printing
 ***********************************************************************/
void output_send ( int dim1, int dim2, MPI_Comm comm, int root, int sproc,
                   int mtag, double *fprnt, int *ierr )
{
    *ierr = psend_d_2d ( fprnt, dim1, dim2, comm, root, sproc, mtag );
}

/***********************************************************************
 * Receive flux message for output
 ***********************************************************************/
void output_recv ( int dim1, int dim2, MPI_Comm comm, int proc, int sproc,
                   int mtag, double *fprnt, int *ierr )
{
    *ierr = precv_d_2d ( fprnt, dim1, dim2, comm, proc, sproc, mtag );
}

/***********************************************************************
 * Print fluxes to file. Either print just first moment or all moments.
 * Root does printing.
 ***********************************************************************/
void output_flux_file ( input_data *input_vars, para_data *para_vars,
                        geom_data *geom_vars, data_data *data_vars,
                        sn_data *sn_vars, control_data *control_vars,
                        mms_data *mms_vars, solvar_data *solvar_vars,
                        sweep_data *sweep_vars, int klb, int kub,
                        int *ierr, char **error, FILE *fp_out )
{
/***********************************************************************
 * Local variable
 ***********************************************************************/
    FILE *fp_flux;
    int i, j, k, g, is, ii, l, mtag, jp, kp, jlb, jub, kloc, rank;

    int tmpStrLen;

    int co[2];

    double *fprnt;

/***********************************************************************
 * Root opens the file, allocates the space for fprnt
 ***********************************************************************/
    *ierr = open_file ( &fp_flux, "flux", "w", error, IPROC, ROOT );

    bcast_i_scalar ( ierr, COMM_SNAP, ROOT, NPROC );

    if ( *ierr != 0 )
    {
        print_error ( fp_out, *error, IPROC, ROOT );

        FREE ( error );

        stop_run ( 3, 3, 0, para_vars, sn_vars, data_vars, mms_vars,
                   geom_vars, solvar_vars, control_vars );
    }

    if ( IPROC == ROOT )
    {
//        ALLOC_1D(co, 2, int, ierr);
        ALLOC_2D(fprnt, NX, NY_GL, double, ierr);
    }

    bcast_i_scalar ( ierr, COMM_SNAP, ROOT, NPROC );

    if ( *ierr != 0 )
    {
        tmpStrLen = strlen ( "***ERROR: OUTPUT_FLUX_FILE: "
                             "Allocation error\n" );

        ALLOC_STR(*error, tmpStrLen + 1, ierr);

        snprintf ( (char *) *error, tmpStrLen + 1,
                   "***ERROR: OUTPUT_FLUX_FILE: "
                   "Allocation error\n" );

        print_error ( fp_out, *error, IPROC, ROOT );

        FREE ( error );

        stop_run ( 3, 3, 0, para_vars, sn_vars, data_vars, mms_vars,
                   geom_vars, solvar_vars, control_vars );
    }

/***********************************************************************
 * Root does all printing. Start with file header comment.
 ***********************************************************************/
    if ( IPROC == ROOT )
    {
        if ( FLUXP == 1 )
        {
            fprintf ( fp_flux, " flux(nx,ny,nz,ng) echo\n"
                      " Column-order loops: x-cells "
                      "(fastest), y-cells, z-cells, groups (slowest)\n");
        }
        else
        {
            fprintf ( fp_flux, " flux(nx,ny,nz,ng) and "
                      "fluxm(cmom-1, nx,ny,nz,ng) echo\n"
                      " Column-order loops by moment:\n"
                      " x-cells (fastest), y-cells, z-cells,"
                      " groups, moments (slowest)\n");
        }
    }

/***********************************************************************
 * Print data from first moment. Use similar technique as mid-plane,
 * but now loop over all planes and all groups. Store all the data of
 * a k-plane, group in one array and print that all at once.
 ***********************************************************************/
    for ( l = 1; l <= MAX(1, (FLUXP-1)*CMOM); l++ )
    {
        if ( IPROC == ROOT)
        {
            fprintf ( fp_flux, "\n  Moment = %i\n", l );
        }

        for ( g = 1; g <= NG; g++ )
        {
            for ( k = 1; k <= NZ_GL; k++ )
            {
                kloc = ((k-1)%NZ) + 1;

                if ( (klb <= k) && (k <= kub) )
                {
                    MTAG = SPROC*NG*NZ + (g-1)*NZ + kloc;

                    if ( l == 1 )
                    {
                        output_send( NX, NY, COMM_SPACE, ROOT, SPROC, MTAG,
                                     &FLUX_4D(0,0,(kloc-1),(g-1)), ierr );
                    }
                    else
                    {
                        output_send( NX, NY, COMM_SPACE, ROOT, SPROC, MTAG,
                                     &FLUXM_5D((l-2),0,0,(kloc-1),(g-1)), ierr );
                    }

                }

                if ( IPROC == ROOT )
                {
                    co[0] = (k-1) / NZ;

                    for ( j = 0; j < NY; j++ )
                    {
                        for (i = 0; i < NX; i++)
                        {
                            FPRNT_2D(i,j) = FLUX_4D(i,j,(kloc-1),(g-1));
                        }
                    }

                    for ( jp = 0; jp <= NPEY-1; jp++ )
                    {
                        jlb = jp*NY + 1;
                        jub = (jp+1) * NY;
                        co[1] = jp;

                        cartrank ( co, &rank, COMM_SPACE );

                        MTAG = rank*NG*NZ + (g-1)*NZ + kloc;

                        output_recv( NX, NY, COMM_SPACE, rank, SPROC, MTAG,
                                     &FPRNT_2D(0,(jlb-1)), ierr );
                    }

                    for ( i = 1; i <= NX; i+=6)
                    {
                        is = i + 6 - 1;

                        if ( is > NX ) is = NX;

                        //fprintf ( fp_flux, "\n");

                        //for ( ii = i; ii <= is; ii++ )
                        //    fprintf( fp_flux, "x    %i      ", ii);



                        for ( j = NY_GL; j >= 1; j-- )
                        {
                            //  fprintf( fp_flux, "\n     %i", j);
                            for ( ii = i; ii <= is; ii++ )
                                fprintf( fp_flux, "   %.10E", FPRNT_2D((ii-1),(j-1)) );

                            fprintf(fp_flux, "\n");
                        }
                    }
                }
            }
        }
    }

/***********************************************************************
 * Cleanup
 ***********************************************************************/
    if ( IPROC == ROOT )
    {
        // FREE(co);
        FREE(fprnt);
    }

    close_file ( fp_flux, "flux", error, IPROC, ROOT );

    bcast_i_scalar ( ierr, COMM_SNAP, ROOT, NPROC );

    if ( *ierr != 0 )
    {
        print_error ( fp_out, *error, IPROC, ROOT );

        FREE ( error );

        stop_run ( 3, 3, 0, para_vars, sn_vars, data_vars, mms_vars,
                   geom_vars, solvar_vars, control_vars );
    }
}
