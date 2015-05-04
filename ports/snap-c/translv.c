/***********************************************************************
 * Solution driver. Contains the time and outer loops. Calls for outer
 * iteration work. Checks convergence and handles eventual output.
 ***********************************************************************/
#include "snap.h"

void translv ( input_data *input_vars, para_data *para_vars, time_data *time_vars,
               geom_data *geom_vars, sn_data *sn_vars, data_data *data_vars,
               control_data *control_vars, solvar_data *solvar_vars, mms_data *mms_vars,
               sweep_data *sweep_vars, dim_sweep_data *dim_sweep_vars,
               FILE *fp_out, int *ierr, char **error )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int cy, otno, i, tot_iits, cy_iits, out_iits;

    int tmpStrLen = 0;

    int ang, x, y, z, oct, g; // indexes for 6D array

    double sf, time, t1, t2, t3, t4, t5, t6, t7, tmp;

    double *ptr_tmp;

    *ierr = 0;

/***********************************************************************
 * Call for data allocations. Some allocations depend on the problem
 * type being requested.
 ***********************************************************************/
    t1 = wtime ();

    geom_alloc ( input_vars, geom_vars, ierr );

    glmax_i ( ierr, COMM_SNAP );

    if ( *ierr != 0 )
    {
        tmpStrLen = strlen ("   ***ERROR: GEOM_ALLOC:"
                            " Allocation error of sweep parameters.\n");
        ALLOC_STR( *error, tmpStrLen + 1, ierr);
        snprintf ( (char *) *error, tmpStrLen + 1,
                   "   ***ERROR: GEOM_ALLOC:"
                   " Allocation error of sweep parameters.\n" );

        print_error ( fp_out, *error, IPROC, ROOT );

        stop_run ( 3, 0, 0, para_vars, sn_vars, data_vars, mms_vars,
                   geom_vars, solvar_vars, control_vars );
    }

    solvar_alloc ( input_vars, sn_vars, solvar_vars, ierr );

    glmax_i ( ierr, COMM_SNAP );

    if ( *ierr != 0 )
    {
        tmpStrLen = strlen ("   ***ERROR: SOLVAR_ALLOC:"
                            " Allocation error of solution.\n");
        ALLOC_STR( *error, tmpStrLen + 1, ierr);
        snprintf ( (char *) *error, tmpStrLen + 1,
                   "   ***ERROR: SOLVAR_ALLOC:"
                   " Allocation error of solution.\n" );

        print_error ( fp_out, *error, IPROC, ROOT );

        stop_run ( 3, 1, 0, para_vars, sn_vars, data_vars, mms_vars,
                   geom_vars, solvar_vars, control_vars );
    }

    control_alloc ( input_vars, control_vars, ierr );

    glmax_i ( ierr, COMM_SNAP );

    if ( *ierr != 0 )
    {
        tmpStrLen = strlen ("   ***ERROR: CONTROL_ALLOC:"
                            " Allocation error of control.\n");
        ALLOC_STR( *error, tmpStrLen + 1, ierr);
        snprintf ( (char *) *error, tmpStrLen + 1,
                   "   ***ERROR: CONTROL_ALLOC:"
                   " Allocation error of control.\n" );

        print_error ( fp_out, *error, IPROC, ROOT );

        stop_run ( 3, 2, 0, para_vars, sn_vars, data_vars, mms_vars,
                   geom_vars, solvar_vars, control_vars );
    }

/***********************************************************************
 * Call for setup of the mini-KBA diagonal map
 ***********************************************************************/
    diag_setup ( input_vars, para_vars, geom_vars, ierr, error );

    glmax_i ( ierr, COMM_SNAP );

    if ( *ierr != 0 )
    {
        tmpStrLen = strlen ("   ***ERROR: DIAG_SETUP:"
                            " Allocation error of diag type array.\n");
        ALLOC_STR( *error, tmpStrLen + 1, ierr);
        snprintf ( (char *) *error, tmpStrLen + 1,
                   "   ***ERROR: DIAG_SETUP:"
                   " Allocation error of diag type array.\n" );

        print_error ( fp_out, *error, IPROC, ROOT );

        stop_run ( 3, 3, 0, para_vars, sn_vars, data_vars, mms_vars,
                   geom_vars, solvar_vars, control_vars );
    }

    t2 = wtime ();

    TPARAM += t2 - t1;

/***********************************************************************
 * The time loop solves the problem for nsteps. If static, there is
 * only one step, and it does not have any time-absorption or -source
 * terms. Set the pointers to angular flux arrays. Set time to one for
 * static for proper multiplication in octsweep.
 ***********************************************************************/
// TODO: Add USEMKL and better parallelize functions
    if ( IPROC == ROOT )
    {
        fprintf ( fp_out,
                  "          Iteration Monitor\n"
                  "****************************************"
                  "****************************************\n" );
    }

    tot_iits = 0;

    // time_loop
    for ( cy = 1; cy <= NSTEPS; cy++ )
    {
        t3 = wtime();

        for ( i = 0; i < NG; i++ )
        {
            VDELT_1D(i) = 0;
        }
        time = 1;

        if ( TIMEDEP == 1 )
        {
            if ( IPROC == ROOT )
            {
                fprintf ( fp_out, " \n******************************\n\n"
                          "  Time Cycle %i\n", cy );
            }

            for ( i = 0; i < NG; i++ )
            {
                VDELT_1D(i) = 2 / (DT * V_1D(i));
            }
            time = DT * (((double) cy) - 0.5 );
        }

        if ( cy > 1 )
        {
            ptr_tmp = PTR_OUT;
            PTR_OUT = PTR_IN;
            PTR_IN = ptr_tmp;
        }

/***********************************************************************
 * Scale the manufactured source for time
 ***********************************************************************/
        if ( SRC_OPT == 3 )
        {
            if ( cy == 1 )
            {
                // TODO: Add USEMKL version
                for ( g = 0; g < NG; g++ )
                {
                    for ( oct = 0; oct < NOCT; oct++ )
                    {
                        for ( z = 0; z < NZ; z++ )
                        {
                            for ( y = 0; y < NY; y++ )
                            {
                                for ( x = 0; x < NX; x++ )
                                {
                                    for ( ang = 0; ang < NANG; ang++ )
                                    {
                                        QIM_6D(ang,x,y,z,oct,g) *= time;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            else
            {
                sf = ((double) ( 2*cy - 1 )) / ((double) ( 2*cy - 3 ));
                // TODO: Add USEMKL version
                for ( g = 1; g <= NG; g++ )
                {
                    for ( oct = 0; oct < NOCT; oct++ )
                    {
                        for ( z = 0; z < NZ; z++ )
                        {
                            for ( y = 0; y < NY; y++ )
                            {
                                for ( x = 0; x < NX; x++ )
                                {
                                    for ( ang = 0; ang < NANG; ang++ )
                                    {
                                        QIM_6D(ang,x,y,z,oct,(g-1)) *= sf;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

/***********************************************************************
 * Zero out flux arrays. Use threads when available.
 ***********************************************************************/
#pragma omp parallel for schedule(dynamic, 1) default(shared) \
    private(g)
        for ( g = 1; g <= NG; g++ )
        {
            for ( z = 0; z < NZ; z++ )
            {
                for ( y = 0; y < NY; y++ )
                {
                    for ( x = 0; x < NX; x++ )
                    {
                        FLUX_4D(x,y,z,(g-1)) = 0;
                        for ( i = 0; i < (CMOM - 1); i++ )
                        {
                            FLUXM_5D(i,x,y,z,(g-1)) = 0;
                        }
                    }
                }
            }
        }

/***********************************************************************
 * Using Jacobi iterations in energy, and the work in the outer loop
 * will be parallelized with threads.
 ***********************************************************************/
        OTRDONE = false;

        cy_iits = 0;

        if ( (IPROC == ROOT) && (IT_DET == 0) ) fprintf(fp_out, "  Outer\n");

        t4 = wtime();

        TPARAM += t4 - t3;

        // outer_loop
        for ( otno = 1; otno <= OITM; otno++ )
        {
            t5 = wtime();

            if ( IPROC==ROOT && IT_DET==1 )
            {
                fprintf(fp_out, " ********************\n  Outer %i\n", otno);
            }

/***********************************************************************
 * Prepare some cross sections: total, in-group scattering, absorption.
 * Keep in the time loop for better consistency with PARTISN. Set up
 * geometric sweep parameters. Parallelize group loop with threads.
 ***********************************************************************/
#pragma omp parallel for schedule(dynamic, 1) default(shared)   \
    private(g)
            for ( g = 1; g <= NG; g++ )
            {
                expxs_reg  ( input_vars, data_vars, solvar_vars, NULL, g, 0, 0 );
                expxs_slgg ( input_vars, data_vars, solvar_vars, g );
                param_calc ( input_vars, sn_vars, solvar_vars, data_vars,
                             geom_vars, g );
            }

/***********************************************************************
 * Perform an outer iteration. Add up inners. Check convergence.
 ***********************************************************************/
            t6 = wtime();
            TPARAM += t6 - t5;

            out_iits = outer ( input_vars, para_vars, geom_vars, time_vars,
                               data_vars, sn_vars, control_vars, solvar_vars,
                               sweep_vars, dim_sweep_vars, fp_out, ierr );

            cy_iits += out_iits;

            if ( IPROC == ROOT )
            {
                if ( otno < 10 )
                {
                    fprintf ( fp_out, "    %i   Dfmxo= %.4E"
                              "    No. Inners=   %i\n", otno, DFMXO, out_iits );
                }
                else if ( otno < 100 )
                {
                    fprintf ( fp_out, "   %i   Dfmxo= %.4E"
                              "    No. Inners=   %i\n", otno, DFMXO, out_iits );
                }
                else if ( otno < 1000 )
                {
                    fprintf ( fp_out, "  %i   Dfmxo= %.4E"
                              "    No. Inners=   %i\n", otno, DFMXO, out_iits );
                }
                else
                {
                    fprintf ( fp_out, " %i   Dfmxo= %.4E"
                              "    No. Inners=   %i\n", otno, DFMXO, out_iits );
                }
            }

            if ( OTRDONE ) break;
        } // end outer_loop

/***********************************************************************
 * Print the time cycle details. Add time cycle iterations.
 ***********************************************************************/
        if ( TIMEDEP == 1 && IPROC == ROOT )
        {
            if ( OTRDONE )
            {
                fprintf ( fp_out, "\n  Cycle= %i"
                          "    Time= %.4E"
                          "    No. Outers= %i"
                          "    No. Inners= %i\n",
                          cy, time, otno, cy_iits );
            }
            else
            {
                fprintf ( fp_out, "\n  ***UNCONVERGED*** Stopping Iterations!!"
                          "\n  Cycle= %i"
                          "    Time= %.4E"
                          "    No. Outers= %i"
                          "    No. Inners= %i\n",
                          cy, time, otno, cy_iits );

            }
        }
        else if ( IPROC == ROOT )
        {
            if ( OTRDONE )
            {
                fprintf ( fp_out,
                          "\n  No. Outers= %i"
                          "    No. Inners= %i\n",
                          otno, cy_iits );
            }
            else
            {
                fprintf ( fp_out, "\n  ***UNCONVERGED*** Stopping Iterations!!"
                          "\n  No. Outers= %i"
                          "    No. Inners= %i\n",
                          otno, cy_iits );

            }
        }

        tot_iits += cy_iits;

        if ( !OTRDONE ) break;
    } // end time_loop


    if ( TIMEDEP == 1 && IPROC == ROOT )
    {
        fprintf ( fp_out, "\n******************************\n"
                  "  Total inners for all time steps, outers = %i\n", tot_iits );
    }

    if ( IPROC == ROOT )
    {
        fprintf ( fp_out, "\n****************************************"
                  "****************************************\n\n");
    }

    t7 = wtime();
    TSLV = t7 - t1;

    tmp = ((double) NX)   * ((double) NY_GL) * ((double) NZ_GL)
        * ((double) NANG) * ((double) NOCT)  * ((double) tot_iits);
    TGRIND = TSLV*1.0E9 / tmp;
}
