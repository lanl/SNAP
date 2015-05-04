/***********************************************************************
 * MODULE: time_module
 * This module contains the variables that measure SNAP's execution
 * times for different pieces of code and the subroutine used to get the
 * time. It also has the timing summary print.
  ***********************************************************************/
#include "snap.h"

/***********************************************************************
 * Initialize time_data struct values to 0
  ***********************************************************************/
void time_data_init ( time_data *time_vars )
{
    time_vars->tsnap    = 0;
    time_vars->tparset  = 0;
    time_vars->tinp     = 0;
    time_vars->tset     = 0;
    time_vars->tslv     = 0;
    time_vars->tparam   = 0;
    time_vars->totrsrc  = 0;
    time_vars->tinners  = 0;
    time_vars->tinrsrc  = 0;
    time_vars->tsweeps  = 0;
    time_vars->tinrmisc = 0;
    time_vars->tslvmisc = 0;
    time_vars->tout     = 0;
    time_vars->tgrind   = 0;
}


/***********************************************************************
 * Get the current walltime
  ***********************************************************************/
double wtime ( )
{
    return MPI_Wtime ( );;
}

void time_summ ( FILE *fp_out, time_data *time_vars )
{
    TINRMISC = TINNERS - ( TINRSRC + TSWEEPS );
    TSLVMISC = TSLV - ( TPARAM + TOTRSRC + TINNERS );

    fprintf ( fp_out, "\n          Timing Summary\n"
              "****************************************"
              "****************************************\n\n" );

    fprintf ( fp_out, "  Code Section                          Time (seconds)\n" );
    fprintf ( fp_out, " **************                        ****************\n" );
    fprintf ( fp_out, "    Parallel Setup                       %.4E\n", TPARSET );
    fprintf ( fp_out, "    Input                                %.4E\n", TINP );
    fprintf ( fp_out, "    Setup                                %.4E\n", TSET );
    fprintf ( fp_out, "    Solve                                %.4E\n", TSLV );
    fprintf ( fp_out, "       Parameter Setup                   %.4E\n", TPARAM );
    fprintf ( fp_out, "       Outer Source                      %.4E\n", TOTRSRC );
    fprintf ( fp_out, "       Inner Iterations                  %.4E\n", TINNERS );
    fprintf ( fp_out, "          Inner Source                   %.4E\n", TINRSRC );
    fprintf ( fp_out, "          Transport Sweeps               %.4E\n", TSWEEPS );
    fprintf ( fp_out, "          Inner Misc Ops                 %.4E\n", TINRMISC );
    fprintf ( fp_out, "       Solution Misc Ops                 %.4E\n", TSLVMISC );
    fprintf ( fp_out, "    Output                               %.4E\n", TOUT );
}
