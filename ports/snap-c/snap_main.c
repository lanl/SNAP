/***********************************************************************
!    SNAP - SN Application Proxy
!
!    Parallel programming model based on PARTISN
!
!
!    SNAP: SN (Discrete Ordinates) Application Proxy
!    Version 1.x
!
!    Author: Aaron Taylor, aaron.d.taylor@intel.com
!
!    This C99 version of SNAP is ported from the Fortran90 SNAP
!    version 1.01 developed by Los Alamos National Labs.
!
!
 ***********************************************************************/
#include "snap.h"

int main ( int argc, char *argv[] )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int ierr    = 0;    // Flag errors
    char *error = NULL; // Error message string

    double t1, t2, t3, t4, t5; // Variables to track runtime

    /* File in/out handling variables */
    char *inputFile = NULL, *outputFile = NULL;
    FILE *fp_in     = NULL, *fp_out     = NULL;

    input_data input_vars;               // Struct to store input variables
    input_data_init ( &input_vars );     // Set variables to initial values

    para_data para_vars;                 // Struct to store parallel variables
    para_data_init ( &para_vars );       // Set variables to initial values

    time_data time_vars;                 // Struct to store time variables
    time_data_init ( &time_vars );       // Set variables to initial values

    geom_data geom_vars;                 // Struct to store geometry variables
    geom_data_init ( &geom_vars );       // Set variables to initial values

    sn_data sn_vars;                     // Struct to store sn variables
    sn_data_init ( &sn_vars );           // Set variables to initial values

    data_data data_vars;                 // Struct to store data variables
    data_data_init ( &data_vars );       // Set variables to initial values

    control_data control_vars;           // Struct to store control variables
    control_data_init ( &control_vars ); // Set variables to initial values

    mms_data mms_vars;                   // Struct to store mms variables
    mms_data_init ( &mms_vars );         // Set variables to initial values

    solvar_data solvar_vars;             // Struct to store solvar variables
    solvar_data_init ( &solvar_vars );   // Set variables to initial values

    sweep_data sweep_vars;               // Struct to store sweep variables
    sweep_data_init ( &sweep_vars );     // Set variables to initial values

    dim_sweep_data dim_sweep_vars;

/***********************************************************************
 * Perform calls that set up the parallel environment in MPI and
 * OpenMP. Also starts the timer. Update parallel setup time.
 ***********************************************************************/
    /* Initialize the parallel environment */
    pinit ( argc, argv, &para_vars, &t1, &ierr );

    t2 = wtime(); // Get the MPI walltime

    /* Calc parallel setup time */
    time_vars.tparset = time_vars.tparset + t2 - t1;


/***********************************************************************
 * Read the command line arguments to get i/o file names.
 * Open the two files.
 ***********************************************************************/
    ierr = cmdarg ( argc, argv, &inputFile, &outputFile, &error,
             para_vars.iproc, para_vars.root );

    bcast_i_scalar ( &ierr, para_vars.comm_snap, para_vars.root, para_vars.nproc );

    /* Ensure arguments are valid */
    if ( ierr != 0 )
    {
        print_error ( NULL, error, para_vars.iproc, para_vars.root );

        FREE ( inputFile );
        FREE ( outputFile );
        FREE ( error );

        stop_run ( 0, 0, 0, &para_vars, &sn_vars, &data_vars, &mms_vars,
                   &geom_vars, &solvar_vars, &control_vars );
    }

    /* Open the input file to read in initial values */
    ierr = open_file ( &fp_in, inputFile, "r", &error, para_vars.iproc, para_vars.root );

    bcast_i_scalar ( &ierr, para_vars.comm_snap, para_vars.root, para_vars.nproc);

    /* Ensure input file is found */
    if ( ierr != 0 )
    {
        print_error ( NULL, error, para_vars.iproc, para_vars.root );

        FREE ( inputFile );
        FREE ( outputFile );
        FREE ( error );

        stop_run ( 0, 0, 0, &para_vars, &sn_vars, &data_vars, &mms_vars,
                   &geom_vars, &solvar_vars, &control_vars );
    }

    /* Open the output file to write results and error messages */
    ierr = open_file ( &fp_out, outputFile, "w", &error, para_vars.iproc, para_vars.root );

    bcast_i_scalar ( &ierr, para_vars.comm_snap, para_vars.root, para_vars.nproc );

    /* Ensure output file can be created */
    if ( ierr != 0 )
    {
        print_error ( NULL, error, para_vars.iproc, para_vars.root );

        FREE ( inputFile );
        FREE ( outputFile );
        FREE ( error );

        stop_run ( 0, 0, 0, &para_vars, &sn_vars, &data_vars, &mms_vars,
                   &geom_vars, &solvar_vars, &control_vars );
    }

/***********************************************************************
 * Write code version and execution time to output.
 ***********************************************************************/
    if ( para_vars.iproc == para_vars.root ) version_print ( fp_out );

/***********************************************************************
 * Read input
 ***********************************************************************/
    ierr = read_input ( fp_in, fp_out, &input_vars, &para_vars, &time_vars );

    ierr = close_file ( fp_in, inputFile, &error, para_vars.iproc,
                        para_vars.root );

    FREE ( inputFile );

    bcast_i_scalar ( &ierr, para_vars.comm_snap, para_vars.root, para_vars.nproc );

    if ( ierr != 0 )
    {
        print_error ( fp_out, error, para_vars.iproc, para_vars.root );

        FREE ( outputFile );
        FREE ( error );

        stop_run ( 0, 0, 0, &para_vars, &sn_vars, &data_vars, &mms_vars,
                   &geom_vars, &solvar_vars, &control_vars );
    }

/***********************************************************************
 * Get nthreads for each proc. Print the warning about resetting
 * nthreads if necessary. Don't stop run. Set up the SDD MPI topology.
 ***********************************************************************/
    t3 = wtime();

    pinit_omp ( para_vars.comm_snap, &input_vars.nthreads, input_vars.nnested,
                &para_vars.do_nested, &ierr, &error );

    if ( ierr != 0 ) print_error ( NULL, error, para_vars.iproc,
                                   para_vars.root );

    pcomm_set ( input_vars.npey, input_vars.npez, &para_vars, &ierr );

    t4 = wtime();

    time_vars.tparset = time_vars.tparset + t4 - t3;

/***********************************************************************
 * Setup problem
 ***********************************************************************/
    setup ( &input_vars, &para_vars, &time_vars, &geom_vars, &sn_vars,
            &data_vars, &solvar_vars, &control_vars, &mms_vars, fp_out,
            &ierr, &error );

/***********************************************************************
 * Call for the problem solution
 ***********************************************************************/
    translv ( &input_vars, &para_vars, &time_vars, &geom_vars, &sn_vars,
              &data_vars, &control_vars, &solvar_vars, &mms_vars,
              &sweep_vars, &dim_sweep_vars, fp_out, &ierr, &error );

/***********************************************************************
 * Output the results. Print the timing summary
 ***********************************************************************/
    output ( &input_vars, &para_vars, &time_vars, &geom_vars, &data_vars,
             &sn_vars, &control_vars, &mms_vars, &solvar_vars, &sweep_vars,
             fp_out, &ierr, &error );

    if ( para_vars.iproc == para_vars.root ) time_summ ( fp_out, &time_vars );

/***********************************************************************
 * Final cleanup: deallocate, close output file, end the program
 ***********************************************************************/
    dealloc_input ( 3, &sn_vars, &data_vars, &mms_vars );
    dealloc_solve ( 3, &geom_vars, &solvar_vars, &control_vars );

    t5 = wtime();
    time_vars.tsnap = t5 - t1;

    if ( para_vars.iproc == para_vars.root )
    {
        fprintf ( fp_out, "  Total Execution time"
                  "                   %.4E\n\n", time_vars.tsnap );
        fprintf ( fp_out, "  Grind Time (nanoseconds)"
                  "         %.4E\n\n", time_vars.tgrind );
        fprintf ( fp_out,
                  "****************************************"
                  "****************************************\n" );
    }

    ierr = close_file ( fp_out, outputFile, &error, para_vars.iproc, para_vars.root );

    FREE ( outputFile );

    bcast_i_scalar ( &ierr, para_vars.comm_snap, para_vars.root, para_vars.nproc );

    if ( ierr != 0 )
    {
        print_error ( 0, error, para_vars.iproc, para_vars.root );

        FREE ( error );

        stop_run ( 0, 0, 0, &para_vars, &sn_vars, &data_vars, &mms_vars,
                   &geom_vars, &solvar_vars, &control_vars );
    }


    if ( control_vars.otrdone )
    {
        FREE ( error );

        stop_run ( 0, 0, 1, &para_vars, &sn_vars, &data_vars, &mms_vars,
                   &geom_vars, &solvar_vars, &control_vars );
    }
    else
    {
        FREE ( error );

        stop_run ( 0, 0, 2, &para_vars, &sn_vars, &data_vars, &mms_vars,
                   &geom_vars, &solvar_vars, &control_vars );
    }

    return 0;
}
