
/***********************************************************************
 * Contains utility subroutines for handling file open/close,
 * errors, command line reading, and program termination.
 ***********************************************************************/
#include "snap.h"
#include <string.h>

/***********************************************************************
 * Reads the command line arguments to get the input
 * and output file names.
 ***********************************************************************/
int cmdarg ( int argc, char *argv[],  char **inputFile, char **outputFile,
              char **error, int iproc, int root )
{
    if ( iproc != root ) return 0;

    // local variables
    int arg_ind, tmpStrLen, ierr = 0;
    char tmpStr[256];

    // Verify the required number of arguments
    // are met, else return error and usage
    if ( argc < 5 )
    {
        ierr = 1;
        tmpStrLen = strlen ( "***ERROR: CMDARG:"
                             " Missing command line entry.\n"
                             "Usage:\n\tmpirun -np [#] [path]/snap"
                             " --fi [infile] --fo [outfile]\n" );

        ALLOC_STR(*error, tmpStrLen + 1, &ierr);

        snprintf ( (char *) *error, tmpStrLen + 1,
                   "***ERROR: CMDARG:"
                  " Missing command line entry.\n"
                   "Usage:\n\tmpirun -np [#] [path]/snap"
                  " --fi [infile] --fo [outfile]\n" );

        return ierr;
     }
    else
    {
        for ( arg_ind=1; arg_ind < argc; arg_ind++ )
        {
            // Argument flags defined on odd input args
            if ( arg_ind % 2 == 1 )
            {
                // Set the inputFile name
                if ( strcmp(argv[arg_ind], "--fi" ) == 0)
                {
                    tmpStrLen = strlen ( argv[arg_ind+1] );

                    ALLOC_STR(*inputFile, tmpStrLen + 1, &ierr);

                    snprintf ( (char *) *inputFile, tmpStrLen + 1,
                               "%s", argv[arg_ind+1] );
                }

                // Set the outputFile name
                else if ( strcmp(argv[arg_ind], "--fo") == 0 )
                {
                    tmpStrLen = strlen ( argv[arg_ind+1] );

                    ALLOC_STR(*outputFile, tmpStrLen + 1, &ierr);

                    snprintf ( (char *) *outputFile, tmpStrLen + 1,
                               "%s", argv[arg_ind+1] );
                }

                // Detect non-valid argument flags and return error
                else
                {
                    tmpStrLen = strlen ( "***ERROR: CMDARG: Argument \"" )
                        + strlen ( argv[arg_ind] )
                        + strlen ( "\" not recognized.\n"
                                   "Usage:\n"
                                   "\tmpirun -np [#] [path]/snap"
                                   " --fi [infile] --fo [outfile]\n" );

                    snprintf ( tmpStr, tmpStrLen + 1,
                               "***ERROR: CMDARG: Argument \""
                               "%s"
                               "\" not recognized.\n"
                               "Usage:\n"
                               "\tmpirun -np [#] [path]/snap"
                               " --fi [infile] --fo [outfile]\n",
                               argv[arg_ind] );

                    tmpStrLen = strlen ( tmpStr );

                    ALLOC_STR(*error, tmpStrLen + 1, &ierr);

                    snprintf ( (char *) *error, tmpStrLen + 1,
                               "%s", tmpStr );

                    return 1;
                }
            }
        }
    }
    return 0;
}

/***********************************************************************
 * Opens file for read/write operations.
 ***********************************************************************/
int open_file ( FILE **fp, char *fileName, char *fileAction,
                char **error, int iproc, int root )
{
    if ( iproc != root) return 0;

    // local variables
    char tmpStr[256];
    int tmpStrLen;
    int ierr = 0;

    // Open the file for specified action
    *fp = fopen ( fileName, fileAction );
    if ( !*fp )
    {
        tmpStrLen = strlen ( "***ERROR: OPEN_FILE: Unable to open file: \"" )
            + strlen ( fileName ) + strlen ( "\"\n" );

        snprintf ( tmpStr, tmpStrLen + 1,
                   "***ERROR: OPEN_FILE: Unable to open file: \"%s\"\n",
                   fileName );

        tmpStrLen = strlen ( tmpStr );

        ALLOC_STR(*error, tmpStrLen + 1, &ierr);

        snprintf ( (char *) *error, tmpStrLen + 1, "%s", tmpStr );

        return 1;
    }

    return 0;
}

/***********************************************************************
 * Closes the file.
 ***********************************************************************/
int close_file ( FILE *fp, char *fileName, char **error,
                 int root, int iproc )
{
    if ( iproc != root ) return 0;

    // local variables
    char tmpStr[256];
    int ierr =0;
    int tmpStrLen;

    // Close the file
    ierr = fclose ( fp );
    if ( ierr != 0 )
    {
        tmpStrLen = strlen ( "***ERROR: CLOSE_FILE: Unable to close file: \"" )
            + strlen ( fileName ) + strlen ( "\"\n" );

        snprintf ( tmpStr, tmpStrLen + 1,
                   "***ERROR: CLOSE_FILE: Unable to close file: \"%s\"\n",
                   fileName );

        tmpStrLen = strlen ( tmpStr );

        ALLOC_STR(*error, tmpStrLen + 1, &ierr);

        snprintf ( (char *) *error, tmpStrLen + 1, "%s", tmpStr );

        return ierr;
    }

    return 0;
}

/***********************************************************************
 * Print error to file, else print to stdout.
 ***********************************************************************/
void print_error ( FILE *fp, char *error, int iproc, int root )
{
    if ( iproc != root ) return;

    if ( !fp ) printf ( "%s\n", error );
    else fprintf( fp, "%s\n", error );
}

/***********************************************************************
 * Verify if a string is empty.
 ***********************************************************************/
int string_empty ( char *stringName )
{
    if ( stringName == NULL )
        return 1;
    else
        return 0;
}

/***********************************************************************
 * Safely end program execution.
 ***********************************************************************/
void stop_run ( int inputFlag, int solveFlag, int statusFlag, para_data *para_vars,
                sn_data *sn_vars, data_data *data_vars, mms_data *mms_vars,
                geom_data *geom_vars, solvar_data *solvar_vars, control_data *control_vars )
{
    if ( inputFlag > 0 )
    {
        dealloc_input ( inputFlag, sn_vars, data_vars, mms_vars );
    }

    if ( solveFlag > 0 )
    {
        dealloc_solve ( solveFlag, geom_vars, solvar_vars, control_vars );
    }

    if ( IPROC == ROOT )
    {
        if ( statusFlag == 0 )
            printf ( "Aww SNAP. Program failed. Try again.\n" );
        else if ( statusFlag == 1 )
            printf ( "Success! Done in a SNAP!\n" );
        else if ( statusFlag == 2 )
            printf ( "Oh SNAP. That did not converge."
                     " But take a look at the Timing Summary anyway!\n");
    }

    pend();

    exit(0);
}

