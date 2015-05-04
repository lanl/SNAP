/***********************************************************************
 * Controls the reading of the input file and checking
 * the input parameters for acceptable values.
 ***********************************************************************/
#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 700
#endif

#include "snap.h"
#include <string.h>
#include <ctype.h>

/***********************************************************************
 * Constructor for input_data struct.
 ***********************************************************************/
void input_data_init ( input_data *input_vars )
{
    // Initialize parallel processing inputs
    NPEY     = 0;
    NPEZ     = 0;
    ICHUNK   = 0;
    NTHREADS = 0;
    NNESTED  = 0;

    // Initialize geometry inputs
    NDIMEN   = 0;
    NX       = 0;
    NY       = 0;
    NZ       = 0;
    LX       = 0;
    LY       = 0;
    LZ       = 0;

    // Intialize Sn inputs
    NMOM     = 0;
    NANG     = 0;

    // Initialize data inputs
    NG       = 0;
    MAT_OPT  = 0;
    SRC_OPT  = 0;
    SCATP    = 0;

    // Initialize control inputs
    EPSI     = 0;
    TF       = 0;
    IITM     = 0;
    OITM     = 0;
    TIMEDEP  = 0;
    NSTEPS   = 0;
    IT_DET   = 0;
    FLUXP    = 0;
    FIXUP    = 0;
}

/***********************************************************************
 * Read the input file.
 ***********************************************************************/
int read_input ( FILE *fp_in, FILE *fp_out, input_data *input_vars,
                 para_data *para_vars, time_data *time_vars )
{
/***********************************************************************
 * Local variables.
 ***********************************************************************/
    double t1, t2;

    int ierr = 0;

    char *error = NULL;

    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    char *tmpData = NULL;

    int tmpStrLen, i;

/***********************************************************************
 * Read the input file. Echo to output file. Call for an input variable
 * check. Only root reads, echoes, checks input.
 ***********************************************************************/

    t1 = wtime ();

    if ( IPROC == ROOT )
    {

        if ( !fp_in  )
        {
            tmpStrLen = strlen ("   ***ERROR: READ_INPUT:"
                                " Problem reading input file.\n");
            ALLOC_STR ( error, tmpStrLen + 1, &ierr );
            snprintf ( error, tmpStrLen + 1,
                       "   ***ERROR: READ_INPUT:"
                       " Problem reading input file.\n" );

            print_error ( fp_out, error, IPROC, ROOT );

            FREE ( error );

            ierr = 1;
        }
        else
        {
            while ( (read = getline(&line, &len, fp_in)) != -1 )
            {
                i = 0;
                while ( isspace(line[i]) )
                {
                    i++;
                }

                // Parallel processing inputs
                // npey: number of process elements in y-dir
                if ( strncmp(&line[i], "npey=", strlen("npey=")) == 0 )
                {
                    get_input_value ( &line[i], "npey=", &tmpData );
                    NPEY = atoi ( tmpData );
                }
                // npez: input number of process elements in z-dir
                else if ( strncmp(&line[i], "npez=", strlen("npez=")) == 0 )
                {
                    get_input_value ( &line[i], "npez=", &tmpData );
                    NPEZ = atoi ( tmpData );
                }
                // ichunk:
                else if ( strncmp(&line[i], "ichunk=", strlen("ichunk=")) == 0 )
                {
                    get_input_value ( &line[i], "ichunk=", &tmpData );
                    ICHUNK = atoi ( tmpData );
                }
                // nthreads: input number of threads
                else if ( strncmp(&line[i], "nthreads=", strlen("nthreads=")) == 0 )
                {
                    get_input_value ( &line[i], "nthreads=", &tmpData );
                    NTHREADS = atoi ( tmpData );
                }
                // nnested:
                else if ( strncmp(&line[i], "nnested=", strlen("nnested=")) == 0 )
                {
                    get_input_value ( &line[i], "nnested=", &tmpData );
                    NNESTED = atoi ( tmpData );
                }

                // Geometry inputs
                // ndimen:
                else if ( strncmp(&line[i], "ndimen=", strlen("ndimen=")) == 0 )
                {
                    get_input_value ( &line[i], "ndimen=", &tmpData );
                    NDIMEN = atoi ( tmpData );
                }
                // nx:
                else if ( strncmp(&line[i], "nx=", strlen("nx=")) == 0 )
                {
                    get_input_value ( &line[i], "nx=", &tmpData );
                    NX = atoi ( tmpData );
                }
                // ny:
                else if ( strncmp(&line[i], "ny=", strlen("ny=")) == 0 )
                {
                    get_input_value ( &line[i], "ny=", &tmpData );
                    NY = atoi ( tmpData );
                }
                // nz:
                else if ( strncmp(&line[i], "nz=", strlen("nz=")) == 0 )
                {
                    get_input_value ( &line[i], "nz=", &tmpData );
                    NZ = atoi ( tmpData );
                }
                // lx:
                else if ( strncmp(&line[i], "lx=", strlen("lx=")) == 0 )
                {
                    get_input_value ( &line[i], "lx=", &tmpData );
                    LX = atof ( tmpData );
                }
                // ly:
                else if ( strncmp(&line[i], "ly=", strlen("ly=")) == 0 )
                {
                    get_input_value ( &line[i], "ly=", &tmpData );
                    LY = atof ( tmpData );
                }
                // lz:
                else if ( strncmp(&line[i], "lz=", strlen("lz=")) == 0 )
                {
                    get_input_value ( &line[i], "lz=", &tmpData );
                    LZ = atof ( tmpData );
                }

                // Sn inputs
                // nmom:
                else if ( strncmp(&line[i], "nmom=", strlen("nmom=")) == 0 )
                {
                    get_input_value ( &line[i], "nmom=", &tmpData );
                    NMOM = atoi ( tmpData );
                }
                // nang:
                else if ( strncmp(&line[i], "nang=", strlen("nang=")) == 0 )
                {
                    get_input_value ( &line[i], "nang=", &tmpData );
                    NANG = atoi ( tmpData );
                }

                // Data inputs
                // ng:
                else if ( strncmp(&line[i], "ng=", strlen("ng=")) == 0 )
                {
                    get_input_value ( &line[i], "ng=", &tmpData );
                    NG = atoi ( tmpData );
                }
                // mat_opt:
                else if ( strncmp(&line[i], "mat_opt=", strlen("mat_opt=")) == 0 )
                {
                    get_input_value ( &line[i], "mat_opt=", &tmpData );
                    MAT_OPT = atoi ( tmpData );
                }
                // src_opt:
                else if ( strncmp(&line[i], "src_opt=", strlen("src_opt=")) == 0 )
                {
                    get_input_value ( &line[i], "src_opt=", &tmpData );
                    SRC_OPT = atoi ( tmpData );
                }
                // scatp:
                else if ( strncmp(&line[i], "scatp=", strlen("scatp=")) == 0 )
                {
                    get_input_value ( &line[i], "scatp=", &tmpData );
                    SCATP = atoi ( tmpData );
                }

                // Control inputs
                // epsi:
                else if ( strncmp(&line[i], "epsi=", strlen("epsi=")) == 0 )
                {
                    get_input_value ( &line[i], "epsi=", &tmpData );
                    EPSI = atof ( tmpData );
                }
                // tf:
                else if ( strncmp(&line[i], "tf=", strlen("tf=")) == 0 )
                {
                    get_input_value ( &line[i], "tf=", &tmpData );
                    TF = atof ( tmpData );
                }
                // iitm:
                else if ( strncmp(&line[i], "iitm=", strlen("iitm=")) == 0 )
                {
                    get_input_value ( &line[i], "iitm=", &tmpData );
                    IITM = atoi ( tmpData );
                }
                // oitm:
                else if ( strncmp(&line[i], "oitm=", strlen("oitm=")) == 0 )
                {
                    get_input_value ( &line[i], "oitm=", &tmpData );
                    OITM = atoi ( tmpData );
                }
                // timedep:
                else if ( strncmp(&line[i], "timedep=", strlen("timedep=")) == 0 )
                {
                    get_input_value ( &line[i], "timedep=", &tmpData );
                    TIMEDEP = atoi ( tmpData );
                }
                // nsteps:
                else if ( strncmp(&line[i], "nsteps=", strlen("nsteps=")) == 0 )
                {
                    get_input_value ( &line[i], "nsteps=", &tmpData );
                    NSTEPS = atoi ( tmpData );
                }
                // it_det:
                else if ( strncmp(&line[i], "it_det=", strlen("it_det=")) == 0 )
                {
                    get_input_value ( &line[i], "it_det=", &tmpData );
                    IT_DET = atoi ( tmpData );
                }
                // fluxp:
                else if ( strncmp(&line[i], "fluxp=", strlen("fluxp=")) == 0 )
                {
                    get_input_value ( &line[i], "fluxp=", &tmpData );
                    FLUXP = atoi ( tmpData );
                }
                // fixup:
                else if ( strncmp(&line[i], "fixup=", strlen("fixup=")) == 0 )
                {
                    get_input_value ( &line[i], "fixup=", &tmpData );
                    FIXUP = atoi ( tmpData );
                }
            }

            // Free temp data from memory
            FREE ( tmpData );
        }
    }

    bcast_i_scalar ( &ierr, COMM_SNAP, ROOT, NPROC );

    if ( ierr != 0 )
    {
        return ierr;
    }

    if ( IPROC == ROOT )
    {
        input_echo ( input_vars, fp_out );
        ierr = input_check ( fp_out, input_vars, para_vars );
    }

/***********************************************************************
 * Broadcast the data to all processes.
 ***********************************************************************/
    ierr = var_bcast ( input_vars, para_vars );

    t2 = wtime();
    time_vars->tinp = t2 - t1;

    return ierr;
}

/* Parse out the value located at current input data line */
void get_input_value ( char *lineData, char *valueID, char **tmpData )
{
    int tmpStrLen = 0;
    int ierr = 0;

    // Trim all excess chars and white space following immediate input value
    while ( !(isspace(lineData[strlen(valueID) + tmpStrLen])) )
    {
        tmpStrLen += 1;
    }

    ALLOC_STR( *tmpData, tmpStrLen, &ierr );

    // Copies value to tmpData string,
    snprintf ( *tmpData, tmpStrLen+1, "%s",
               &lineData[strlen(valueID)] );
}

/***********************************************************************
 * Print the input variables to the output file
 ***********************************************************************/
void input_echo ( input_data *input_vars, FILE *fp_out )
{
    fprintf ( fp_out, "****************************************"
             "****************************************\n\n"
             "          Input Echo - Values from input or default\n"
             "****************************************"
             "****************************************\n\n" );

    fprintf ( fp_out, "  NML=invar\n" );
    fprintf ( fp_out, "     npey=     %i\n", NPEY );
    fprintf ( fp_out, "     npez=     %i\n", NPEZ );
    fprintf ( fp_out, "     ichunk=   %i\n", ICHUNK );
    fprintf ( fp_out, "     nthreads= %i\n", NTHREADS );
    fprintf ( fp_out, "     nnested=  %i\n", NNESTED );
    fprintf ( fp_out, "     ndimen=   %i\n", NDIMEN );
    fprintf ( fp_out, "     nx=       %i\n", NX );
    fprintf ( fp_out, "     ny=       %i\n", NY );
    fprintf ( fp_out, "     nz=       %i\n", NZ );
    fprintf ( fp_out, "     lx=       %.4E\n", LX );
    fprintf ( fp_out, "     ly=       %.4E\n", LY );
    fprintf ( fp_out, "     lz=       %.4E\n", LZ );
    fprintf ( fp_out, "     nmom=     %i\n", NMOM );
    fprintf ( fp_out, "     nang=     %i\n", NANG );
    fprintf ( fp_out, "     ng=       %i\n", NG );
    fprintf ( fp_out, "     mat_opt=  %i\n", MAT_OPT );
    fprintf ( fp_out, "     src_opt=  %i\n", SRC_OPT );
    fprintf ( fp_out, "     scatp=    %i\n", SCATP );
    fprintf ( fp_out, "     epsi=     %.4E\n", EPSI );
    fprintf ( fp_out, "     iitm=     %i\n", IITM );
    fprintf ( fp_out, "     oitm=     %i\n", OITM );
    fprintf ( fp_out, "     timedep=  %i\n", TIMEDEP );
    fprintf ( fp_out, "     tf=       %.4E\n", TF );
    fprintf ( fp_out, "     nsteps=   %i\n", NSTEPS );
    fprintf ( fp_out, "     it_det=   %i\n", IT_DET );
    fprintf ( fp_out, "     fluxp=    %i\n", FLUXP );
    fprintf ( fp_out, "     fixup=    %i\n\n", FIXUP );
    fprintf ( fp_out, "****************************************"
              "****************************************\n\n" );
}

/***********************************************************************
 * Checks input for valid entries.
 ***********************************************************************/
int input_check ( FILE *fp_out, input_data *input_vars, para_data *para_vars )
{
    // Local variables
    char *error = NULL;
    int ierr = 0;

    int tmpStrLen;

    // Check the parallel environment variables.
    // Parallel processing inputs.

    if ( NPEY < 1 )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NPEY must be positive\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NPEY must be positive\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( NPEZ < 1 )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NPEZ must be positive\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NPEZ must be positive\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (NDIMEN > 1) && (NY % NPEY != 0) )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NPEY must divide evenly into NY\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NPEY must divide evenly into NY\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (NDIMEN > 2) && (NZ % NPEZ != 0) )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NPEZ must divide evenly into NZ\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NPEZ must divide evenly into NZ\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (NDIMEN < 2) && (NPEY != 1) )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NPEY must be 1 if not 2-D problem\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NPEY must be 1 if not 2-D problem\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (NDIMEN < 3) && (NPEZ != 1) )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NPEZ must be 1 if not 3-D problem\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NPEZ must be 1 if not 3-D problem\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( NPEY * NPEZ != NPROC )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NPEY*NPEZ must equal MPI NPROC\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1 ,
                   "   ***ERROR: INPUT_CHECK:"
                   " NPEY*NPEZ must equal MPI NPROC\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( ICHUNK > NX )
    {
        ICHUNK = NX;

        tmpStrLen = strlen ( "   *WARNING: INPUT_CHECK:"
                             " ICHUNK cannot exceed NX; reset to NX\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   *WARNING: INPUT_CHECK:"
                   " ICHUNK cannot exceed NX; reset to NX\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (NDIMEN == 1) && (ICHUNK != NX) )
        ICHUNK = NX;

    if ( NX % ICHUNK != 0 )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " ICHUNK must divide evenly into NX\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " ICHUNK must divide evenly into NX\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( NTHREADS < 1 )
    {
        NTHREADS = 1;

        tmpStrLen = strlen ( "   *WARNING: INPUT_CHECK:"
                             " NTHREADS must be positive;"
                             " reset to 1\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   *WARNING: INPUT_CHECK:"
                   " NTHREADS must be positive;"
                   " reset to 1\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( NNESTED < 0 )
    {
        NNESTED = 0;

        tmpStrLen = strlen ( "   *WARNING: INPUT_CHECK:"
                             " NNESTED must be non-negative;"
                             " reset to 0\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   *WARNING: INPUT_CHECK:"
                   " NNESTED must be non-negative;"
                   " reset to 0\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( NNESTED == 1 )
    {
        NNESTED = 0;

        tmpStrLen = strlen ( "   *WARNING: INPUT_CHECK:"
                             " NNESTED=1 same as 0+overhead;"
                             " reset to 0\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   *WARNING: INPUT_CHECK:"
                   " NNESTED=1 same as 0+overhead;"
                   " reset to 0\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    // Geometry inputs.
    if ( (NDIMEN < 1) || (NDIMEN > 3) )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NDIMEN must be 1, 2, or 3\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NDIMEN must be 1, 2, or 3\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( NX < 4 )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NX must be at least 4\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NX must be at least 4\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (NY < 4) && (NDIMEN > 1) )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NY must be at least 4\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NY must be at least 4\n");

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (NZ < 4) && (NDIMEN > 2) )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NZ must be at least 4\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NZ must be at least 4\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( LX <= 0 )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " LX must be positive\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " LX must be positive\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( LY <= 0 )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " LY must be positive\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
            "   ***ERROR: INPUT_CHECK:"
                   " LY must be positive\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( LZ <= 0 )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " LZ must be positive\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " LZ must be positive\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (NDIMEN < 2) && ((NY > 1) || (LY != 1)) )
    {
        NY = 1;
        LY = 0;

        tmpStrLen = strlen ( "   *WARNING: INPUT_CHECK:"
                             " NY and LY reset for 1-D problem\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   *WARNING: INPUT_CHECK:"
                   "NY and LY reset for 1-D problem\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (NDIMEN < 3) && ((NZ > 1) || (LZ != 1)) )
    {
        NZ = 1;
        LZ = 0;

        tmpStrLen = strlen ( "   *WARNING: INPUT_CHECK:"
                             " NZ and LZ reset for 1/2-D problem\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   *WARNING: INPUT_CHECK:"
                   " NZ and LZ reset for 1/2-D problem\n" );

      print_error ( fp_out, error, IPROC, ROOT );
    }

    // Sn inputs.
    if ( (NMOM < 1) || (NMOM > 4) )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NMOM must be positive and less than 5\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NMOM must be positive and less than 5\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( NANG < 1 )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NANG must be positive\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NANG must be positive\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    // Data inputs.
    if ( NG < 1 )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NG must be positive\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NG must be positive\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( NTHREADS > NG )
    {
        NTHREADS = NG;

        tmpStrLen = strlen ( "   *WARNING: INPUT_CHECK:"
                             " NTHREADS should be <= NG;"
                             " reset to NG\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   *WARNING: INPUT_CHECK:"
                   " NTHREADS should be <= NG;"
                   " reset to NG\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (MAT_OPT < 0) || (MAT_OPT > 2) )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " MAT_OPT must be 0/1/2\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " MAT_OPT must be 0/1/2\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (SRC_OPT < 0) || (SRC_OPT > 3) )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " SRC_OPT must be 0/1/2/3\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " SRC_OPT must be 0/1/2/3\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

// Commented out in original code
//    if ( (SRC_OPT == 3) && (MAT_OPT != 0) )
//    {
//        MAT_OPT = 0;
//
//        tmpStrLen = strlen ( "   *WARNING: INPUT_CHECK:"
//                             " MAT_OPT must be 0 for SRC_OPT=3\n" );
//        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
//        snprintf ( error, tmpStrLen + 1,
//                   "   *WARNING: INPUT_CHECK:"
//                   " MAT_OPT must be 0 for SRC_OPT=3\n" );
//
//      print_error ( fp_out, error, IPROC, ROOT );
//    }

    if ( (SCATP != 0) && (SCATP != 1) )
    {
        SCATP = 0;

        tmpStrLen = strlen ( "   *WARNING: INPUT_CHECK:"
                             " SCATP must be 0/1; reset to 0\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   *WARNING: INPUT_CHECK:"
                   " SCATP must be 0/1; reset to 0\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    // Control inputs.
    if ( (EPSI <= 0) || (EPSI >= 1.0E-2) )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " EPSI must be positive, less than 1.0E-2\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " EPSI must be positive, less than 1.0E-2\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( IITM < 1 )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " IITM must be positive\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " IITM must be positive\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( OITM < 1 )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " OITM must be positive\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " OITM must be positive\n" );

        print_error ( fp_out, error,  IPROC, ROOT );
    }

    if ( (TIMEDEP != 0) && (TIMEDEP != 1) )
    {
        TIMEDEP = 0;

        tmpStrLen = strlen ( "   *WARNING: INPUT_CHECK:"
                             " TIMEDEP must be 0/1; reset to 0\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   *WARNING: INPUT_CHECK:"
                   " TIMEDEP must be 0/1; reset to 0\n" );

      print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (TIMEDEP == 1) && (TF <= 0) )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " TF must be positive\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " TF must be positive\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (TIMEDEP == 1) && (NSTEPS < 1) )
    {
        ierr = ierr + 1;

        tmpStrLen = strlen ( "   ***ERROR: INPUT_CHECK:"
                             " NSTEPS must be positive\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***ERROR: INPUT_CHECK:"
                   " NSTEPS must be positive\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (TIMEDEP == 0) && (NSTEPS != 1) )
    {
        NSTEPS = 1;

        tmpStrLen = strlen ( "   ***WARNING: INPUT_CHECK:"
                             " NSTEPS reset to 1 for static calc\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***WARNING: INPUT_CHECK:"
                   " NSTEPS reset to 1 for static calc\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (IT_DET != 0) && (IT_DET != 1) )
    {
        IT_DET = 0;

        tmpStrLen = strlen ( "   ***WARNING: INPUT_CHECK:"
                             " IT_DET must equal 0/1; reset to 0\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***WARNING: INPUT_CHECK:"
                   " IT_DET must equal 0/1; reset to 0\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (FLUXP < 0) || (FLUXP > 2) )
    {
        FLUXP = 0;

        tmpStrLen = strlen ( "   ***WARNING: INPUT_CHECK:"
                             " FLUXP must equal 0/1/2; reset to 0\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***WARNING: INPUT_CHECK:"
                   " FLUXP must equal 0/1/2; reset to 0\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    if ( (FIXUP != 0) && (FIXUP != 1) )
    {
        FIXUP = 0;

        tmpStrLen = strlen ( "   ***WARNING: INPUT_CHECK:"
                             " FIXUP must equal 0/1; reset to 0\n" );
        ALLOC_STR ( error, tmpStrLen + 1, &ierr );
        snprintf ( error, tmpStrLen + 1,
                   "   ***WARNING: INPUT_CHECK:"
                   " FIXUP must equal 0/1; reset to 0\n" );

        print_error ( fp_out, error, IPROC, ROOT );
    }

    return ierr;
}

/***********************************************************************
 * Broadcast input variables from root to all other procs
 * To do: Create MPI_Type_create_struct to skip packing
 * struct data into array.
 ***********************************************************************/
int var_bcast ( input_data *input_vars, para_data *para_vars )
{
    int ierr = 0;

    int *ipak;
    int ilen = 30;
    ALLOC_1D(ipak, ilen, int, &ierr);

    double *dpak;
    int dlen = 15;
    ALLOC_1D(dpak, dlen, double, &ierr);

    if ( IPROC == ROOT )
    {
        ipak[0] = NPEY;
        ipak[1] = NPEZ;
        ipak[2] = ICHUNK;
        ipak[3] = NTHREADS;
        ipak[4] = NDIMEN;
        ipak[5] = NX;
        ipak[6] = NY;
        ipak[7] = NZ;
        ipak[8] = NMOM;
        ipak[9] = NANG;
        ipak[10] = NG;
        ipak[11] = TIMEDEP;
        ipak[12] = NSTEPS;
        ipak[13] = IITM;
        ipak[14] = OITM;
        ipak[15] = MAT_OPT;
        ipak[16] = SRC_OPT;
        ipak[17] = SCATP;
        ipak[18] = IT_DET;
        ipak[19] = FLUXP;
        ipak[20] = FIXUP;
        ipak[21] = NNESTED;

        dpak[0] = LX;
        dpak[1] = LY;
        dpak[2] = LZ;
        dpak[3] = TF;
        dpak[4] = EPSI;
    }

/***********************************************************************
 * Broadcast data.
 ***********************************************************************/
#ifdef DEBUG
#if ((DEBUG > 1) && (DEBUG < 3)) || (DEBUG > 10)
    int i = 0;
    if ( IPROC == ROOT )
    {
        for ( i = 0; i < ilen; i++ )
        {
            printf ( "Int value at %i: %i, from %i proc\n", i, ipak[i], IPROC );
        }
        for ( i = 0; i < dlen; i++ )
        {
            printf ( "Double value at %i: %.4E, from %i proc\n", i, dpak[i], IPROC );
        }
    }
#endif
#endif

    bcast_i_1d ( ipak, ilen, COMM_SNAP, ROOT, NPROC );
    bcast_d_1d ( dpak, dlen, COMM_SNAP, ROOT, NPROC );

    if ( IPROC != ROOT )
    {
        NPEY     =  ipak[0];
        NPEZ     =  ipak[1];
        ICHUNK   =  ipak[2];
        NTHREADS =  ipak[3];
        NDIMEN   =  ipak[4];
        NX       =  ipak[5];
        NY       =  ipak[6];
        NZ       =  ipak[7];
        NMOM     =  ipak[8];
        NANG     =  ipak[9];
        NG       = ipak[10];
        TIMEDEP  = ipak[11];
        NSTEPS   = ipak[12];
        IITM     = ipak[13];
        OITM     = ipak[14];
        MAT_OPT  = ipak[15];
        SRC_OPT  = ipak[16];
        SCATP    = ipak[17];
        IT_DET   = ipak[18];
        FLUXP    = ipak[19];
        FIXUP    = ipak[20];
        NNESTED  = ipak[21];

        LX   = dpak[0];
        LY   = dpak[1];
        LZ   = dpak[2];
        TF   = dpak[3];
        EPSI = dpak[4];
    }

    ierr = barrier ( COMM_SNAP );

#ifdef DEBUG
#if ((DEBUG > 1) && (DEBUG < 3)) || (DEBUG > 10)
    if ( IPROC != ROOT )
    {
        for ( i = 0; i < ilen; i++ )
        {
            printf ( "Int value at %i: %i, from %i proc\n", i, ipak[i], IPROC );
        }
        for ( i = 0; i < dlen; i++ )
        {
            printf ( "Double value at %i: %.4E, from %i proc\n", i, dpak[i], IPROC );
        }
}
#endif
#endif

    FREE (ipak);
    FREE (dpak);

    return ierr;
}
