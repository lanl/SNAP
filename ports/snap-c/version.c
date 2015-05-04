/***********************************************************************
 * Handles version information
 ***********************************************************************/
#include "snap.h"
#include <time.h>

// SNAP release number and release date.
int version = 101;
char *cvers = "1.01";
char *vdate = "11-10-2014";

/***********************************************************************
 * Print version information
 ***********************************************************************/
void version_print ( FILE *fp_out )
{
    // Local variables.
    time_t t = time ( NULL );
    struct tm tm = *localtime ( &t );

    // Call intrinsics to get current date and time.
    // Print version information to out file.
    fprintf ( fp_out, " SNAP: SN (Discrete Ordinates) Application Proxy\n" );
    fprintf ( fp_out, " Version Number..  %s\n", cvers );
    fprintf ( fp_out, " Version Date..  %s\n", vdate);
    fprintf ( fp_out, " Ran on %d-%d-%d", tm.tm_mon + 1, tm.tm_mday, tm.tm_year);
    fprintf ( fp_out, " at time %d:%d:%d\n\n", tm.tm_hour, tm.tm_min, tm.tm_sec );
}
