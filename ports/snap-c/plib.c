/*******************************************************************************
 * Module: plib.c
 * Contains the variables that control parallel decomposition and the
 * subroutines for parallel enviornment setup. Only module that requires MPI
 * library interaction except for time.c (MPI_WTIME).
 *******************************************************************************/
#include "snap.h"
#include <math.h>
#include <string.h>

/*******************************************************************************
 * Constructor for para_data struct
 *******************************************************************************/
void para_data_init ( para_data *para_vars )
{
    ROOT = 0;
    G_OFF = (int) pow ( 2, 14 );

    NPROC = 0;
    IPROC = 0;
    SPROC = 0;
    YCOMM = 0;
    ZCOMM = 0;
    YPROC = 0;
    ZPROC = 0;
    YLOP = 0;
    YHIP = 0;
    ZLOP = 0;
    ZHIP = 0;
    THREAD_LEVEL = 0;
    THREAD_SINGLE = 0;
    THREAD_FUNNELED = 0;
    THREAD_SERIALIZED = 0;
    THREAD_MULTIPLE = 0;
    MAX_THREADS = 0;
    NUM_GRTH = 0;

    FIRSTY = false;
    LASTY = false;
    FIRSTZ = false;
    LASTZ = false;
    DO_NESTED = false;
}

/*******************************************************************************
 * Initialize the MPI process environment. Replicate MPI_COMM_WORLD to
 * local communicator. Get each process its rank and total size.
 *******************************************************************************/
void pinit ( int argc, char *argv[], para_data *para_vars, double *time, int *ierr )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/

/*******************************************************************************
 * Set thread support levels if you have a thread multiple MPI.
 * These and the MPI_INIT_THREAD are commented out since they are
 * not supported everywhere. Comment them back in if you have a
 * compiler with this support and want to use them.
 *******************************************************************************/
//    thread_single     = MPI_THREAD_SINGLE;
//    thread_funneled   = MPI_THREAD_FUNNELED;
//    thread_serialized = MPI_THREAD_SERIALIZED;
//    thread_multiple   = MPI_THREAD_MULTIPLE;

/*******************************************************************************
* Initialize MPI and return available thread support level. Prefer
* thread_multiple but require thread_serialized. Abort for insufficient
* thread support. Don't want to use MPI_COMM_WORLD everywhere, so
* duplicate to comm_snap. Start the timer.
 *******************************************************************************/
// *ierr = MPI_Init_thread ( thread_serialized, thread_level );
    *ierr = MPI_Init ( &argc, &argv );

    *time = wtime(); // Get the MPI walltime

    *ierr = MPI_Comm_dup( MPI_COMM_WORLD, &COMM_SNAP ); // Duplicate MPI comm

/*******************************************************************************
 * Get the communicator size and each process' rank within the main
 * communicator
 *******************************************************************************/
    *ierr = MPI_Comm_size ( COMM_SNAP, &NPROC );
    *ierr = MPI_Comm_rank ( COMM_SNAP, &IPROC );

 /*******************************************************************************
 * Put a barrier for every process to reach this point.
 *******************************************************************************/
    *ierr = barrier ( COMM_SNAP );
}

/*******************************************************************************
 * MPI barrier.
 *******************************************************************************/
int barrier ( MPI_Comm comm )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int ierr = 0;

    ierr = MPI_Barrier( comm );

    return ierr;
}

/*******************************************************************************
 * Setup the SDD commmunicator.
 *******************************************************************************/
void pcomm_set( int npey, int npez,  para_data *para_vars, int *ierr )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int dims[2], periods[2], remain[2], reorder = 1;

/*******************************************************************************
 * Use MPI routines to create a Cartesian topology
 *******************************************************************************/
    dims[0] = npez;
    dims[1] = npey;
    periods[0] = 0;
    periods[1] = 0;

    *ierr = MPI_Cart_create( COMM_SNAP, 2, dims, periods, reorder, &COMM_SPACE );

/*******************************************************************************
 * Set up the sub-communicators of the cartesian grid
 *******************************************************************************/
    remain[0] = 0;
    remain[1] = 1;
    *ierr = MPI_Cart_sub( COMM_SPACE, remain, &YCOMM );

    remain[0] = 1;
    remain[1] = 0;
    *ierr = MPI_Cart_sub( COMM_SPACE, remain, &ZCOMM );

/*******************************************************************************
 * Get comm_space, ycomm, and zcomm ranks
 *******************************************************************************/
    *ierr = MPI_Comm_rank( COMM_SPACE, &SPROC );
    *ierr = MPI_Comm_rank( YCOMM, &YPROC );
    *ierr = MPI_Comm_rank( ZCOMM, &ZPROC );

/*******************************************************************************
 * Set some variables used during solution
 *******************************************************************************/
    if ( YPROC == 0 )
    {
        FIRSTY = true;
        YLOP = YPROC;
    }
    else
    {
        FIRSTY = false;
        YLOP = YPROC - 1;
    }

    if ( YPROC == npey - 1 )
    {
        LASTY = true;
        YHIP = YPROC;
    }
    else
    {
        LASTY = false;
        YHIP = YPROC + 1;
    }

    if ( ZPROC == 0 )
    {
        FIRSTZ = true;
        ZLOP = ZPROC;
    }
    else
    {
        FIRSTZ = false;
        ZLOP = ZPROC - 1;
    }

    if ( ZPROC == npez - 1 )
    {
        LASTZ = true;
        ZHIP = ZPROC;
    }
    else
    {
        LASTZ = false;
        ZHIP = ZPROC + 1;
    }
}

/*******************************************************************************
 * Call to end MPI processes
 *******************************************************************************/
void pend ()
{
    MPI_Finalize();
}

/*******************************************************************************
 * All reduce global max value (integer). Use specified communicator.
 *******************************************************************************/
int glmax_i ( int *value, MPI_Comm comm )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int ierr = 0, x;

/*******************************************************************************/

    if ( comm == MPI_COMM_NULL) return ierr;

    ierr = MPI_Allreduce( value, &x, 1, MPI_INTEGER, MPI_MAX, comm );
    *value = x;

    return ierr;
}

/*******************************************************************************
 * All reduce global max value (double precision float). Use specified
 * communicator.
 *******************************************************************************/
int glmax_d ( double *value, MPI_Comm comm )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int ierr = 0;
    double x;

/*******************************************************************************/

    if ( comm == MPI_COMM_NULL ) return ierr;

    ierr = MPI_Allreduce( value, &x, 1, MPI_DOUBLE_PRECISION, MPI_MAX, comm );
    *value = x;

    return ierr;
}

/*******************************************************************************
 * All reduce global max value (double precision float) for 1-d array.
 * Use specified communicator.
 *******************************************************************************/
int glmax_d_1d ( double *value, int dlen, MPI_Comm comm )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int ierr = 0;
    int i;
    double x[dlen];

/*******************************************************************************/

    if ( comm == MPI_COMM_NULL ) return ierr;

    ierr = MPI_Allreduce ( value, x, dlen, MPI_DOUBLE_PRECISION, MPI_MAX, comm );

    for ( i = 0; i < dlen; i++ )
    {
        value[i] = x[i];
    }

    return ierr;
}

/*******************************************************************************
 * All reduce global min value (integer). Use specified communicator.
 *******************************************************************************/
int glmin_i ( int *value, MPI_Comm comm )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int ierr = 0, x;

/*******************************************************************************/

    if ( comm == MPI_COMM_NULL ) return ierr;

    ierr = MPI_Allreduce( value, &x, 1, MPI_INTEGER, MPI_MIN, comm );
    *value = x;

    return ierr;
}

/*******************************************************************************
 * All reduce global min value (double precision float). Use specified
 * communicator.
 *******************************************************************************/
int glmin_d( double *value, MPI_Comm comm )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/

    int ierr = 0;
    double x = 0;

/*******************************************************************************/

    if ( comm == MPI_COMM_NULL ) return ierr;

    ierr = MPI_Allreduce ( value, &x, 1, MPI_DOUBLE_PRECISION, MPI_MIN, comm );
    *value = x;

    return ierr;
}


/*******************************************************************************
 * Broadcast (integer scalar). Use specified communicator and casting proc.
 *******************************************************************************/
int bcast_i_scalar ( int *value, MPI_Comm comm, int bproc, int nproc )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int ierr = 0;

/*******************************************************************************/

    if ( nproc == 1 ) return ierr;
    if ( comm == MPI_COMM_NULL ) return ierr;

    ierr = MPI_Bcast ( value, 1, MPI_INT, bproc, comm );

    return ierr;
}

/*******************************************************************************
 * Broadcast (double precision float scalar). Use specified communicator
 * and casting proc.
 *******************************************************************************/
int bcast_d_scalar ( double *value, MPI_Comm comm, int bproc, int nproc )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int ierr = 0;

/*******************************************************************************/

    if ( nproc == 1 ) return ierr;
    if ( comm == MPI_COMM_NULL ) return ierr;

    ierr = MPI_Bcast ( value, 1, MPI_DOUBLE_PRECISION, bproc, comm );

    return ierr;
}

/*******************************************************************************
 * Broadcast (integer 1d array). Use specified communicator and casting proc.
 *******************************************************************************/
int bcast_i_1d ( int *value, int ilen, MPI_Comm comm, int bproc, int nproc )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int ierr = 0;

/*******************************************************************************/

    if ( nproc == 1 ) return ierr;
    if ( comm == MPI_COMM_NULL ) return ierr;

    ierr = MPI_Bcast ( value, ilen, MPI_INT, bproc, comm );

    return ierr;
}

/*******************************************************************************
 * Broadcast (double precision float 1d array). Use specified communicator
 * and casting proc.
 *******************************************************************************/
int bcast_d_1d ( double *value, int dlen, MPI_Comm comm, int bproc, int nproc )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int ierr = 0;

/*******************************************************************************/

    if ( nproc == 1 ) return ierr;
    if ( comm == MPI_COMM_NULL ) return ierr;

    ierr = MPI_Bcast ( value, dlen, MPI_DOUBLE_PRECISION, bproc, comm );

    return ierr;
}

/*******************************************************************************
 * Send a rank-2 double presicision array.
 *******************************************************************************/
int psend_d_2d ( double *value, int d1, int d2, MPI_Comm comm,
                 int proc, int myproc, int mtag)
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int dlen, ierr = 0;

/*******************************************************************************/

    if ( proc == myproc || comm == MPI_COMM_NULL ) return ierr;

    dlen = d1*d2;

    ierr = MPI_Send ( value, dlen, MPI_DOUBLE_PRECISION, proc, mtag, comm );

    return ierr;
}

/*******************************************************************************
 * Send a rank-3 double presicision array.
 *******************************************************************************/
int psend_d_3d ( double *value, int d1, int d2, int d3, MPI_Comm comm,
                 int proc, int myproc, int mtag)
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int dlen, ierr = 0;

/*******************************************************************************/

    if ( proc == myproc || comm == MPI_COMM_NULL ) return ierr;

    dlen = d1*d2*d3;

    ierr = MPI_Send ( value, dlen, MPI_DOUBLE_PRECISION, proc, mtag, comm );

    return ierr;
}

/*******************************************************************************
 * Receive a rank-2 double presicision array.
 *******************************************************************************/
int precv_d_2d ( double *value, int d1, int d2, MPI_Comm comm,
                 int proc, int myproc, int mtag)
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int dlen, ierr = 0;
    MPI_Status istat;

/*******************************************************************************/
    if ( proc == myproc || comm == MPI_COMM_NULL ) return ierr;

    dlen = d1*d2;

    ierr = MPI_Recv ( value, dlen, MPI_DOUBLE_PRECISION, proc, mtag,
                      comm, &istat );

    return ierr;
}

/*******************************************************************************
 * Receive a rank-3 double presicision array.
 *******************************************************************************/
int precv_d_3d ( double *value, int d1, int d2, int d3, MPI_Comm comm,
                 int proc, int myproc, int mtag)
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int dlen, ierr = 0;
    MPI_Status istat;

/*******************************************************************************/

    if ( proc == myproc || comm == MPI_COMM_NULL ) return ierr;

    dlen = d1*d2*d3;

    ierr = MPI_Recv ( value, dlen, MPI_DOUBLE_PRECISION, proc, mtag,
                      comm, &istat );

    return ierr;
}

/*******************************************************************************
 * Return the rank of a proc defined by the coordinates of the Cartesian
 * communicator.
 *******************************************************************************/
int cartrank ( int *coord, int *rank, MPI_Comm comm )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int ierr = 0;

/*******************************************************************************/

    if ( comm == MPI_COMM_NULL ) return ierr;

    ierr = MPI_Cart_rank ( comm, coord, rank );

    return ierr;
}

/*******************************************************************************
 * Setup the number of OpenMP threads. Check if any proc is exceeding
 * max threads. Reset and report if so.
 *******************************************************************************/
void pinit_omp( MPI_Comm comm, int *nthreads, int nnested, bool *do_nested,
                int *ierr, char **error )
{
/*******************************************************************************
 * Local variables
 *******************************************************************************/
    int max_threads;
    int tmpStrLen;

/*******************************************************************************/

#ifdef USEMKL
    max_threads = mkl_get_max_threads();
#else
    max_threads = omp_get_max_threads();
#endif

    if ( *nthreads > max_threads )
    {
        *ierr = 1;
        *nthreads = max_threads;
    }

    glmax_i ( ierr, comm );

    if ( *ierr != 0 )
    {
        tmpStrLen = strlen ( "*WARNING: PINIT_OMP:"
                             " NTHREADS>MAX_THREADS;"
                             " reset to MAX_THREADS\n" );

        ALLOC_STR(*error, tmpStrLen + 1, ierr);

        snprintf ( (char *) *error, tmpStrLen + 1,
                   "*WARNING: PINIT_OMP:"
                   " NTHREADS>MAX_THREADS;"
                   " reset to MAX_THREADS\n" );
    }

#ifdef USEMKL
    mkl_set_num_threads ( *nthreads );
#else
    omp_set_num_threads ( *nthreads );
#endif

/*******************************************************************************
 * Setup for nested threading
 *******************************************************************************/
    if ( nnested > 0 ) *do_nested = true;

    omp_set_nested ( *do_nested );
}

/*******************************************************************************
 * Operate on an OpenMP lock
 *******************************************************************************/
void plock_omp ( char *dowhat, omp_lock_t *lock )
{
    if ( strncmp(dowhat, "init", 4) == 0)
        omp_init_lock( lock );
    else if ( strncmp(dowhat, "set", 3) == 0)
        omp_set_lock( lock );
    else if ( strncmp(dowhat, "unset", 5) == 0)
        omp_unset_lock( lock );
    else if ( strncmp(dowhat, "destroy", 7) == 0)
        omp_destroy_lock( lock );
    else
        return;
}

/*******************************************************************************
 * Return thread number of caller, [0, nthreads-1]. Maintains separation
 * of main code and OpenMP by placing here.
 *******************************************************************************/
int thread_num()
{
    int thread_num;

    thread_num = omp_get_thread_num();

    return thread_num;
}
