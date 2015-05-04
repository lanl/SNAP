/***********************************************************************
 * Header file for c version of SNAP.
 ***********************************************************************/
#ifndef _SNAP_H
#define _SNAP_H

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#ifdef USEMKL
#include "mkl.h"
#define INTEL_BB 64
#define VML_ACCURACY VML_EP
#define VML_HANDLING VML_FTZDAZ_OFF
#define VML_ERROR VML_ERRMODE_DEFAULT
#define VECLEN_MIN 40
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/***********************************************************************
 * Typedef functions
 ***********************************************************************/
// plib.c
/***********************************************************************
 * Parallel Run-time Varibales
 *
 * Note all ranks are zero based
 *
 * root       - root process for comm_snap, 0
 *
 * nproc      - number of MPI processes
 * iproc      - rank of calling process in base communicator
 *
 * comm_snap  - base communicator, duplicated from MPI_COMM_WORLD
 * comm_space - SDD communicator, ndimen-1 grid for 2-D (x-y) or
 *              3-D (x-y-z) problems. Non-existent for 1-D (x) problems.
 * sproc      - rank of calling process in comm_space
 *
 * ycomm      - y-dimension process communicator
 * zcomm      - z-dimension process communicator
 * yproc      - PE column in SDD 2-D PE mesh (comm_space)
 * zproc      - PE row in SDD 2-D PE mesh (comm_space)
 * firsty     - logical determining if lowest yproc
 * lasty      - logical determining if highest yproc
 * firstz     - logical determining if lowest zproc
 * lastz      - logical determining if highest zproc
 * ylop       - rank of preceding yproc in ycomm
 * yhip       - rank of succeeding yproc in ycomm
 * zlop       - rank of preceding zproc in zcomm
 * zhip       - rank of succeeding zproc in zcomm
 *
 * g_off      - group offset for message tags
 *
 * thread_level       - level of MPI thread support
 * thread_single      - MPI_THREAD_SINGLE
 * thread_funneled    - MPI_THREAD_FUNNELED
 * thread_serialized  - MPI_THREAD_SERIALIZED
 * thread_multiple    - MPI_THREAD_MULTIPLE
 * lock               - OpenMP lock
 *
 * num_grth   - minimum number of nthreads and ng; used to ensure loop
 *              over groups with communications is sized properly
 * do_nested  - true/false use nested threading, i.e., mini-KBA
 ***********************************************************************/
typedef struct para_data_
{
    int root, g_off;

    MPI_Comm comm_snap, comm_space;

    int nproc, iproc, sproc, ycomm,
        zcomm, yproc, zproc, ylop, yhip, zlop, zhip, thread_level,
        thread_single, thread_funneled, thread_serialized, thread_multiple,
        max_threads, num_grth;

    omp_lock_t lock;

    bool firsty, lasty, firstz, lastz, do_nested;

} para_data;

// input.c
/***********************************************************************
 * Input Variables
 *
 * Parallel processing inputs:
 * npey       - number of PEs in y-x plain
 * npez       - number of PEs in z-x plain
 * ichunk     -
 * nthreads   - number of OpenMP threads
 * nnested    - number of nested threads
 *
 * Geometry inputs:
 * ndimen     - number of spatial dimensions 1/2/3
 * nx         - number of x-dir spatial cells (global)
 * ny         - number of y-dir spatial cells
 *              (global on input, reset to per PE in setup)
 * nz         - number of z-dir spatial cells
 *              (global on input, reset to per PE in setup)
 * lx         - total length of x domain
 * ly         - total length of y domain
 * lz         - total length of z domain
 *
 * Sn inputs:
 * nmom       - number of discrete ordinates per octant
 * nang       - scattering order
 *
 * Data inputs:
 * ng         - number of groups
 * mat_opt    - material layout, 0/1/2=homogeneous/center/corner, with
 *              two materials, and material 2 nowhere/center/corner
 * src_opt    - source layout, 0/1/2=homogenous/src-center/src-corner=
 *              source everywhere/center of problem/corner, strength=10.0
 * scatp      - 0/1=no/yes print the full scattering matrix to file 'slgg'
 *
 * Control inputs:
 * epsi       -
 * tf         -
 * iitm       -
 * oitm       -
 * timedep    -
 * nsteps     -
 * it_det     -
 * fluxp      -
 * fixup      -
 ***********************************************************************/
typedef struct input_data_
{
    // Parallel processing inputs
    int npey, npez, ichunk, nthreads, nnested;

    // Geometry inputs
    int ndimen, nx, ny, nz;
    double lx, ly, lz;

    // Sn inputs
    int nmom, nang;

    // Data inputs
    int ng, mat_opt, src_opt, scatp;

    // Control inputs
    double epsi, tf;
    int iitm, oitm, timedep, nsteps, it_det, fluxp, fixup;

} input_data;

// time.c
/***********************************************************************
 * Time run-time variables
 *
 * tsnap    - total SNAP run time
 * tparset  - parallel environment setup time
 * tinp     - input run time
 * tset     - setup run time
 * tslv     - total solution run time
 * tparam   - time for setting up solve parameters
 * totrsrc  - time for outer source computations
 * tinners  - total time spent on inner iterations
 * tinrsrc  - time for inner source computations
 * tsweeps  - time for transport sweeps, including angular sourc compute
 * tinrmisc - time for miscellaneous inner ops
 * tslvmisc - time for miscellaneous solution ops
 * tout     - output run time
 * tgrind   - transport grind time
 ***********************************************************************/
typedef struct time_data_
{
    double tsnap, tparset, tinp, tset, tslv, tparam, totrsrc, tinners,
        tinrsrc, tsweeps, tinrmisc, tslvmisc, tout, tgrind;

} time_data;

// geom.c
/***********************************************************************
 * Geometry run-time variables
 *
 * ny_gl    - global number of y-dir spatial cells
 * nz_gl    - global number of z-dir spatial cells
 * jlb      - global index of local lower y bound
 * jub      - global index of local upper y bound
 * klb      - global index of local lower z bound
 * kub      - global index of local upper z bound
 *
 * dx       - x width of spatial cell
 * dy       - y width of spatial cell
 * dz       - z width of spatial cell
 *
 * nc       - number of i-chunks, nx/ichunk
 *
 * hi       - Spatial DD x-coefficient
 * hj(nang) - Spatial DD y-coefficient
 * hk(nang) - Spatial DD z-coefficient
 *
 * dinv(nang,nx,ny,nz,ng) - Sweep denominator, pre-computed/inverted
 *
 * ndiag    - number of diagonals of mini-KBA sweeps in nested threading
 ***********************************************************************/
typedef struct cell_id_type_
{
    int ic, jc, kc;

} cell_id_type;

typedef struct diag_type_
{
    int lenc;

    cell_id_type *cell_id_vars;

} diag_type;

typedef struct geom_data_
{
    int ny_gl, nz_gl, jlb, jub, klb, kub, nc, ndiag;

    double dx, dy, dz, hi;

    double *hj, *hk; // 1D arrays

    double *dinv; // 5D array

    diag_type *diag_vars; // 1D array

} geom_data;


// sn.c
/***********************************************************************
 * SN run-time variables
 *
 * cmom       - computational number of moments according to nmom & ndimen
 * noct       - number of directional octants
 * mu(nang)   - x direction cosines
 * eta(nang)  - y direction cosines
 * xi(nang)   - z direction cosines
 * w(nang)    - angle weights
 *
 * wmu(nang)  - w*mu
 * weta(nang) - w*eta
 * wxi(nang)  - w*xi
 *
 * ec(nang,cmom,noct) - Scattering expansion coefficients
 * lma(nmom)          - number of 'm' moments per order l
 ***********************************************************************/
typedef struct sn_data_
{
    int cmom, noct;

    double *mu, *eta, *xi, *w, *wmu,*weta, *wxi; // 1-D arrays

    double *ec; // 3-D array

    int *lma;

} sn_data;

// data.c
/***********************************************************************
 * Data run-time variables
 *
 * v(ng)         - mock velocity array
 * nmat          - number of materials
 * mat(nx,ny,nz) - material identifier array
 *
 * qi(nx,ny,nz,ng)             - fixed source array for src_opt<3
 * qim(nang,nx,ny,nz,noct,ng)  - fixed source array for src_opt>=3

 * sigt(nmat,ng)          - total interaction
 * siga(nmat,ng)          - absorption
 * sigs(nmat,ng)          - scattering, total
 * slgg(nmat,nmom,ng,ng)  - scattering matrix, all moments/groups
 * vdelt(ng)              - time-absorption coefficient
 ***********************************************************************/
typedef struct data_data_
{
    int nmat;
    int *mat;  // 3-D array

    double *v, *vdelt;          // 1-D arrays
    double *sigt, *siga, *sigs; // 2-D arrays
    double *qi, *slgg;          // 4-D arrays
    double *qim;                // 6-D arrays

} data_data;

// conrol.c
/***********************************************************************
 * control run-time variables
 *
 * dt       - time-step size
 *
 * tolr      - parameter, small number used for determining how to
 *             compute flux error
 * dfmxi(ng) - max error of inner iteration
 * dfmxo     - max error of outer iteration
 *
 * inrdone(ng)  - logical for inners being complete
 * otrdone      - logical for outers being complete
 ***********************************************************************/
typedef struct control_data_
{
    bool otrdone;
    bool *inrdone;     // 1-D array

    double dt, dfmxo, tolr;

    double *dfmxi;     // 1-D array

} control_data;

// mms.c
/***********************************************************************
 * mms variables
 *
 * ref_flux(nx,ny,nz,ng)          - Manufactured solution
 * ref_fluxm(cmom-1,nx,ny,nz,ng)  - Manufactured solution moments
 *
 * a_const       - i function constant
 * b_const       - j function constant
 * c_const       - k function constant
 *
 * ib(nx+1)      - i cell boundaries
 * jb(ny+1)      - j cell boundaries
 * kb(nz+1)      - k cell boundaries
 ***********************************************************************/
typedef struct mms_data_
{
    double *ref_flux;   // 4-D array
    double *ref_fluxm;  // 5-D array

    double a_const, b_const, c_const;

    double *ib, *jb, *kb; // 1-D arrays

} mms_data;

// solvar.c
/***********************************************************************
 * Solvar Module variables
 *
 * ptr_in(nang,nx,ny,nz,noct,ng)   - Incoming time-edge flux pointer
 * ptr_out(nang,nx,ny,nz,noct,ng)  - Outgoing time-edge flux pointer
 *
 * flux(nx,ny,nz,ng)          - Scalar flux moments array
 * fluxpo(nx,ny,nz,ng)        - Previous outer copy of scalar flux array
 * fluxpi(nx,ny,nz,ng)        - Previous inner copy of scalar flux array
 * fluxm(cmom-1,nx,ny,nz,ng)  - Flux moments array
 *
 * q2grp(cmom,nx,ny,nz,ng)  - Out-of-group scattering + fixed sources
 * qtot(cmom,nx,ny,nz,ng)   - Total source: q2grp + within-group source
 *
 * t_xs(nx,ny,nz,ng)       - Total cross section on mesh
 * a_xs(nx,ny,nz,ng)       - Absorption cross section on mesh
 * s_xs(nmom,nx,ny,nz,ng)  - In-group scattering cross section on mesh
 *
 * psii(nang,ny,nz,ng)     - Working psi_x array
 * psij(nang,ichunk,nz,ng) - Working psi_y array
 * psik(nang,ichunk,ny,ng) - Working psi_z array
 *
 * jb_in(nang,ichunk,nz,ng)  - y-dir boundary flux in from comm
 * jb_out(nang,ichunk,nz,ng) - y-dir boundary flux out to comm
 * kb_in(nang,ichunk,ny,ng)  - z-dir boundary flux in from comm
 * kb_out(nang,ichunk,ny,ng) - z-dir boundary flux out to comm
 *
 * flkx(nx+1,ny,nz,ng)     - x-dir leakage array
 * flky(nx,ny+1,nz,ng)     - y-dir leakage array
 * flkz(nx,ny,nz+1,ng)     - z-dir leakage array
 *
 ***********************************************************************/
typedef struct solvar_data_
{
    double *flux, *fluxpo, *fluxpi, *t_xs, *a_xs, // 4-D arrays
        *psii, *psij, *psik, *jb_in, *jb_out,
        *kb_in, *kb_out, *flkx, *flky, *flkz;

    double *qtot, *q2grp, *fluxm, *s_xs;          // 5-D arrays

    double *ptr_in, *ptr_out;                     // 6-D arrays
} solvar_data;


// dim1_sweep.c and dim3_sweep.c
/***********************************************************************
 * dim_sweep module variable
 *
 * fmin        - min scalar flux. Dummy for now, not used elsewhere.
 * fmax        - max scalar flux. Dummy for now, not used elsewhere.
 ***********************************************************************/
typedef struct dim_sweep_data_
{
    double fmin, fmax;
} dim_sweep_data;


// sweep.c
/***********************************************************************
 * Module variables
 *
 * mtag               - message tag
 * <y or z>p_snd      - rank of process sending to
 * <y or z>p_rcv      - rank of process receiving from
 * incoming<y or z>   - logical determining if proc will receive a message
 * outgoing<y or z>   - logical determining if proc will send a message
 ***********************************************************************/
typedef struct sweep_data_
{
    int mtag, yp_snd, yp_rcv, zp_snd, zp_rcv;

    bool incomingy, incomingz, outgoingy, outgoingz;

} sweep_data;

/***********************************************************************
 * Function prototypes
 ***********************************************************************/
/* plib.c: Contains the variables that control parallel decomposition and the
 subroutines for parallel enviornment setup. Only module that requires MPI
 library interaction except for time.c (MPI_WTIME). */
// Constructor for para_data type
void para_data_init ( para_data *para_vars );

// Initialize the MPI process environment
void pinit ( int argc, char *argv[], para_data *para_vars,
             double *time, int *ierr );

// Call to execute an MPI_Barrier
int barrier ( MPI_Comm comm );

// Setup the SDD communicator
void pcomm_set( int npey, int npez,  para_data *para_vars, int *ierr );

// Call to execute MPI_Finalize
void pend ( void );

// All reduce global max value (integer)
int glmax_i ( int *value, MPI_Comm comm );

// All reduce global max value (double)
int glmax_d ( double *value, MPI_Comm comm );

// All reduce global max value (double) for 1-d array
int glmax_d_1d ( double *value, int dlen, MPI_Comm comm );

// All reduce global min value (integer)
int glmin_i ( int *value, MPI_Comm comm );

// All reduce global min value (double)
int glmin_d ( double *value, MPI_Comm comm );

// Broadcast (integer) scalar
int bcast_i_scalar ( int *value, MPI_Comm comm, int bproc, int nproc );

// Broadcast (double) scalar
int bcast_d_scalar ( double *value, MPI_Comm comm,
                     int bproc, int nproc );

// Broadcast (integer) 1-d array
int bcast_i_1d ( int *value, int ilen, MPI_Comm comm,
                 int bproc, int nproc );

// Broadcast (double) 1-d array
int bcast_d_1d ( double *value, int dlen, MPI_Comm comm,
                 int bproc, int nproc );

// Send a rank-2 double presicision array
int psend_d_2d ( double *value, int d1, int d2, MPI_Comm comm,
                 int proc, int myproc, int mtag);

// Send a rank-3 double presicision array
int psend_d_3d ( double *value, int d1, int d2, int d3, MPI_Comm comm,
                 int proc, int myproc, int mtag);

// Receive a rank-2 double presicision array
int precv_d_2d ( double *value, int d1, int d2, MPI_Comm comm,
                 int proc, int myproc, int mtag);

// Receive a rank-3 double presicision array
int precv_d_3d ( double *value, int d1, int d2, int d3, MPI_Comm comm,
                 int proc, int myproc, int mtag);

// Return rank of proc defined by coordinates of Cartesian communicator
int cartrank ( int *coord, int *rank, MPI_Comm comm );

// Setup the number of OpenMP threads. Check if any proc is exceeding
// max threads. Reset and report if so.
void pinit_omp( MPI_Comm comm, int *nthreads, int nnested,
                bool *do_nested, int *ierr, char **error );

// Operate on an OpenMP lock
void plock_omp ( char *dowhat, omp_lock_t *lock );

// Return thread number of caller
int thread_num( void );


// utils.c
int cmdarg ( int argc, char *argv[], char **inputFile, char **outputFile,
             char **error, int iproc, int root );

int open_file ( FILE **fp, char *fileName, char *fileAction,
                char **error, int iproc, int root );

int close_file ( FILE *fp, char *fileName, char **error,
                 int iproc, int root );

void print_error ( FILE *fp, char *error, int iproc, int root );

int string_empty ( char *stringName );

void stop_run ( int inputFlag, int solveFlag, int statusFlag, para_data *para_vars,
                sn_data *sn_vars, data_data *data_vars, mms_data *mms_vars,
                geom_data *geom_vars, solvar_data *solvar_vars, control_data *control_vars );


// dealloc.c
void dealloc_input ( int selectFlag, sn_data *sn_vars,
                     data_data *data_vars, mms_data *mms_vars );

void dealloc_solve ( int selectFlag, geom_data *geom_vars,
                     solvar_data *solvar_vars, control_data *control_vars );


// version.c
void version_print ( FILE *fp_out );


// input.c
void input_data_init ( input_data *input_vars );

int read_input ( FILE *fp_in, FILE *fp_out, input_data *input_vars,
                 para_data *para_vars, time_data *time_vars );

void get_input_value ( char *lineData, char *valueID, char **tmpData );

void input_echo ( input_data *input_vars, FILE *fp_out );

int input_check ( FILE *fp_out, input_data *input_vars, para_data *para_vars );

int var_bcast ( input_data *input_vars, para_data *para_vars );


// time.c
void time_data_init ( time_data *time_vars );

double wtime ( void );

void time_summ ( FILE *fp_out, time_data *time_vars );


//setup.c
void setup ( input_data *input_vars, para_data *para_vars, time_data *time_vars,
             geom_data *geom_vars, sn_data *sn_vars, data_data *data_vars,
             solvar_data *solvar_vars, control_data *control_vars,
             mms_data *mms_vars, FILE *fp_out, int *ierr, char **error );

void setup_alloc( input_data *input_vars, para_data *para_vars, sn_data *sn_vars,
                  data_data *data_vars, int *flg, int *ierr, char **error );

void setup_delta( input_data *input_vars, geom_data *geom_vars,
                  control_data *control_vars );

void setup_vel ( input_data *input_vars, data_data *data_vars );

void setup_angle ( input_data *input_vars, sn_data *sn_vars);

void setup_mat ( input_data *input_vars, geom_data *geom_vars, data_data *data_vars,
                 int *i1, int *i2, int *j1, int *j2, int *k1, int *k2 );

void setup_data ( input_data *input_vars, data_data *data_vars);

void setup_src ( input_data *input_vars, para_data *para_vars, geom_data *geom_vars,
                 sn_data *sn_vars, data_data *data_vars, control_data *control_vars,
                 mms_data *mms_vars, int *i1, int *i2, int *j1, int *j2, int *k1,
                 int *k2, int *ierr, char **error );

void setup_echo ( FILE *fp_out, input_data *input_vars, para_data *para_vars,
                  geom_data *geom_vars, data_data *data_vars, sn_data *sn_vars,
                  control_data *control_vars, int mis, int mie, int mjs, int mje, int mks,
                  int mke, int qis, int qie, int qjs, int qje, int qks, int qke );

void setup_scatp( input_data *input_vars, para_data *para_vars,
                  data_data *data_vars, int *ierr, char **error );


// geom.c
void geom_data_init ( geom_data *geom_vars );

void geom_alloc ( input_data *input_vars, geom_data *geom_vars, int *ierr );

void geom_dealloc ( geom_data *geom_vars );

void param_calc ( input_data *input_vars, sn_data *sn_vars,
                  solvar_data *solvar_vars, data_data *data_vars,
                  geom_data *geom_vars, int ng_indx );

void diag_setup ( input_data *input_vars, para_data *para_vars,
                  geom_data *geom_vars, int *ierr, char **error );


// sn.c
void sn_data_init ( sn_data *sn_vars );

void sn_allocate ( sn_data *sn_vars, input_data *input_vars, int *ierr );

void sn_deallocate ( sn_data *sn_vars );

void expcoeff ( input_data *input_vars, sn_data *sn_vars, int *ndimen );


/* data.c: contains the variables and setup subroutines for the mock
 cross section data. It establishes the number of groups and constructs
 the cross section arrays.*/
// Constructor for data_data type
void data_data_init (data_data *data_vars );

// Allocate data module arrays
void data_allocate ( data_data *data_vars, input_data *input_vars,
                     sn_data *sn_vars, int *ierr );

// Deallocate the data module arrays
void data_deallocate ( data_data *data_vars );


// control.c
void control_data_init ( control_data *control_vars );

void control_alloc ( input_data *input_vars, control_data *control_vars, int *ierr);

void control_dealloc ( control_data *control_vars );


// mms.c
void mms_data_init ( mms_data *mms_vars );

void mms_setup ( input_data *input_vars, para_data *para_vars, geom_data *geom_vars,
                 data_data *data_vars, sn_data *sn_vars, control_data *control_vars,
                 mms_data *mms_vars, int *ierr, char **error );

void mms_allocate ( input_data *input_vars, sn_data *sn_vars, mms_data *mms_vars,
                    int *ierr, char **error );

void mms_deallocate (  mms_data *mms_vars );

void mms_cells ( input_data *input_vars, geom_data *geom_vars, mms_data *mms_vars );

void mms_flux_1 ( input_data *input_vars, geom_data *geom_vars,
                  sn_data *sn_vars, mms_data *mms_vars );

void mms_trigint ( char *trig, int lc, double d, double del,
                   double *cb, double *fn );

void mms_src_1( input_data *input_vars, geom_data *geom_vars, data_data *data_vars,
                sn_data *sn_vars, mms_data *mms_vars );

void mms_flux_1_2 ( input_data *input_vars, control_data *control_vars,
                    mms_data *mms_vars );

void mms_verify_1 ( input_data *input_vars, para_data *para_vars, control_data *control_vars,
                    mms_data *mms_vars, solvar_data *solvar_vars, FILE *fp_out );


// translv.c
void translv ( input_data *input_vars, para_data *para_vars, time_data *time_vars,
               geom_data *geom_vars, sn_data *sn_vars, data_data *data_vars,
               control_data *control_vars, solvar_data *solvar_vars, mms_data *mms_vars,
               sweep_data *sweep_vars, dim_sweep_data *dim_sweep_vars,
               FILE *fp_out, int *ierr, char **error );

// solvar.c
void solvar_data_init ( solvar_data *solvar_vars );

void solvar_alloc ( input_data *input_vars, sn_data* sn_vars, solvar_data *solvar_vars,
                    int *ierr );

void solvar_dealloc ( solvar_data *solvar_vars );


// dim1_sweep.c
void dim1_sweep_data_init ( dim_sweep_data *dim_sweep_vars );

void dim1_sweep ( input_data *input_vars, geom_data *geom_vars, sn_data *sn_vars, data_data *data_vars,
                  control_data *control_vars, solvar_data *solvar_vars, dim_sweep_data *dim_sweep_vars,
                  int id, int oct, int d1, int d2, int d3, int d4, int i1, int i2, int g, int *ierr );

// dim3_sweep.c
void dim3_sweep_data_init ( dim_sweep_data *dim_sweep_vars );

void dim3_sweep ( input_data *input_vars, para_data *para_vars,
                  geom_data *geom_vars, sn_data *sn_vars,
                  data_data *data_vars, control_data *control_vars,
                  solvar_data *solvar_vars, dim_sweep_data *dim_sweep_vars,
                  int ich, int id, int d1, int d2, int d3, int d4, int jd,
                  int kd, int jlo, int klo, int jhi, int khi, int jst, int kst,
                  int i1, int i2, int oct, int g, int *ierr );


// sweep.c
void sweep_data_init (sweep_data *sweep_vars );

void sweep ( input_data *input_vars, para_data *para_vars, geom_data *geom_vars,
             sn_data *sn_vars, data_data *data_vars, control_data *control_vars,
             solvar_data *solvar_vars, sweep_data *sweep_vars,
             dim_sweep_data *dim_sweep_vars, int *ierr );
/*
void sweep_recv_bdry ( para_data *para_vars, sweep_data *sweep_vars,solvar_data *solvar_vars,
                       input_data *input_vars,
                       double *value, int d1, int d2, int d3, MPI_Comm comm,
                       int proc, int myproc, int g, int iop );

void sweep_send_bdry ( para_data *para_vars, sweep_data *sweep_vars, solvar_data *solvar_vars,
                       input_data *input_vars,
                       double *value, int d1, int d2, int d3, MPI_Comm comm,
                       int proc, int myproc, int g, int iop );
*/
void sweep_recv_bdry ( input_data *input_vars, para_data *para_vars,
                       solvar_data *solvar_vars, sweep_data *sweep_vars, int g, int iop );

void sweep_send_bdry ( input_data *input_vars, para_data *para_vars,
                       solvar_data *solvar_vars, sweep_data *sweep_vars, int g, int iop );


// octsweep.c
void octsweep ( input_data *input_vars, para_data *para_vars, geom_data *geom_vars, sn_data *sn_vars,
                data_data *data_vars, control_data *control_vars,
                solvar_data *solvar_vars, dim_sweep_data *dim_sweep_vars,
                int g, int iop, int jd, int kd, int jlo, int jhi, int jst,
                int klo, int khi, int kst, int *ierr );


// expxs.c
#ifdef USEMACRO
void expxs_reg ( double *xs, int *map, double *cs, int nmat_size, int nx_size,
                 int ny_size, int nz_size, int ng_indx, int ng_size );

void expxs_slgg ( double *scat, int *map, double *cs, int nmat_size, int nmom_size,
                  int nx_size, int ny_size, int nz_size, int ng_indx, int ng_size );
#else
void expxs_reg ( input_data *input_vars, data_data *data_vars, solvar_data *solvar_vars,
                 double *cs, int ng_indx, int gp_indx, int l_indx );

void expxs_slgg ( input_data *input_vars, data_data *data_vars,
                  solvar_data *solvar_vars, int ng_indx );
#endif


// outer.c
int outer ( input_data *input_vars, para_data *para_vars, geom_data *geom_vars,
            time_data *time_vars, data_data *data_vars, sn_data *sn_vars,
            control_data *control_vars, solvar_data *solvar_vars,
            sweep_data *sweep_vars, dim_sweep_data *dim_sweep_vars,
            FILE *fp_out, int *ierr );

void otr_src ( input_data *input_vars, data_data *data_vars,
               sn_data *sn_vars, solvar_data *solvar_vars, int *ierr );

void otr_src_scat ( input_data *input_vars, data_data *data_vars, sn_data *sn_vars,
                    solvar_data *solvar_vars, int g, int gp, int *ierr );

void otr_conv ( input_data *input_vars, para_data *para_vars, control_data *control_vars,
                solvar_data *solvar_vars, int *ierr );


// inner.c
void inner ( input_data *input_vars, para_data *para_vars, geom_data *geom_vars,
             time_data *time_vars, data_data *data_vars, sn_data *sn_vars,
             control_data *control_vars, solvar_data *solvar_vars,
             sweep_data *sweep_vars, dim_sweep_data *dim_sweep_vars, int inno,
             int *iits, FILE *fp_out, int *ierr );

void inr_src ( input_data *input_vars, sn_data *sn_vars,
               control_data *control_vars, solvar_data *solvar_vars, para_data *para_vars );

void inr_src_scat ( input_data *input_vars, sn_data *sn_vars,
                    solvar_data *solvar_vars, int g, para_data *para_vars );

void inr_conv ( input_data *input_vars, para_data *para_vars,
                control_data *control_vars, solvar_data *solvar_vars,
                int inno, int *iits, FILE *fp_out, int *ierr );


// output.c
void output ( input_data *input_vars, para_data *para_vars, time_data *time_vars,
              geom_data *geom_vars, data_data *data_vars, sn_data *sn_vars,
              control_data *control_vars, mms_data *mms_vars, solvar_data *solvar_vars,
              sweep_data *sweep_vars, FILE *fp_out, int *ierr, char **error );

//void output_send ( input_data *input_vars, para_data *para_vars,
//                   control_data *control_vars, sweep_data *sweep_vars,
//                   double *fprnt, int *ierr );

//void output_recv ( input_data *input_vars, para_data *para_vars,
//                   control_data *control_vars, sweep_data *sweep_vars,
//                   double *fprnt, int *ierr );

void output_send ( int dim1, int dim2, MPI_Comm comm, int root, int sproc,
                   int mtag, double *fprnt, int *ierr );

void output_recv ( int dim1, int dim2, MPI_Comm comm, int proc, int sproc,
                   int mtag, double *fprnt, int *ierr );


void output_flux_file ( input_data *input_vars, para_data *para_vars,
                        geom_data *geom_vars, data_data *data_vars,
                        sn_data *sn_vars, control_data *control_vars,
                        mms_data *mms_vars, solvar_data *solvar_vars,
                        sweep_data *sweep_vars, int klb, int kub,
                        int *ierr, char **error, FILE *fp_out );


/***********************************************************************
 * Memory allocation macros
 ***********************************************************************/
#ifdef USEMKL

#define ALLOC_STR(PNTR, STRLEN, IERR)                           \
    if ( !PNTR )                                                \
    {                                                           \
        PNTR = (char *) mkl_malloc ( STRLEN, INTEL_BB );        \
        if (!PNTR )                                             \
        {                                                       \
            perror("ALLOC_STR");                                \
            fprintf(stderr,                                     \
                    "Allocation failed for " #PNTR              \
                    ".  Terminating...\n");                     \
            *IERR = 1; /* exit(-1); */                         \
        }                                                       \
    }                                                           \
    else                                                        \
    {                                                           \
        PNTR = (char *) mkl_realloc ( PNTR, STRLEN );           \
        if (!PNTR )                                             \
        {                                                       \
            perror("ALLOC_STR");                                \
            fprintf(stderr,                                     \
                    "Re-allocation failed for " #PNTR           \
                    ".  Terminating...\n");                     \
            *IERR = 1; /* exit(-1); */                         \
        }                                                       \
    }

#define REALLOC_STR(PNTR, STRLEN, IERR)                 \
    PNTR = (char *) mkl_realloc ( PNTR, STRLEN );       \
    if (!PNTR)                                          \
    {                                                   \
        perror("REALLOC_STR");                          \
        fprintf(stderr,                                 \
                "Re-allocation failed for " #PNTR       \
                ".  Terminating...\n");                 \
        *IERR = 1; /* exit(-1); */                      \
    }

#define FREE(PNTR)                                     \
    if ( PNTR )                                        \
    {                                                  \
        mkl_free ( PNTR );                             \
    }

#define REALLOC_2D(PNTR, NUMX, NUMY, TYPE, IERR)                        \
    PNTR = (TYPE *)mkl_realloc(PNTR, NUMX*NUMY*sizeof(TYPE));                 \
    if (!PNTR)                                                          \
    {                                                                   \
        perror("REALLOC_2D");                                             \
        fprintf(stderr,                                                 \
                "Reallocation failed for " #PNTR ".  Terminating...\n");  \
        *IERR = 1; /* exit(-1); */                                     \
    }


#define ALLOC_1D(PNTR, NUM, TYPE, IERR)                                 \
    PNTR = (TYPE *)mkl_calloc(NUM, sizeof(TYPE), INTEL_BB);             \
    if (!PNTR)                                                          \
    {                                                                   \
        perror("ALLOC_1D");                                             \
        fprintf(stderr,                                                 \
                "Allocation failed for " #PNTR ".  Terminating...\n");  \
        *IERR = 1; /* exit(-1); */                                     \
    }

#define ALLOC_2D(PNTR, NUMX, NUMY, TYPE, IERR)                          \
    PNTR = (TYPE *)mkl_calloc((NUMX) * (NUMY),                          \
                              sizeof(TYPE), INTEL_BB);                  \
    if (!PNTR)                                                          \
    {                                                                   \
        perror("ALLOC_2D");                                             \
        fprintf(stderr,                                                 \
                "Allocation failed for " #PNTR ".  Terminating...\n");  \
        *IERR = 1; /* exit(-1); */                                     \
    }

#define ALLOC_3D(PNTR, NUMX, NUMY, NUMZ, TYPE, IERR)                    \
    PNTR = (TYPE *)mkl_calloc((NUMX) * (NUMY) * (NUMZ),                 \
                              sizeof(TYPE), INTEL_BB);                  \
    if (!PNTR)                                                          \
    {                                                                   \
        perror("ALLOC_3D");                                             \
        fprintf(stderr,                                                 \
                "Allocation failed for " #PNTR ".  Terminating...\n");  \
        *IERR = 1; /* exit(-1); */                                     \
    }

#define ALLOC_4D(PNTR, NUMW, NUMX, NUMY, NUMZ, TYPE, IERR)              \
    PNTR = (TYPE *)mkl_calloc((NUMW) * (NUMX) * (NUMY) * (NUMZ),        \
                              sizeof(TYPE), INTEL_BB);                  \
    if (!PNTR)                                                          \
    {                                                                   \
        perror("ALLOC_4D");                                             \
        fprintf(stderr,                                                 \
                "Allocation failed for " #PNTR ".  Terminating...\n");  \
        *IERR = 1; /* exit(-1); */                                     \
    }

#define ALLOC_5D(PNTR, NUMV, NUMW, NUMX, NUMY, NUMZ, TYPE, IERR)        \
    PNTR = (TYPE *)mkl_calloc((NUMV) * (NUMW)                           \
                              * (NUMX) * (NUMY) * (NUMZ),               \
                              sizeof(TYPE), INTEL_BB);                  \
    if (!PNTR)                                                          \
    {                                                                   \
        perror("ALLOC_5D");                                             \
        fprintf(stderr,                                                 \
                "Allocation failed for " #PNTR ".  Terminating...\n");  \
        *IERR = 1; /* exit(-1); */                                     \
    }

#define ALLOC_6D(PNTR, NUMU, NUMV, NUMW, NUMX, NUMY, NUMZ, TYPE, IERR)  \
    PNTR = (TYPE *)mkl_calloc((NUMU) * (NUMV) *(NUMW)                   \
                              * (NUMX) * (NUMY) * (NUMZ),               \
                              sizeof(TYPE), INTEL_BB);                  \
    if (!PNTR)                                                          \
    {                                                                   \
        perror("ALLOC_6D");                                             \
        fprintf(stderr,                                                 \
                "Allocation failed for " #PNTR ".  Terminating...\n");  \
        *IERR = 1; /* exit(-1); */                                     \
    }

#else

#define ALLOC_STR(PNTR, STRLEN, IERR)                   \
    if ( !PNTR )                                        \
    {                                                   \
        PNTR = (char *) malloc ( STRLEN );              \
        if (!PNTR )                                     \
        {                                               \
            perror("ALLOC_STR");                        \
            fprintf(stderr,                             \
                    "Allocation failed for " #PNTR      \
                    ".  Terminating...\n");             \
            *IERR = 1; /* exit(-1); */                 \
        }                                               \
    }                                                   \
    else                                                \
    {                                                   \
        PNTR = (char *) realloc ( PNTR, STRLEN );       \
        if (!PNTR )                                     \
        {                                               \
            perror("ALLOC_STR");                        \
            fprintf(stderr,                             \
                    "Re-allocation failed for " #PNTR   \
                    ".  Terminating...\n");             \
            *IERR = 1; /* exit(-1); */                 \
        }                                               \
    }

#define REALLOC_STR(PNTR, STRLEN, IERR)                 \
    PNTR = (char *) realloc ( PNTR, STRLEN );           \
    if (!PNTR)                                          \
    {                                                   \
        perror("REALLOC_STR");                          \
        fprintf(stderr,                                 \
                "Re-allocation failed for " #PNTR       \
                ".  Terminating...\n");                 \
        *IERR = 1; /* exit(-1); */                     \
    }

#define FREE(PNTR)                              \
    if ( PNTR )                                 \
    {                                           \
        free ( PNTR );                          \
    }

#define REALLOC_2D(PNTR, NUMX, NUMY, TYPE, IERR)                        \
    PNTR = (TYPE *)realloc(PNTR, NUMX*NUMY*sizeof(TYPE));              \
    if (!PNTR)                                                          \
    {                                                                   \
     perror("REALLOC_2D");                                                \
     fprintf(stderr,                                                    \
                 "Reallocation failed for " #PNTR ". Terminating...\n");  \
     *IERR = 1; /* exit(-1); */                                         \
     }

#define ALLOC_1D(PNTR, NUM, TYPE, IERR)                                 \
                PNTR = (TYPE *)calloc(NUM, sizeof(TYPE));               \
                if (!PNTR)                                              \
                {                                                       \
                    perror("ALLOC_1D");                                 \
                    fprintf(stderr,                                     \
                            "Allocation failed for " #PNTR ". Terminating...\n"); \
                    *IERR = 1; /* exit(-1); */                         \
                }

#define ALLOC_2D(PNTR, NUMX, NUMY, TYPE, IERR)                          \
    PNTR = (TYPE *)calloc((NUMX) * (NUMY), sizeof(TYPE));               \
    if (!PNTR)                                                          \
    {                                                                   \
     perror("ALLOC_2D");                                                \
     fprintf(stderr,                                                    \
                 "Allocation failed for " #PNTR ". Terminating...\n");  \
     *IERR = 1; /* exit(-1); */                                        \
     }

#define ALLOC_3D(PNTR, NUMX, NUMY, NUMZ, TYPE, IERR)                    \
    PNTR = (TYPE *)calloc((NUMX) * (NUMY) * (NUMZ), sizeof(TYPE));      \
    if (!PNTR)                                                          \
    {                                                                   \
        perror("ALLOC_3D");                                             \
        fprintf(stderr,                                                 \
                "Allocation failed for " #PNTR ".  Terminating...\n");  \
        *IERR = 1; /* exit(-1); */                                     \
    }

#define ALLOC_4D(PNTR, NUMW, NUMX, NUMY, NUMZ, TYPE, IERR)              \
    PNTR = (TYPE *)calloc((NUMW) * (NUMX) * (NUMY) * (NUMZ),            \
                          sizeof(TYPE));                                \
    if (!PNTR)                                                          \
    {                                                                   \
        perror("ALLOC_4D");                                             \
        fprintf(stderr,                                                 \
                "Allocation failed for " #PNTR ".  Terminating...\n");  \
        *IERR = 1; /* exit(-1); */                                     \
    }

#define ALLOC_5D(PNTR, NUMV, NUMW, NUMX, NUMY, NUMZ, TYPE, IERR)        \
    PNTR = (TYPE *)calloc((NUMV) * (NUMW) * (NUMX) * (NUMY) * (NUMZ),   \
                              sizeof(TYPE));                            \
    if (!PNTR)                                                          \
    {                                                                   \
        perror("ALLOC_5D");                                             \
        fprintf(stderr,                                                 \
                "Allocation failed for " #PNTR ".  Terminating...\n");  \
        *IERR = 1; /* exit(-1); */                                     \
    }

#define ALLOC_6D(PNTR, NUMU, NUMV, NUMW, NUMX, NUMY, NUMZ, TYPE, IERR)  \
    PNTR = (TYPE *)calloc((NUMU) * (NUMV) *(NUMW)                       \
                          * (NUMX) * (NUMY) * (NUMZ),                   \
                              sizeof(TYPE));                            \
    if (!PNTR)                                                          \
    {                                                                   \
     perror("ALLOC_6D");                                                \
     fprintf(stderr,                                                    \
                 "Allocation failed for " #PNTR ".  Terminating...\n"); \
     *IERR = 1; /* exit(-1); */                                         \
     }

#endif

// plib.c
/* root */
#define ROOT_PARA(PARA)              PARA->root
// Assuming 'PARA=para_vars'
#define ROOT                         ROOT_PARA(para_vars)

/* g_off */
#define G_OFF_PARA(PARA)             PARA->g_off
// Assuming 'PARA=para_vars'
#define G_OFF                        G_OFF_PARA(para_vars)

/* comm_snap */
#define COMM_SNAP_PARA(PARA)         PARA->comm_snap
// Assuming 'PARA=para_vars'
#define COMM_SNAP                    COMM_SNAP_PARA(para_vars)

/* comm_space */
#define COMM_SPACE_PARA(PARA)        PARA->comm_space
// Assuming 'PARA=para_vars'
#define COMM_SPACE                   COMM_SPACE_PARA(para_vars)

/* nproc */
#define NPROC_PARA(PARA)             PARA->nproc
// Assuming 'PARA=para_vars'
#define NPROC                        NPROC_PARA(para_vars)

/* iproc */
#define IPROC_PARA(PARA)             PARA->iproc
// Assuming 'PARA=para_vars'
#define IPROC                        IPROC_PARA(para_vars)

/* sproc */
#define SPROC_PARA(PARA)             PARA->sproc
// Assuming 'PARA=para_vars'
#define SPROC                        SPROC_PARA(para_vars)

/* ycomm */
#define YCOMM_PARA(PARA)             PARA->ycomm
// Assuming 'PARA=para_vars'
#define YCOMM                        YCOMM_PARA(para_vars)

/* zcomm */
#define ZCOMM_PARA(PARA)             PARA->zcomm
// Assuming 'PARA=para_vars'
#define ZCOMM                        ZCOMM_PARA(para_vars)

/* yproc */
#define YPROC_PARA(PARA)             PARA->yproc
// Assuming 'PARA=para_vars'
#define YPROC                        YPROC_PARA(para_vars)

/* zproc */
#define ZPROC_PARA(PARA)             PARA->zproc
// Assuming 'PARA=para_vars'
#define ZPROC                        ZPROC_PARA(para_vars)

/* ylop */
#define YLOP_PARA(PARA)              PARA->ylop
// Assuming 'PARA=para_vars'
#define YLOP                         YLOP_PARA(para_vars)

/* yhip */
#define YHIP_PARA(PARA)              PARA->yhip
// Assuming 'PARA=para_vars'
#define YHIP                         YHIP_PARA(para_vars)

/* zlop */
#define ZLOP_PARA(PARA)              PARA->zlop
// Assuming 'PARA=para_vars'
#define ZLOP                         ZLOP_PARA(para_vars)

/* zhip */
#define ZHIP_PARA(PARA)              PARA->zhip
// Assuming 'PARA=para_vars'
#define ZHIP                         ZHIP_PARA(para_vars)

/* thread_level */
#define THREAD_LEVEL_PARA(PARA)      PARA->thread_level
// Assuming 'PARA=para_vars'
#define THREAD_LEVEL                 THREAD_LEVEL_PARA(para_vars)

/* thread_single */
#define THREAD_SINGLE_PARA(PARA)     PARA->thread_single
// Assuming 'PARA=para_vars'
#define THREAD_SINGLE                THREAD_SINGLE_PARA(para_vars)

/* thread_funneled */
#define THREAD_FUNNELED_PARA(PARA)   PARA->thread_funneled
// Assuming 'PARA=para_vars'
#define THREAD_FUNNELED              THREAD_FUNNELED_PARA(para_vars)

/* thread_serialized */
#define THREAD_SERIALIZED_PARA(PARA) PARA->thread_serialized
// Assuming 'PARA=para_vars'
#define THREAD_SERIALIZED            THREAD_SERIALIZED_PARA(para_vars)

/* thread_multiple */
#define THREAD_MULTIPLE_PARA(PARA)   PARA->thread_multiple
// Assuming 'PARA=para_vars'
#define THREAD_MULTIPLE              THREAD_MULTIPLE_PARA(para_vars)

/* max_threads */
#define MAX_THREADS_PARA(PARA)       PARA->max_threads
// Assuming 'PARA=para_vars'
#define MAX_THREADS                  MAX_THREADS_PARA(para_vars)

/* lock */
#define LOCK_PARA(PARA)              PARA->lock
// Assuming 'PARA=para_vars'
#define LOCK                         LOCK_PARA(para_vars)

/* num_grth */
#define NUM_GRTH_PARA(PARA)          PARA->num_grth
// Assuming 'PARA=para_vars'
#define NUM_GRTH                     NUM_GRTH_PARA(para_vars)

/* firsty */
#define FIRSTY_PARA(PARA)            PARA->firsty
// Assuming 'PARA=para_vars'
#define FIRSTY                       FIRSTY_PARA(para_vars)

/* lasty */
#define LASTY_PARA(PARA)             PARA->lasty
// Assuming 'PARA=para_vars'
#define LASTY                        LASTY_PARA(para_vars)

/* firstz */
#define FIRSTZ_PARA(PARA)            PARA->firstz
// Assuming 'PARA=para_vars'
#define FIRSTZ                       FIRSTZ_PARA(para_vars)

/* lastz */
#define LASTZ_PARA(PARA)             PARA->lastz
// Assuming 'PARA=para_vars'
#define LASTZ                        LASTZ_PARA(para_vars)

/* do_nested */
#define DO_NESTED_PARA(PARA)         PARA->do_nested
// Assuming 'PARA=para_vars'
#define DO_NESTED                    DO_NESTED_PARA(para_vars)


// input.c
/* npey */
#define NPEY_IN(IN)     IN->npey
// Assuming 'IN=input_vars'
#define NPEY            NPEY_IN(input_vars)

/* npez */
#define NPEZ_IN(IN)     IN->npez
// Assuming 'IN=input_vars'
#define NPEZ            NPEZ_IN(input_vars)

/* ichunk */
#define ICHUNK_IN(IN)   IN->ichunk
// Assuming 'IN=input_vars'
#define ICHUNK          ICHUNK_IN(input_vars)

/* nthreads */
#define NTHREADS_IN(IN) IN->nthreads
// Assuming 'IN=input_vars'
#define NTHREADS        NTHREADS_IN(input_vars)

/* nnested */
#define NNESTED_IN(IN)  IN->nnested
// Assuming 'IN=input_vars'
#define NNESTED         NNESTED_IN(input_vars)

/* ndimen */
#define NDIMEN_IN(IN)   IN->ndimen
// Assuming 'IN=input_vars'
#define NDIMEN          NDIMEN_IN(input_vars)

/* nx */
#define NX_IN(IN)       IN->nx
// Assuming 'IN=input_vars'
#define NX              NX_IN(input_vars)

/* ny */
#define NY_IN(IN)       IN->ny
// Assuming 'IN=input_vars'
#define NY              NY_IN(input_vars)

/* nz */
#define NZ_IN(IN)       IN->nz
// Assuming 'IN=input_vars'
#define NZ              NZ_IN(input_vars)

/* lx */
#define LX_IN(IN)       IN->lx
// Assuming 'IN=input_vars'
#define LX              LX_IN(input_vars)

/* ly */
#define LY_IN(IN)       IN->ly
// Assuming 'IN=input_vars'
#define LY              LY_IN(input_vars)

/* lz */
#define LZ_IN(IN)       IN->lz
// Assuming 'IN=input_vars'
#define LZ              LZ_IN(input_vars)

/* nmom */
#define NMOM_IN(IN)     IN->nmom
// Assuming 'IN=input_vars'
#define NMOM            NMOM_IN(input_vars)

/* nang */
#define NANG_IN(IN)     IN->nang
// Assuming 'IN=input_vars'
#define NANG            NANG_IN(input_vars)

/* ng */
#define NG_IN(IN)       IN->ng
// Assuming 'IN=input_vars'
#define NG              NG_IN(input_vars)

/* mat_opt */
#define MAT_OPT_IN(IN)  IN->mat_opt
// Assuming 'IN=input_vars'
#define MAT_OPT         MAT_OPT_IN(input_vars)

/* src_opt */
#define SRC_OPT_IN(IN)  IN->src_opt
// Assuming 'IN=input_vars'
#define SRC_OPT         SRC_OPT_IN(input_vars)

/* scatp */
#define SCATP_IN(IN)    IN->scatp
// Assuming 'IN=input_vars'
#define SCATP           SCATP_IN(input_vars)

/* epsi */
#define EPSI_IN(IN)     IN->epsi
// Assuming 'IN=input_vars'
#define EPSI            EPSI_IN(input_vars)

/* tf */
#define TF_IN(IN)       IN->tf
// Assuming 'IN=input_vars'
#define TF              TF_IN(input_vars)

/* iitm */
#define IITM_IN(IN)     IN->iitm
// Assuming 'IN=input_vars'
#define IITM            IITM_IN(input_vars)

/* oitm */
#define OITM_IN(IN)     IN->oitm
// Assuming 'IN=input_vars'
#define OITM            OITM_IN(input_vars)

/* timedep */
#define TIMEDEP_IN(IN)  IN->timedep
// Assuming 'IN=input_vars'
#define TIMEDEP         TIMEDEP_IN(input_vars)

/* nsteps */
#define NSTEPS_IN(IN)   IN->nsteps
// Assuming 'IN=input_vars'
#define NSTEPS          NSTEPS_IN(input_vars)

/* it_det */
#define IT_DET_IN(IN)   IN->it_det
// Assuming 'IN=input_vars'
#define IT_DET          IT_DET_IN(input_vars)

/* fluxp */
#define FLUXP_IN(IN)    IN->fluxp
// Assuming 'IN=input_vars'
#define FLUXP           FLUXP_IN(input_vars)

/* fixup */
#define FIXUP_IN(IN)    IN->fixup
// Assuming 'IN=input_vars'
#define FIXUP           FIXUP_IN(input_vars)

/* tsnap */
// time.c
#define TSNAP_TIME(TIME)    TIME->tsnap
// Assuming 'TIME=time_vars'
#define TSNAP               TSNAP_TIME(time_vars)

/* tparset */
#define TPARSET_TIME(TIME)  TIME->tparset
// Assuming 'TIME=time_vars'
#define TPARSET             TPARSET_TIME(time_vars)

/* tinp */
#define TINP_TIME(TIME)     TIME->tinp
// Assuming 'TIME=time_vars'
#define TINP                TINP_TIME(time_vars)

/* tset */
#define TSET_TIME(TIME)     TIME->tset
// Assuming 'TIME=time_vars'
#define TSET                TSET_TIME(time_vars)

/* tslv */
#define TSLV_TIME(TIME)     TIME->tslv
// Assuming 'TIME=time_vars'
#define TSLV                TSLV_TIME(time_vars)

/* tparam */
#define TPARAM_TIME(TIME)   TIME->tparam
// Assuming 'TIME=time_vars'
#define TPARAM              TPARAM_TIME(time_vars)

/* totrsrc */
#define TOTRSRC_TIME(TIME)  TIME->totrsrc
// Assuming 'TIME=time_vars'
#define TOTRSRC             TOTRSRC_TIME(time_vars)

/* tinners */
#define TINNERS_TIME(TIME)  TIME->tinners
// Assuming 'TIME=time_vars'
#define TINNERS             TINNERS_TIME(time_vars)

/* tinrsrc */
#define TINRSRC_TIME(TIME)  TIME->tinrsrc
// Assuming 'TIME=time_vars'
#define TINRSRC             TINRSRC_TIME(time_vars)

/* tsweeps */
#define TSWEEPS_TIME(TIME)  TIME->tsweeps
// Assuming 'TIME=time_vars'
#define TSWEEPS             TSWEEPS_TIME(time_vars)

/* tinrmisc */
#define TINRMISC_TIME(TIME) TIME->tinrmisc
// Assuming 'TIME=time_vars'
#define TINRMISC            TINRMISC_TIME(time_vars)

/* tslvmisc */
#define TSLVMISC_TIME(TIME) TIME->tslvmisc
// Assuming 'TIME=time_vars'
#define TSLVMISC            TSLVMISC_TIME(time_vars)

/* tout */
#define TOUT_TIME(TIME)     TIME->tout
// Assuming 'TIME=time_vars'
#define TOUT                TOUT_TIME(time_vars)

/* tgrind */
#define TGRIND_TIME(TIME)   TIME->tgrind
// Assuming 'TIME=time_vars'
#define TGRIND              TGRIND_TIME(time_vars)


// setup.c
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

// geom.c
/* ny_gl */
#define NY_GL_GEOM(GEOM)    GEOM->ny_gl
// Assuming 'GEOM=geom_vars'
#define NY_GL               NY_GL_GEOM(geom_vars)

/* nz_gl */
#define NZ_GL_GEOM(GEOM)    GEOM->nz_gl
// Assuming 'GEOM=geom_vars'
#define NZ_GL               NZ_GL_GEOM(geom_vars)

/* jlb */
#define JLB_GEOM(GEOM)      GEOM->jlb
// Assuming 'GEOM=geom_vars'
#define JLB                 JLB_GEOM(geom_vars)

/* jub */
#define JUB_GEOM(GEOM)      GEOM->jub
// Assuming 'GEOM=geom_vars'
#define JUB                 JUB_GEOM(geom_vars)

/* klb */
#define KLB_GEOM(GEOM)      GEOM->klb
// Assuming 'GEOM=geom_vars'
#define KLB                 KLB_GEOM(geom_vars)

/* kub */
#define KUB_GEOM(GEOM)      GEOM->kub
// Assuming 'GEOM=geom_vars'
#define KUB                 KUB_GEOM(geom_vars)

/* nc */
#define NC_GEOM(GEOM)       GEOM->nc
// Assuming 'GEOM=geom_vars'
#define NC                  NC_GEOM(geom_vars)

/* ndiag */
#define NDIAG_GEOM(GEOM)    GEOM->ndiag
// Assuming 'GEOM=geom_vars'
#define NDIAG               NDIAG_GEOM(geom_vars)

/* dx */
#define DX_GEOM(GEOM)       GEOM->dx
// Assuming 'GEOM=geom_vars'
#define DX                  DX_GEOM(geom_vars)

/* dy */
#define DY_GEOM(GEOM)       GEOM->dy
// Assuming 'GEOM=geom_vars'
#define DY                  DY_GEOM(geom_vars)

/* dz */
#define DZ_GEOM(GEOM)       GEOM->dz
// Assuming 'GEOM=geom_vars'
#define DZ                  DZ_GEOM(geom_vars)

/* hi */
#define HI_GEOM(GEOM)       GEOM->hi
// Assuming 'GEOM=geom_vars'
#define HI                  HI_GEOM(geom_vars)

/* hj(nang) */
#define HJ_1D_GEOM(GEOM, ANG) GEOM->hj[ANG]
#define HJ_GEOM(GEOM)         GEOM->hj
// Assuming 'GEOM=geom_vars'
#define HJ_1D(ANG)            HJ_1D_GEOM(geom_vars, ANG)
#define HJ                    HJ_GEOM(geom_vars)

/* hk(nang) */
#define HK_1D_GEOM(GEOM, ANG) GEOM->hk[ANG]
#define HK_GEOM(GEOM)         GEOM->hk
// Assuming 'GEOM=geom_vars'
#define HK_1D(ANG)            HK_1D_GEOM(geom_vars, ANG)
#define HK                    HK_GEOM(geom_vars)

/* dinv(nang,nx,ny,nz,ng) */
#ifdef ROWORDER
#define DINV_5D_GEOM(IN, GEOM, ANG, X, Y, Z, G)                         \
    GEOM->dinv[ ANG * NX_IN(IN) * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)     \
                + X * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)                 \
                + Y * NZ_IN(IN) * NG_IN(IN)                             \
                + Z * NG_IN(IN)                                         \
                + G ]
#else
#define DINV_5D_GEOM(IN, GEOM, ANG, X, Y, Z, G)                         \
    GEOM->dinv[ G   * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN) * NANG_IN(IN)   \
                + Z * NY_IN(IN) * NX_IN(IN) * NANG_IN(IN)               \
                + Y * NX_IN(IN) * NANG_IN(IN)                           \
                + X * NANG_IN(IN)                                       \
                + ANG ]
#endif
#define DINV_GEOM(GEOM)     GEOM->dinv
// Assuming 'GEOM=geom_vars' and 'IN=input_vars'
#define DINV_5D(ANG, X, Y, Z, G)                                \
    DINV_5D_GEOM(input_vars, geom_vars, ANG, X, Y, Z, G)
#define DINV                DINV_GEOM(geom_vars)

/* diag(ndiag) */
#define DIAG_1D_GEOM(GEOM, DIAG) GEOM->diag_vars[DIAG]
#define DIAG_GEOM(GEOM)          GEOM->diag_vars
// Assuming 'GEOM=geom_vars'
#define DIAG_1D(DIAG)            DIAG_1D_GEOM(geom_vars, DIAG)
#define DIAG                     DIAG_GEOM(geom_vars)


// sn.c
/* cmom */
#define CMOM_SN(SN) SN->cmom
// Assuming 'SN=sn_vars'
#define CMOM CMOM_SN(sn_vars)

/* noct */
#define NOCT_SN(SN) SN->noct
// Assuming 'SN=sn_vars'
#define NOCT NOCT_SN(sn_vars)

/* mu(nang) */
#define MU_1D_SN(SN, ANG)   SN->mu[ANG]
#define MU_SN(SN)           SN->mu
// Assuming 'SN=sn_vars'
#define MU_1D(ANG)          MU_1D_SN(sn_vars, ANG)
#define MU                  MU_SN(sn_vars)

/* eta(nang) */
#define ETA_1D_SN(SN, ANG)  SN->eta[ANG]
#define ETA_SN(SN)          SN->eta
// Assuming 'SN=sn_vars'
#define ETA_1D(ANG)         ETA_1D_SN(sn_vars, ANG)
#define ETA                 ETA_SN(sn_vars)

/* xi(nang) */
#define XI_1D_SN(SN, ANG)   SN->xi[ANG]
#define XI_SN(SN)           SN->xi
// Assuming 'SN=sn_vars'
#define XI_1D(ANG)          XI_1D_SN(sn_vars, ANG)
#define XI                  XI_SN(sn_vars)

/* w(nang) */
#define W_1D_SN(SN, ANG)    SN->w[ANG]
#define W_SN(SN)            SN->w
// Assuming 'SN=sn_vars'
#define W_1D(ANG)           W_1D_SN(sn_vars, ANG)
#define W                   W_SN(sn_vars)

/* wmu(nang) */
#define WMU_1D_SN(SN, ANG)  SN->wmu[ANG]
#define WMU_SN(SN)          SN->wmu
// Assuming 'SN=sn_vars'
#define WMU_1D(ANG)         WMU_1D_SN(sn_vars, ANG)
#define WMU                 WMU_SN(sn_vars)

/* weta(nang)*/
#define WETA_1D_SN(SN, ANG) SN->weta[ANG]
#define WETA_SN(SN)         SN->weta
// Assuming 'SN=sn_vars'
#define WETA_1D(ANG)        WETA_1D_SN(sn_vars, ANG)
#define WETA                WETA_SN(sn_vars)

/* wxi(nang) */
#define WXI_1D_SN(SN, ANG)  SN->wxi[ANG]
#define WXI_SN(SN)          SN->wxi
// Assuming 'SN=sn_vars'
#define WXI_1D(ANG)         WXI_1D_SN(sn_vars, ANG)
#define WXI                 WXI_SN(sn_vars)

/* ec(nang,cmom,noct) */
#ifdef ROWORDER
#define EC_3D_SN(SN, ANG, MOM, OCT)             \
    SN->ec[ ANG   * CMOM_SN(SN) * NOCT_SN(SN)   \
            + MOM * NOCT_SN(SN)                 \
            + OCT ]
#else
#define EC_3D_SN(IN, SN, ANG, MOM, OCT)         \
    SN->ec[ OCT   * CMOM_SN(SN) * NANG_IN(IN)   \
            + MOM * NANG_IN(IN)                 \
            + ANG ]
#endif
#define EC_SN(SN)         SN->ec
// Assuming 'SN=sn_vars'
#ifdef ROWORDER
#define EC_3D(ANG, MOM, OCT) EC_3D_SN(sn_vars, ANG, MOM, OCT)
#else
#define EC_3D(ANG, MOM, OCT) EC_3D_SN(input_vars, sn_vars, ANG, MOM, OCT)
#endif
#define EC                EC_SN(sn_vars)

/* lma(nmom) */
#define LMA_1D_SN(SN, MOM)  SN->lma[MOM]
#define LMA_SN(SN)        SN->lma
// Assuming 'SN=sn_vars'
#define LMA_1D(MOM)         LMA_1D_SN(sn_vars, MOM)
#define LMA               LMA_SN(sn_vars)


// data.c
/* v(ng) */
#define V_1D_DATA(DATA, G) DATA->v[G]
#define V_DATA(DATA)       DATA->v
// Assuming 'DATA=data_vars'
#define V_1D(G)            V_1D_DATA(data_vars, G)
#define V                  V_DATA(data_vars)

/* nmat */
#define NMAT_DATA(DATA)    DATA->nmat
// Assuming 'DATA=data_vars'
#define NMAT               NMAT_DATA(data_vars)

/* mat(nx,ny,nz) */
#ifdef ROWORDER
#define MAT_3D_DATA(IN, DATA, X, Y, Z)          \
    DATA->mat[ X   * NY_IN(IN) * NZ_IN(IN)      \
               + Y * NZ_IN(IN)                  \
               + Z ]
#else
#define MAT_3D_DATA(IN, DATA, X, Y, Z)          \
    DATA->mat[ Z   * NY_IN(IN) * NX_IN(IN)      \
               + Y * NX_IN(IN)                  \
               + X ]
#endif
#define MAT_DATA(DATA) DATA->mat
// Assuming 'DATA=data_vars' and 'IN=input_vars'
#define MAT_3D(X, Y, Z) MAT_3D_DATA(input_vars, data_vars, X, Y, Z)
#define MAT MAT_DATA(data_vars)

/* qi(nx,ny,nz,ng) */
#ifdef ROWORDER
#define QI_4D_DATA(IN, DATA, X, Y, Z, G)                \
    DATA->qi[ X   * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)   \
              + Y * NZ_IN(IN) * NG_IN(IN)               \
              + Z * NG_IN(IN)                           \
              + G ]
#else
#define QI_4D_DATA(IN, DATA, X, Y, Z, G)                \
    DATA->qi[ G   * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN)   \
              + Z * NY_IN(IN) * NX_IN(IN)               \
              + Y * NX_IN(IN)                           \
              + X ]
#endif
#define QI_DATA(DATA) DATA->qi
// Assuming 'DATA=data_vars' and 'IN=input_vars'
#define QI_4D(X, Y, Z, G) QI_4D_DATA(input_vars, data_vars, X, Y, Z, G)
#define QI QI_DATA(data_vars)

/* qim(nang,nx,ny,nz,noct,ng) */
#ifdef ROWORDER
#define QIM_6D_DATA(IN, SN, DATA, ANG, X, Y, Z, OCT, G)                 \
    DATA->qim[ ANG   * NX_IN(IN)   * NY_IN(IN)   * NZ_IN(IN)   * NOCT_SN(SN) * NG_IN(IN) \
               + X   * NY_IN(IN)   * NZ_IN(IN)   * NOCT_SN(SN) * NG_IN(IN) \
               + Y   * NZ_IN(IN)   * NOCT_SN(SN) * NG_IN(IN)            \
               + Z   * NOCT_SN(SN) * NG_IN(IN)                          \
               + OCT * NG_IN(IN)                                        \
               + G ]
#else
#define QIM_6D_DATA(IN, SN, DATA, ANG, X, Y, Z, OCT, G)                 \
    DATA->qim[ G     * NOCT_SN(SN) * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN) * NANG_IN(IN) \
               + OCT * NZ_IN(IN)   * NY_IN(IN) * NX_IN(IN) * NANG_IN(IN) \
               + Z   * NY_IN(IN)   * NX_IN(IN) * NANG_IN(IN)            \
               + Y   * NX_IN(IN)   * NANG_IN(IN)                        \
               + X   * NANG_IN(IN)                                      \
               + ANG ]
#endif
#define QIM_DATA(DATA) DATA->qim
// Assuming 'DATA=data_vars' and 'IN=input_vars'
#define QIM_6D(ANG, X, Y, Z, OCT, G)                                    \
    QIM_6D_DATA(input_vars, sn_vars, data_vars, ANG, X, Y, Z, OCT, G)
#define QIM QIM_DATA(data_vars)

/* sigt(nmat,ng) */
#ifdef ROWORDER
#define SIGT_2D_DATA(IN, DATA, N_MAT, G)        \
    DATA->sigt[ N_MAT * NG_IN(IN)               \
                + G ]
#else
#define SIGT_2D_DATA( DATA, N_MAT, G)           \
    DATA->sigt[ G * NMAT_DATA(DATA)             \
                + N_MAT ]
#endif
#define SIGT_DATA(DATA) DATA->sigt
// Assuming 'DATA=data_vars'
#ifdef ROWORDER
#define SIGT_2D(N_MAT, G) SIGT_2D_DATA(input_vars, data_vars, N_MAT, G)
#else
#define SIGT_2D(N_MAT, G) SIGT_2D_DATA(data_vars, N_MAT, G)
#endif
#define SIGT SIGT_DATA(data_vars)

/* siga(nmat,ng) */
#ifdef ROWORDER
#define SIGA_2D_DATA(IN, DATA, N_MAT, G)        \
    DATA->siga[ N_MAT * NG_IN(IN)               \
                + G ]
#else
#define SIGA_2D_DATA(DATA, N_MAT, G)            \
    DATA->siga[ G * NMAT_DATA(DATA)             \
                + N_MAT ]
#endif
#define SIGA_DATA(DATA) DATA->siga
// Assuming 'DATA=data_vars'
#ifdef ROWORDER
#define SIGA_2D(N_MAT, G) SIGA_2D_DATA(input_vars, data_vars, N_MAT, G)
#else
#define SIGA_2D(N_MAT, G) SIGA_2D_DATA(data_vars, N_MAT, G)
#endif
#define SIGA SIGA_DATA(data_vars)

/* sigs(nmat,ng) */
#ifdef ROWORDER
#define SIGS_2D_DATA(IN, DATA, N_MAT, G)        \
    DATA->sigs[ N_MAT * NG_IN(IN)               \
                + G ]
#else
#define SIGS_2D_DATA( DATA, N_MAT, G)           \
    DATA->sigs[ G * NMAT_DATA(DATA)             \
                + N_MAT ]
#endif
#define SIGS_DATA(DATA) DATA->sigs
// Assuming 'DATA=data_vars'
#ifdef ROWORDER
#define SIGS_2D(N_MAT, G) SIGS_2D_DATA(input_vars, data_vars, N_MAT, G)
#else
#define SIGS_2D(N_MAT, G) SIGS_2D_DATA(data_vars, N_MAT, G)
#endif
#define SIGS SIGS_DATA(data_vars)

/* slgg(nmat,nmom,ng,ng) */
#ifdef ROWORDER
#define SLGG_4D_DATA(IN, DATA, N_MAT, MOM, G1, G2)              \
    DATA->slgg[ N_MAT   * NMOM_IN(IN) * NG_IN(IN) * NG_IN(IN)   \
                + MOM * NG_IN(IN) * NG_IN(IN)                   \
                + G1  * NG_IN(IN)                               \
                + G2 ]
#else
#define SLGG_4D_DATA(IN, DATA, N_MAT, MOM, G1, G2)                      \
    DATA->slgg[ G2    * NG_IN(IN)   * NMOM_IN(IN) * NMAT_DATA(DATA)     \
                + G1  * NMOM_IN(IN) * NMAT_DATA(DATA)                   \
                + MOM * NMAT_DATA(DATA)                                 \
                + N_MAT ]
#endif
#define SLGG_DATA(DATA) DATA->slgg
// Assuming 'DATA=data_vars' and 'IN=input_vars'
#define SLGG_4D(N_MAT, MOM, G1, G2) SLGG_4D_DATA(input_vars, data_vars, N_MAT, MOM, G1, G2)
#define SLGG SLGG_DATA(data_vars)

/* vdelt(ng) */
#define VDELT_1D_DATA(DATA, X)     DATA->vdelt[X]
#define VDELT_DATA(DATA)           DATA->vdelt
// Assuming 'DATA=data_vars'
#define VDELT_1D(X)                VDELT_1D_DATA(data_vars, X)
#define VDELT                      VDELT_DATA(data_vars)


// control.c
/* dt */
#define DT_CNTRL(CNTRL)            CNTRL->dt
// Assuming 'CNTRL=control_vars'
#define DT                         DT_CNTRL(control_vars)

/* tolr */
#define TOLR_CNTRL(CNTRL)          CNTRL->tolr
// Assuming 'CNTRL=control_vars'
#define TOLR                       TOLR_CNTRL(control_vars)

/* dfmxi(ng) */
#define DFMXI_1D_CNTRL(CNTRL, X)   CNTRL->dfmxi[X]
#define DFMXI_CNTRL(CNTRL)         CNTRL->dfmxi
// Assuming 'CNTRL=control_vars'
#define DFMXI_1D(X)                DFMXI_1D_CNTRL(control_vars, X)
#define DFMXI                      DFMXI_CNTRL(control_vars)

/* dfmxo */
#define DFMXO_CNTRL(CNTRL)         CNTRL->dfmxo
// Assuming 'CNTRL=control_vars'
#define DFMXO                      DFMXO_CNTRL(control_vars)

/* inrdone(ng) */
#define INRDONE_1D_CNTRL(CNTRL, X) CNTRL->inrdone[X]
#define INRDONE_CNTRL(CNTRL)       CNTRL->inrdone
// Assuming 'CNTRL=control_vars'
#define INRDONE_1D(X)              INRDONE_1D_CNTRL(control_vars, X)
#define INRDONE                    INRDONE_CNTRL(control_vars)

/* otrdone */
#define OTRDONE_CNTRL(CNTRL)       CNTRL->otrdone
// Assuming 'CNTRL=control_vars'
#define OTRDONE                    OTRDONE_CNTRL(control_vars)


// mms.c
/* ref_flux(nx,ny,nz,ng) */
#ifdef ROWORDER
#define REF_FLUX_4D_MMS(IN, MMS, X, Y, Z, G)                    \
    MMS->ref_flux[ X   * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)      \
                   + Y * NZ_IN(IN) * NG_IN(IN)                  \
                   + Z * NG_IN(IN)                              \
                   + G ]
#else
#define REF_FLUX_4D_MMS(IN, MMS, X, Y, Z, G)                    \
    MMS->ref_flux[ G   * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN)      \
                   + Z * NY_IN(IN) * NX_IN(IN)                  \
                   + Y * NX_IN(IN)                              \
                   + X ]

#endif
#define REF_FLUX_MMS(MMS) MMS->ref_flux
// Assuming 'MMS=mms_vars' and 'IN=input_vars'
#define REF_FLUX_4D(X, Y, Z, G)                         \
    REF_FLUX_4D_MMS(input_vars, mms_vars, X, Y, Z, G)
#define REF_FLUX REF_FLUX_MMS(mms_vars)

/* ref_fluxm(cmom-1,nx,ny,nz,ng) */
#ifdef ROWORDER
#define REF_FLUXM_5D_MMS(IN, MMS, MOM1, X, Y, Z, G)                     \
    MMS->ref_fluxm[ MOM1 * NX_IN(IN) * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN) \
                    + X  * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)            \
                    + Y  * NZ_IN(IN) * NG_IN(IN)                        \
                    + Z  * NG_IN(IN)                                    \
                    + G ]
#else
#define REF_FLUXM_5D_MMS(IN, SN, MMS, MOM1, X, Y, Z, G)                 \
    MMS->ref_fluxm[ G    * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN) * (CMOM_SN(SN)-1) \
                    + Z  * NY_IN(IN) * NX_IN(IN) * (CMOM_SN(SN)-1)      \
                    + Y  * NX_IN(IN) * (CMOM_SN(SN)-1)                  \
                    + X  * (CMOM_SN(SN)-1)                              \
                    + MOM1 ]
#endif
#define REF_FLUXM_MMS(MMS) MMS->ref_fluxm
// Assuming 'MMS=mms_vars' and 'IN=input_vars'
#ifdef ROWORDER
#define REF_FLUXM_5D(MOM1, X, Y, Z, G)                          \
    REF_FLUXM_5D_MMS(input_vars, mms_vars, MOM1, X, Y, Z, G)
#else
#define REF_FLUXM_5D(MOM1, X, Y, Z, G)                                  \
    REF_FLUXM_5D_MMS(input_vars, sn_vars, mms_vars, MOM1, X, Y, Z, G)
#endif
#define REF_FLUXM REF_FLUXM_MMS(mms_vars)

/* a_const */
#define A_CONST_MMS(MMS)   MMS->a_const
// Assuming 'MMS=mms_vars'
#define A_CONST            A_CONST_MMS(mms_vars)

/* b_const */
#define B_CONST_MMS(MMS)   MMS->b_const
// Assuming 'MMS=mms_vars'
#define B_CONST            B_CONST_MMS(mms_vars)

/* c_const */
#define C_CONST_MMS(MMS)   MMS->c_const
// Assuming 'MMS=mms_vars'
#define C_CONST            C_CONST_MMS(mms_vars)

/* ib(nx+1) */
#define IB_1D_MMS(MMS, X1) MMS->ib[X1]
#define IB_MMS(MMS)        MMS->ib
// Assuming 'MMS=mms_vars'
#define IB_1D(X1)          IB_1D_MMS(mms_vars, X1)
#define IB                 IB_MMS(mms_vars)

/* jb(ny+1) */
#define JB_1D_MMS(MMS, Y1) MMS->jb[Y1]
#define JB_MMS(MMS)        MMS->jb
// Assuming 'MMS=mms_vars'
#define JB_1D(Y1)          JB_1D_MMS(mms_vars, Y1)
#define JB                 JB_MMS(mms_vars)

/* kb(nz+1) */
#define KB_1D_MMS(MMS, Z1) MMS->kb[Z1]
#define KB_MMS(MMS)        MMS->kb
// Assuming 'MMS=mms_vars'
#define KB_1D(Z1)          KB_1D_MMS(mms_vars, Z1)
#define KB                 KB_MMS(mms_vars)

// solvar.c
/* flux(nx,ny,nz,ng) */
#ifdef ROWORDER
#define FLUX_4D_SOL(IN, SOL, X, Y, Z, G)                \
    SOL->flux[ X   * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)  \
               + Y * NZ_IN(IN) * NG_IN(IN)              \
               + Z * NG_IN(IN)                          \
               + G ]
#else
#define FLUX_4D_SOL(IN, SOL, X, Y, Z, G)                \
    SOL->flux[ G   * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN)  \
               + Z * NY_IN(IN) * NX_IN(IN)              \
               + Y * NX_IN(IN)                          \
               + X ]
#endif
#define FLUX_SOL(SOL) SOL->flux
// Assuming 'SOL=solvar_vars'
#define FLUX_4D(X, Y, Z, G) FLUX_4D_SOL(input_vars, solvar_vars, X, Y, Z, G)
#define FLUX FLUX_SOL(solvar_vars)

/* fluxpo(nx,ny,nz,ng) */
#ifdef ROWORDER
#define FLUXPO_4D_SOL(IN, SOL, X, Y, Z, G)                      \
    SOL->fluxpo[ X   * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)        \
                 + Y * NZ_IN(IN) * NG_IN(IN)                    \
                 + Z * NG_IN(IN)                                \
                 + G ]
#else
#define FLUXPO_4D_SOL(IN, SOL, X, Y, Z, G)                      \
    SOL->fluxpo[ G   * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN)        \
                 + Z * NY_IN(IN) * NX_IN(IN)                    \
                 + Y * NX_IN(IN)                                \
                 + X ]
#endif
#define FLUXPO_SOL(SOL) SOL->fluxpo
// Assuming 'SOL=solvar_vars'
#define FLUXPO_4D(X, Y, Z, G) FLUXPO_4D_SOL(input_vars, solvar_vars, X, Y, Z, G)
#define FLUXPO FLUXPO_SOL(solvar_vars)

/* fluxpi(nx,ny,nz,ng) */
#ifdef ROWORDER
#define FLUXPI_4D_SOL(IN, SOL, X, Y, Z, G)                      \
    SOL->fluxpi[ X   * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)        \
                 + Y * NZ_IN(IN) * NG_IN(IN)                    \
                 + Z * NG_IN(IN)                                \
                 + G ]
#else
#define FLUXPI_4D_SOL(IN, SOL, X, Y, Z, G)                      \
    SOL->fluxpi[ G   * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN)        \
                 + Z * NY_IN(IN) * NX_IN(IN)                    \
                 + Y * NX_IN(IN)                                \
                 + X ]
#endif
#define FLUXPI_SOL(SOL) SOL->fluxpi
// Assuming 'SOL=solvar_vars'
#define FLUXPI_4D(X, Y, Z, G) FLUXPI_4D_SOL(input_vars, solvar_vars, X, Y, Z, G)
#define FLUXPI FLUXPI_SOL(solvar_vars)

/* fluxm(cmom-1,nx,ny,nz,ng) */
#ifdef ROWORDER
#define FLUXM_5D_SOL(IN, SOL, MOM1, X, Y, Z, G)                         \
    SOL->fluxm[ MOM1 *NX_IN(IN) * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)     \
                + X * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)                 \
                + Y * NZ_IN(IN) * NG_IN(IN)                             \
                + Z * NG_IN(IN)                                         \
                + G ]
#else
#define FLUXM_5D_SOL(IN, SN, SOL, MOM1, X, Y, Z, G)                     \
    SOL->fluxm[ G   * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN) * (CMOM_SN(SN)-1) \
                + Z * NY_IN(IN) * NX_IN(IN) * (CMOM_SN(SN)-1)           \
                + Y * NX_IN(IN) * (CMOM_SN(SN)-1)                       \
                + X * (CMOM_SN(SN)-1)                                   \
                + MOM1 ]
#endif
#define FLUXM_SOL(SOL) SOL->fluxm
// Assuming 'SOL=solvar_vars'
#ifdef ROWORDER
#define FLUXM_5D(MOM1, X, Y, Z, G)                              \
    FLUXM_5D_SOL(input_vars, solvar_vars, MOM1, X, Y, Z, G)
#else
#define FLUXM_5D(MOM1, X, Y, Z, G)                                      \
    FLUXM_5D_SOL(input_vars, sn_vars, solvar_vars, MOM1, X, Y, Z, G)
#endif
#define FLUXM FLUXM_SOL(solvar_vars)

/* q2grp(cmom,nx,ny,nz,ng) */
#ifdef ROWORDER
#define Q2GRP_5D_SOL(IN, SOL, MOM1, X, Y, Z, G)                         \
    SOL->q2grp[ MOM1 * NX_IN(IN) * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)    \
                + X  * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)                \
                + Y  * NZ_IN(IN) * NG_IN(IN)                            \
                + Z  * NG_IN(IN)                                        \
                + G ]
#else
#define Q2GRP_5D_SOL(IN, SN, SOL, MOM1, X, Y, Z, G)                     \
    SOL->q2grp[ G   * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN) * CMOM_SN(SN)   \
                + Z * NY_IN(IN) * NX_IN(IN) * CMOM_SN(SN)               \
                + Y * NX_IN(IN) * CMOM_SN(SN)                           \
                + X * CMOM_SN(SN)                                       \
                + MOM1 ]
#endif
#define Q2GRP_SOL(SOL) SOL->q2grp
// Assuming 'SOL=solvar_vars'
#ifdef ROWORDER
#define Q2GRP_5D(MOM1, X, Y, Z, G)                              \
    Q2GRP_5D_SOL(input_vars, solvar_vars, MOM1, X, Y, Z, G)
#else
#define Q2GRP_5D(MOM1, X, Y, Z, G)                                      \
    Q2GRP_5D_SOL(input_vars, sn_vars, solvar_vars, MOM1, X, Y, Z, G)
#endif
#define Q2GRP Q2GRP_SOL(solvar_vars)

/* qtot(cmom,nx,ny,nz,ng) */
#ifdef ROWORDER
#define QTOT_5D_SOL(IN, SOL, MOM1, X, Y, Z, G)                          \
    SOL->qtot[ MOM1 * NX_IN(IN) * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)     \
               + X  * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)                 \
               + Y  * NZ_IN(IN) * NG_IN(IN)                             \
               + Z  * NG_IN(IN)                                         \
               + G ]
#else
#define QTOT_5D_SOL(IN, SN, SOL, MOM1, X, Y, Z, G)                      \
    SOL->qtot[ G   * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN) * CMOM_SN(SN)    \
               + Z * NY_IN(IN) * NX_IN(IN) * CMOM_SN(SN)                \
               + Y * NX_IN(IN) * CMOM_SN(SN)                            \
               + X * CMOM_SN(SN)                                        \
               + MOM1 ]
#endif
#define QTOT_SOL(SOL) SOL->qtot
// Assuming 'SOL=solvar_vars'
#ifdef ROWORDER
#define QTOT_5D(MOM1, X, Y, Z, G)                               \
    QTOT_5D_SOL(input_vars, solvar_vars, MOM1, X, Y, Z, G)
#else
#define QTOT_5D(MOM1, X, Y, Z, G)                                       \
    QTOT_5D_SOL(input_vars, sn_vars, solvar_vars, MOM1, X, Y, Z, G)
#endif
#define QTOT QTOT_SOL(solvar_vars)

/* t_xs(nx,ny,nz,ng) */
#ifdef ROWORDER
#define T_XS_4D_SOL(IN, SOL, X, Y, Z, G)                \
    SOL->t_xs[ X   * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)  \
               + Y * NZ_IN(IN) * NG_IN(IN)              \
               + Z * NG_IN(IN)                          \
               + G ]
#else
#define T_XS_4D_SOL(IN, SOL, X, Y, Z, G)                \
    SOL->t_xs[ G   * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN)  \
               + Z * NY_IN(IN) * NX_IN(IN)              \
               + Y * NX_IN(IN)                          \
               + X ]
#endif
#define T_XS_SOL(SOL) SOL->t_xs
// Assuming 'SOL=solvar_vars'
#define T_XS_4D(X, Y, Z, G) T_XS_4D_SOL(input_vars, solvar_vars, X, Y, Z, G)
#define T_XS T_XS_SOL(solvar_vars)

/* a_xs(nx,ny,nz,ng) */
#ifdef ROWORDER
#define A_XS_4D_SOL(IN, SOL, X, Y, Z, G)                \
    SOL->a_xs[ X   * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)  \
               + Y * NZ_IN(IN) * NG_IN(IN)              \
               + Z * NG_IN(IN)                          \
               + G ]
#else
#define A_XS_4D_SOL(IN, SOL, X, Y, Z, G)                \
    SOL->a_xs[ G   * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN)  \
               + Z * NY_IN(IN) * NX_IN(IN)              \
               + Y * NX_IN(IN)                          \
               + X ]
#endif
#define A_XS_SOL(SOL) SOL->a_xs
// Assuming 'SOL=solvar_vars'
#define A_XS_4D(X, Y, Z, G) A_XS_4D_SOL(input_vars, solvar_vars, X, Y, Z, G)
#define A_XS A_XS_SOL(solvar_vars)

/* s_xs(nmom,nx,ny,nz,ng) */
#ifdef ROWORDER
#define S_XS_5D_SOL(IN, SOL, MOM, X, Y, Z, G)                           \
    SOL->s_xs[ MOM * NX_IN(IN) * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)      \
               + X * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)                  \
               + Y * NZ_IN(IN) * NG_IN(IN)                              \
               + Z * NG_IN(IN)                                          \
               + G ]
#else
#define S_XS_5D_SOL(IN, SOL, MOM, X, Y, Z, G)                           \
    SOL->s_xs[ G   * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN) * NMOM_IN(IN)    \
               + Z * NY_IN(IN) * NX_IN(IN) * NMOM_IN(IN)                \
               + Y * NX_IN(IN) * NMOM_IN(IN)                            \
               + X * NMOM_IN(IN)                                        \
               + MOM ]
#endif
#define S_XS_SOL(SOL) SOL->s_xs
// Assuming 'SOL=solvar_vars'
#define S_XS_5D(MOM, X, Y, Z, G)                                \
    S_XS_5D_SOL(input_vars, solvar_vars, MOM, X, Y, Z, G)
#define S_XS S_XS_SOL(solvar_vars)

/* psii(nang,ny,nz,ng) */
#ifdef ROWORDER
#define PSII_4D_SOL(IN, SOL, ANG, Y, Z, G)                      \
    SOL->psii[ ANG   * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)        \
               + Y * NZ_IN(IN) * NG_IN(IN)                      \
               + Z * NG_IN(IN)                                  \
               + G ]
#else
#define PSII_4D_SOL(IN, SOL, ANG, Y, Z, G)                      \
    SOL->psii[ G   * NZ_IN(IN) * NY_IN(IN) * NANG_IN(IN)        \
               + Z * NY_IN(IN) * NANG_IN(IN)                    \
               + Y * NANG_IN(IN)                                \
               + ANG ]
#endif
#define PSII_SOL(SOL) SOL->psii
// Assuming 'SOL=solvar_vars'
#define PSII_4D(ANG, Y, Z, G) PSII_4D_SOL(input_vars, solvar_vars, ANG, Y, Z, G)
#define PSII PSII_SOL(solvar_vars)

/* psij(nang,ichunk,nz,ng) */
#ifdef ROWORDER
#define PSIJ_4D_SOL(IN, SOL, ANG, CHUNK, Z, G)                  \
    SOL->psij[ ANG   * ICHUNK_IN(IN) * NZ_IN(IN) * NG_IN(IN)    \
               + CHUNK * NZ_IN(IN) * NG_IN(IN)                  \
               + Z * NG_IN(IN)                                  \
               + G ]
#else
#define PSIJ_4D_SOL(IN, SOL, ANG, CHUNK, Z, G)                  \
    SOL->psij[ G   * NZ_IN(IN) * ICHUNK_IN(IN) * NANG_IN(IN)    \
               + Z * ICHUNK_IN(IN) * NANG_IN(IN)                \
               + CHUNK * NANG_IN(IN)                            \
               + ANG ]
#endif
#define PSIJ_SOL(SOL) SOL->psij
// Assuming 'SOL=solvar_vars'
#define PSIJ_4D(ANG, CHUNK, Z, G) PSIJ_4D_SOL(input_vars, solvar_vars, ANG, CHUNK, Z, G)
#define PSIJ PSIJ_SOL(solvar_vars)

/* psik(nang,ichunk,ny,ng) */
#ifdef ROWORDER
#define PSIK_4D_SOL(IN, SOL, ANG, CHUNK, Y, G)                  \
    SOL->psik[ ANG   * ICHUNK_IN(IN) * NY_IN(IN) * NG_IN(IN)    \
               + CHUNK * NY_IN(IN) * NG_IN(IN)                  \
               + Y * NG_IN(IN)                                  \
               + G ]
#else
#define PSIK_4D_SOL(IN, SOL, ANG, CHUNK, Y, G)                  \
    SOL->psik[ G   * NY_IN(IN) * ICHUNK_IN(IN) * NANG_IN(IN)    \
               + Y * ICHUNK_IN(IN) * NANG_IN(IN)                \
               + CHUNK * NANG_IN(IN)                            \
               + ANG ]
#endif
#define PSIK_SOL(SOL) SOL->psik
// Assuming 'SOL=solvar_vars'
#define PSIK_4D(ANG, CHUNK, Y, G) PSIK_4D_SOL(input_vars, solvar_vars, ANG, CHUNK, Y, G)
#define PSIK PSIK_SOL(solvar_vars)

/* jb_in(nang,ichunk,nz,ng) */
#ifdef ROWORDER
#define JB_IN_4D_SOL(IN, SOL, ANG, CHUNK, Z, G)                 \
    SOL->jb_in[ ANG   * ICHUNK_IN(IN) * NZ_IN(IN) * NG_IN(IN)   \
                + CHUNK * NZ_IN(IN) * NG_IN(IN)                 \
                + Z * NG_IN(IN)                                 \
                + G ]
#else
#define JB_IN_4D_SOL(IN, SOL, ANG, CHUNK, Z, G)                 \
    SOL->jb_in[ G   * NZ_IN(IN) * ICHUNK_IN(IN) * NANG_IN(IN)   \
                + Z * ICHUNK_IN(IN) * NANG_IN(IN)               \
                + CHUNK * NANG_IN(IN)                           \
                + ANG ]
#endif
#define JB_IN_SOL(SOL) SOL->jb_in
// Assuming 'SOL=solvar_vars'
#define JB_IN_4D(ANG, CHUNK, Z, G) JB_IN_4D_SOL(input_vars, solvar_vars, ANG, CHUNK, Z, G)
#define JB_IN JB_IN_SOL(solvar_vars)

/* jb_out(nang,ichunk,nz,ng) */
#ifdef ROWORDER
#define JB_OUT_4D_SOL(IN, SOL, ANG, CHUNK, Z, G)                \
    SOL->jb_out[ ANG   * ICHUNK_IN(IN) * NZ_IN(IN) * NG_IN(IN)  \
                 + CHUNK * NZ_IN(IN) * NG_IN(IN)                \
                 + Z * NG_IN(IN)                                \
                 + G ]
#else
#define JB_OUT_4D_SOL(IN, SOL, ANG, CHUNK, Z, G)                \
    SOL->jb_out[ G   * NZ_IN(IN) * ICHUNK_IN(IN) * NANG_IN(IN)  \
                 + Z * ICHUNK_IN(IN) * NANG_IN(IN)              \
                 + CHUNK * NANG_IN(IN)                          \
                 + ANG ]
#endif
#define JB_OUT_SOL(SOL) SOL->jb_out
// Assuming 'SOL=solvar_vars'
#define JB_OUT_4D(ANG, CHUNK, Z, G) JB_OUT_4D_SOL(input_vars, solvar_vars, ANG, CHUNK, Z, G)
#define JB_OUT JB_OUT_SOL(solvar_vars)

/* kb_in(nang,ichunk,ny,ng) */
#ifdef ROWORDER
#define KB_IN_4D_SOL(IN, SOL, ANG, CHUNK, Y, G)                 \
    SOL->kb_in[ ANG   * ICHUNK_IN(IN) * NY_IN(IN) * NG_IN(IN)   \
                + CHUNK * NY_IN(IN) * NG_IN(IN)                 \
                + Y * NG_IN(IN)                                 \
                + G ]
#else
#define KB_IN_4D_SOL(IN, SOL, ANG, CHUNK, Y, G)                 \
    SOL->kb_in[ G   * NY_IN(IN) * ICHUNK_IN(IN) * NANG_IN(IN)   \
                + Y * ICHUNK_IN(IN) * NANG_IN(IN)               \
                + CHUNK * NANG_IN(IN)                           \
                + ANG ]
#endif
#define KB_IN_SOL(SOL) SOL->kb_in
// Assuming 'SOL=solvar_vars'
#define KB_IN_4D(ANG, CHUNK, Y, G) KB_IN_4D_SOL(input_vars, solvar_vars, ANG, CHUNK, Y, G)
#define KB_IN KB_IN_SOL(solvar_vars)

/* kb_out(nang,ichunk,ny,ng) */
#ifdef ROWORDER
#define KB_OUT_4D_SOL(IN, SOL, ANG, CHUNK, Y, G)                \
    SOL->kb_out[ ANG   * ICHUNK_IN(IN) * NY_IN(IN) * NG_IN(IN)  \
                 + CHUNK * NY_IN(IN) * NG_IN(IN)                \
                 + Y * NG_IN(IN)                                \
                 + G ]
#else
#define KB_OUT_4D_SOL(IN, SOL, ANG, CHUNK, Y, G)                \
    SOL->kb_out[ G   * NY_IN(IN) * ICHUNK_IN(IN) * NANG_IN(IN)  \
                 + Y * ICHUNK_IN(IN) * NANG_IN(IN)              \
                 + CHUNK * NANG_IN(IN)                          \
                 + ANG ]
#endif
#define KB_OUT_SOL(SOL) SOL->kb_out
// Assuming 'SOL=solvar_vars'
#define KB_OUT_4D(ANG, CHUNK, Y, G) KB_OUT_4D_SOL(input_vars, solvar_vars, ANG, CHUNK, Y, G)
#define KB_OUT KB_OUT_SOL(solvar_vars)

/* flkx(nx+1,ny,nz,ng) */
#ifdef ROWORDER
#define FLKX_4D_SOL(IN, SOL, X, Y, Z, G)                \
    SOL->flkx[ X   * NY_IN(IN) * NZ_IN(IN) * NG_IN(IN)  \
               + Y * NZ_IN(IN) * NG_IN(IN)              \
               + Z * NG_IN(IN)                          \
               + G ]
#else
#define FLKX_4D_SOL(IN, SOL, X, Y, Z, G)                        \
    SOL->flkx[ G   * NZ_IN(IN) * NY_IN(IN) * (NX_IN(IN)+1)      \
               + Z * NY_IN(IN) * (NX_IN(IN)+1)                  \
               + Y * (NX_IN(IN)+1)                              \
               + X ]
#endif
#define FLKX_SOL(SOL) SOL->flkx
// Assuming 'SOL=solvar_vars'
#define FLKX_4D(X, Y, Z, G) FLKX_4D_SOL(input_vars, solvar_vars, X, Y, Z, G)
#define FLKX FLKX_SOL(solvar_vars)

/* flky(nx,ny+1,nz,ng) */
#ifdef ROWORDER
#define FLKY_4D_SOL(IN, SOL, X, Y, Z, G)                        \
    SOL->flky[ X   * (NY_IN(IN)+1) * NZ_IN(IN) * NG_IN(IN)      \
               + Y * NZ_IN(IN) * NG_IN(IN)                      \
               + Z * NG_IN(IN)                                  \
               + G ]
#else
#define FLKY_4D_SOL(IN, SOL, X, Y, Z, G)                        \
    SOL->flky[ G   * NZ_IN(IN) * (NY_IN(IN)+1) * NX_IN(IN)      \
               + Z * (NY_IN(IN)+1) * NX_IN(IN)                  \
               + Y * NX_IN(IN)                                  \
               + X ]
#endif
#define FLKY_SOL(SOL) SOL->flky
// Assuming 'SOL=solvar_vars'
#define FLKY_4D(X, Y, Z, G) FLKY_4D_SOL(input_vars, solvar_vars, X, Y, Z, G)
#define FLKY FLKY_SOL(solvar_vars)

/* flkz(nx,ny,nz+1,ng) */
#ifdef ROWORDER
#define FLKZ_4D_SOL(IN, SOL, X, Y, Z, G)                        \
    SOL->flkz[ X   * NY_IN(IN) * (NZ_IN(IN)+1) * NG_IN(IN)      \
               + Y * (NZ_IN(IN)+1) * NG_IN(IN)                  \
               + Z * NG_IN(IN)                                  \
               + G ]
#else
#define FLKZ_4D_SOL(IN, SOL, X, Y, Z, G)                        \
    SOL->flkz[ G   * (NZ_IN(IN)+1) * NY_IN(IN) * NX_IN(IN)      \
               + Z * NY_IN(IN) * NX_IN(IN)                      \
               + Y * NX_IN(IN)                                  \
               + X ]
#endif
#define FLKZ_SOL(SOL) SOL->flkz
// Assuming 'SOL=solvar_vars'
#define FLKZ_4D(X, Y, Z, G) FLKZ_4D_SOL(input_vars, solvar_vars, X, Y, Z, G)
#define FLKZ FLKZ_SOL(solvar_vars)

/* ptr_in(nang,nx,ny,nz,noct,ng) */
#ifdef ROWORDER
#define PTR_IN_6D_SOL(IN, SN, SOL, ANG, X, Y, Z, OCT, G)                \
    SOL->ptr_in[ ANG   * NX_IN(IN)   * NY_IN(IN)   * NZ_IN(IN)   * NOCT_SN(SN) * NG_IN(IN) \
                 + X   * NY_IN(IN)   * NZ_IN(IN)   * NOCT_SN(SN) * NG_IN(IN) \
                 + Y   * NZ_IN(IN)   * NOCT_SN(SN) * NG_IN(IN)          \
                 + Z   * NOCT_SN(SN) * NG_IN(IN)                        \
                 + OCT * NG_IN(IN)                                      \
                 + G ]
#else
#define PTR_IN_6D_SOL(IN, SN, SOL, ANG, X, Y, Z, OCT, G)                \
    SOL->ptr_in[ G     * NOCT_SN(SN) * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN) * NANG_IN(IN) \
                 + OCT * NZ_IN(IN)   * NY_IN(IN) * NX_IN(IN) * NANG_IN(IN) \
                 + Z   * NY_IN(IN)   * NX_IN(IN) * NANG_IN(IN)          \
                 + Y   * NX_IN(IN)   * NANG_IN(IN)                      \
                 + X   * NANG_IN(IN)                                    \
                 + ANG ]
#endif
#define PTR_IN_SOL(SOL) SOL->ptr_in
// Assuming 'SOL=solvar_vars' and 'IN=input_vars'
#define PTR_IN_6D(ANG, X, Y, Z, OCT, G)                                 \
    PTR_IN_6D_SOL(input_vars, sn_vars, solvar_vars, ANG, X, Y, Z, OCT, G)
#define PTR_IN PTR_IN_SOL(solvar_vars)

/* ptr_out(nang,nx,ny,nz,noct,ng) */
#ifdef ROWORDER
#define PTR_OUT_6D_SOL(IN, SN, SOL, ANG, X, Y, Z, OCT, G)               \
    SOL->ptr_out[ ANG   * NX_IN(IN)   * NY_IN(IN)   * NZ_IN(IN)   * NOCT_SN(SN) * NG_IN(IN) \
                  + X   * NY_IN(IN)   * NZ_IN(IN)   * NOCT_SN(SN) * NG_IN(IN) \
                  + Y   * NZ_IN(IN)   * NOCT_SN(SN) * NG_IN(IN)         \
                  + Z   * NOCT_SN(SN) * NG_IN(IN)                       \
                  + OCT * NG_IN(IN)                                     \
                  + G ]
#else
#define PTR_OUT_6D_SOL(IN, SN, SOL, ANG, X, Y, Z, OCT, G)               \
    SOL->ptr_out[ G     * NOCT_SN(SN) * NZ_IN(IN) * NY_IN(IN) * NX_IN(IN) * NANG_IN(IN) \
                  + OCT * NZ_IN(IN)   * NY_IN(IN) * NX_IN(IN) * NANG_IN(IN) \
                  + Z   * NY_IN(IN)   * NX_IN(IN) * NANG_IN(IN)         \
                  + Y   * NX_IN(IN)   * NANG_IN(IN)                     \
                  + X   * NANG_IN(IN)                                   \
                  + ANG ]
#endif
#define PTR_OUT_SOL(SOL) SOL->ptr_out
// Assuming 'SOL=solvar_vars' and 'IN=input_vars'
#define PTR_OUT_6D(ANG, X, Y, Z, OCT, G)                                \
    PTR_OUT_6D_SOL(input_vars, sn_vars, solvar_vars, ANG, X, Y, Z, OCT, G)
#define PTR_OUT PTR_OUT_SOL(solvar_vars)


// dim1_sweep.c and dim3_sweep.c
/* fmin */
#define FMIN_DIM(DIM) DIM->fmin
// Assuming 'DIM=dim_sweep_vars'
#define FMIN          FMIN_DIM(dim_sweep_vars)

/* fmax */
#define FMAX_DIM(DIM) DIM->fmax
// Assuming 'DIM=dim_sweep_vars'
#define FMAX          FMAX_DIM(dim_sweep_vars)


// sweep.c
/* mtag */
#define MTAG_SWP(SWP)      SWP->mtag
// Assuming 'SWP=sweep_vars'
#define MTAG               MTAG_SWP(sweep_vars)

/* yp_snd */
#define YP_SND_SWP(SWP)    SWP->yp_snd
// Assuming 'SWP=sweep_vars'
#define YP_SND             YP_SND_SWP(sweep_vars)

/* yp_rcv */
#define YP_RCV_SWP(SWP)    SWP->yp_rcv
// Assuming 'SWP=sweep_vars'
#define YP_RCV             YP_RCV_SWP(sweep_vars)

/* zp_snd */
#define ZP_SND_SWP(SWP)    SWP->zp_snd
// Assuming 'SWP=sweep_vars'
#define ZP_SND             ZP_SND_SWP(sweep_vars)

/* zp_rcv */
#define ZP_RCV_SWP(SWP)    SWP->zp_rcv
// Assuming 'SWP=sweep_vars'
#define ZP_RCV             ZP_RCV_SWP(sweep_vars)

/* incomingy */
#define INCOMINGY_SWP(SWP) SWP->incomingy
// Assuming 'SWP=sweep_vars'
#define INCOMINGY          INCOMINGY_SWP(sweep_vars)

/* incomingz */
#define INCOMINGZ_SWP(SWP) SWP->incomingz
// Assuming 'SWP=sweep_vars'
#define INCOMINGZ          INCOMINGZ_SWP(sweep_vars)

/* outgoingy */
#define OUTGOINGY_SWP(SWP) SWP->outgoingy
// Assuming 'SWP=sweep_vars'
#define OUTGOINGY          OUTGOINGY_SWP(sweep_vars)

/* outgoingz */
#define OUTGOINGZ_SWP(SWP) SWP->outgoingz
// Assuming 'SWP=sweep_vars'
#define OUTGOINGZ          OUTGOINGZ_SWP(sweep_vars)

#endif
