/***********************************************************************
 * Module: sweep.c
 *
 * This module contains all the subroutines related to the mesh sweep.
 ***********************************************************************/
#include "snap.h"

void sweep_data_init ( sweep_data *sweep_vars )
{
    MTAG   = 0;
    YP_SND = 0;
    YP_RCV = 0;
    ZP_SND = 0;
    ZP_RCV = 0;

    INCOMINGY = false;
    INCOMINGZ = false;
    OUTGOINGY = false;
    OUTGOINGZ = false;
}

/***********************************************************************
 * Driver for the mesh sweeps. Manages the loops over octant pairs.
 ***********************************************************************/
void sweep ( input_data *input_vars, para_data *para_vars, geom_data *geom_vars,
             sn_data *sn_vars, data_data *data_vars, control_data *control_vars,
             solvar_data *solvar_vars, sweep_data *sweep_vars,
             dim_sweep_data *dim_sweep_vars, int *ierr )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int jd, kd, jlo, jhi, jst, klo, khi, kst, iop;

    int g, k, j, i, g_indx;

    int *gnext;

    bool use_lock = false, cont_dogrp_loop = true;

    int grp_act[NG], dogrp[NUM_GRTH];

/***********************************************************************
 * Set up OpenMP lock if necessary.
 ***********************************************************************/
    use_lock = (NPROC>1)
        && (NTHREADS>1)
        && (THREAD_LEVEL!=THREAD_MULTIPLE);

    if ( use_lock )
    {
        plock_omp ( "init", &LOCK );
    }

/***********************************************************************
 * Start OpenMP parallel region for entire sweep subroutine. Then clear
 * leakage arrays with threaded do loop.
 ***********************************************************************/
//#pragma omp parallel default(shared)  {
#pragma omp for schedule(dynamic, 1) private(g, k, j, i)
        for ( g = 1; g <= NG; g++ )
        {
            if ( !INRDONE_1D(g-1) )
            {
                for ( k = 0; k < NZ; k++ )
                {
                    for ( j = 0; j < NY; j++ )
                    {
                        for ( i = 0; i < NX+1; i++ )
                        {
                            FLKX_4D(i,j,k,(g-1)) = 0;
                        }
                    }
                }
                for ( k = 0; k < NZ; k++ )
                {
                    for ( j = 0; j < NY+1; j++ )
                    {
                        for ( i = 0; i < NX; i++ )
                        {
                            FLKY_4D(i,j,k,(g-1)) = 0;
                        }
                    }
                }
                for ( k = 0; k < NZ+1; k++ )
                {
                    for ( j = 0; j < NY; j++ )
                    {
                        for ( i = 0; i < NX; i++ )
                        {
                            FLKZ_4D(i,j,k,(g-1)) = 0;
                        }
                    }
                }
            }
        }

/***********************************************************************
 * Loop over octant pairs, according to ndimen. Set up the sweep order.
 * Place a barrier at start of new octant to make sure all threads have
 * same kd and jd. Only one thread needs to set shared values of loop
 * bounds, strides, and communication parameters.
 ***********************************************************************/
        // kd_loop
        for ( kd = 1; kd <= MAX((NDIMEN-1),1); kd++ )
        {
            // jd_loop
            for ( jd = 1; jd <= MIN(NDIMEN, 2); jd++)
            {
//#pragma omp barrier

//#pragma omp single
                {
                    if ( jd == 1 )
                    {
                        jlo = NY;
                        jhi = 1;
                        jst = -1;
                        YP_SND = YLOP;
                        YP_RCV = YHIP;
                    }
                    else
                    {
                        jlo = 1;
                        jhi = NY;
                        jst = 1;
                        YP_SND = YHIP;
                        YP_RCV = YLOP;
                    }

                    if ( kd == 1 )
                    {
                        klo = NZ;
                        khi = 1;
                        kst = -1;
                        ZP_SND = ZLOP;
                        ZP_RCV = ZHIP;
                    }
                    else
                    {
                        klo = 1;
                        khi = NZ;
                        kst = 1;
                        ZP_SND = ZHIP;
                        ZP_RCV = ZLOP;
                    }

                    INCOMINGY = !( (jd==1 && LASTY) || (jd==2 && FIRSTY) );
                    INCOMINGZ = !( (kd==1 && LASTZ) || (kd==2 && FIRSTZ) );
                    OUTGOINGY = !( (jd==1 && FIRSTY) || (jd==2 && LASTY) );
                    OUTGOINGZ = !( (kd==1 && FIRSTZ) || (kd==2 && LASTZ) );

                    MTAG = 2*NC * (jd-1 + 2*(kd-1));

/***********************************************************************
 * Set up groups to be swept. Only num_grth can be swept at a time
 * due to communications.
 ***********************************************************************/
                    for (g_indx = 0; g_indx < NG; g_indx++ )
                    {
                        grp_act[g_indx] = 1;

                        if ( INRDONE_1D(g_indx) ) grp_act[g_indx] = 0;
                    }

                    for ( j = 1; j <= NUM_GRTH; j++ )
                    {
                        dogrp[j-1] = 0;
                    }

                    for ( i = 1; i <= NUM_GRTH; i++ )
                    {
                        g = 1;

                        gnext = grp_act;
                        *gnext = *grp_act;

                        for ( j = 1; j < NG; j++ )
                        {
                            if ( *(grp_act + j) > *gnext  )
                            {
                                gnext = (grp_act + j);
                                g = j + 1;
                            }
                        }

                        if ( grp_act[g-1] == 0 ) break;

                        grp_act[g-1] = 0;
                        dogrp[i-1] = g;

                    }
                }

/***********************************************************************
 * Loop as long as groups that are not converged still need to be
 * done. Set an operation as looping over a chunk of i-cells.
 ***********************************************************************/
                // dogrp_loop
                while (true)
                {
                    //iop_loop
                    for ( iop = 1; iop <= 2*NC; iop++ )
                    {

/***********************************************************************
 * If supported, thread_multiple allows all threads over groups
 * to handle independent MPI communications. Loop over groups.
 ***********************************************************************/
                        if ( THREAD_LEVEL == THREAD_MULTIPLE )
                        {
//#pragma omp for schedule(static, 1) private(i) nowait
                            for ( i = 1; i <= NUM_GRTH; i++ )
                            {
                                if ( dogrp[i-1] != 0 )
                                {
                                    sweep_recv_bdry (input_vars, para_vars, solvar_vars,
                                                     sweep_vars, dogrp[i-1], iop );

                                    octsweep ( input_vars, para_vars, geom_vars, sn_vars, data_vars,
                                               control_vars, solvar_vars, dim_sweep_vars,
                                               dogrp[i-1], iop, jd, kd, jlo, jhi, jst,
                                               klo, khi, kst, ierr );

                                    sweep_send_bdry (input_vars, para_vars, solvar_vars,
                                                     sweep_vars, dogrp[i-1], iop );
                                }
                            }
                        }

                        else
                        {

/***********************************************************************
 * Otherwise, thread_serialized makes sure only one thread calls
 * for MPI communications at a time. Use locks to make sure all
 * sends/receives are done before next receives/sends. Use
 * ordered loops to match threads' receives with sends.
 ***********************************************************************/
                            if ( INCOMINGY || INCOMINGZ )
                            {
//#pragma omp for schedule(static, 1) ordered private(i) nowait
                                for ( i = 1; i <= NUM_GRTH; i++ )
                                {
                                    if ( i == 1 && use_lock )
                                        plock_omp( "set", &LOCK );

//#pragma omp ordered
                                    {
                                        if ( dogrp[i-1] != 0 )
                                        {
                                            sweep_recv_bdry( input_vars, para_vars, solvar_vars,
                                                             sweep_vars, dogrp[i-1], iop );
                                        }
                                    }

                                    if ( i == NUM_GRTH && use_lock )
                                        plock_omp( "unset", &LOCK );
                                }
                            }

//#pragma omp for schedule(static, 1) private(i) nowait
                            for ( i = 1; i <= NUM_GRTH; i++ )
                            {
                                if ( dogrp[i-1] != 0 )
                                {
                                    octsweep ( input_vars, para_vars, geom_vars, sn_vars, data_vars,
                                               control_vars, solvar_vars, dim_sweep_vars,
                                               dogrp[i-1], iop, jd, kd, jlo, jhi, jst,
                                               klo, khi, kst, ierr );
                                }
                            }

                            if ( OUTGOINGY || OUTGOINGZ )
                            {
//#pragma omp for schedule(static, 1) ordered private(i) nowait
                                for ( i = 1; i <= NUM_GRTH; i++ )
                                {
                                    if ( i == 1 && use_lock )
                                        plock_omp( "set", &LOCK );

//#pragma omp ordered
                                    {
                                        if ( dogrp[i-1] != 0 )
                                        {
                                            sweep_send_bdry( input_vars, para_vars, solvar_vars,
                                                             sweep_vars, dogrp[i-1], iop );
                                        }
                                    }

                                    if ( i == NUM_GRTH && use_lock )
                                        plock_omp( "unset", &LOCK );
                                }
                            }

                        }
                    } // end iop_loop

/***********************************************************************
 * End sweep for octant pair. Use barrier statement to sync threads
 * and use single block to assign next set of groups, if any left
 ***********************************************************************/
//#pragma omp barrier

//#pragma omp single
                        // {
                    for ( j = 1; j <= NUM_GRTH; j++ )
                    {
                        dogrp[j-1] = 0;
                    }

                    for ( j = 1; j <= NUM_GRTH; j++ )
                    {
                        g = 1;

                        gnext = grp_act;
                        *gnext = *grp_act;

                        for ( i = 1; i < NG; i++ )
                        {
                            if ( *(grp_act + i) > *gnext  )
                            {
                                gnext = (grp_act + i);
                                g = i + 1;
                            }
                        }

                        if ( grp_act[g-1] == 0 ) break;

                        grp_act[g-1] = 0;
                        dogrp[j-1] = g;
                    }


                    for ( i = 1; i <= NUM_GRTH; i++ )
                    {
                        if ( dogrp[i-1] > 0 )
                        {
                            cont_dogrp_loop = true;
                            break;
                        }
                        else
                        {
                            cont_dogrp_loop = false;
                        }
                    }

                    if ( !cont_dogrp_loop )
                    {
                        break;
                    }
                        //}
                } // end dogrp_loop
            } // end jd_loop
        } // end kd_loop
        //} // end parallel

/***********************************************************************
 * Destroy the lock
 ***********************************************************************/
    if (use_lock)
    {
        plock_omp ( "destroy", &LOCK );
    }
}

/***********************************************************************
 * Receive flux from upstream boundaries
 ***********************************************************************/
void sweep_recv_bdry ( input_data *input_vars, para_data *para_vars,
                       solvar_data *solvar_vars, sweep_data *sweep_vars,
                       int g, int iop )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int i;
    int ierr = 0;

/***********************************************************************/

    i = iop + g*G_OFF + MTAG;

    precv_d_3d( &JB_IN_4D(0,0,0,(g-1)), NANG, ICHUNK, NZ, YCOMM, YP_RCV, YPROC, i );
    precv_d_3d( &KB_IN_4D(0,0,0,(g-1)), NANG, ICHUNK, NY, ZCOMM, ZP_RCV, ZPROC, i );
}

/***********************************************************************
 * Send flux from upstream boundaries
 ***********************************************************************/
void sweep_send_bdry ( input_data *input_vars, para_data *para_vars,
                       solvar_data *solvar_vars, sweep_data *sweep_vars,
                       int g, int iop )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int i;
    int ierr = 0;

/***********************************************************************/

    i = iop + g*G_OFF + MTAG;

    psend_d_3d( &JB_OUT_4D(0,0,0,(g-1)), NANG, ICHUNK, NZ, YCOMM, YP_SND, YPROC, i );
    psend_d_3d( &KB_OUT_4D(0,0,0,(g-1)), NANG, ICHUNK, NY, ZCOMM, ZP_SND, ZPROC, i );
}
