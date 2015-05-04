/***********************************************************************
 * Module: control.c
 * This module contains the variables that control SNAP's solver
 * routines. This includes the time-dependent variables.
 ***********************************************************************/
#include "snap.h"

void control_data_init ( control_data *control_vars )
{
    DT      = 0;
    TOLR    = 1.02E-12;
    DFMXI   = NULL;
    DFMXO   = -1;
    INRDONE = NULL;
    OTRDONE = false;
}

/***********************************************************************
 * Allocate control module variables.
 ***********************************************************************/
void control_alloc ( input_data *input_vars, control_data *control_vars, int *ierr )
{
    int i;

    ALLOC_1D(DFMXI,   NG, double, ierr);
    ALLOC_1D(INRDONE, NG, bool,   ierr);

    if ( *ierr != 0 ) return;

    for ( i = 0; i < NG; i++ )
    {
        INRDONE_1D(i) = false;
        DFMXI_1D(i) = -1;
    }

    DFMXO = -1;
}

/***********************************************************************
 * Deallocate control module variables.
 ***********************************************************************/
void control_dealloc ( control_data *control_vars )
{
    FREE(DFMXI);
    FREE(INRDONE);
}
