/***********************************************************************
 * Module: octsweep_sweep.c
 *
 * This module controls the setup and calls for sweeping a single octant
 * pair. It calls for the actual sweep logic depending on the spatial
 * dimensionality of the problem.
 ***********************************************************************/
#include "snap.h"

/***********************************************************************
 * Call for the appropriate sweeper depending on the dimensionality. Let
 * the actual sweep routine know the octant info to get the order right.
 ***********************************************************************/
void octsweep ( input_data *input_vars, para_data *para_vars, geom_data *geom_vars,
                sn_data *sn_vars, data_data *data_vars, control_data *control_vars,
                solvar_data *solvar_vars, dim_sweep_data *dim_sweep_vars,
                int g, int iop, int jd, int kd, int jlo, int jhi, int jst,
                int klo, int khi, int kst, int *ierr )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int id, oct, ich, d1, d2, d3, d4, i1, i2;

/***********************************************************************
 * Determine octant and chunk index.
 ***********************************************************************/
    id = 1 + (iop-1)/NC;
    oct = id + 2*(jd-1) + 4*(kd-1);

    if ( id == 1 )
    {
        ich = NC - iop + 1;
    }
    else
    {
        ich = iop - NC;
    }

/***********************************************************************
 * Send ptr_in and ptr_out dimensions dependent on timedep because they
 * are not allocated if static problem
 ***********************************************************************/
    d1 = 0;
    d2 = 0;
    d3 = 0;
    d4 = 0;
    i1 = 1;
    i2 = 1;
    if ( TIMEDEP == 1 )
    {
      d1 = NANG;
      d2 = NX;
      d3 = NY;
      d4 = NZ;
      i1 = oct;
      i2 = g;
    }

/***********************************************************************
 * Call for the actual sweeper. Ensure proper size/bounds of time-dep
 * arrays is given to avoid errors.
 ***********************************************************************/
    if ( NDIMEN == 1 )
    {
        dim1_sweep ( input_vars, geom_vars, sn_vars, data_vars,
                     control_vars, solvar_vars, dim_sweep_vars,
                     id, oct, d1, d2, d3, d4, i1, i2, g, ierr );
    }
    else
    {
        dim3_sweep ( input_vars, para_vars, geom_vars, sn_vars, data_vars,
                     control_vars, solvar_vars, dim_sweep_vars, ich, id,
                     d1, d2, d3, d4, jd, kd, jlo, klo, jhi, khi, jst, kst,
                     i1, i2, oct, g, ierr );
    }
}
