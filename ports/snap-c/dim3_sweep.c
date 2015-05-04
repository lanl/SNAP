/***********************************************************************
 * Module: dim3_sweep.c
 *
 * This module contains the 2D and 3D mesh sweep logic.
 ***********************************************************************/
#include "snap.h"

// Local variable array macro
#define PSI_1D(ANG)   psi[ANG]
#define PC_1D(ANG)    pc[ANG]
#define DEN_1D(ANG)   den[ANG]

#ifdef ROWORDER
#define HV_2D(ANG, X) hv[ ANG*4                 \
                          + X ]
#else
#define HV_2D(ANG, X) hv[ X*NANG                \
                          + ANG ]
#endif

#ifdef ROWORDER
#define FXHV_2D(ANG, X) fxhv[ ANG*4             \
                              + X ]
#else
#define FXHV_2D(ANG, X) fxhv[ X*NANG            \
                              + ANG ]
#endif

// Simplify array indexing when certain values constant throughout module
#define PSII_3D(ANG, Y, Z)       PSII_4D(ANG, Y, Z, (g-1))
#define PSIJ_3D(ANG, CHUNK, Z)   PSIJ_4D(ANG, CHUNK, Z, (g-1))
#define PSIK_3D(ANG, CHUNK, Y)   PSIK_4D(ANG, CHUNK, Y, (g-1))
#define QTOT_4D(MOM1, X, Y, Z)   QTOT_5D(MOM1, X, Y, Z, (g-1))
#define EC_2D(ANG, MOM1)         EC_3D(ANG, MOM1, (oct-1))
#define VDELT_CONST              VDELT_1D(g-1)
#define PTR_IN_4D(ANG, X, Y, Z)  PTR_IN_6D(ANG, X, Y, Z, (i1-1), (i2-1))
#define PTR_OUT_4D(ANG, X, Y, Z) PTR_OUT_6D(ANG, X, Y, Z, (i1-1), (i2-1))
#define DINV_4D(ANG, X, Y, Z)    DINV_5D(ANG, X, Y, Z, (g-1))
#define FLUX_3D(X, Y, Z)         FLUX_4D(X, Y, Z, (g-1))
#define FLUXM_4D(MOM1, X, Y, Z)  FLUXM_5D(MOM1, X, Y, Z, (g-1))
#define JB_IN_3D(ANG, CHUNK, Z)  JB_IN_4D(ANG, CHUNK, Z, (g-1))
#define JB_OUT_3D(ANG, CHUNK, Z) JB_OUT_4D(ANG, CHUNK, Z, (g-1))
#define KB_IN_3D(ANG, CHUNK, Y)  KB_IN_4D(ANG, CHUNK, Y, (g-1))
#define KB_OUT_3D(ANG, CHUNK, Y) KB_OUT_4D(ANG, CHUNK, Y, (g-1))
#define FLKX_3D(X, Y, Z)         FLKX_4D(X, Y, Z, (g-1))
#define FLKY_3D(X, Y, Z)         FLKY_4D(X, Y, Z, (g-1))
#define FLKZ_3D(X, Y, Z)         FLKZ_4D(X, Y, Z, (g-1))
#define T_XS_3D(X, Y, Z)         T_XS_4D(X, Y, Z, (g-1))

void dim3_sweep_data_init ( dim_sweep_data *dim_sweep_vars )
{
    FMIN = 0;
    FMAX = 0;
}

/***********************************************************************
 *  3-D slab mesh sweeper.
 ***********************************************************************/
void dim3_sweep ( input_data *input_vars, para_data *para_vars,
                  geom_data *geom_vars, sn_data *sn_vars,
                  data_data *data_vars, control_data *control_vars,
                  solvar_data *solvar_vars, dim_sweep_data *dim_sweep_vars,
                  int ich, int id, int d1, int d2, int d3, int d4, int jd,
                  int kd, int jlo, int klo, int jhi, int khi, int jst, int kst,
                  int i1, int i2, int oct, int g, int *ierr )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int ist, d, n, ic, i, j, k, l, ibl, ibr, ibb, ibt, ibf, ibk;

    int z_ind, y_ind, ic_ind, ang, indx1 = 4;

    double sum_hv = 0, sum_hv_tmp = 0, sum_wpsi = 0, sum_ecwpsi = 0,
        sum_wmupsii = 0, sum_wetapsij = 0, sum_wxipsik = 0;

    double psi[NANG], pc[NANG], den[NANG];
    double hv[NANG*4], fxhv[NANG*4];

    double vec1_vec2_tmp[NANG], PSI_2X[NANG], hv_p1[NANG*4],
        mu_hv[NANG], hj_hv[NANG], hk_hv[NANG], w_psi[NANG];

    double unit_vec[indx1];
    for ( i = 0; i < indx1; i++ )
    {
        unit_vec[i] = 1;
    }


#ifdef USEVML
    double PC_2X[NANG];

    double PSII_HI_tmp[NANG], PSII_MU_HI_tmp[NANG],
        PSIJ_HJ_tmp[NANG], PSIK_HK_tmp[NANG],
        PSII_PSIJ_tmp[NANG], PSII_PSIJ_PSIK_tmp[NANG];
#endif

/***********************************************************************
 * Set up the sweep order in the i-direction.
 ***********************************************************************/
    ist = -1;
    if ( id == 2 ) ist = 1;

/***********************************************************************
 * Zero out the outgoing boundary arrays and fixup array
 ***********************************************************************/
    for ( z_ind = 0; z_ind < NZ; z_ind++ )
    {
        for ( ic_ind = 0; ic_ind < ICHUNK; ic_ind++ )
        {
            for ( ang = 0; ang < NANG; ang++ )
            {
                JB_OUT_3D(ang,ic_ind,z_ind) = 0;
            }
        }
    }

    for ( y_ind = 0; y_ind < NY; y_ind++ )
    {
        for ( ic_ind = 0; ic_ind < ICHUNK; ic_ind++ )
        {
            for ( ang = 0; ang < NANG; ang++ )
            {
                KB_OUT_3D(ang,ic_ind,y_ind) = 0;
            }
        }
    }

    for ( i = 0; i < 4; i++)
    {
        for ( ang = 0; ang < NANG; ang++ )
        {
            FXHV_2D(ang, i) = 0;
        }
    }

/***********************************************************************
 * Loop over cells along the diagonals. When only 1 diagonal, it's
 * normal sweep order. Otherwise, nested threading performs mini-KBA.
 ***********************************************************************/
/***********************************************************************
 * Commented out all nested OMP statements because not all compilers support
 * these put them back in if you want.
 ***********************************************************************/
// #pragma omp parallel num_threads(NNESTED) default(shared) firstprivate(fxhv)
//    {
//    #endif
    // diagonal loop
    for ( d = 1; d <= NDIAG; d++ )
    {

         #pragma omp for schedule(static, 1) private(n,ic,i,j,k,l,psi,pc,sum_hv,hv,den)
        //  line_loop
        for ( n = 1; n <= (DIAG_1D(d-1).lenc); n++ )
        {
            ic = DIAG_1D(d-1).cell_id_vars[n-1].ic;

            if ( ist < 0 )
            {
                i = ich*ICHUNK - ic + 1;
            }
            else
            {
                i = (ich-1)*ICHUNK + ic;
            }

            if ( i <= NX )
            {
                j = DIAG_1D(d-1).cell_id_vars[n-1].jc;

                if ( jst < 0 )
                {
                    j = NY - j + 1;
                }

                k = DIAG_1D(d-1).cell_id_vars[n-1].kc;

                if ( kst < 0 )
                {
                    k = NZ - k + 1;
                }

/***********************************************************************
 * Left/right boundary conditions, always vacuum.
 ***********************************************************************/
                ibl = 0;
                ibr = 0;

                if ( (i == NX) && (ist == -1) )
                {
                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        PSII_3D(ang,(j-1),(k-1)) = 0;
                    }
                }
                else if ( i == 1 && ist == 1 )
                {
                    switch ( ibl )
                    {
                    case 0:
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSII_3D(ang,(j-1),(k-1)) = 0;
                        }
                    case 1:
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSII_3D(ang,(j-1),(k-1)) = 0;
                        }
                    }
                }

/***********************************************************************
 * Top/bottom boundary condtions. Vacuum at global boundaries, but
 * set to some incoming flux from neighboring proc.
 ***********************************************************************/
                ibb = 0;
                ibt = 0;
                if ( j == jlo )
                {
                    if ( jd == 1 && LASTY )
                    {
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSIJ_3D(ang,(ic-1),(k-1)) = 0;
                        }
                    }
                    else if ( jd == 2 && FIRSTY )
                    {
                        switch ( ibb )
                        {
                        case 0:
                            for ( ang = 0; ang < NANG; ang++ )
                            {
                                PSIJ_3D(ang,(ic-1),(k-1)) = 0;
                            }
                        case 1:
                            for ( ang = 0; ang < NANG; ang++ )
                            {
                                PSIJ_3D(ang,(ic-1),(k-1)) = 0;
                            }
                        }


                    }

                    else
                    {
#ifdef USEMKL
                        cblas_dcopy(NANG, &JB_IN_3D(0,(ic-1),(k-1)), 1,
                                    &PSIJ_3D(0,(ic-1),(k-1)), 1);

#elif defined USEBLAS
                        dcopy(NANG, &JB_IN_3D(0,(ic-1),(k-1)), 1,
                                    &PSIJ_3D(0,(ic-1),(k-1)), 1);

#else
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSIJ_3D(ang,(ic-1),(k-1))
                                = JB_IN_3D(ang,(ic-1),(k-1));
                        }
#endif
                    }
                }

/***********************************************************************
 * Front/back boundary condtions. Vacuum at global boundaries, but
 * set to some incoming flux from neighboring proc.
 ***********************************************************************/
                ibf = 0;
                ibk = 0;
                if ( k == klo )
                {
                    if ( (kd == 1 && LASTZ) || NDIMEN < 3 )
                    {
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSIK_3D(ang,(ic-1),(j-1)) = 0;
                        }
                    }
                    else if ( kd == 2 && FIRSTZ )
                    {
                        switch ( ibf )
                        {
                        case 0:
                            for ( ang = 0; ang < NANG; ang++ )
                            {
                                PSIK_3D(ang,(ic-1),(j-1)) = 0;
                            }
                        case 1:
                            for ( ang = 0; ang < NANG; ang++ )
                            {
                                PSIK_3D(ang,(ic-1),(j-1)) = 0;
                            }
                        }
                    }

                    else
                    {
#ifdef USEMKL
                        cblas_dcopy(NANG, &KB_IN_3D(0,(ic-1),(j-1)), 1,
                                    &PSIK_3D(0,(ic-1),(j-1)), 1);

#elif defined USEBLAS
                        dcopy(NANG, &KB_IN_3D(0,(ic-1),(j-1)), 1,
                              &PSIK_3D(0,(ic-1),(j-1)), 1);

#else
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSIK_3D(ang,(ic-1),(j-1))
                                = KB_IN_3D(ang,(ic-1),(j-1));
                        }
#endif
                    }
                }

/***********************************************************************
 * Compute the angular source
 ***********************************************************************/
#ifdef USEMKL
                for ( ang = 0; ang < NANG; ang++ )
                {
                    PSI_1D(ang) = QTOT_4D(0,(i-1),(j-1),(k-1));
                }

                if ( SRC_OPT == 3 )
                {
                    cblas_daxpy(NANG, 1,
                                &QIM_6D(0,(i-1),(j-1),(k-1),(oct-1),(g-1)),
                                1, psi, 1);
                }

                cblas_dgemv(CblasColMajor, CblasNoTrans, NANG, (CMOM-1), 1,
                            &EC_2D(0,1), NANG, &QTOT_4D(1,(i-1),(j-1),(k-1)),
                            1, 1, psi, 1);

#else
                for ( ang = 0; ang < NANG; ang++ )
                {
                    PSI_1D(ang) = QTOT_4D(0,(i-1),(j-1),(k-1));

                    if ( SRC_OPT == 3 )
                    {
                        PSI_1D(ang) +=
                            QIM_6D(ang,(i-1),(j-1),(k-1),(oct-1),(g-1));
                    }
                }

                for ( l = 2; l <=CMOM; l++ )
                {
                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        PSI_1D(ang) +=
                            EC_2D(ang,(l-1))
                            *QTOT_4D((l-1),(i-1),(j-1),(k-1));
                    }
                }
#endif

/***********************************************************************
 * Compute the numerator for the update formula
 ***********************************************************************/
#ifdef USEMKL
            #ifdef USEVML_unchecked
                // Use VML for vector lengths exceeding certain length
                if ( NANG > VECLEN_MIN )
                {
                #ifdef MKLUPDATE
                    cblas_dcopy(NANG, psi, 1, pc, 1);

                    vmdMul(NANG, &PSII_3D(0,(j-1),(k-1)), MU, vec1_vec2_tmp,
                           VML_ACCURACY | VML_HANDLING | VML_ERROR );

                    cblas_daxpy(NANG, HI, vec1_vec2_tmp, 1, pc, 1);

                    vmdMul(NANG, &PSIJ_3D(0,(ic-1),(k-1)), HJ, vec1_vec2_tmp,
                           VML_ACCURACY | VML_HANDLING | VML_ERROR );

                    cblas_daxpy(NANG, 1, vec1_vec2_tmp, 1, pc, 1);

                    vmdMul(NANG, &PSIK_3D(0,(ic-1),(j-1)), HK, vec1_vec2_tmp,
                           VML_ACCURACY | VML_HANDLING | VML_ERROR );

                    cblas_daxpy(NANG, 1, vec1_vec2_tmp, 1, pc, 1);

                #else
                    vmdMul(NANG, &PSII_3D(0,(j-1),(k-1)), MU, PSII_MU_HI_tmp,
                           VML_ACCURACY | VML_HANDLING | VML_ERROR );

                    cblas_dscal(NANG, HI, PSII_MU_HI_tmp, 1);

                    vmdMul(NANG, &PSIJ_3D(0,(ic-1),(k-1)), HJ, PSIJ_HJ_tmp,
                           VML_ACCURACY | VML_HANDLING | VML_ERROR );

                    vmdMul(NANG, &PSIK_3D(0,(ic-1),(k-1)), HK, PSIK_HK_tmp,
                           VML_ACCURACY | VML_HANDLING | VML_ERROR );

                    vmdAdd(NANG, PSIK_HK_tmp, PSIJ_HJ_tmp, PSIJ_HJ_tmp,
                        VML_ACCURACY | VML_HANDLING | VML_ERROR );

                    vmdAdd(NANG, PSII_MU_HI_tmp, PSIJ_HJ_tmp, PSIJ_HJ_tmp,
                        VML_ACCURACY | VML_HANDLING | VML_ERROR );

                    vmdAdd(NANG, psi, PSIJ_HJ_tmp, pc,
                        VML_ACCURACY | VML_HANDLING | VML_ERROR );

                #endif
                }

                else
                {
                #ifdef MKLUPDATE
                    cblas_dcopy(NANG, psi, 1, pc, 1);

                    cblas_dsbmv(CblasColMajor, CblasLower, NANG, 0, HI,
                                &PSII_3D(0,(j-1),(k-1)), 1, MU, 1, 1, pc, 1);

                    cblas_dsbmv(CblasColMajor, CblasLower, NANG, 0, 1,
                                &PSIJ_3D(0,(ic-1),(k-1)), 1, HJ, 1, 1, pc, 1);

                    cblas_dsbmv(CblasColMajor, CblasLower, NANG, 0, 1,
                                &PSIK_3D(0,(ic-1),(j-1)), 1, HK, 1, 1, pc, 1);

                #else
                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        PC_1D(ang) = PSI_1D(ang)
                            + PSII_3D(ang,(j-1),(k-1)) *MU_1D(ang)*HI
                            + PSIJ_3D(ang,(ic-1),(k-1))*HJ_1D(ang)
                            + PSIK_3D(ang,(ic-1),(j-1))*HK_1D(ang);
                    }
                #endif
                }

                if ( VDELT_CONST != 0 )
                {
                    cblas_daxpy(NANG, VDELT_CONST,
                                &PTR_IN_6D(0,(i-1),(j-1),(k-1),(i1-1),(i2-1)), 1, pc, 1);
                }

            #else
                for ( ang = 0; ang < NANG; ang++ )
                {
                    PC_1D(ang) = PSI_1D(ang)
                        + PSII_3D(ang,(j-1),(k-1)) *MU_1D(ang)*HI
                        + PSIJ_3D(ang,(ic-1),(k-1))*HJ_1D(ang)
                        + PSIK_3D(ang,(ic-1),(j-1))*HK_1D(ang);
                }

                if ( VDELT_CONST != 0 )
                {
                    cblas_daxpy(NANG, VDELT_CONST,
                                &PTR_IN_6D(0,(i-1),(j-1),(k-1),(i1-1),(i2-1)), 1, pc, 1);
                }
            #endif

#else
                for ( ang = 0; ang < NANG; ang++ )
                {
                    PC_1D(ang) = PSI_1D(ang)
                        + PSII_3D(ang,(j-1),(k-1)) *MU_1D(ang)*HI
                        + PSIJ_3D(ang,(ic-1),(k-1))*HJ_1D(ang)
                        + PSIK_3D(ang,(ic-1),(j-1))*HK_1D(ang);

                    if ( VDELT_CONST != 0 )
                    {
                        PC_1D(ang) += VDELT_CONST
                            *PTR_IN_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1));
                    }
                }
#endif

/***********************************************************************
 * Compute the solution of the center. Use DD for edges. Use fixup
 * if requested.
 ***********************************************************************/
                if ( FIXUP == 0 )
                {
//#ifdef USEMKL
                #ifdef USEVML
                    if ( NANG > VECLEN_MIN )
                    {
                        vmdMul(NANG, pc, &DINV_4D(0,(i-1),(j-1),(k-1)), psi,
                               VML_ACCURACY | VML_HANDLING | VML_ERROR );

                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSI_2X[ang] = 2*PSI_1D(ang);
                        }

                        vmdSub(NANG, PSI_2X, &PSII_3D(0, (j-1), (k-1)),
                               &PSII_3D(0, (j-1), (k-1)),
                               VML_ACCURACY | VML_HANDLING | VML_ERROR );

                        vmdSub(NANG, PSI_2X, &PSIJ_3D(0, (ic-1), (k-1)),
                               &PSIJ_3D(0, (ic-1), (k-1)),
                               VML_ACCURACY | VML_HANDLING | VML_ERROR );

                        if ( NDIMEN == 3)
                        {
                            vmdSub(NANG, PSI_2X, &PSIK_3D(0, (ic-1), (j-1)),
                                   &PSIK_3D(0, (ic-1), (j-1)),
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );
                        }

                        if ( VDELT_CONST != 0 )
                        {
                            vmdSub(NANG, PSI_2X, &PTR_IN_6D(0,(i-1),(j-1),(k-1),(i1-1),(i2-1)),
                                   &PTR_OUT_6D(0,(i-1),(j-1),(k-1),(i1-1),(i2-1)),
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );
                        }
                    }

                    else
                    {
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSI_1D(ang)
                                = PC_1D(ang)*DINV_4D(ang,(i-1),(j-1),(k-1));

                            PSII_3D(ang,(j-1),(k-1))
                                = 2*PSI_1D(ang) - PSII_3D(ang,(j-1),(k-1));

                            PSIJ_3D(ang,(ic-1),(k-1))
                                = 2*PSI_1D(ang) - PSIJ_3D(ang,(ic-1),(k-1));

                            if ( NDIMEN == 3 )
                            {
                                PSIK_3D(ang,(ic-1),(j-1))
                                    = 2*PSI_1D(ang) - PSIK_3D(ang,(ic-1),(j-1));
                            }

                            if ( VDELT_CONST != 0 )
                            {
                                PTR_OUT_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1))
                                    = 2*PSI_1D(ang)
                                    - PTR_IN_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1));
                            }
                        }
                    }

/*                #else
                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        PSI_1D(ang)
                            = PC_1D(ang)*DINV_4D(ang,(i-1),(j-1),(k-1));
                    }

                    cblas_dscal(NANG, -1, &PSII_3D(0, (j-1), (k-1)), 1);

                    cblas_daxpy(NANG, 2, psi, 1, &PSII_3D(0, (j-1), (k-1)), 1);

                    cblas_dscal(NANG, -1, &PSIJ_3D(0, (ic-1), (k-1)), 1);

                    cblas_daxpy(NANG, 2, psi, 1, &PSIJ_3D(0, (ic-1), (k-1)), 1);

                    if ( NDIMEN == 3 )
                    {
                        cblas_dscal(NANG, -1, &PSIK_3D(0, (ic-1), (j-1)), 1);

                        cblas_daxpy(NANG, 2, psi, 1, &PSIK_3D(0, (ic-1), (j-1)), 1);
                    }

                    if ( VDELT_CONST != 0 )
                    {
                         cblas_dscal(NANG, -1,
                                     &PTR_IN_6D(0,(i-1),(j-1),(k-1),(i1-1),(i2-1)), 1);

                         cblas_daxpy(NANG, 2, psi, 1,
                                     &PTR_IN_6D(0,(i-1),(j-1),(k-1),(i1-1),(i2-1)), 1);
                     }
                 #endif
*/
#elif defined MKLUPDATE
                     //cblas_dsbmv(CblasColMajor, CblasLower, NANG, 0, 1,
                     //       &DINV_4D(0,(i-1),(j-1),(k-1)), 1, pc, 1, 0, psi, 1);

                     vmdMul(NANG, pc, &DINV_4D(0,(i-1),(j-1),(k-1)), psi,
                            VML_ACCURACY | VML_HANDLING | VML_ERROR);

                     cblas_dscal(NANG, -1, &PSII_3D(0, (j-1), (k-1)), 1);

                     cblas_daxpy(NANG, 2, psi, 1, &PSII_3D(0, (j-1), (k-1)), 1);

                     cblas_dscal(NANG, -1, &PSIJ_3D(0, (ic-1), (k-1)), 1);

                     cblas_daxpy(NANG, 2, psi, 1, &PSIJ_3D(0, (ic-1), (k-1)), 1);

                     if ( NDIMEN == 3 )
                     {
                         cblas_dscal(NANG, -1, &PSIK_3D(0, (ic-1), (j-1)), 1);

                         cblas_daxpy(NANG, 2, psi, 1, &PSIK_3D(0, (ic-1), (j-1)), 1);
                     }

                     if ( VDELT_CONST != 0 )
                     {
                         cblas_dscal(NANG, -1,
                                     &PTR_IN_6D(0,(i-1),(j-1),(k-1),(i1-1),(i2-1)), 1);

                         cblas_daxpy(NANG, 2, psi, 1,
                                     &PTR_IN_6D(0,(i-1),(j-1),(k-1),(i1-1),(i2-1)), 1);
                     }

#else
                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        PSI_1D(ang)
                            = PC_1D(ang)*DINV_4D(ang,(i-1),(j-1),(k-1));

                        PSII_3D(ang,(j-1),(k-1))
                            = 2*PSI_1D(ang) - PSII_3D(ang,(j-1),(k-1));

                        PSIJ_3D(ang,(ic-1),(k-1))
                            = 2*PSI_1D(ang) - PSIJ_3D(ang,(ic-1),(k-1));

                        if ( NDIMEN == 3 )
                        {
                            PSIK_3D(ang,(ic-1),(j-1))
                                = 2*PSI_1D(ang) - PSIK_3D(ang,(ic-1),(j-1));
                        }

                        if ( VDELT_CONST != 0 )
                        {
                            PTR_OUT_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1))
                                = 2*PSI_1D(ang)
                                - PTR_IN_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1));
                        }
                    }
#endif
                }

                else
                {

/***********************************************************************
 * Multi-pass set to zero + rebalance fixup. Determine angles
 * that will need fixup first.
 ***********************************************************************/
                    sum_hv = 0;
#ifdef USEMKL

                #ifdef USEVML
                    for ( indx1 = 0; indx1 < 4; indx1++ )
                    {
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            HV_2D(ang, indx1) = 1;
                        }

                    }
                    sum_hv = cblas_dasum(NANG*4, hv, 1);

                    vmdMul( NANG, pc, &DINV_4D(0,(i-1),(j-1),(k-1)), pc,
                            VML_ACCURACY | VML_HANDLING | VML_ERROR );

                #else
                    for ( indx1 = 0; indx1 < 4; indx1++ )
                    {
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            HV_2D(ang, indx1) = 1;
                        }
                    }

                    sum_hv = cblas_dasum(NANG*4, hv, 1);

                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        PC_1D(ang) = PC_1D(ang)
                            * DINV_4D(ang,(i-1),(j-1),(k-1));
                    }
                 #endif

#else
                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        for ( indx1 = 0; indx1 < 4; indx1++ )
                        {
                            HV_2D(ang, indx1) = 1;
                            sum_hv += HV_2D(ang,indx1);
                        }

                        PC_1D(ang) = PC_1D(ang)
                            * DINV_4D(ang,(i-1),(j-1),(k-1));
                    }
#endif

                    // fixup_loop
                    while (true)
                    {
                        sum_hv_tmp = 0;

#ifdef USEMKL
                    #ifdef USEVML
                        if (NANG > VECLEN_MIN)
                        {
                            for (ang = 0; ang < NANG; ang++ )
                            {
                                PC_2X[ang] = 2*PC_1D(ang);
                            }

                            vmdSub(NANG, PC_2X, &PSII_3D(0,(j-1),(k-1)),
                                   &FXHV_2D(0,0),
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );

                            vmdSub(NANG, PC_2X, &PSIJ_3D(0,(ic-1),(k-1)),
                                   &FXHV_2D(0,1),
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );

                            if ( NDIMEN == 3 )
                            {
                                vmdSub(NANG, PC_2X, &PSIK_3D(0,(ic-1),(j-1)),
                                       &FXHV_2D(0,2),
                                       VML_ACCURACY | VML_HANDLING | VML_ERROR );
                            }

                            if ( VDELT_CONST != 0 )
                            {
                                vmdSub(NANG, PC_2X,
                                       &PTR_IN_6D(0,(i-1),(j-1),(k-1),(i1-1),(i2-1)),
                                       &FXHV_2D(0,3),
                                       VML_ACCURACY | VML_HANDLING | VML_ERROR );
                            }

                            for ( indx1 = 0; indx1 < 4; indx1++ )
                            {
                                for (ang = 0; ang < NANG; ang++ )
                                {
                                    if ( FXHV_2D(ang,indx1) < 0 )
                                    {
                                        HV_2D(ang,indx1) = 0;
                                    }
                                }
                            }

                            sum_hv_tmp = cblas_dasum(NANG*4, hv, 1);
                        }

                        else
                        {
                            for ( ang = 0; ang < NANG; ang++ )
                            {
                                FXHV_2D(ang,0) =  2*PC_1D(ang)
                                    - PSII_3D(ang,(j-1),(k-1));

                                FXHV_2D(ang,1) =  2*PC_1D(ang)
                                    - PSIJ_3D(ang,(ic-1),(k-1));

                                if ( NDIMEN == 3 )
                                {
                                    FXHV_2D(ang,2) = 2*PC_1D(ang)
                                        - PSIK_3D(ang,(ic-1),(j-1));
                                }

                                if ( VDELT_CONST != 0 )
                                {
                                    FXHV_2D(ang,3) = 2*PC_1D(ang)
                                        - PTR_IN_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1));
                                }

                                for ( indx1 = 0; indx1 < 4; indx1++ )
                                {
                                    if ( FXHV_2D(ang,indx1) < 0 )
                                    {
                                        HV_2D(ang,indx1) = 0;
                                    }
                                    sum_hv_tmp += HV_2D(ang,indx1);
                                }
                            }


                        }

                   #else
                        unit_vec[0] = 1;
                        unit_vec[1] = 1;
                        unit_vec[2] = 0;
                        unit_vec[3] = 0;

                        cblas_dcopy(NANG, &PSII_3D(0,(j-1),(k-1)),
                                    1, &FXHV_2D(0,0), 1);
                        cblas_dcopy(NANG, &PSIJ_3D(0,(ic-1),(k-1)),
                                    1, &FXHV_2D(0,1), 1);

                        cblas_dscal(NANG*2, -1, &FXHV_2D(0,0), 1);

                        if ( NDIMEN == 3 )
                        {
                            cblas_dcopy(NANG, &PSIK_3D(0,(ic-1),(j-1)),
                                        1, &FXHV_2D(0,2), 1);
                            cblas_dscal(NANG, -1, &FXHV_2D(0,2), 1);
                            unit_vec[2] = 1;
                        }

                        if ( VDELT_CONST != 0 )
                        {
                            cblas_dcopy(NANG, &PTR_IN_6D(0,(i-1),(j-1),(k-1),(i1-1),(i2-1)),
                                        1, &FXHV_2D(0,3), 1);
                            cblas_dscal(NANG, -1, &FXHV_2D(0,3), 1);
                            unit_vec[3] = 1;
                        }

                        cblas_dger(CblasColMajor, NANG, 4, 2, pc, 1, unit_vec, 1, fxhv, NANG);

                        for ( indx1 = 0; indx1 < 4; indx1++ )
                        {
                            for ( ang = 0; ang < NANG; ang++ )
                            {
                                if ( FXHV_2D(ang,indx1) < 0 )
                                {
                                    HV_2D(ang,indx1) = 0;
                                }
                            }
                        }

                        sum_hv_tmp = cblas_dasum(NANG*4, hv, 1);
                    #endif

#else
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            FXHV_2D(ang,0) =  2*PC_1D(ang)
                                - PSII_3D(ang,(j-1),(k-1));

                            FXHV_2D(ang,1) =  2*PC_1D(ang)
                                - PSIJ_3D(ang,(ic-1),(k-1));

                            if ( NDIMEN == 3 )
                            {
                                FXHV_2D(ang,2) = 2*PC_1D(ang)
                                    - PSIK_3D(ang,(ic-1),(j-1));
                            }

                            if ( VDELT_CONST != 0 )
                            {
                                FXHV_2D(ang,3) = 2*PC_1D(ang)
                                    - PTR_IN_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1));
                            }

                            for ( indx1 = 0; indx1 < 4; indx1++ )
                            {
                                if ( FXHV_2D(ang,indx1) < 0 )
                                {
                                    HV_2D(ang,indx1) = 0;
                                }
                                sum_hv_tmp += HV_2D(ang,indx1);
                            }
                        }
#endif

/***********************************************************************
 * Exit loop when all angles are fixed up
 ***********************************************************************/
                        if ( sum_hv == sum_hv_tmp ) break;

                        sum_hv = sum_hv_tmp;

/***********************************************************************
 * Recompute balance equation numerator and denominator and get
 * new cell average flux
 ***********************************************************************/
#ifdef USEVML
                        if (NANG > VECLEN_MIN)
                        {
                            cblas_dcopy(NANG*4, hv, 1, hv_p1, 1);
                            for ( ang = 0; ang < NANG*4; ang++ )
                            {
                                hv_p1[ang] += 1;
                            }

                            vmdMul(NANG, MU, &hv_p1[NANG*0], &hv_p1[NANG*0],
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );

                            vmdMul(NANG, &PSII_3D(0,(j-1),(k-1)),
                                   &hv_p1[NANG*0], &hv_p1[NANG*0],
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );

                            cblas_dscal(NANG, HI, &hv_p1[NANG*0], 1);

                            vmdMul(NANG, HJ, &hv_p1[NANG*1], &hv_p1[NANG*1],
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );

                            vmdMul(NANG, &PSIJ_3D(0,(ic-1),(k-1)),
                                   &hv_p1[NANG*1], &hv_p1[NANG*1],
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );

                            vmdMul(NANG, HK, &hv_p1[NANG*2], &hv_p1[NANG*2],
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );

                            vmdMul(NANG, &PSIK_3D(0,(ic-1),(j-1)),
                                   &hv_p1[NANG*2], &hv_p1[NANG*2],
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );

                            vmdAdd(NANG, &hv_p1[NANG*0], &hv_p1[NANG*1], pc,
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );


                            vmdAdd(NANG, &hv_p1[NANG*2], pc, pc,
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );

                            if (VDELT_CONST != 0 )
                            {
                                vmdMul(NANG, &PTR_IN_6D(0,(i-1),(j-1),(k-1),(i1-1),(i2-1)),
                                       &hv_p1[NANG*3], &hv_p1[NANG*3],
                                       VML_ACCURACY | VML_HANDLING | VML_ERROR );
                                cblas_daxpy(NANG, VDELT_CONST, &hv_p1[NANG*3], 1,
                                            pc, 1);
                            }

                            cblas_dscal(NANG, 0.5, pc, 1);
                            cblas_daxpy(NANG, 1, psi, 1, pc, 1);

                            for ( ang = 0; ang < NANG; ang++ )
                            {
                                DEN_1D(ang) = T_XS_3D((i-1),(j-1),(k-1))
                                    + MU_1D(ang)  * HI * HV_2D(ang,0)
                                    + HJ_1D(ang)  * HV_2D(ang,1)
                                    + HK_1D(ang)  * HV_2D(ang,2)
                                    + VDELT_CONST * HV_2D(ang,3);

                                if ( DEN_1D(ang) > TOLR )
                                {
                                    PC_1D(ang) /= DEN_1D(ang);
                                }
                                else
                                {
                                    PC_1D(ang) = 0;
                                }
                            }
                        }

                        else
                        {
                            for ( ang = 0; ang < NANG; ang++ )
                            {
                                PC_1D(ang) = PSII_3D(ang,(j-1),(k-1))
                                    * MU_1D(ang) * HI * (1+HV_2D(ang,0))
                                    + PSIJ_3D(ang,(ic-1),(k-1))
                                    * HJ_1D(ang) * (1+HV_2D(ang,1))
                                    + PSIK_3D(ang,(ic-1),(j-1))
                                    * HK_1D(ang) * (1+HV_2D(ang,2));

                                if ( VDELT_CONST != 0 )
                                {
                                    PC_1D(ang) += VDELT_CONST
                                        * PTR_IN_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1))
                                        * (1+HV_2D(ang,3));
                                }

                                PC_1D(ang) = PSI_1D(ang) + 0.5*PC_1D(ang);

                                DEN_1D(ang) = T_XS_3D((i-1),(j-1),(k-1))
                                    + MU_1D(ang)  * HI * HV_2D(ang,0)
                                    + HJ_1D(ang)  * HV_2D(ang,1)
                                    + HK_1D(ang)  * HV_2D(ang,2)
                                    + VDELT_CONST * HV_2D(ang,3);

                                if ( DEN_1D(ang) > TOLR )
                                {
                                    PC_1D(ang) /= DEN_1D(ang);
                                }
                                else
                                {
                                    PC_1D(ang) = 0;
                                }
                            }

                        }

#elif defined MKLUPDATE
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PC_1D(ang) = PSII_3D(ang,(j-1),(k-1))
                                * MU_1D(ang) * HI * (1+HV_2D(ang,0))
                                + PSIJ_3D(ang,(ic-1),(k-1))
                                * HJ_1D(ang) * (1+HV_2D(ang,1))
                                + PSIK_3D(ang,(ic-1),(j-1))
                                * HK_1D(ang) * (1+HV_2D(ang,2));

                            if ( VDELT_CONST != 0 )
                            {
                                PC_1D(ang) += VDELT_CONST
                                    * PTR_IN_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1))
                                    * (1+HV_2D(ang,3));
                            }

                            DEN_1D(ang) = T_XS_3D((i-1),(j-1),(k-1))
                                + MU_1D(ang)  * HI * HV_2D(ang,0)
                                + HJ_1D(ang)  * HV_2D(ang,1)
                                + HK_1D(ang)  * HV_2D(ang,2)
                                + VDELT_CONST * HV_2D(ang,3);

                        }

                        cblas_dscal(NANG, 0.5, pc, 1);
                        cblas_daxpy(NANG, 1, psi, 1, pc, 1);

                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            if ( DEN_1D(ang) > TOLR )
                            {
                                PC_1D(ang) /= DEN_1D(ang);
                            }
                            else
                            {
                                PC_1D(ang) = 0;
                            }
                        }

#else
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PC_1D(ang) = PSII_3D(ang,(j-1),(k-1))
                                * MU_1D(ang) * HI * (1+HV_2D(ang,0))
                                + PSIJ_3D(ang,(ic-1),(k-1))
                                * HJ_1D(ang) * (1+HV_2D(ang,1))
                                + PSIK_3D(ang,(ic-1),(j-1))
                                * HK_1D(ang) * (1+HV_2D(ang,2));

                            if ( VDELT_CONST != 0 )
                            {
                                PC_1D(ang) += VDELT_CONST
                                    * PTR_IN_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1))
                                    * (1+HV_2D(ang,3));
                            }

                            PC_1D(ang) = PSI_1D(ang) + 0.5*PC_1D(ang);

                            DEN_1D(ang) = T_XS_3D((i-1),(j-1),(k-1))
                                + MU_1D(ang)  * HI * HV_2D(ang,0)
                                + HJ_1D(ang)  * HV_2D(ang,1)
                                + HK_1D(ang)  * HV_2D(ang,2)
                                + VDELT_CONST * HV_2D(ang,3);

                            if ( DEN_1D(ang) > TOLR )
                            {
                                PC_1D(ang) /= DEN_1D(ang);
                            }
                            else
                            {
                                PC_1D(ang) = 0;
                            }
                        }
#endif

                    } // end fixup_loop

/***********************************************************************
 * Fixup done, compute edges
 ***********************************************************************/
#ifdef USEVML
                    if (NANG > VECLEN_MIN)
                    {
                        cblas_dcopy(NANG, pc, 1, psi, 1);

                        vmdMul(NANG, &FXHV_2D(0,0), &HV_2D(0,0),
                               &PSII_3D(0,(j-1),(k-1)),
                               VML_ACCURACY | VML_HANDLING | VML_ERROR );

                        vmdMul(NANG, &FXHV_2D(0,1), &HV_2D(0,1),
                               &PSIJ_3D(0,(ic-1),(k-1)),
                               VML_ACCURACY | VML_HANDLING | VML_ERROR );

                        if ( NDIMEN == 3 )
                        {
                            vmdMul(NANG, &FXHV_2D(0,2), &HV_2D(0,2),
                                   &PSIK_3D(0,(ic-1),(j-1)),
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );
                        }

                        if ( VDELT_CONST != 0 )
                        {
                            vmdMul(NANG, &FXHV_2D(0,3), &HV_2D(0,3),
                                   &PTR_OUT_6D(0,(i-1),(j-1),(k-1),(i1-1),(i2-1)),
                                   VML_ACCURACY | VML_HANDLING | VML_ERROR );
                        }
                    }

                    else
                    {
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSI_1D(ang) = PC_1D(ang);

                            PSII_3D(ang,(j-1),(k-1))
                                = FXHV_2D(ang,0) * HV_2D(ang,0);

                            PSIJ_3D(ang,(ic-1),(k-1))
                                = FXHV_2D(ang,1) * HV_2D(ang,1);

                            if ( NDIMEN == 3 )
                            {
                                PSIK_3D(ang,(ic-1),(j-1))
                                    = FXHV_2D(ang,2) * HV_2D(ang,2);
                            }

                            if ( VDELT_CONST != 0 )
                            {
                                PTR_OUT_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1))
                                    = FXHV_2D(ang,3) * HV_2D(ang,3);
                            }
                        }

                    }
#else
                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        PSI_1D(ang) = PC_1D(ang);

                        PSII_3D(ang,(j-1),(k-1))
                            = FXHV_2D(ang,0) * HV_2D(ang,0);

                        PSIJ_3D(ang,(ic-1),(k-1))
                            = FXHV_2D(ang,1) * HV_2D(ang,1);

                        if ( NDIMEN == 3 )
                        {
                            PSIK_3D(ang,(ic-1),(j-1))
                                = FXHV_2D(ang,2) * HV_2D(ang,2);
                        }

                        if ( VDELT_CONST != 0 )
                        {
                            PTR_OUT_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1))
                                = FXHV_2D(ang,3) * HV_2D(ang,3);
                        }
                    }
#endif
                }

/***********************************************************************
 * Clear the flux arrays
 ***********************************************************************/
                if ( oct == 1 )
                {
                    FLUX_4D((i-1),(j-1),(k-1),(g-1)) = 0;

                    for ( indx1 = 0; indx1 < (CMOM-1); indx1++ )
                    {
                        FLUXM_5D(indx1,(i-1),(j-1),(k-1),(g-1)) = 0;
                    }
                }

/***********************************************************************
 * Compute the flux moments
 ***********************************************************************/
#ifdef USEMKL
                vmdMul(NANG, W, psi, w_psi,
                       VML_ACCURACY | VML_HANDLING | VML_ERROR);

                FLUX_4D((i-1),(j-1),(k-1),(g-1)) += cblas_ddot(NANG, W, 1, psi, 1);

                for ( l = 1; l <= (CMOM-1); l++ )
                {
                    FLUXM_5D((l-1),(i-1),(j-1),(k-1),(g-1)) += cblas_ddot(NANG, &EC_2D(0,l), 1, w_psi,1);
                }

#else
                sum_wpsi = 0;

                for ( ang = 0; ang < NANG; ang++ )
                {
                    sum_wpsi += W_1D(ang)*PSI_1D(ang);
                }

                FLUX_4D((i-1),(j-1),(k-1),(g-1)) += sum_wpsi;

                for ( l = 1; l <= (CMOM-1); l++ )
                {
                    sum_ecwpsi = 0;

                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        sum_ecwpsi += EC_2D(ang,(l))*W_1D(ang)*PSI_1D(ang);
                    }

                    FLUXM_5D((l-1),(i-1),(j-1),(k-1),(g-1)) += sum_ecwpsi;
                }
#endif

/***********************************************************************
 * Calculate min and max scalar fluxes (not used elsewhere
 * currently)
 ***********************************************************************/
                if ( oct == NOCT )
                {
                    FMIN = MIN( FMIN, FLUX_3D((i-1),(j-1),(k-1)) );
                    FMAX = MAX( FMAX, FLUX_3D((i-1),(j-1),(k-1)) );
                }

/***********************************************************************
 * Save edge fluxes (dummy if checks for unused non-vacuum BCs)
 ***********************************************************************/
                if ( j == jhi )
                {
                    if ( jd==2 && LASTY )
                    {
                        // CONTINUE
                    }
                    else if ( jd == 1 && FIRSTY )
                    {
                        if ( ibb == 1 )
                        {
                            // CONTINUE
                        }
                    }
                    else
                    {
#ifdef USEMKL
                        cblas_dcopy(NANG, &PSIJ_3D(0,(ic-1),(k-1)), 1,
                            &JB_OUT_3D(0,(ic-1),(k-1)), 1);
#else
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            JB_OUT_3D(ang,(ic-1),(k-1))
                                = PSIJ_3D(ang,(ic-1),(k-1));
                        }
#endif
                    }
                }

                if ( k == khi )
                {
                    if ( kd == 2 && LASTZ )
                    {
                        // CONTINUE
                    }
                    else if ( kd==1 && FIRSTZ )
                    {
                        if ( ibf == 1 )
                        {
                            // CONTINUE
                        }
                    }
                    else
                    {
#ifdef USEMKL
                        cblas_dcopy(NANG, &PSIK_3D(0,(ic-1),(j-1)), 1,
                            &KB_OUT_3D(0,(ic-1),(j-1)), 1);
#else
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            KB_OUT_3D(ang,(ic-1),(j-1))
                                = PSIK_3D(ang,(ic-1),(j-1));
                        }
#endif
                    }
                }

/***********************************************************************
 * Compute leakages (not used elsewhere currently)
 ***********************************************************************/
                if ( ((i+id-1) == 1) || ((i+id-1) == (NX+1)) )
                {
#ifdef USEMKL
                    FLKX_3D((i+id-1-1),(j-1),(k-1))
                        += ist*cblas_ddot(NANG, &WMU_1D(0), 1,
                                          &PSII_3D(0,(j-1),(k-1)), 1);
#else
                    sum_wmupsii = 0;

                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        sum_wmupsii
                            += WMU_1D(ang) * PSII_3D(ang,(j-1),(k-1));
                    }

                    FLKX_3D((i+id-1-1),(j-1),(k-1))
                        += ist*sum_wmupsii;
#endif
                }

                if ( (jd==1 && FIRSTY) || (jd==2 && LASTY) )
                {
#ifdef USEMKL
                    FLKY_3D((i-1),(j+jd-1-1),(k-1))
                        += jst*cblas_ddot(NANG, &WETA_1D(0), 1,
                            &PSIJ_3D(0,(ic-1),(k-1)), 1);
#else
                    sum_wetapsij = 0;

                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        sum_wetapsij
                            += WETA_1D(ang) * PSIJ_3D(ang,(ic-1),(k-1));
                    }

                    FLKY_3D((i-1),(j+jd-1-1),(k-1))
                        += jst*sum_wetapsij;
#endif
                }

                if ( ((kd == 1 && FIRSTZ) || (kd == 2 && LASTZ)) && NDIMEN == 3 )
                {
#ifdef USEMKL
                    FLKZ_3D((i-1),(j-1),(k+kd-1-1))
                        += kst*cblas_ddot(NANG, &WXI_1D(0), 1,
                                          &PSIK_3D(0,(ic-1),(j-1)), 1);
#else
                    sum_wxipsik = 0;

                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        sum_wxipsik
                            += WXI_1D(ang) * PSIK_3D(ang,(ic-1),(j-1));
                    }

                    FLKZ_3D((i-1),(j-1),(k+kd-1-1))
                        += kst*sum_wxipsik;
#endif
                }
            }

/***********************************************************************
 * Finish the loops
 ***********************************************************************/
            } // end line_loop
        } // end diagonal_loop
// } // omp end parallel
}
