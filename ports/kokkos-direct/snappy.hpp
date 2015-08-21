#ifndef SNAPPY_HPP
#define SNAPPY_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>
#include <Kokkos_Serial.hpp>

#include <impl/Kokkos_Timer.hpp>

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <sched.h>

double const c0 = 0.0;
double const c1 = 1.0;
double const c2 = 2.0;
double const m1 = -c1;
double const p5 = c1/c2;
double const tolr = 1.0e-12;

using Kokkos::ALL;
using Kokkos::pair;
using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::subview;
using std::cout;
using std::endl;
using std::vector;

class cell_id_c {
  public:
    int ic;
    int j;
    int k;
  
    cell_id_c(){
      ic = 0;
      j = 0;
      k = 0;
    }
    
    cell_id_c(int ic_, int j_, int k_){
      ic = ic_;
      j = j_;
      k = k_;
    }
};

class diag_c {
  public:
    int len;
    vector<cell_id_c> cell_id;
    
    diag_c(int len_, vector<cell_id_c> cell_id_){
      len = len_;
      cell_id = cell_id_;
    }
};


namespace SNAPPY {

void populate_diag( const int ichunk, const int ny, const int nz, vector<diag_c>& diag ) {
  
// mini-kba sweep
{
//       ndiag = ichunk + ny + nz - 2
// 
//       ALLOCATE( diag(ndiag), indx(ndiag), STAT=ierr )
//       IF ( ierr /= 0 ) RETURN
// 
//       diag%len = 0
//       indx = 0
}

{
// !_______________________________________________________________________
// !
// !     Cells of same diagonal all have same value according to i+j+k-2
// !     formula. Use that to compute len for each diagonal. Use ichunk.
// !_______________________________________________________________________
// 
//       DO k = 1, nz
//       DO j = 1, ny
//       DO i = 1, ichunk
//         nn = i + j + k - 2
//         diag(nn)%len = diag(nn)%len + 1
//       END DO
//       END DO
//       END DO
}

{
// !_______________________________________________________________________
// !
// !     Next allocate cell_id array within diag type according to len
// !_______________________________________________________________________
// 
//       DO nn = 1, ndiag
//         ing = diag(nn)%len
//         ALLOCATE( diag(nn)%cell_id(ing), STAT=ierr )
//         IF ( ierr /= 0 ) RETURN
//       END DO
}

{
// !_______________________________________________________________________
// !
// !     Lastly, set each cell's actual ijk indices in this diagonal map
// !_______________________________________________________________________
// 
//       DO k = 1, nz
//       DO j = 1, ny
//       DO i = 1, ichunk
//         nn = i + j + k - 2
//         indx(nn) = indx(nn) + 1
//         ing = indx(nn)
//         diag(nn)%cell_id(ing)%ic = i
//         diag(nn)%cell_id(ing)%j  = j
//         diag(nn)%cell_id(ing)%k  = k
//       END DO
//       END DO
//       END DO
// 
//       DEALLOCATE( indx )
}

  int ndiag = ichunk + ny + nz - 2;
  for (int nn = 1; nn <= ndiag; nn++) {
    vector<cell_id_c> a;
    diag.emplace_back( 0, a );
  }

  for (int kk = 1; kk <= nz; kk++) {
    for (int jj = 1; jj <= ny; jj++) {
      for (int ii = 1; ii <= ichunk; ii++) {
        int nn = ii + jj + kk - 2;
        diag[nn-1].len += 1;
        diag[nn-1].cell_id.emplace_back(ii-1,jj-1,kk-1);
      }
    }
  }
  
}

template<typename Device, class ViewType1S>
struct scale {

  typedef Device device_type;
  ViewType1S inout_a;
  double scalar_a;
  
  scale(ViewType1S inout_a_, double scalar_a_):inout_a(inout_a_),scalar_a(scalar_a_) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    inout_a(i) = scalar_a * inout_a(i);
  }
};

// template<typename Device, typename ... Rest>
// struct sum_elementwise_product{};

template<typename Device, class ViewTypeA, class ViewTypeB, class ViewTypeC>
struct elementwise_product {

  typedef Device device_type;
  ViewTypeA inout_a;
  ViewTypeB input_b;
  ViewTypeC input_c;
  double scalar_a;
  double scalar_b;
  double scalar_c;
  
  elementwise_product(ViewTypeA inout_a_, double scalar_a_, ViewTypeB input_b_, double scalar_b_, ViewTypeC input_c_, double scalar_c_):inout_a(inout_a_),scalar_a(scalar_a_),input_b(input_b_),scalar_b(scalar_b_),input_c(input_c_),scalar_c(scalar_c_) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
     //printf("cpu: %d\n",sched_getcpu());
    inout_a(i) = scalar_a * inout_a(i) + ( ( scalar_b *input_b(i) ) * ( scalar_c * input_c(i) ) );
  }
};

template<typename Device, class ViewTypeA>
struct sum_1d{

  typedef Device device_type;
  typename ViewTypeA::const_type input_a;
  typedef double value_type;
  
  sum_1d(ViewTypeA input_a_):input_a(input_a_) {};
  
	KOKKOS_INLINE_FUNCTION
	void operator()( int i, value_type& contrib ) const {
		contrib += input_a(i);
	} // this thread's contribution
	
	KOKKOS_INLINE_FUNCTION
	void init( value_type& contrib ) const {
		contrib = c0;
	}
		
	KOKKOS_INLINE_FUNCTION
	void join( volatile value_type& contrib, volatile const value_type& input ) const {
		contrib = contrib + input;
	}
	  
};

template<typename Device, class ViewTypeA>
struct sum_2d{

  typedef Device device_type;
  typename ViewTypeA::const_type input_a;
  typedef double value_type;
  
  sum_2d(ViewTypeA input_a_):input_a(input_a_) {};
  
	KOKKOS_INLINE_FUNCTION
	void operator()( int i, value_type& contrib ) const {
	  for (int jj = 0; jj < input_a.dimension_1(); jj++) {
		  contrib += input_a(i,jj);
		}
	} // this thread's contribution
	
	KOKKOS_INLINE_FUNCTION
	void init( value_type& contrib ) const {
		contrib = c0;
	}
		
	KOKKOS_INLINE_FUNCTION
	void join( volatile value_type& contrib, volatile const value_type& input ) const {
		contrib = contrib + input;
	}
	  
};

template<typename Device, typename ... Rest>
struct sum_elementwise_product{};

template<typename Device, class ViewTypeA, class ViewTypeB>
struct sum_elementwise_product<Device, ViewTypeA, ViewTypeB>{

  typedef Device device_type;
  typename ViewTypeA::const_type input_a;
  typename ViewTypeB::const_type input_b;
  typedef double value_type;
  
  sum_elementwise_product(ViewTypeA input_a_, ViewTypeB input_b_):input_a(input_a_),input_b(input_b_) {};
  
	KOKKOS_INLINE_FUNCTION
	void operator()( int i, value_type& contrib ) const {
		contrib += ( input_a(i) * input_b(i) );
	} // this thread's contribution
	
	KOKKOS_INLINE_FUNCTION
	void init( value_type& contrib ) const {
		contrib = c0;
	}
		
	KOKKOS_INLINE_FUNCTION
	void join( volatile value_type& contrib, volatile const value_type& input ) const {
		contrib = contrib + input;
	}
	  
};

template<typename Device, class ViewTypeA, class ViewTypeB, class ViewTypeC>
struct sum_elementwise_product<Device, ViewTypeA, ViewTypeB, ViewTypeC>{

  typedef Device device_type;
  typename ViewTypeA::const_type input_a;
  typename ViewTypeB::const_type input_b;
  typename ViewTypeC::const_type input_c;
  typedef double value_type;
  
  sum_elementwise_product(ViewTypeA input_a_, ViewTypeB input_b_, ViewTypeC input_c_):input_a(input_a_),input_b(input_b_),input_c(input_c_) {};
  
	KOKKOS_INLINE_FUNCTION
	void operator()( int i, value_type& contrib ) const {
		contrib += ( input_a(i) * input_b(i) * input_c(i) );
	} // this thread's contribution
	
	KOKKOS_INLINE_FUNCTION
	void init( value_type& contrib ) const {
		contrib = c0;
	}
		
	KOKKOS_INLINE_FUNCTION
	void join( volatile value_type& contrib, volatile const value_type& input ) const {
		contrib = contrib + input;
	}
	  
};

template<typename Device, typename ... Rest>
struct update{

};

template<typename Device, class ViewTypeA, class ViewTypeB>
struct update<Device, ViewTypeA, ViewTypeB>{

  typedef Device device_type;
  ViewTypeA inout_a;
  ViewTypeB input_b;
  double scalar_a;
  double scalar_b;
  
  update(ViewTypeA inout_a_, double scalar_a_, ViewTypeB input_b_, double scalar_b_):inout_a(inout_a_),scalar_a(scalar_a_),input_b(input_b_),scalar_b(scalar_b_) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    inout_a(i) = scalar_a * inout_a(i) + scalar_b *input_b(i);
  }

};

template<typename Device, class ViewTypeA, class ViewTypeB, class ViewTypeC>
struct update<Device, ViewTypeA, ViewTypeB, ViewTypeC>{

  typedef Device device_type;
  ViewTypeA inout_a;
  ViewTypeB input_b;
  ViewTypeC input_c;
  double scalar_a;
  double scalar_b;
  double scalar_c;
  
  update(ViewTypeA inout_a_, double scalar_a_, ViewTypeB input_b_, double scalar_b_, ViewTypeC input_c_, double scalar_c_):inout_a(inout_a_),scalar_a(scalar_a_),input_b(input_b_),scalar_b(scalar_b_),input_c(input_c_),scalar_c(scalar_c_) {};
  
  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    inout_a(i) = scalar_a * inout_a(i) + scalar_b *input_b(i) + scalar_c *input_c(i);
  }

};

template< typename Device, class ViewTypeA,
          class ViewTypeB,
          class ViewTypeC1, class ViewTypeC2,
          class ViewTypeD1, class ViewTypeD2,
          class ViewTypeE1, class ViewTypeE2 >
struct update_Aa_Bb_C1C2c_D1D2d_E1E2e{

  typedef Device device_type;
  ViewTypeA inout_a;
  ViewTypeB input_b;
  ViewTypeC1 input_c1;
  ViewTypeC2 input_c2;
  ViewTypeD1 input_d1;
  ViewTypeD2 input_d2;
  ViewTypeE1 input_e1;
  ViewTypeE2 input_e2;
  double scalar_a;
  double scalar_b;
  double scalar_c;
  double scalar_d;
  double scalar_e;
  
  update_Aa_Bb_C1C2c_D1D2d_E1E2e( ViewTypeA inout_a_, double scalar_a_,
                                  ViewTypeB input_b_, double scalar_b_,
                                  ViewTypeC1 input_c1_, ViewTypeC2 input_c2_, double scalar_c_,
                                  ViewTypeD1 input_d1_, ViewTypeD2 input_d2_, double scalar_d_,
                                  ViewTypeE1 input_e1_, ViewTypeE2 input_e2_, double scalar_e_ ):
                                  inout_a(inout_a_), scalar_a(scalar_a_),
                                  input_b(input_b_), scalar_b(scalar_b_),
                                  input_c1(input_c1_), input_c2(input_c2_), scalar_c(scalar_c_),
                                  input_d1(input_d1_), input_d2(input_d2_), scalar_d(scalar_d_),
                                  input_e1(input_e1_), input_e2(input_e2_), scalar_e(scalar_e_) {};
                            
  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    inout_a(i)  = scalar_a * inout_a(i)
                + scalar_b * input_b(i)
                + scalar_c * input_c1(i) * input_c2(i)
                + scalar_d * input_d1(i) * input_d2(i)
                + scalar_e * input_e1(i) * input_e2(i);
  }
  
};

template<typename Device, class ViewTypeA, class ViewTypeB1, class ViewTypeB2, class ViewTypeB3>
struct update_Aa_B1B2pB3b{

  typedef Device device_type;
  ViewTypeA inout_a;
  ViewTypeB1 input_b1;
  ViewTypeB2 input_b2;
  ViewTypeB3 input_b3;
  double scalar_a;
  double scalar_b;
  
  update_Aa_B1B2pB3b( ViewTypeA inout_a_, double scalar_a_,
                      ViewTypeB1 input_b1_, ViewTypeB2 input_b2_, ViewTypeB3 input_b3_, double scalar_b_ ):
                      inout_a(inout_a_), scalar_a(scalar_a_),
                      input_b1(input_b1_), input_b2(input_b2_), input_b3(input_b3_), scalar_b(scalar_b_) {};
                      
  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    inout_a(i)  = scalar_a * inout_a(i)
                + scalar_b * input_b1(i) * ( input_b2(i) + input_b3(i) );
  }
  
};

template< typename Device, class ViewTypeA,
          class ViewTypeB1, class ViewTypeB2, class ViewTypeB3, class ViewTypeB4,
          class ViewTypeC1, class ViewTypeC2, class ViewTypeC3, class ViewTypeC4, 
          class ViewTypeD1, class ViewTypeD2, class ViewTypeD3, class ViewTypeD4 >
struct update_Aa_B1B2B3pB4b_C1C2C3pC4c_D1D2C3pD4d{

  typedef Device device_type;
  ViewTypeA inout_a;
  ViewTypeB1 input_b1;
  ViewTypeB2 input_b2;
  ViewTypeB3 input_b3;
  ViewTypeB4 input_b4;
  ViewTypeC1 input_c1;
  ViewTypeC2 input_c2;
  ViewTypeC3 input_c3;
  ViewTypeC4 input_c4;
  ViewTypeD1 input_d1;
  ViewTypeD2 input_d2;
  ViewTypeD3 input_d3;
  ViewTypeD4 input_d4;
  double scalar_a;
  double scalar_b;
  double scalar_c;
  double scalar_d;
  
  update_Aa_B1B2B3pB4b_C1C2C3pC4c_D1D2C3pD4d(
    ViewTypeA inout_a_, double scalar_a_,
    ViewTypeB1 input_b1_, ViewTypeB2 input_b2_, ViewTypeB3 input_b3_, ViewTypeB4 input_b4_, double scalar_b_,
    ViewTypeC1 input_c1_, ViewTypeC2 input_c2_, ViewTypeC3 input_c3_, ViewTypeC4 input_c4_, double scalar_c_,
    ViewTypeD1 input_d1_, ViewTypeD2 input_d2_, ViewTypeD3 input_d3_, ViewTypeD4 input_d4_, double scalar_d_ ):
    inout_a(inout_a_), scalar_a(scalar_a_),
    input_b1(input_b1_), input_b2(input_b2_), input_b3(input_b3_), input_b4(input_b4_), scalar_b(scalar_b_),
    input_c1(input_c1_), input_c2(input_c2_), input_c3(input_c3_), input_c4(input_c4_), scalar_c(scalar_c_),
    input_d1(input_d1_), input_d2(input_d2_), input_d3(input_d3_), input_d4(input_d4_), scalar_d(scalar_d_) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    inout_a(i)  = scalar_a * inout_a(i)
                + scalar_b * input_b1(i) * input_b2(i) * input_b3(i) * input_b4(i)
                + scalar_c * input_c1(i) * input_c2(i) * ( input_c3(i) + input_c4(i) )
                + scalar_d * input_d1(i) * input_d2(i) * ( input_d3(i) + input_d4(i) );
  }
  
};

template< typename Device,
          class ViewTypeA,
          class ViewTypeB,
          class ViewTypeC,
          class ViewTypeD1, class ViewTypeD2,
          class ViewTypeE1, class ViewTypeE2,
          class ViewTypeF1, class ViewTypeF2 >
struct update_Aa_Bb_Cc_D1D2d_E1E2e_F1F2f{

  typedef Device device_type;
  ViewTypeA inout_a;
  ViewTypeB input_b;
  ViewTypeC input_c;
  ViewTypeD1 input_d1;
  ViewTypeD2 input_d2;
  ViewTypeE1 input_e1;
  ViewTypeE2 input_e2;
  ViewTypeF1 input_f1;
  ViewTypeF2 input_f2;
  double scalar_a;
  double scalar_b;
  double scalar_c;
  double scalar_d;
  double scalar_e;
  double scalar_f;
  
  update_Aa_Bb_Cc_D1D2d_E1E2e_F1F2f(
    ViewTypeA inout_a_, double scalar_a_,
    ViewTypeB input_b_, double scalar_b_,
    ViewTypeC input_c_, double scalar_c_,
    ViewTypeD1 input_d1_, ViewTypeD2 input_d2_, double scalar_d_,
    ViewTypeE1 input_e1_, ViewTypeE2 input_e2_, double scalar_e_,
    ViewTypeF1 input_f1_, ViewTypeF2 input_f2_, double scalar_f_ ):
    inout_a(inout_a_), scalar_a(scalar_a_),
    input_b(input_b_), scalar_b(scalar_b_),
    input_c(input_c_), scalar_c(scalar_c_),
    input_d1(input_d1_), input_d2(input_d2_), scalar_d(scalar_d_),
    input_e1(input_e1_), input_e2(input_e2_), scalar_e(scalar_e_),
    input_f1(input_f1_), input_f2(input_f2_), scalar_f(scalar_f_) {};
    
    KOKKOS_INLINE_FUNCTION
    void operator()(int i) const {
      inout_a(i)  = scalar_a * inout_a(i)
                  + scalar_b * input_b(i)
                  + scalar_c * input_c(i)
                  + scalar_d * input_d1(i) * input_d2(i)
                  + scalar_e * input_e1(i) * input_e2(i)
                  + scalar_f * input_f1(i) * input_f2(i);
    }    
  
};

template< typename Device, class ViewTypeA, class ViewTypeB >
struct where_den_gt_tolr{

  typedef Device device_type;
  ViewTypeA inout_a;
  ViewTypeB input_b;
  double scalar;
  
  where_den_gt_tolr( ViewTypeA inout_a_, ViewTypeB input_b_, double scalar_ ):
    inout_a(inout_a_), input_b(input_b_), scalar(scalar_) {};
    
  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    inout_a(i) = ( ( input_b(i) > scalar ) ? ( inout_a(i) / input_b(i) ) : ( c0 ) );
  }
  
};

template< typename Device, class ViewTypeA, class ViewTypeB >
struct where_fxhv_lt_zero{

  typedef Device device_type;
  ViewTypeA inout_a;
  ViewTypeB input_b;
  
  where_fxhv_lt_zero( ViewTypeA inout_a_, ViewTypeB input_b_ ):
    inout_a(inout_a_), input_b(input_b_) {};
  
  // probably should be a compile time error if second dimensions of a,b don't match
  
  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    for (int jj = 0; jj < inout_a.dimension_1(); jj++) {
      inout_a(i,jj) = ( ( input_b(i,jj) < c0 ) ? ( c0 ) : ( inout_a(i,jj) ) );
    }
  }
  
};
// template<typename Device, typename ... Rest>
// struct el_p{};
// 
// template<typename Device, class ViewTypeA, typename ... Rest>
// struct el_p<Device, ViewTypeA, Rest...>{
// 
//   typedef Device device_type;
//   ViewTypeA input_a;
//   
//   el_p(ViewTypeA input_a_):input_a(input_a_) {};
//   
//   KOKKOS_INLINE_FUNCTION
//   void operator()(int i) const {
//     return input_a(i)*el_p<Rest>(...);
//   }
//   
// };
// 
// template<typename Device, class ViewTypeA>
// struct el_p<Device, ViewTypeA>{
// 
//   typedef Device device_type;
//   ViewTypeA input_a;
//   
//   el_p(ViewTypeA input_a_):input_a(input_a_) {};
//   
//   KOKKOS_INLINE_FUNCTION
//   void operator()(int i) const {
//     return input_a(i);
//   }
// 
// };

template <  typename Device, typename view_type_1d, typename view_type_2d,
            typename view_type_3d, typename view_type_4d, typename view_type_5d,
            typename view_type_6d, typename view_type_1d_s >
void dim3_sweep(  const int ichunk, const bool firsty, const bool lasty,          // PLIB
                const bool firstz, const bool lastz, const int nnested,           // PLIB
                const int nx, const double hi, const view_type_1d hj,             // GEOM
                const view_type_1d hk, const int ndimen, const int ny,            // GEOM
                const int nz, const int ndiag, const vector<diag_c>& diag,        // GEOM
                const int cmom, const int nang, const view_type_1d mu,            // SN
                const view_type_1d w, const int noct,                             // SN
                const int src_opt, const int ng, const view_type_6d qim,          // DATA
                const int fixup,                                                  // CONTROL
                const int ich, const int id, const int d1, const int d2,
                const int d3, const int d4, const int jd, const int kd,
                const int jlo, const int klo, const int oct, const int g,
                const int jhi, const int khi, const int jst, const int kst,
                view_type_3d psii, view_type_3d psij, view_type_3d psik,
                const view_type_4d qtot, const view_type_2d ec,
                const double vdelt, const view_type_4d ptr_in,
                view_type_4d ptr_out, const view_type_4d dinv,
                view_type_3d flux, view_type_4d fluxm,
                const view_type_3d jb_in, view_type_3d jb_out,
                const view_type_3d kb_in, view_type_3d kb_out,
                const view_type_1d wmu, const view_type_1d weta,
                const view_type_1d wxi, view_type_3d flkx, view_type_3d flky,
                view_type_3d flkz, const view_type_3d t_xs )
                
{
//////////////////////////////////////////////////////////////////////////////////////////
// 00 - preamble
//////////////////////////////////////////////////////////////////////////////////////////
	typedef Device device_type;
  typedef view_type_1d view_t_1d;
  typedef view_type_2d view_t_2d;
  typedef view_type_3d view_t_3d;
  typedef view_type_4d view_t_4d;
  typedef view_type_5d view_t_5d;
  typedef view_type_6d view_t_6d;
  typedef view_type_1d_s view_t_1d_s;
  
//////////////////////////////////////////////////////////////////////////////////////////
// local variables
//////////////////////////////////////////////////////////////////////////////////////////
{
//     INTEGER(i_knd) :: ist, d, n, ic, i, j, k, l, ibl, ibr, ibb, ibt,   &
//       ibf, ibk
// 
//     REAL(r_knd) :: sum_hv
// 
//     REAL(r_knd), DIMENSION(nang) :: psi, pc, den
// 
//     REAL(r_knd), DIMENSION(nang,4) :: hv, fxhv
}
	// need initialized to values
	int j;
	int k;
	int i;
	int l;
	int ibl;
	int ibr;
	int ibb;
	int ibt;
	int ibf;
	int ibk;
	int ist;
	int ic;
	bool fixup_condition;
	double sum_w_psi;
	double sum_ec_w_psi;
	double sum_wmu_psii_j_k;
  double sum_weta_psij_ic_k;
  double sum_wxi_psik_ic_j;
  double sum_hv;
  double temp_sum_hv;
	double fmin = c0;
	double fmax = c0;
  view_t_1d psi( "psi", nang );
  view_t_1d pc( "pc", nang );
  view_t_1d den( "den", nang );
  view_t_2d hv( "hv", nang, 4 );
  view_t_2d fxhv( "fxhv", nang, 4 );
  view_t_1d ones( "ones", nang );
  deep_copy(ones, c1);
  
//   cout << " d " << endl;
  
  // local subviews
  view_t_1d_s psii_slice;
  view_t_1d_s psij_slice;
  view_t_1d_s psik_slice;
  view_t_1d_s jb_in_slice;
  view_t_1d_s kb_in_slice;
  view_t_1d_s jb_out_slice;
  view_t_1d_s kb_out_slice;
  view_t_1d_s qim_slice;
  view_t_1d_s ec_slice;
  view_t_1d_s dinv_slice;
  view_t_1d_s ptr_in_slice;
  view_t_1d_s ptr_out_slice;
  view_t_1d_s fxhv_0_slice;
  view_t_1d_s fxhv_1_slice;
  view_t_1d_s fxhv_2_slice;
  view_t_1d_s fxhv_3_slice;
  view_t_1d_s hv_0_slice;
  view_t_1d_s hv_1_slice;
  view_t_1d_s hv_2_slice;
  view_t_1d_s hv_3_slice;
  view_t_1d_s fluxm_slice;
  
//////////////////////////////////////////////////////////////////////////////////////////
// A - set up the sweep order in the i-direction
//////////////////////////////////////////////////////////////////////////////////////////
{
//     ist = -1
//     IF ( id == 2 ) ist = 1
}
  ist = -1;
  if ( id == 2 ) ist = 1;

//   cout << " A " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// B - zero out the outgoing boundary arrays and fixup array
//////////////////////////////////////////////////////////////////////////////////////////
{
//     jb_out = zero
//     kb_out = zero
//     fxhv = zero
}
  deep_copy(jb_out, c0);
  deep_copy(kb_out, c0);
  deep_copy(fxhv, c0);

//   cout << " B " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// C- loop over cells along the diagonals. When only 1 diagonal, it's
// normal sweep order. Otherwise, nested threading performs mini-KBA.
//////////////////////////////////////////////////////////////////////////////////////////
{
//     diagonal_loop: DO d = 1, ndiag
// 
//       line_loop: DO n = 1, diag(d)%len
}

//   cout << " ndiag " << diag.size() << endl;
  
  for (int d = 0; d < diag.size(); d++ ) { // diagonal_loop
   
    for (int n = 0; n < diag[d].len; n++ ) { // line_loop
//       cout << " current diag: " << d << ", len: " << diag[d].len << endl;

{
//         ic = diag(d)%cell_id(n)%ic
}
      ic = diag[d].cell_id[n].ic;
            ist = -1;

//         cout << "ist " << ist << " ich " << ich << " ichunk " << ichunk << " ic " << ic << endl;
                  
{
//         IF ( ist < 0 ) THEN
//           i = ich*ichunk - ic + 1
//         ELSE
//           i = (ich-1)*ichunk + ic
//         END IF
}     
      if ( ist < 0 ) {
        i = ich * ichunk - ic;
      } else {
        i = ich * ichunk + ic;
      }
//       cout << "i " << i << " nx " << nx << endl;
      
{
//         IF ( i > nx ) CYCLE line_loop
}
      if ( i > nx-1 ) continue;

{
//         j = diag(d)%cell_id(n)%j
//         IF ( jst < 0 ) j = ny - j + 1
// 
//         k = diag(d)%cell_id(n)%k
//         IF ( kst < 0 ) k = nz - k + 1
}
      j = diag[d].cell_id[n].j;
      if ( jst < 0 ) j = ny - j + 1;
      
      k = diag[d].cell_id[n].k;
      if ( kst < 0 ) k = nz - k + 1;

//       cout << "i " << i << " j " << j << " k " << k << endl;      
//       cout << "id " << id << " ist " << ist << " i " << i << endl;
//       cout << "jd " << jd << " jst " << jst << " j " << j << endl;
//       cout << "kd " << kd << " kst " << kst << " k " << k << endl;
//       cout << "ich " << ich << " ichunk " << ichunk << endl;

//    cout << " C " << endl; 
//////////////////////////////////////////////////////////////////////////////////////////
// D - left/right boundary conditions, always vacuum
//////////////////////////////////////////////////////////////////////////////////////////
{
//         ibl = 0; ibr = 0
//         IF ( i==nx .AND. ist==-1 ) THEN
//           psii(:,j,k) = zero
//         ELSE IF ( i==1 .AND. ist==1 ) THEN
//           SELECT CASE ( ibl )
//             CASE ( 0 )
//               psii(:,j,k) = zero
//             CASE ( 1 )
//               psii(:,j,k) = zero
//           END SELECT
//         END IF
}
      ibl = 0;
      ibr = 0;
      if ( ( i == nx ) && ( ist == -1 ) ) {
        psii_slice = subview( psii, ALL(), j, k );
        parallel_for( nang, scale<device_type,view_t_1d_s>(psii_slice, c0) );
      } else if ( ( i == 1 ) && ( ist == 1 ) ) {
        psii_slice = subview( psii, ALL(), j, k );
        switch ( ibl ) {
          case 0:            
            parallel_for( nang, scale<device_type,view_t_1d_s>(psii_slice, c0) );
            break;
          case 1:
            parallel_for( nang, scale<device_type,view_t_1d_s>(psii_slice, c0) );
            break;          
        }
      }
//   cout << " D " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// E - top/bottom boundary condtions. Vacuum at global boundaries, but
// set to some incoming flux from neighboring proc
//////////////////////////////////////////////////////////////////////////////////////////
{
//         ibb = 0; ibt = 0
//         IF ( j == jlo ) THEN
//           IF ( jd==1 .AND. lasty ) THEN
//             psij(:,ic,k) = zero
//           ELSE IF ( jd==2 .AND. firsty ) THEN
//             SELECT CASE ( ibb )
//               CASE ( 0 )
//                 psij(:,ic,k) = zero
//               CASE ( 1 )
//                 psij(:,ic,k) = zero
//             END SELECT
//           ELSE
//             psij(:,ic,k) = jb_in(:,ic,k)
//           END IF
//         END IF
}
      ibb = 0;
      ibt = 0;
      if ( j == jlo ) {
        if ( ( jd == 1 ) && ( lasty ) ) {
          psij_slice = subview( psij, ALL(), ic, k );
          parallel_for( nang, scale<device_type,view_t_1d_s>(psij_slice, c0) );
//           std::cout << " E1 " << std::endl;
        } else if ( ( jd == 2 ) && ( firsty ) ) {
            psij_slice = subview( psij, ALL(), ic, k );
          switch ( ibb ) {
            case 0:
              parallel_for( nang, scale<device_type,view_t_1d_s>(psij_slice, c0) );
              break;
            case 1:
              parallel_for( nang, scale<device_type,view_t_1d_s>(psij_slice, c0) );
              break;
          }
        } else {
          psij_slice = subview( psij, ALL(), ic, k );
          jb_in_slice = subview( jb_in, ALL(), ic, k );
          parallel_for( nang, update<device_type,view_t_1d_s,view_t_1d_s>(psij_slice, c0, jb_in_slice, c1) );
        }
      } // end if ( j == jlo )
      
//   cout << " E " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// F - front/back boundary condtions. Vacuum at global boundaries, but
// set to some incoming flux from neighboring proc
//////////////////////////////////////////////////////////////////////////////////////////
{
//         ibf = 0; ibk = 0
//         IF ( k == klo ) THEN
//           IF ( (kd==1 .AND. lastz) .OR. ndimen<3 ) THEN
//             psik(:,ic,j) = zero
//           ELSE IF ( kd==2 .AND. firstz ) THEN
//             SELECT CASE ( ibf )
//               CASE ( 0 )
//                 psik(:,ic,j) = zero
//               CASE ( 1 )
//                 psik(:,ic,j) = zero
//             END SELECT
//           ELSE
//             psik(:,ic,j) = kb_in(:,ic,j)
//           END IF
//         END IF
}
      ibf = 0; ibk = 0;
      if ( k == klo ) {
        if ( ( ( kd == 1 ) && ( lastz ) ) || ( ndimen < 3 ) ) {
          psik_slice = subview( psik, ALL(), ic, j );
          parallel_for( nang, scale<device_type,view_t_1d_s>(psik_slice, c0) );
        } else if ( ( kd == 2 ) && ( firstz ) ) {
          psik_slice = subview( psik, ALL(), ic, j );
          switch ( ibf ) {
            case 0:
              parallel_for( nang, scale<device_type,view_t_1d_s>(psik_slice, c0) );
              break;
            case 1:
              parallel_for( nang, scale<device_type,view_t_1d_s>(psik_slice, c0) );
              break;
          }
        } else {
          psik_slice = subview( psik, ALL(), ic, j );
          kb_in_slice = subview( kb_in, ALL(), ic, j ); 
          parallel_for( nang, update<device_type,view_t_1d_s,view_t_1d_s>(psik_slice, c0, kb_in_slice, c1) );
        }
      } // end if ( k == klo )
//   cout << " F " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// G - compute the angular source
//////////////////////////////////////////////////////////////////////////////////////////
{
//         psi = qtot(1,i,j,k) // psi(nang), qtot(cmom,nx,ny,nz)
//         IF ( src_opt == 3 ) psi = psi + qim(:,i,j,k,oct,g)
// 
//         DO l = 2, cmom
//           psi = psi + ec(:,l)*qtot(l,i,j,k)
//         END DO
}
      deep_copy( psi, qtot(1,i,j,k) );
//   cout << " G1 " << endl;      
      if ( src_opt == 3 ) {
//    cout << " G2 " << endl;     
        qim_slice = subview( qim, ALL(), i, j, k, oct, g );
//    cout << " G3 " << endl;      
        parallel_for( nang, update<device_type,view_t_1d,view_t_1d_s>(psi, c1, qim_slice, c1) );
//         cout << " G4 " << endl;
      }
      
      for (l = 1; l < cmom; l++) {
//       cout << " l " << l << " cmom " << cmom << endl;
//    cout << " G5 " << endl;     
        ec_slice = subview( ec, ALL(), l );
//   cout << " G6 " << endl;        
        parallel_for( nang, update<device_type,view_t_1d,view_t_1d_s>(psi, c1, ec_slice, qtot(l,i,j,k)) );
//   cout << " G7 " << endl;
      }
//   cout << " G " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// H - compute the numerator for the update formula
//////////////////////////////////////////////////////////////////////////////////////////
{
//         pc = psi + psii(:,j,k)*mu*hi + psij(:,ic,k)*hj + psik(:,ic,j)*hk // pc(nang)
//         IF ( vdelt /= zero ) pc = pc + vdelt*ptr_in(:,i,j,k)
}
      psii_slice = subview( psii, ALL(), j, k );
      psij_slice = subview( psij, ALL(), ic, k );
      psik_slice = subview( psik, ALL(), ic, j );
      parallel_for( nang, update_Aa_Bb_C1C2c_D1D2d_E1E2e< device_type, view_t_1d,       // pc
                                                    view_t_1d,                          // psi
                                                    view_t_1d_s, view_t_1d,             // psii_slice
                                                    view_t_1d_s, view_t_1d,             // psij_slice
                                                    view_t_1d_s, view_t_1d >            // psik_slice
                                                  ( pc, c0,
                                                    psi, c1,
                                                    psii_slice, mu, hi,
                                                    psij_slice, hj, c1,
                                                    psik_slice, hk, c1 ) );
      if ( vdelt != c0 ) { // fp not equal is bad practice always, should be checking for a tolerance
        ptr_in_slice = subview( ptr_in, ALL(), i, j, k );
        parallel_for( nang, update<device_type,view_t_1d,view_t_1d_s>( pc, c1, ptr_in_slice, vdelt ) );
      }
//   cout << " H " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// I - compute the solution of the center. Use DD for edges. Use fixup if requested.
//////////////////////////////////////////////////////////////////////////////////////////
{
//         IF ( fixup == 0 ) THEN
}
      if ( fixup == 0 ) {
{
//  
//           psi = pc*dinv(:,i,j,k)
// 
//           psii(:,j,k) = two*psi - psii(:,j,k)
//           psij(:,ic,k) = two*psi - psij(:,ic,k)
//           IF ( ndimen == 3 ) psik(:,ic,j) = two*psi - psik(:,ic,j)
//           IF ( vdelt /= zero )                                         &
//             ptr_out(:,i,j,k) = two*psi - ptr_in(:,i,j,k)
//
}
        dinv_slice = subview( dinv, ALL(), i, j, k );
        parallel_for( nang, elementwise_product<device_type, view_t_1d, view_t_1d, view_t_1d_s>(psi, c0, pc, c1, dinv_slice, c1) );
        
        psii_slice = subview( psii, ALL(), j, k );
        psij_slice = subview( psij, ALL(), ic, k );
        parallel_for( nang, update<device_type,view_t_1d_s,view_t_1d>(psii_slice, m1, psi, c2) );
        parallel_for( nang, update<device_type,view_t_1d_s,view_t_1d>(psij_slice, m1, psi, c2) );
        if ( ndimen == 3 ) {
          ptr_in_slice = subview( ptr_in, ALL(), i, j, k );
          ptr_out_slice = subview( ptr_out, ALL(), i, j, k );
          parallel_for( nang, update<device_type,view_t_1d_s,view_t_1d,view_t_1d_s>(ptr_out_slice, c0, psi, c2, ptr_in_slice, m1) );
        }
        if ( vdelt != c0 ) { // fp not equal is bad practice always, should be checking for a tolerance
          ptr_out_slice = subview( ptr_out, ALL(), i, j, k );
          ptr_in_slice = subview( ptr_in, ALL(), i, j, k );
          parallel_for( nang, elementwise_product<device_type,view_t_1d_s,view_t_1d,view_t_1d_s>(ptr_out_slice, c0, psi, c2, ptr_in_slice, m1 ) );      
        }

//         ELSE
//   cout << " I " << endl;
      } else {
//////////////////////////////////////////////////////////////////////////////////////////
// J - multi-pass set to zero + rebalance fixup, determine angles that will need fixup first
//////////////////////////////////////////////////////////////////////////////////////////
{
//           hv = one; sum_hv = SUM( hv )
// 
//           pc = pc * dinv(:,i,j,k)
// 
//           fixup_loop: DO
// 
//             fxhv(:,1) = two*pc - psii(:,j,k)
//             fxhv(:,2) = two*pc - psij(:,ic,k)
//             IF ( ndimen == 3 ) fxhv(:,3) = two*pc - psik(:,ic,j)
//             IF ( vdelt /= zero ) fxhv(:,4) = two*pc - ptr_in(:,i,j,k)
// 
//             WHERE ( fxhv < zero ) hv = zero
}
        deep_copy(hv, c1);
        parallel_reduce( nang, sum_2d<device_type,view_t_2d>(hv), sum_hv );
        
        fixup_condition = true;
        while ( fixup_condition == true ) {
            
          fxhv_0_slice = subview( fxhv, ALL(), 0 );
          psii_slice = subview( psii, ALL(), j, k );
          parallel_for( nang, update<device_type,view_t_1d_s,view_t_1d,view_t_1d_s>(fxhv_0_slice, c0, pc, c2, psii_slice, m1) );
          fxhv_1_slice = subview( fxhv, ALL(), 1 );
          psij_slice = subview( psij, ALL(), ic, k );
          parallel_for( nang, update<device_type,view_t_1d_s,view_t_1d,view_t_1d_s>(fxhv_1_slice, c0, pc, c2, psij_slice, m1) );          
          if ( ndimen == 3 ) {
            fxhv_2_slice = subview( fxhv, ALL(), 2 );
            psik_slice = subview( psik, ALL(), ic, j );
            parallel_for( nang, update<device_type,view_t_1d_s,view_t_1d,view_t_1d_s>(fxhv_2_slice, c0, pc, c2, psik_slice, m1) );
          }
          if ( vdelt != c0 ) { // fp not equal is bad practice always, should be checking for a tolerance
            fxhv_3_slice = subview( fxhv, ALL(), 3 );
            ptr_in_slice = subview( ptr_in, ALL(), i, j, k );
            parallel_for( nang, update<device_type,view_t_1d_s,view_t_1d,view_t_1d_s>(fxhv_3_slice, c0, pc, c2, ptr_in_slice, m1) );
          }
          
          parallel_for( nang, where_fxhv_lt_zero<device_type,view_t_2d,view_t_2d>(hv, fxhv) );
//   cout << " J " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// K - exit loop when all angles are fixed up
//////////////////////////////////////////////////////////////////////////////////////////
{
//             IF ( sum_hv == SUM( hv ) ) EXIT fixup_loop
//             sum_hv = SUM( hv )
}
          parallel_reduce( nang, sum_2d<device_type,view_t_2d>(hv), temp_sum_hv );
          if (sum_hv == temp_sum_hv) break;
          sum_hv = temp_sum_hv;
//   cout << " K " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// L - recompute balance equation numerator and denominator and get new cell average flux
//////////////////////////////////////////////////////////////////////////////////////////
{
//             pc = psii(:,j,k)*mu*hi*(one+hv(:,1)) + psij(:,ic,k)*hj*(one+hv(:,2)) + psik(:,ic,j)*hk*(one+hv(:,3))
//             IF ( vdelt /= zero )                                       &
//               pc = pc + vdelt*ptr_in(:,i,j,k)*(one+hv(:,4))
//             pc = psi + half*pc
// 
//             den = t_xs(i,j,k) + mu*hi*hv(:,1) + hj*hv(:,2) + hk*hv(:,3) + vdelt*hv(:,4)
// 
//             WHERE( den > tolr )
//               pc = pc/den
//             ELSEWHERE
//               pc = zero
//             END WHERE
}
          psii_slice = subview( psii, ALL(), j, k );
          psij_slice = subview( psij, ALL(), ic, k );
          psik_slice = subview( psik, ALL(), ic, j );
          hv_0_slice = subview( hv, ALL(), 0 );
          hv_1_slice = subview( hv, ALL(), 1 );
          hv_2_slice = subview( hv, ALL(), 2 );
          parallel_for( nang, update_Aa_B1B2B3pB4b_C1C2C3pC4c_D1D2C3pD4d<device_type,
            view_t_1d,
            view_t_1d_s, view_t_1d, view_t_1d, view_t_1d_s,
            view_t_1d_s, view_t_1d, view_t_1d, view_t_1d_s,
            view_t_1d_s, view_t_1d, view_t_1d, view_t_1d_s>
            ( pc, c0,
              psii_slice, mu, ones, hv_0_slice, hi,
              psij_slice, hj, ones, hv_1_slice, c1,
              psik_slice, hk, ones, hv_2_slice, c1) );
              
          if ( vdelt != c0 ) { // fp not equal is bad practice always, should be checking for a tolerance
            ptr_in_slice = subview( ptr_in, ALL(), i, j, k );
            hv_3_slice = subview( hv, ALL(), 3 );
            parallel_for( nang, update_Aa_B1B2pB3b<device_type,
              view_t_1d,
              view_t_1d_s, view_t_1d, view_t_1d_s>
              ( pc, c0, 
                ptr_in_slice, ones, hv_3_slice, vdelt) );
            
          }
          parallel_for( nang, update<device_type,view_t_1d,view_t_1d>(pc, p5, psi, c1) );
 
          hv_0_slice = subview( hv, ALL(), 0 );
          hv_1_slice = subview( hv, ALL(), 1 );
          hv_2_slice = subview( hv, ALL(), 2 );
          hv_3_slice = subview( hv, ALL(), 3 );
          parallel_for( nang, update_Aa_Bb_Cc_D1D2d_E1E2e_F1F2f< device_type,
            view_t_1d,
            view_t_1d,
            view_t_1d_s,
            view_t_1d, view_t_1d_s,
            view_t_1d, view_t_1d_s,
            view_t_1d, view_t_1d_s >
            ( den, c0,
              ones, t_xs(i,j,k),
              hv_3_slice, vdelt,
              mu, hv_0_slice, hi,
              hj, hv_1_slice, c1,
              hk, hv_2_slice, c1) );
          
          parallel_for( nang, where_den_gt_tolr<device_type,view_t_1d,view_t_1d>( pc, den, tolr ) );
                            
{
//           END DO fixup_loop
}
        }
//   cout << " L " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// M - fixup done, compute edges
//////////////////////////////////////////////////////////////////////////////////////////
{
//           psi = pc
// 
//           psii(:,j,k) = fxhv(:,1) * hv(:,1)
//           psij(:,ic,k) = fxhv(:,2) * hv(:,2)
//           IF ( ndimen == 3 ) psik(:,ic,j) = fxhv(:,3) * hv(:,3)
//           IF ( vdelt /= zero ) ptr_out(:,i,j,k) = fxhv(:,4) * hv(:,4)
//
}
      deep_copy(psi, pc);

      psii_slice = subview( psii, ALL(), j, k );
      fxhv_0_slice = subview( fxhv, ALL(), 0 );
      hv_0_slice = subview( hv, ALL(), 0 );
      parallel_for( nang, elementwise_product<device_type,view_t_1d_s,view_t_1d_s,view_t_1d_s>(psii_slice, c0, fxhv_0_slice, c1, hv_0_slice, c1 ) );

      psij_slice = subview( psij, ALL(), ic, k );
      fxhv_1_slice = subview( fxhv, ALL(), 1 );
      hv_1_slice = subview( hv, ALL(), 1 );
      parallel_for( nang, elementwise_product<device_type,view_t_1d_s,view_t_1d_s,view_t_1d_s>(psij_slice, c0, fxhv_1_slice, c1, hv_1_slice, c1 ) );

      if ( ndimen == 3 ) {
        psik_slice = subview( psik, ALL(), ic, j );
        fxhv_2_slice = subview( fxhv, ALL(), 2 );
        hv_2_slice = subview( hv, ALL(),2 );
        parallel_for( nang, elementwise_product<device_type,view_t_1d_s,view_t_1d_s,view_t_1d_s>(psik_slice, c0, fxhv_2_slice, c1, hv_2_slice, c1 ) );
      }

      if ( vdelt != c0 ) { // fp not equal is bad practice always, should be checking for a tolerance
        ptr_out_slice = subview( ptr_out, ALL(), i, j, k );
        fxhv_3_slice = subview( fxhv, ALL(), 3 );
        hv_3_slice = subview( hv, ALL(), 3 );
        parallel_for( nang, elementwise_product<device_type,view_t_1d_s,view_t_1d_s,view_t_1d_s>(ptr_out_slice, c0, fxhv_3_slice, c1, hv_3_slice, c1 ) );      
      }

{
//         END IF
}
//   cout << " M " << endl;
      } // end if ( fixup == 0 )
//////////////////////////////////////////////////////////////////////////////////////////
// N - clear the flux arrays
//////////////////////////////////////////////////////////////////////////////////////////
{
//         IF ( oct == 1 ) THEN
//           flux(i,j,k) = zero
//           fluxm(:,i,j,k) = zero
//         END IF
}
      if ( oct == 1 ) {
        flux(i,j,k) = c0;
        fluxm_slice = subview( fluxm, ALL(), i, j, k );
        parallel_for( cmom-1, scale<device_type,view_t_1d_s>(fluxm_slice, c0) );
      }
//   cout << " N " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// O - compute the flux moments
//////////////////////////////////////////////////////////////////////////////////////////
{
//         flux(i,j,k) = flux(i,j,k) + SUM( w*psi )
//         DO l = 1, cmom-1
//           fluxm(l,i,j,k) = fluxm(l,i,j,k) + SUM( ec(:,l+1)*w*psi )
//         END DO
}
      parallel_reduce( nang, sum_elementwise_product<device_type, view_t_1d, view_t_1d>(w, psi), sum_w_psi );
      flux(i,j,k) = flux(i,j,k) + sum_w_psi;
      for (l = 0; l < cmom-1; l++) {
        ec_slice = subview( ec, ALL(), l+1 );
        parallel_reduce( nang, sum_elementwise_product<device_type,view_t_1d_s,view_t_1d,view_t_1d>(ec_slice, w, psi), sum_ec_w_psi );
        fluxm(l,i,j,k) = fluxm(l,i,j,k) + sum_ec_w_psi;
      }
//   cout << " O " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// P - calculate min and max scalar fluxes (not used elsewhere currently)
//////////////////////////////////////////////////////////////////////////////////////////
{
//         IF ( oct == noct ) THEN
//           fmin = MIN( fmin, flux(i,j,k) )
//           fmax = MAX( fmax, flux(i,j,k) )
//         END IF
}
      if ( oct == noct ) {
        fmin = std::min( fmin, flux(i,j,k) );
        fmax = std::max( fmax, flux(i,j,k) );
      }
//   cout << " P " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// Q - save edge fluxes (dummy if checks for unused non-vacuum BCs)
//////////////////////////////////////////////////////////////////////////////////////////
{
//         IF ( j == jhi ) THEN
//           IF ( jd==2 .AND. lasty ) THEN
//             CONTINUE
//           ELSE IF ( jd==1 .AND. firsty ) THEN
//             IF ( ibb == 1 ) CONTINUE
//           ELSE
//             jb_out(:,ic,k) = psij(:,ic,k)
//           END IF
//         END IF
}
      if ( j = jhi ) {
        if ( ( jd == 2 ) && ( lasty ) ) {
        } else if ( ( jd == 1 ) && ( firsty ) ) {
          if ( ibb == 1 ) {}
        } else {
          jb_out_slice = subview( jb_out, ALL(), ic, k );
          psij_slice = subview( psij, ALL(), ic, k );
          parallel_for( nang, update<device_type,view_t_1d_s,view_t_1d_s>(jb_out_slice, c0, psij_slice, c1) );
        }
      }

{
//         IF ( k == khi ) THEN
//           IF ( kd==2 .AND. lastz ) THEN
//             CONTINUE
//           ELSE IF ( kd==1 .AND. firstz ) THEN
//             IF ( ibf == 1 ) CONTINUE
//           ELSE
//             kb_out(:,ic,j) = psik(:,ic,j)
//           END IF
//         END IF
}
      if ( k = khi ) {
        if ( ( kd == 2 ) && ( lastz ) ) {
        } else if ( ( kd == 1 ) && ( firstz ) ) {
          if ( ibf == 1 ) {}
        } else {
          kb_out_slice = subview( kb_out, ALL(), ic, j );
          psik_slice = subview( psik, ALL(), ic, j );
          parallel_for( nang, update<device_type,view_t_1d_s,view_t_1d_s>(kb_out_slice, c0, psik_slice, c1 ) );
        }
      }
//   cout << " Q " << endl;
//////////////////////////////////////////////////////////////////////////////////////////
// R - compute leakages (not used elsewhere currently)
//////////////////////////////////////////////////////////////////////////////////////////
{
//         IF ( i+id-1==1 .OR. i+id-1==nx+1 ) THEN
//           flkx(i+id-1,j,k) = flkx(i+id-1,j,k) +                        &
//             ist*SUM( wmu*psii(:,j,k) )
//         END IF
// 
//         IF ( (jd==1 .AND. firsty) .OR. (jd==2 .AND. lasty) ) THEN
//           flky(i,j+jd-1,k) = flky(i,j+jd-1,k) +                        &
//             jst*SUM( weta*psij(:,ic,k) )
//         END IF
// 
//         IF ( ((kd==1 .AND. firstz) .OR. (kd==2 .AND. lastz)) .AND.     &
//              ndimen==3 ) THEN
//           flkz(i,j,k+kd-1) = flkz(i,j,k+kd-1) +                        &
//             kst*SUM( wxi*psik(:,ic,j) )
//         END IF
}
      if ( ( ( i + id - 1 ) == 1 ) || ( ( i + id - 1 ) == ( nx + 1 ) ) ) {
        psii_slice = subview( psii, ALL(), j, k );
        parallel_reduce( nang, sum_elementwise_product<device_type, view_t_1d, view_t_1d_s>(wmu, psii_slice), sum_wmu_psii_j_k );
        flkx(i+id-1,j,k) = flkx(i+id-1,j,k) + ist * sum_wmu_psii_j_k;
      }
      
      if ( ( ( jd == 1 ) && ( firsty ) ) || ( ( jd == 2 ) && ( lasty ) ) ) {
        psij_slice = subview( psij, ALL(), ic, k );
        parallel_reduce( nang, sum_elementwise_product<device_type, view_t_1d, view_t_1d_s>(weta, psij_slice), sum_weta_psij_ic_k );
        flky(i,j+jd-1,k) = flky(i,j+jd-1,k) + jst * sum_weta_psij_ic_k;
      }
      
      if ( ( ( ( kd == 1 ) && ( firstz ) ) || ( ( kd == 2 ) && ( lastz ) ) ) && ( ndimen == 3 ) ) {
        psik_slice = subview( psik, ALL(), ic, j );
        parallel_reduce( nang, sum_elementwise_product<device_type, view_t_1d, view_t_1d_s>(wxi, psik_slice), sum_wxi_psik_ic_j );
        flkz(i,j,k+kd-1) = flkz(i,j,k+kd-1) + kst * sum_wxi_psik_ic_j;
      }

//   cout << " R " << endl;    
//////////////////////////////////////////////////////////////////////////////////////////
// S - finish the loops
//////////////////////////////////////////////////////////////////////////////////////////
{
//      END DO line_loop
// 
//    END DO diagonal_loop
}

    } // end line_loop

  } // end diagonal_loop


} // end dim3_sweep()


} // namespace SNAPPY


#endif //SNAPPY_HPP

