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
#include <algorithm>

#include <sched.h>

#define FIXUP 1 //cuda workaround

// NOTE - don't declare a view in a parallel region, will compile, then will runtime error
// NOTE - passing one parm to 2d array results in template error
// NOTE - declaring a 2d array with a 1d type is fine (that is how you find the above error)

double const c0 = 0.0;
double const c1 = 1.0;
double const c2 = 2.0;
double const m1 = -c1;
double const p5 = c1/c2;
double const tolr = 1.0e-12;

bool const ibl = false;
bool const ibr = false;
bool const ibb = false;
bool const ibt = false;
bool const ibf = false;
bool const ibk = false;
//bool fixup = false;

int const ndimen = 3;
int const src_opt = 3;

bool const lasty = true; // TODO MUST BE PASSED IN
bool const firsty = false; // TODO MUST BE PASSED IN
bool const lastz = true; // TODO MUST BE PASSED IN
bool const firstz = false; // TODO MUST BE PASSED IN

using Kokkos::ALL;
using Kokkos::pair;
using Kokkos::TeamPolicy;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;
using Kokkos::View;
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

void populate_diag( const int ichunk, const int ny, const int nz, vector<diag_c>& diag,
                    int& ndiag, int& ndiag_entries ) {
  
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

  ndiag = ichunk + ny + nz - 2;
  ndiag_entries = ichunk * ny * nz;
  
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

void count_memory(  const int ndiag, const int ndiag_entries, const int nang, const int ng,
                                 const int nx, const int ny, const int nz, const int nmom, const int cmom,
                                 const int noct, const int ichunk, const int M ) {

	long int double_count = 0;
	double_count +=  ndiag; // diag_start(ndiag) 
	double_count += ndiag; // diag_len(ndiag) 
	double_count += ndiag_entries; // diag_ic(ndiag_entries)
	double_count += ndiag_entries; // diag_j(ndiag_entries)
	double_count += ndiag_entries; // diag_k(ndiag_entries)
	double_count += nang * ny * nz * ng; // psii(nang,ny,nz,ng)
	double_count += nang * ichunk * nz * ng; // psij(nang,ichunk,nz,ng) 
	double_count += nang * ichunk * ny * ng; // psik(nang,ichunk,ny,ng)
	double_count += nang * ichunk * nz * ng; // jb_in(nang,ichunk,nz,ng)   
	double_count += nang * ichunk * ny * ng; // kb_in(nang,ichunk,ny,ng)
	double_count += nang * nx * ny *nz * noct * ng; // qim(nang,nx,ny,nz,noct,ng)  
	double_count += cmom * nx * ny * nz * ng; // qtot(cmom,nx,ny,nz,ng)
	double_count += nang * cmom; // ec(nang,cmom)
	double_count += nang; // mu(nang) 
	double_count += nang; // w(nang)
	double_count += nang; // wmu(nang)
	double_count += nang; // weta(nang)
	double_count += nang; // wxi(nang)  
	double_count += nang; // hj(nang) 
	double_count += nang; // hk(nang)  
	double_count += ng; // vdelt(ng)
	double_count += nang * nx * ny * nz * noct * ng; // ptr_in(nang,nx,ny,nz,noct,ng)
	double_count += nang * nx * ny * nz * noct * ng; // ptr_out(nang,nx,ny,nz,noct,ng)
	double_count += nx * ny * nz * ng; // flux(nx,ny,nz,ng)
	double_count += (cmom-1) * nx * ny * nz * ng; // fluxm(cmom-1,nx,ny,nz,ng) 
	double_count += nang * M; // psi(nang,M)    
	double_count += nang * M; // pc(nang,M)    
	double_count += nang * ichunk * nz * ng; // jb_out(nang,ichunk,nz,ng)
	double_count += nang * ichunk * ny * ng; // kb_out(nang,ichunk,ny,ng)
	double_count += (nx+1) * ny * nz * ng; // flkx(nx+1,ny,nz,ng)
	double_count += nx * (ny+1) * nz * ng; // flky(nx,ny+1,nz,ng)
	double_count += nx * ny * (nz+1) * ng; // flkz(nx,ny,nz+1,ng)
	double_count += nang * 4 * M; // hv(nang,4,M)  
	double_count += nang * 4 * M; // fxhv(nang,4,M)  
	double_count += nang * nx * ny * nz * ng; // dinv(nang,nx,ny,nz,ng)
	double_count += nang * M; // den(nang,M)
	double_count += nx * ny * nz * ng; // t_xs(nx,ny,nz,ng)
	
	cout << "count_memory(): " << double_count << " doubles will be allocated for ";
	cout << double_count * 8 << " bytes ( " << (double_count * 8) / (1024 * 1024);
	cout << " MB ) " << endl;                    
} 
template< typename view_1d_t >
void flatten_diag(  const int ndiag, const int ndiag_entries, const vector<diag_c>& diag,
                    view_1d_t diag_start, view_1d_t diag_len,
                    view_1d_t diag_ic, view_1d_t diag_j, view_1d_t diag_k ){
/*
  cout << "flatten_diag():" << endl;
  cout << "STL" << endl;
  for (int dd = 0; dd < diag.size(); dd++) {
    for (int nn = 0; nn < diag[dd].len; nn++) {
      cout << " dd " << dd;
      cout << " nn " << nn;
      cout << " ic " << diag[dd].cell_id[nn].ic;
      cout << " j " << diag[dd].cell_id[nn].j;
      cout << " k " << diag[dd].cell_id[nn].k;
      cout << endl;
    }
  }
*/
  int start_counter = 0;
  for (int dd = 0; dd < diag.size(); dd++) {
    diag_start(dd) = start_counter;
    diag_len(dd) = diag[dd].len;
    for (int nn = 0; nn < diag[dd].len; nn++) {
      diag_ic(start_counter) = diag[dd].cell_id[nn].ic;
      diag_j(start_counter) = diag[dd].cell_id[nn].j;
      diag_k(start_counter) = diag[dd].cell_id[nn].k;
      start_counter++;   
    }
  }

/*
  cout << "FLATTENED" << endl;
  for (int dd = 0; dd < ndiag; dd++) {
    for (int nn = 0; nn < diag_len[dd]; nn++) {
      int idx = diag_start[dd] + nn;
      cout << " dd " << dd;
      cout << " nn " << nn;
      cout << " ic " << diag_ic[idx];
      cout << " j " << diag_j[idx];
      cout << " k " << diag_k[idx];
      cout << endl;
    }
  }
*/
}
template< typename policy_type,
          typename view_1d_t, typename view_2d_t, typename view_3d_t, typename view_4d_t,
          typename view_5d_t, typename view_6d_t, typename view_7d_t >
struct dim3_sweep2 {
  typedef typename view_1d_t::execution_space device_t;
//   typedef View<double*,Kokkos::LayoutLeft,typename device_t::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > view_1d_scratch_t;
//   typedef View<double**,Kokkos::LayoutLeft,typename device_t::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > view_2d_scratch_t;
//   typedef View<double***,Kokkos::LayoutLeft,typename device_t::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > view_3d_scratch_t;


  typedef policy_type local_policy_t;
  typedef typename local_policy_t::member_type team_member_t; 
    
  const int M;
  const int L;
  const int ng;
  const int cmom;
  const int noct;
  const int nx;
  const int ny;
  const int nz;
  const int ichunk;
//  const vector<diag_c> diag;
  const int ndiag;
  const int ndiag_entries;
  const int id;
  const int ich;
  const int oct;
  const int jlo;
  const int jhi;
  const int jst;
  const int jd;
  const int klo;
  const int khi;
  const int kst;
  const int kd;
  const double hi;
  
  view_1d_t diag_start; // diag_start(ndiag)                RO
  view_1d_t diag_len;   // diag_len(ndiag)                  RO
  view_1d_t diag_ic;    // diag_ic(ndiag_entries)           RO
  view_1d_t diag_j;     // diag_j(ndiag_entries)            RO
  view_1d_t diag_k;     // diag_k(ndiag_entries)            RO
  view_4d_t psii;       // psii(nang,ny,nz,ng)              RW
  view_4d_t psij;       // psij(nang,ichunk,nz,ng)          RW
  view_4d_t psik;       // psik(nang,ichunk,ny,ng)          RW
  view_4d_t jb_in;      // jb_in(nang,ichunk,nz,ng)         RO
  view_4d_t kb_in;      // kb_in(nang,ichunk,ny,ng)         RO
  view_6d_t qim;        // qim(nang,nx,ny,nz,noct,ng)       RO  
  view_5d_t qtot;       // qtot(cmom,nx,ny,nz,ng)           RO
  view_2d_t ec;         // ec(nang,cmom)                    RO
  view_1d_t mu;         // mu(nang)                         RO
  view_1d_t w;          // w(nang)                          RO
  view_1d_t wmu;        // wmu(nang)                        RO
  view_1d_t weta;       // weta(nang)                       RO
  view_1d_t wxi;        // wxi(nang)                        RO
  view_1d_t hj;         // hj(nang)                         RO
  view_1d_t hk;         // hk(nang)                         RO
  view_1d_t vdelt;      // vdelt(ng)                        RO
  view_6d_t ptr_in;     // ptr_in(nang,nx,ny,nz,noct,ng)    RO
  view_6d_t ptr_out;    // ptr_out(nang,nx,ny,nz,noct,ng)    RO
  view_4d_t flux;       // flux(nx,ny,nz,ng)                RW
  view_5d_t fluxm;      // fluxm(cmom-1,nx,ny,nz,ng)        RW
  view_2d_t psi;        // psi(nang,M)                      RW
  view_2d_t pc;         // pc(nang,M)                       RW
  view_4d_t jb_out;     // jb_out(nang,ichunk,nz,ng)        RW
  view_4d_t kb_out;     // kb_out(nang,ichunk,ny,ng)        RW
  view_4d_t flkx;       // flkx(nx+1,ny,nz,ng)              RW
  view_4d_t flky;       // flky(nx,ny+1,nz,ng)              RW
  view_4d_t flkz;       // flkz(nx,ny,nz+1,ng)              RW
  view_3d_t hv;         // hv(nang,4,M)                     RW
  view_3d_t fxhv;       // fxhv(nang,4,M)                   RW
  view_5d_t dinv;       // dinv(nang,nx,ny,nz,ng)           RW
  view_2d_t den;        // den(nang,M)                      RW
  view_4d_t t_xs;       // t_xs(nx,ny,nz,ng)                RW
  
  dim3_sweep2(  const int M_, const int L_,
                const int ng_, const int cmom_, const int noct_,
                const int nx_, const int ny_, const int nz_, const int ichunk_,
                const int ndiag_, const int ndiag_entries_,
                const int id_, const int ich_, const int oct_,
                const int jlo_, const int jhi_, const int jst_, const int jd_,
                const int klo_, const int khi_, const int kst_, const int kd_,
                view_1d_t diag_start_, view_1d_t diag_len_,
                view_1d_t diag_ic_, view_1d_t diag_j_, view_1d_t diag_k_,
                view_4d_t psii_, view_4d_t psij_, view_4d_t psik_,
                view_4d_t jb_in_, view_4d_t kb_in_,
                view_6d_t qim_, view_5d_t qtot_, view_2d_t ec_,
                view_1d_t mu_, view_1d_t w_,
                view_1d_t wmu_, view_1d_t weta_, view_1d_t wxi_,
                const double hi_, view_1d_t hj_, view_1d_t hk_,
                view_1d_t vdelt_, view_6d_t ptr_in_, view_6d_t ptr_out_,
                view_4d_t flux_, view_5d_t fluxm_, view_2d_t psi_, view_2d_t pc_,
                view_4d_t jb_out_, view_4d_t kb_out_,
                view_4d_t flkx_, view_4d_t flky_, view_4d_t flkz_,
                view_3d_t hv_, view_3d_t fxhv_, view_5d_t dinv_,
                view_2d_t den_, view_4d_t t_xs_ ):
                M(M_), L(L_),
                ng(ng_), cmom(cmom_), noct(noct_),
                nx(nx_), ny(ny_), nz(nz_), ichunk(ichunk_),
                ndiag(ndiag_), ndiag_entries(ndiag_entries_),
                id(id_), ich(ich_), oct(oct_),
                jlo(jlo_), jhi(jhi_), jst(jst_), jd(jd_),
                klo(klo_), khi(khi_), kst(kst_), kd(kd_),
                diag_start(diag_start_), diag_len(diag_len_),
                diag_ic(diag_ic_), diag_j(diag_j_), diag_k(diag_k_),
                psii(psii_), psij(psij_), psik(psik_),
                jb_in(jb_in_), kb_in(kb_in_),
                qim(qim_), qtot(qtot_), ec(ec_),
                mu(mu_), w(w_),
                wmu(wmu_), weta(weta_), wxi(wxi_),
                hi(hi_), hj(hj_), hk(hk_),
                vdelt(vdelt_), ptr_in(ptr_in_), ptr_out(ptr_out_),
                flux(flux_), fluxm(fluxm_), psi(psi_), pc(pc_),
                jb_out(jb_out_), kb_out(kb_out_),
                flkx(flkx_), flky(flky_), flkz(flkz_),
                hv(hv_), fxhv(fxhv_), dinv(dinv_),
                den(den_), t_xs(t_xs_){}
  
//   size_t team_shmem_size( int team_size ) const {
//     return sizeof(double)*(L*M*12 + 7*L);
//   }

  KOKKOS_INLINE_FUNCTION
  void operator() ( const team_member_t& team_member) const {

//     view_2d_scratch_t psi(team_member.team_shmem(),L,M);
//     view_2d_scratch_t ec(team_member.team_shmem(),L,M);
//     view_2d_scratch_t pc(team_member.team_shmem(),L,M);
//     view_2d_scratch_t den(team_member.team_shmem(),L,M);
// 
//     view_1d_scratch_t mu(team_member.team_shmem(),L);
//     view_1d_scratch_t hj(team_member.team_shmem(),L);
//     view_1d_scratch_t hk(team_member.team_shmem(),L);
//     view_1d_scratch_t w(team_member.team_shmem(),L);
//     view_1d_scratch_t wmu(team_member.team_shmem(),L);
//     view_1d_scratch_t weta(team_member.team_shmem(),L);
//     view_1d_scratch_t wxi(team_member.team_shmem(),L);
// 
//     view_3d_scratch_t fxhv(team_member.team_shmem(),L,4,M);
//     view_3d_scratch_t hv(team_member.team_shmem(),L,4,M);

    //LxMx12 + 7*L

    const int nn = team_member.league_rank();
    const int g = nn;
//     const int nang = L; // gw comment out to suppress compiler warning
       
//     cout << " current group: " << g << endl;
    
    int jst = 1;
    int kst = 1;
  
//////////////////////////////////////////////////////////////////////////////////////////
// A - set up the sweep order in the i-direction
//////////////////////////////////////////////////////////////////////////////////////////
{
//     ist = -1
//     IF ( id == 2 ) ist = 1
}
    int ist = -1;
    if ( id == 2 ) ist = 1;

    
//////////////////////////////////////////////////////////////////////////////////////////
// B - zero out the outgoing boundary arrays and fixup array
//////////////////////////////////////////////////////////////////////////////////////////
{
//     jb_out = zero
//     kb_out = zero
//     fxhv = zero
}
// TODO THINK ABOUT THIS, DOES IT LIVE HERE?
//     deep_copy(jb_out, c0);
//     deep_copy(kb_out, c0);
//     deep_copy(fxhv, c0);
    
    parallel_for( Kokkos::TeamThreadRange( team_member, M ), KOKKOS_LAMBDA( const int& mm ) { // diag loop
    
      int i;  // current x location, since each thread handles a different diag, ok to live here
              // i updated each line in the diag
      double sum_w_psi;
      double sum_ec_w_psi;
      double sum_wmu_psii_j_k;
      double sum_weta_psij_ic_k;
      double sum_wxi_psik_ic_j;
      double sum_hv;
      double temp_sum_hv;
      double fmin; // dummy, min scalar flux
      double fmax; // dummy, max scalar flux
      
//////////////////////////////////////////////////////////////////////////////////////////
// C- loop over cells along the diagonals. When only 1 diagonal, it's
// normal sweep order. Otherwise, nested threading performs mini-KBA.
//////////////////////////////////////////////////////////////////////////////////////////
{
//     diagonal_loop: DO d = 1, ndiag
// 
//       line_loop: DO n = 1, diag(d)%len
}
//       cout << " current diag: " << mm << ", len: " << diag[mm].len << endl;
//      for (int dd = 0; dd < diag[mm].len; dd++) { // line loop
      for (int dd = 0; dd < diag_len(mm); dd++) { // line loop
{
//         ic = diag(d)%cell_id(n)%ic
}
//        const int ic = diag[mm].cell_id[dd].ic;
        int ic_idx = diag_start(mm) + dd;
        const int ic = diag_ic( ic_idx );
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
//         cout << " i " << i << " nx " << nx << endl;
      
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
//        int j = diag[mm].cell_id[dd].j;
        int j = diag_j( ic_idx );
        if ( jst < 0 ) j = ny - j + 1;
    
//        int k = diag[mm].cell_id[dd].k;
        int k = diag_k( ic_idx );
        if ( kst < 0 ) k = nz - k + 1;
 
//   printf("C\n");
        
//         cout << "i " << i << " j " << j << " k " << k << endl;

          
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
        parallel_for( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll ) {
          if ( ( i == nx ) && ( ist == -1 ) ) {
            psii(ll,j,k,g) = c0;
          } else if ( ( i == 1 ) && ( ist == 1 ) ) {
            switch ( ibl ) {
              case 0:            
                psii(ll,j,k,g) = c0;
                break;
              case 1:
                psii(ll,j,k,g) = c0;
                break;          
            }
          }
        }); // ThreadVectorRange D

//   printf("D\n");
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
        parallel_for( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll ) {
          if ( j == jlo ) {
            if ( ( jd == 1 ) && ( lasty ) ) {
              psij(ll,ic,k,g) = c0;
            } else if ( ( jd == 2 ) && ( firsty ) ) {
              switch ( ibb ) {
                case 0:
                  psij(ll,ic,k,g) = c0;
                  break;
                case 1:
                  psij(ll,ic,k,g) = c0;
                  break;
              }
            } else {
              psij(ll,ic,k,g) = jb_in(ll,ic,k,g);
            }
          } // end if ( j == jlo )
        }); // ThreadVectorRange E

//   printf("E\n");
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
        parallel_for( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll ) {
          if ( k == klo ) {
            if ( ( ( kd == 1 ) && ( lastz ) ) || ( ndimen < 3 ) ) {
              psik(ll,ic,j,g) = c0;
            } else if ( ( kd == 2 ) && ( firstz ) ) {
              switch ( ibf ) {
                case 0:
                  psik(ll,ic,j,g) = c0;
                  break;
                case 1:
                  psik(ll,ic,j,g) = c0;
                  break;
              }
            } else {
              psik(ll,ic,j,g) = kb_in(ll,ic,j,g);
            }
          } // end if ( k == klo )
        }); // ThreadVectorRange F

//   printf("F\n");
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
        parallel_for( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll ) {
          psi(ll,mm) = qtot(0,i,j,k,g);
          if ( src_opt == 3 ) psi(ll,mm) = psi(ll,mm) + qim(ll,i,j,k,oct,g);

          for (int l = 1; l < cmom; l++) {
            psi(ll,mm) = psi(ll,mm) + ec(ll,l)*qtot(l,i,j,k,g);
          }
        }); // ThreadVectorRange G

//   printf("G\n");
//////////////////////////////////////////////////////////////////////////////////////////
// H - compute the numerator for the update formula
//////////////////////////////////////////////////////////////////////////////////////////
{
//         pc = psi + psii(:,j,k)*mu*hi + psij(:,ic,k)*hj + psik(:,ic,j)*hk // pc(nang)
//         IF ( vdelt /= zero ) pc = pc + vdelt*ptr_in(:,i,j,k)
}
        parallel_for( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll ) {
          pc(ll,mm) = psi(ll,mm) + psii(ll,j,k,g)*mu(ll)*hi + psij(ll,ic,k,g)*hj(ll) + psik(ll,ic,j,g)*hk(ll);
          if ( vdelt(g) != c0 ) { // fp not equal is bad practice always, should be checking for a tolerance
            pc(ll,mm) = pc(ll,mm) + vdelt(g) * ptr_in(ll,i,j,k,oct,g);
          }
        }); // ThreadVectorRange H
        
//   printf("H\n");
//////////////////////////////////////////////////////////////////////////////////////////
// I - compute the solution of the center. Use DD for edges. Use fixup if requested.
//////////////////////////////////////////////////////////////////////////////////////////
{
//         IF ( fixup == 0 ) THEN
}
         if ( !FIXUP ) {

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
          parallel_for( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll ) {
            psi(ll,mm) = pc(ll,mm) * dinv(ll,i,j,k,g);
            psii(ll,j,k,g) = c2 * psi(ll,mm) - psii(ll,j,k,g);
            psij(ll,ic,k,g) = c2 * psi(ll,mm) - psij(ll,ic,k,g);
            if ( ndimen == 3 ) psik(ll,ic,j,g) = c2 * psi(ll,mm) - psik(ll,ic,j,g);
            if ( vdelt(g) != c0 ) ptr_out(ll,i,j,k,oct,g) = c2 * psi(ll,mm) - ptr_in(ll,i,j,k,oct,g);
          });

{
//         ELSE
}
//   printf("I\n");
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
          parallel_for( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll ) {
            hv(ll,0,mm) = c0;
            hv(ll,1,mm) = c0;
            hv(ll,2,mm) = c0;
            hv(ll,3,mm) = c0;
          });      
          parallel_reduce( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll, double& v_sum_hv ) {
            v_sum_hv += ( hv(ll,0,mm) + hv(ll,1,mm) + hv(ll,2,mm) + hv(ll,3,mm) );
          }, sum_hv);
          parallel_for( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll ) {
            pc(ll,mm) = pc(ll,mm) * dinv(ll,i,j,k,g);
          });
          
          bool fixup_condition = true;
          while ( fixup_condition == true ) {
            
            parallel_for( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll ) {
            
              fxhv(ll,0,mm) = c2 * pc(ll,mm) - psii(ll,j,k,g);
              fxhv(ll,1,mm) = c2 * pc(ll,mm) - psij(ll,ic,k,g);
              if ( ndimen == 3 ) fxhv(ll,2,mm) = c2 * pc(ll,mm) - psik(ll,ic,j,g);
              if ( vdelt(g) != c0 ) fxhv(ll,3,mm) = c2 * pc(ll,mm) - ptr_in(ll,i,j,k,oct,g);
          
              if ( fxhv(ll,0,mm) < c0 ) hv(ll,0,mm) = c0;
              if ( fxhv(ll,1,mm) < c0 ) hv(ll,1,mm) = c0;
              if ( fxhv(ll,2,mm) < c0 ) hv(ll,2,mm) = c0;
              if ( fxhv(ll,3,mm) < c0 ) hv(ll,3,mm) = c0;
              
            });
            
//   printf("J\n");
//////////////////////////////////////////////////////////////////////////////////////////
// K - exit loop when all angles are fixed up
//////////////////////////////////////////////////////////////////////////////////////////
{
//             IF ( sum_hv == SUM( hv ) ) EXIT fixup_loop
//             sum_hv = SUM( hv )
}
              parallel_reduce( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll, double& v_temp_sum_hv ) {
                v_temp_sum_hv += ( hv(ll,0,mm) + hv(ll,1,mm) + hv(ll,2,mm) + hv(ll,3,mm) );
              }, temp_sum_hv);
              if (sum_hv == temp_sum_hv) break;
              sum_hv = temp_sum_hv;
              
//   printf("K\n");
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
              parallel_for( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll ) {
                
                pc(ll,mm) = psii(ll,j,k,g)*mu(ll)*hi*(c1+hv(ll,0,mm)) + psij(ll,ic,k,g)*hj(ll)*(c1+hv(ll,1,mm)) + psik(ll,ic,j,g)*hk(ll)*(c1+hv(ll,2,mm));
                if ( vdelt(g) != c0 ) pc(ll,mm) = pc(ll,mm) + vdelt(g)*ptr_in(ll,i,j,k,oct,g)*(c1+hv(ll,3,mm));
                
                den(ll,mm) = t_xs(i,j,k,g) + mu(ll)*hi*hv(ll,0,mm) + hj(ll)*hv(ll,1,mm) + hk(ll)*hv(ll,2,mm) + vdelt(g)*hv(ll,3,mm);
              
                ( den(ll,mm) > tolr ) ? ( pc(ll,mm) = pc(ll,mm) / den(ll,mm) ) : ( pc(ll,mm) = c0 ) ;
                
              });
                            
{
//           END DO fixup_loop
}
          }
//   printf("L\n");
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
          parallel_for( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll ) {
            psi(ll,mm) = pc(ll,mm);
            psii(ll,j,k,g) = fxhv(ll,0,mm) * hv(ll,0,mm);
            psij(ll,ic,k,g) = fxhv(ll,1,mm) * hv(ll,1,mm);
            if ( ndimen == 3 ) psik(ll,ic,j,g) = fxhv(ll,2,mm) * hv(ll,2,mm);
            if ( vdelt(g) != c0 ) ptr_out(ll,i,j,k,oct,g) = fxhv(ll,3,mm) * hv(ll,3,mm);
          });
          
{
//         END IF
}
//   printf("M\n");
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
          flux(i,j,k,g) = c0;
          for (int l = 0; l < cmom-1; l++) {
            fluxm(l,i,j,k,g) = c0;
          } 
        }

//   printf("N\n");
//////////////////////////////////////////////////////////////////////////////////////////
// O - compute the flux moments
//////////////////////////////////////////////////////////////////////////////////////////
{
//         flux(i,j,k) = flux(i,j,k) + SUM( w*psi )
//         DO l = 1, cmom-1
//           fluxm(l,i,j,k) = fluxm(l,i,j,k) + SUM( ec(:,l+1)*w*psi )
//         END DO
}
        parallel_reduce( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll, double& v_sum_w_psi ) {
          v_sum_w_psi += w(ll) * psi(ll,mm);
        }, sum_w_psi);
        flux(i,j,k,g) = flux(i,j,k,g) + sum_w_psi;
        for (int l = 0; l < cmom-1; l++) {
          parallel_reduce( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll, double& v_sum_ec_w_psi ) {
            v_sum_ec_w_psi += ec(ll,l+1) * w(ll) * psi(ll,mm);
          }, sum_ec_w_psi);
          fluxm(l,i,j,k,g) = fluxm(l,i,j,k,g) + sum_ec_w_psi;
        }

//   printf("O\n");
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
//          fmin = std::min( fmin, flux(i,j,k,g) );
//          fmax = std::max( fmax, flux(i,j,k,g) );
          fmin = ( fmin < flux(i,j,k,g) ) ? fmin : flux(i,j,k,g);
          fmax = ( fmax > flux(i,j,k,g) ) ? fmax : flux(i,j,k,g);
        }
        
//   printf("P\n");
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
        parallel_for( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll ) {
        
          if ( j == jhi ) {
            if ( ( jd == 2 ) && ( lasty ) ) {
            } else if ( ( jd == 1 ) && ( firsty ) ) {
              if ( ibb == 1 ) {}
            } else {
              jb_out(ll,ic,k,g) += psij(ll,ic,k,g);
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
          if ( k == khi ) {
            if ( ( kd == 2 ) && ( lastz ) ) {
            } else if ( ( kd == 1 ) && ( firstz ) ) {
              if ( ibf == 1 ) {}
            } else {
              kb_out(ll,ic,j,g) += psik(ll,ic,j,g);
            }
          }
        }); // end parallel_for Q
        
//   printf("Q\n");

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
          parallel_reduce( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll, double& v_sum_wmu_psii_j_k ) {
            v_sum_wmu_psii_j_k += wmu(ll) * psii(ll,j,k,g);
          }, sum_wmu_psii_j_k);
          flkx(i+id-1,j,k,g) = flkx(i+id-1,j,k,g) + ist * sum_wmu_psii_j_k;
        }
      
        if ( ( ( jd == 1 ) && ( firsty ) ) || ( ( jd == 2 ) && ( lasty ) ) ) {
          parallel_reduce( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll, double& v_sum_weta_psij_ic_k ) {
            v_sum_weta_psij_ic_k += weta(ll) * psij(ll,ic,k,g);
          }, sum_weta_psij_ic_k);
          flky(i,j+jd-1,k,g) = flky(i,j+jd-1,k,g) + jst * sum_weta_psij_ic_k;
        }
      
        if ( ( ( ( kd == 1 ) && ( firstz ) ) || ( ( kd == 2 ) && ( lastz ) ) ) && ( ndimen == 3 ) ) {
          parallel_reduce( Kokkos::ThreadVectorRange( team_member, L ), [&]( const int& ll, double& v_sum_wxi_psik_ic_j ) {
            v_sum_wxi_psik_ic_j += wxi(ll) * psik(ll,ic,j,g);
          }, sum_wxi_psik_ic_j);
          flkz(i,j,k+kd-1,g) = flkz(i,j,k+kd-1,g) + kst * sum_wxi_psik_ic_j;
        }

//   printf("R\n");

//////////////////////////////////////////////////////////////////////////////////////////
// S - finish the loops
//////////////////////////////////////////////////////////////////////////////////////////
{
//      END DO line_loop
// 
//    END DO diagonal_loop
}
      } // end line loop


    }); // TeamThreadRange
  
  } // end () overload (diagonal loop)
    
};

} // namespace SNAPPY


#endif //SNAPPY_HPP

