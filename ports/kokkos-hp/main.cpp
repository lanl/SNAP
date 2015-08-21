#include <time.h>
#include <iostream>

#include <Kokkos_Core.hpp>
#include <Kokkos_Serial.hpp>
#include <Kokkos_Threads.hpp>

#ifdef KOKKOS_HAVE_OPENMP
#include <Kokkos_OpenMP.hpp>
#endif

#include <Kokkos_hwloc.hpp>

#include <snappy.hpp>

const int n_test_iter = 1;
time_t timer_start;
time_t timer_end;

namespace SNAPPY {

// void run_serial(  const int N, const int M, const int L, const int hyper_threads, const int vector_lanes,
//                   const int nx, const int ny, const int nz, const int ichunk,
//                   const int nang, const int noct, const int ng, const int nmom, const int cmom,
//                   const vector<diag_c>& diag )
// {
//   
// 	typedef Kokkos::Serial device_t;
// 	typedef TeamPolicy<device_t> team_policy_t;
// 	typedef View<double*, device_t> view_1d_t;
// 	typedef View<double**, device_t> view_2d_t;
// 	typedef View<double***, device_t> view_3d_t;
// 	typedef View<double****, device_t> view_4d_t;
// 	typedef View<double*****, device_t> view_5d_t;
// 	typedef View<double******, device_t> view_6d_t;
// 	typedef View<double*******, device_t> view_7d_t;
// 
// 	int id = 1;
// 	int ich = 1;
// 	int jlo = 1;
// 	int jhi = ny-1;
// 	int jst = 1;
// 	int jd = 2;
// 	int klo = 1;
// 	int khi = nz-1;
// 	int kst = 1;
// 	int kd = 2;
// 	double hi = c1;
// 	
// 	view_4d_t psii( "psii", nang, ny, nz, ng );
// 	view_4d_t psij( "psij", nang, ichunk, nz, ng );
// 	view_4d_t psik( "psik", nang, ichunk, ny, ng );
// 	view_4d_t jb_in( "jb_in", nang, ichunk, ny, ng ); // jb_in(nang,ichunk,nz,ng)
// 	view_4d_t kb_in( "kb_in", nang, ichunk, ny, ng ); // kb_in(nang,ichunk,nz,ng)
// 	view_6d_t qim( "qim", nang, nx, ny, nz, noct, ng ); // qim(nang,nx,ny,nz,noct,ng)
//   view_5d_t qtot( "qtot", cmom, nx, ny, nx, ng ); // qtot(cmom,nx,ny,nz,ng)
// 	view_2d_t ec( "ec", nang, cmom ); // ec(nang,cmom)
// 	view_1d_t mu( "mu", nang ); // mu(nang)
// 	view_1d_t w( "w", nang ); // w(nang)
// 	view_1d_t wmu( "wmu", nang ); // wmu(nang)
// 	view_1d_t weta( "weta", nang ); // weta(nang)
// 	view_1d_t wxi( "wxi", nang ); // wxi(nang)
// 	view_1d_t hj( "hj", nang ); // hj(nang)
// 	view_1d_t hk( "hk", nang ); // hk(nang)
// 	view_1d_t vdelt( "vdelt", ng ); // vdelt(ng)
// 	view_6d_t ptr_in( "ptr_in", nang, nx, ny, nz, noct, ng ); // ptr_in(nang,nx,ny,nz,noct,ng)
// 	view_6d_t ptr_out( "ptr_out", nang, nx, ny, nz, noct, ng ); // ptr_out(nang,nx,ny,nz,noct,ng)
// 	view_4d_t flux( "flux", nx, ny, nz, ng ); // flux(nx,ny,nz,ng)
// 	view_5d_t fluxm( "fluxm", cmom-1, nx, ny, nz, ng ); //fluxm(cmom-1,nx,ny,nz,ng)
// 	view_2d_t psi( "psi", nang, M );
// 	view_2d_t pc( "pc", nang, M );
// 	view_4d_t jb_out( "jb_out", nang, ichunk, nz, ng );
// 	view_4d_t kb_out( "kb_out", nang, ichunk, ny, ng );
// 	view_4d_t flkx( "flkx", nx+1, ny, nz, ng );
// 	view_4d_t flky( "flky", nx, ny+1, nz, ng );
// 	view_4d_t flkz( "flkz", nx, ny, nz+1, ng );
//   view_3d_t hv( "hv", nang, 4, M ); // hv(nang,4,M)
//   view_3d_t fxhv( "fxhv", nang, 4, M ); // fxhv(nang,4,M)
//   view_5d_t dinv( "dinv", nang, nx, ny, nz, ng ); // dinv(nang,nx,ny,nz,ng)
//   view_2d_t den( "den", nang, M ); // den(nang,M)
//   view_4d_t t_xs( "t_xs", nx, ny, nz, ng ); // t_xs(nx,ny,nz,ng)
//     	
//   const team_policy_t policy( N, hyper_threads, vector_lanes );
//   
//   for (int ii = 0; ii < n_test_iter; ii++) {
//     time(&timer_start);
//     
//     for (int oct = 0; oct < noct; oct++) {
//       parallel_for( policy, dim3_sweep2<  team_policy_t,
//                                           view_1d_t, view_2d_t, view_3d_t, view_4d_t,
//                                           view_5d_t, view_6d_t, view_7d_t >
//                                         ( M, L,
//                                           ng, cmom, noct,
//                                           nx, ny, nz, ichunk,
//                                           diag,
//                                           id, ich, oct,
//                                           jlo, jhi, jst, jd,
//                                           klo, khi, kst, kd,
//                                           psii, psij, psik,
//                                           jb_in, kb_in,
//                                           qim, qtot, ec,
//                                           mu, w,
//                                           wmu, weta, wxi,
//                                           hi, hj, hk,
//                                           vdelt, ptr_in, ptr_out,
//                                           flux, fluxm, psi, pc,
//                                           jb_out, kb_out,
//                                           flkx, flky, flkz,
//                                           hv, fxhv, dinv,
//                                           den, t_xs ) );
//     }// end noct
//     
//     time(&timer_end);
//     std::cout << " ii " << ii << " elapsed time " << difftime(timer_end, timer_start) << std::endl;
//   } // end n_test_iter
// 	
// }

// void run_openmp(  const int N, const int M, const int L, const int hyper_threads, const int vector_lanes,
//                   const int nx, const int ny, const int nz, const int ichunk,
//                   const int nang, const int noct, const int ng, const int nmom, const int cmom,
//                   const vector<diag_c>& diag )
// {
//   #ifdef KOKKOS_HAVE_OPENMP
//   
// 	typedef Kokkos::OpenMP device_t;
// 	typedef TeamPolicy<device_t> team_policy_t;
// 	typedef View<double*, device_t> view_1d_t;
// 	typedef View<double**, device_t> view_2d_t;
// 	typedef View<double***, device_t> view_3d_t;
// 	typedef View<double****, device_t> view_4d_t;
// 	typedef View<double*****, device_t> view_5d_t;
// 	typedef View<double******, device_t> view_6d_t;
// 	typedef View<double*******, device_t> view_7d_t;
// 
// 	int id = 1;
// 	int ich = 1;
// 	int jlo = 0;
// 	int jhi = ny-1;
// 	int jst = 1;
// 	int jd = 2;
// 	int klo = 0;
// 	int khi = nz-1;
// 	int kst = 1;
// 	int kd = 2;
// 	double hi = c1;
// 		
// 	Kokkos::OpenMP::initialize();
// 	
// 	view_4d_t psii( "psii", nang, ny, nz, ng );
// 	view_4d_t psij( "psij", nang, ichunk, nz, ng );
// 	view_4d_t psik( "psik", nang, ichunk, ny, ng );
// 	view_4d_t jb_in( "jb_in", nang, ichunk, ny, ng ); // jb_in(nang,ichunk,nz,ng)
// 	view_4d_t kb_in( "kb_in", nang, ichunk, ny, ng ); // kb_in(nang,ichunk,nz,ng)
// 	view_6d_t qim( "qim", nang, nx, ny, nz, noct, ng ); // qim(nang,nx,ny,nz,noct,ng)
//   view_5d_t qtot( "qtot", cmom, nx, ny, nx, ng ); // qtot(cmom,nx,ny,nz,ng)
// 	view_2d_t ec( "ec", nang, cmom ); // ec(nang,cmom)
// 	view_1d_t mu( "mu", nang ); // mu(nang)
// 	view_1d_t w( "w", nang ); // w(nang)
// 	view_1d_t wmu( "wmu", nang ); // wmu(nang)
// 	view_1d_t weta( "weta", nang ); // weta(nang)
// 	view_1d_t wxi( "wxi", nang ); // wxi(nang)
// 	view_1d_t hj( "hj", nang ); // hj(nang)
// 	view_1d_t hk( "hk", nang ); // hk(nang)
// 	view_1d_t vdelt( "vdelt", ng ); // vdelt(ng)
// 	view_6d_t ptr_in( "ptr_in", nang, nx, ny, nz, noct, ng ); // ptr_in(nang,nx,ny,nz,noct,ng)
// 	view_6d_t ptr_out( "ptr_out", nang, nx, ny, nz, noct, ng ); // ptr_out(nang,nx,ny,nz,noct,ng)
// 	view_4d_t flux( "flux", nx, ny, nz, ng ); // flux(nx,ny,nz,ng)
// 	view_5d_t fluxm( "fluxm", cmom-1, nx, ny, nz, ng ); //fluxm(cmom-1,nx,ny,nz,ng)
// 	view_2d_t psi( "psi", nang, M );
// 	view_2d_t pc( "pc", nang, M );
// 	view_4d_t jb_out( "jb_out", nang, ichunk, nz, ng );
// 	view_4d_t kb_out( "kb_out", nang, ichunk, ny, ng );
// 	view_4d_t flkx( "flkx", nx+1, ny, nz, ng );
// 	view_4d_t flky( "flky", nx, ny+1, nz, ng );
// 	view_4d_t flkz( "flkz", nx, ny, nz+1, ng );
//   view_3d_t hv( "hv", nang, 4, M ); // hv(nang,4,M)
//   view_3d_t fxhv( "fxhv", nang, 4, M ); // fxhv(nang,4,M)
//   view_5d_t dinv( "dinv", nang, nx, ny, nz, ng ); // dinv(nang,nx,ny,nz,ng)
//   view_2d_t den( "den", nang, M ); // den(nang,M)
//   view_4d_t t_xs( "t_xs", nx, ny, nz, ng ); // t_xs(nx,ny,nz,ng)
//    	
//   const team_policy_t policy( N, hyper_threads, vector_lanes );
// 
//   for (int ii = 0; ii < n_test_iter; ii++) {
//     time(&timer_start);
//     
//     for (int oct = 0; oct < noct; oct++) {
//       parallel_for( policy, dim3_sweep2<  team_policy_t,
//                                           view_1d_t, view_2d_t, view_3d_t, view_4d_t,
//                                           view_5d_t, view_6d_t, view_7d_t >
//                                         ( M, L,
//                                           ng, cmom, noct,
//                                           nx, ny, nz, ichunk,
//                                           diag,
//                                           id, ich, oct,
//                                           jlo, jhi, jst, jd,
//                                           klo, khi, kst, kd,
//                                           psii, psij, psik,
//                                           jb_in, kb_in,
//                                           qim, qtot, ec,
//                                           mu, w,
//                                           wmu, weta, wxi,
//                                           hi, hj, hk,
//                                           vdelt, ptr_in, ptr_out,
//                                           flux, fluxm, psi, pc,
//                                           jb_out, kb_out,
//                                           flkx, flky, flkz,
//                                           hv, fxhv, dinv,
//                                           den, t_xs ) );
//     }// end noct
//     
//     time(&timer_end);
//     std::cout << " ii " << ii << " elapsed time " << difftime(timer_end, timer_start) << std::endl;
//   } // end n_test_iter
//   
// 	Kokkos::OpenMP::finalize();
// 
//   #else
//     return 0;
//   #endif
// }

void run_threads( const int N, const int M, const int L, const int hyper_threads, const int vector_lanes,
                  const int nx, const int ny, const int nz, const int ichunk,
                  const int nang, const int noct, const int ng, const int nmom, const int cmom,
                  const vector<diag_c>& diag )
{
  #ifdef KOKKOS_HAVE_PTHREAD
  
	typedef Kokkos::Threads device_t;
	typedef TeamPolicy<device_t> team_policy_t;
	typedef View<double*, device_t> view_1d_t;
	typedef View<double**, device_t> view_2d_t;
	typedef View<double***, device_t> view_3d_t;
	typedef View<double****, device_t> view_4d_t;
	typedef View<double*****, device_t> view_5d_t;
	typedef View<double******, device_t> view_6d_t;
	typedef View<double*******, device_t> view_7d_t;

	int id = 1;
	int ich = 1;
	int jlo = 0;
	int jhi = ny-1;
	int jst = 1;
	int jd = 2;
	int klo = 0;
	int khi = nz-1;
	int kst = 1;
	int kd = 2;
	double hi = c1;
	
	Kokkos::Threads::initialize(  Kokkos::hwloc::get_available_numa_count() *
                                Kokkos::hwloc::get_available_cores_per_numa() *
                                Kokkos::hwloc::get_available_threads_per_core()   );
	Kokkos::Threads::print_configuration(cout);
	
	view_4d_t psii( "psii", nang, ny, nz, ng );
	view_4d_t psij( "psij", nang, ichunk, nz, ng );
	view_4d_t psik( "psik", nang, ichunk, ny, ng );
	view_4d_t jb_in( "jb_in", nang, ichunk, ny, ng ); // jb_in(nang,ichunk,nz,ng)
	view_4d_t kb_in( "kb_in", nang, ichunk, ny, ng ); // kb_in(nang,ichunk,nz,ng)
	view_6d_t qim( "qim", nang, nx, ny, nz, noct, ng ); // qim(nang,nx,ny,nz,noct,ng)
  view_5d_t qtot( "qtot", cmom, nx, ny, nx, ng ); // qtot(cmom,nx,ny,nz,ng)
	view_2d_t ec( "ec", nang, cmom ); // ec(nang,cmom)
	view_1d_t mu( "mu", nang ); // mu(nang)
	view_1d_t w( "w", nang ); // w(nang)
	view_1d_t wmu( "wmu", nang ); // wmu(nang)
	view_1d_t weta( "weta", nang ); // weta(nang)
	view_1d_t wxi( "wxi", nang ); // wxi(nang)
	view_1d_t hj( "hj", nang ); // hj(nang)
	view_1d_t hk( "hk", nang ); // hk(nang)
	view_1d_t vdelt( "vdelt", ng ); // vdelt(ng)
	view_6d_t ptr_in( "ptr_in", nang, nx, ny, nz, noct, ng ); // ptr_in(nang,nx,ny,nz,noct,ng)
	view_6d_t ptr_out( "ptr_out", nang, nx, ny, nz, noct, ng ); // ptr_out(nang,nx,ny,nz,noct,ng)
	view_4d_t flux( "flux", nx, ny, nz, ng ); // flux(nx,ny,nz,ng)
	view_5d_t fluxm( "fluxm", cmom-1, nx, ny, nz, ng ); //fluxm(cmom-1,nx,ny,nz,ng)
	view_2d_t psi( "psi", nang, M );
	view_2d_t pc( "pc", nang, M );
	view_4d_t jb_out( "jb_out", nang, ichunk, nz, ng );
	view_4d_t kb_out( "kb_out", nang, ichunk, ny, ng );
	view_4d_t flkx( "flkx", nx+1, ny, nz, ng );
	view_4d_t flky( "flky", nx, ny+1, nz, ng );
	view_4d_t flkz( "flkz", nx, ny, nz+1, ng );
  view_3d_t hv( "hv", nang, 4, M ); // hv(nang,4,M)
  view_3d_t fxhv( "fxhv", nang, 4, M ); // fxhv(nang,4,M)
  view_5d_t dinv( "dinv", nang, nx, ny, nz, ng ); // dinv(nang,nx,ny,nz,ng)
  view_2d_t den( "den", nang, M ); // den(nang,M)
  view_4d_t t_xs( "t_xs", nx, ny, nz, ng ); // t_xs(nx,ny,nz,ng)
   	  
  const team_policy_t policy( N, hyper_threads, vector_lanes );
  
  for (int ii = 0; ii < n_test_iter; ii++) {
    time(&timer_start);
    
    for (int oct = 0; oct < noct; oct++) {
      parallel_for( policy, dim3_sweep2<  team_policy_t,
                                          view_1d_t, view_2d_t, view_3d_t, view_4d_t,
                                          view_5d_t, view_6d_t, view_7d_t >
                                        ( M, L,
                                          ng, cmom, noct,
                                          nx, ny, nz, ichunk,
                                          diag,
                                          id, ich, oct,
                                          jlo, jhi, jst, jd,
                                          klo, khi, kst, kd,
                                          psii, psij, psik,
                                          jb_in, kb_in,
                                          qim, qtot, ec,
                                          mu, w,
                                          wmu, weta, wxi,
                                          hi, hj, hk,
                                          vdelt, ptr_in, ptr_out,
                                          flux, fluxm, psi, pc,
                                          jb_out, kb_out,
                                          flkx, flky, flkz,
                                          hv, fxhv, dinv,
                                          den, t_xs ) );
    }// end noct
    
    time(&timer_end);
    std::cout << " ii " << ii << " elapsed time " << difftime(timer_end, timer_start) << std::endl;
  } // end n_test_iter	
	
	Kokkos::Threads::finalize();
	
  #else
    return 0;
  #endif
}

} // namespace SNAPPY

int main(int argc, char *argv[])
{

  int hyper_threads = 2; // 2 haswell, 4 phi, 8 power8
  int vector_lanes = 4; // 4 haswell, 8 phi

  int nx = 32;
  int ny = 32;
  int nz = 32;
  int ichunk = 32;
  int nang = 120 * vector_lanes;
  int noct = 8;
  int ng = 120;
  int nmom = 4;
  int cmom = nmom * nmom;
  
  vector<diag_c> diag;

  SNAPPY::populate_diag( ichunk, ny, nz, diag );

//   cout << "populate_diag():" << endl;
//   for (int dd = 0; dd < diag.size(); dd++) {
//     for (int nn = 0; nn < diag[dd].len; nn++) {
//       cout << " dd " << dd << " nn " << nn;
//       cout << " ic " << diag[dd].cell_id[nn].ic;
//       cout << " j " << diag[dd].cell_id[nn].j;
//       cout << " k " << diag[dd].cell_id[nn].k;
//       cout << endl;
//     }
//   }

   if (Kokkos::hwloc::available()) {
     int threads_count = Kokkos::hwloc::get_available_numa_count() *
                     Kokkos::hwloc::get_available_cores_per_numa() *
                     Kokkos::hwloc::get_available_threads_per_core();
     std::cout << "Kokkos::hwloc::get_available_numa_count(): " << Kokkos::hwloc::get_available_numa_count() << std::endl;
     std::cout << "Kokkos::hwloc::get_available_cores_per_numa(): " << Kokkos::hwloc::get_available_cores_per_numa() << std::endl;
     std::cout << "Kokkos::hwloc::get_available_threads_per_core(): " << Kokkos::hwloc::get_available_threads_per_core() << std::endl;
   }


  int N = ng;		// oversubscribed, work units, divvied out to thread teams
  int M = diag.size();		// divvied among threads in a team?
  int L = nang;		// fastest varying dimension, multiple of vector_lanes

  cout << " N,ng " << N << " M,diag.size " << M << " L,nang " << L << endl;
  
//   cout << " Kokkos::Serial " << endl;
//   SNAPPY::run_serial( N, M, L, hyper_threads, vector_lanes,
//                       nx, ny, nz, ichunk, nang, noct, ng, nmom, cmom,
//                       diag ); 
  
//   #ifdef KOKKOS_HAVE_OPENMP
//   cout << " Kokkos::OpenMP " << endl;
//   SNAPPY::run_openmp( N, M, L, hyper_threads, vector_lanes,
//                       nx, ny, nz, ichunk, nang, noct, ng, nmom, cmom,
//                       diag );  
//   #endif

  #ifdef KOKKOS_HAVE_PTHREAD
  cout << " Kokkos::Threads " << endl;
  SNAPPY::run_threads(  N, M, L, hyper_threads, vector_lanes,
                        nx, ny, nz, ichunk, nang, noct, ng, nmom, cmom,
                        diag );  
  #endif  
  
}
