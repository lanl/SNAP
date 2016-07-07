#include <time.h>
#include <iostream>

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <snappy.hpp>

const int n_test_iter = 1;
time_t timer_start;
time_t timer_end;

namespace SNAPPY {


void run( const int N, const int M, const int L, const int hyper_threads, const int vector_lanes,
                  const int nx, const int ny, const int nz, const int ichunk,
                  const int nang, const int noct, const int ng, const int nmom, const int cmom,
                  const vector<diag_c>& diag, const int ndiag, const int ndiag_entries )
{
  
	typedef typename Kokkos::DefaultExecutionSpace device_t;
	typedef TeamPolicy<device_t> team_policy_t;
	typedef View<double*, device_t> view_1d_t;
	typedef View<double**, Kokkos::LayoutLeft, device_t> view_2d_t;
	typedef View<double***, Kokkos::LayoutLeft, device_t> view_3d_t;
	typedef View<double****, Kokkos::LayoutLeft, device_t> view_4d_t;
	typedef View<double*****, Kokkos::LayoutLeft, device_t> view_5d_t;
	typedef View<double******, Kokkos::LayoutLeft, device_t> view_6d_t;
	typedef View<double*******, Kokkos::LayoutLeft, device_t> view_7d_t;


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
	
	Kokkos::initialize();
	Kokkos::DefaultExecutionSpace::print_configuration(cout);
	
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
  
  view_1d_t diag_start( "diag_start", ndiag );
  view_1d_t diag_len( "diag_len", ndiag );
  view_1d_t diag_ic( "diag_ic", ndiag_entries );
  view_1d_t diag_j( "diag_j", ndiag_entries );
  view_1d_t diag_k( "diag_k", ndiag_entries );
  
   const team_policy_t policy( N, hyper_threads, vector_lanes );
  
  SNAPPY::flatten_diag<view_1d_t>(  ndiag, ndiag_entries, diag,
                                    diag_start, diag_len, diag_ic, diag_j, diag_k );
                    
   for (int ii = 0; ii < n_test_iter; ii++) {
     Kokkos::Impl::Timer timer;
     
     for (int oct = 0; oct < noct; oct++) {
       parallel_for( policy, dim3_sweep2<  team_policy_t,
                                           view_1d_t, view_2d_t, view_3d_t, view_4d_t,
                                           view_5d_t, view_6d_t, view_7d_t >
                                         ( M, L,
                                           ng, cmom, noct,
                                           nx, ny, nz, ichunk,
                                           ndiag, ndiag_entries,
                                           id, ich, oct,
                                           jlo, jhi, jst, jd,
                                           klo, khi, kst, kd,
                                           diag_start, diag_len,
                                           diag_ic, diag_j, diag_k,
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
     
     std::cout << " ii " << ii << " elapsed time " << timer.seconds() << std::endl;
   } // end n_test_iter	
	
	Kokkos::finalize();
}

} // namespace SNAPPY

int main(int argc, char *argv[])
{

  int hyper_threads = 16; // 2 haswell, 4 phi, 8 power8, 16 K40
  int vector_lanes = 32; // 4 haswell, 8 phi, 1 K40, 32 K40

  int nx = 8;
  int ny = 8;
  int nz = 8;
  int ichunk = 8;
  int nang = 480;
  int noct = 8;
  int ng = 120;
  int nmom = 4;
  int cmom = nmom * nmom;
  
  int ndiag;
  int ndiag_entries;
  
  // Read command line arguments
  for(int i=0; i<argc; i++) {
           if( (strcmp(argv[i], "-n") == 0) ) {
      ichunk = nx = ny = nz = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-nang") == 0) {
      nang = atoi(argv[++i]);
    } else if( strcmp(argv[i], "-ng") == 0) {
      ng = atoi(argv[++i]);
    } else if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0)) {
      printf("Nearest Point Options:\n");
      printf("  -num_points (-p)  <int>: number of points (default: 100000)\n");
      printf("  -nrepeat <int>:          number of test invocations (default: 10)\n");
      printf("  -help (-h):              print this message\n");
    }
  }

  vector<diag_c> diag;

  SNAPPY::populate_diag( ichunk, ny, nz, diag, ndiag, ndiag_entries );

   if (Kokkos::hwloc::available()) {
     int threads_count = Kokkos::hwloc::get_available_numa_count() *
                     Kokkos::hwloc::get_available_cores_per_numa() *
                     Kokkos::hwloc::get_available_threads_per_core();
     std::cout << "Kokkos::hwloc::get_available_numa_count(): " << Kokkos::hwloc::get_available_numa_count() << std::endl;
     std::cout << "Kokkos::hwloc::get_available_cores_per_numa(): " << Kokkos::hwloc::get_available_cores_per_numa() << std::endl;
     std::cout << "Kokkos::hwloc::get_available_threads_per_core(): " << Kokkos::hwloc::get_available_threads_per_core() << std::endl;
   }

// for gpu, N = ng/hyper_threads, for cpu, N = ng
  int N = ng/hyper_threads;		// oversubscribed, work units, divvied out to thread teams
  int M = diag.size();		// divvied among threads in a team?
  int L = nang;		// fastest varying dimension, multiple of vector_lanes

  cout << " N,ng " << N << " M,diag.size " << M << " L,nang " << L << endl;
  
  SNAPPY::count_memory( ndiag, ndiag_entries, nang, ng, nx, ny, nz, nmom, cmom, noct, ichunk, M );
  
  SNAPPY::run(  N, M, L, hyper_threads, vector_lanes,
                        nx, ny, nz, ichunk, nang, noct, ng, nmom, cmom,
                        diag, ndiag, ndiag_entries );  
  
}
