#include <time.h>

#include <Kokkos_Serial.hpp>
#include <Kokkos_Threads.hpp>

#ifdef KOKKOS_HAVE_OPENMP
#include <Kokkos_OpenMP.hpp>
#endif

#ifdef KOKKOS_HAVE_CUDA
#include <Kokkos_Cuda.hpp>
#endif

#include <Kokkos_hwloc.hpp>

//vtune
//#include "ittnotify.h"

#include <snappy.hpp>

const int n_test_iter = 1;
const int nang = 240; // TEST
const int nmom = 4; // TEST
const int noct = 8; // TEST
const int ng = 40; // TEST

time_t timer_start;
time_t timer_end;

namespace SNAPPY {

void run_serial(  int nx, int ny, int nz, int ndiag, const vector<diag_c>& diag,
                  int ndimen, int id, int jd, int kd,
                  int nang, int nmom, int noct, int ng,
                  int ich, int ichunk,
                  int jlo, int klo, int jhi, int khi, int jst, int kst,
                  bool firsty, bool lasty, bool firstz, bool lastz,
                  int nnested, int src_opt, int fixup,
                  double hi, double vdelt )
{

	typedef Kokkos::Serial device_type;
	typedef Kokkos::View<double*, Kokkos::LayoutLeft, device_type> serial_view_t_1d;
	typedef Kokkos::View<double**, Kokkos::LayoutLeft, device_type> serial_view_t_2d;
	typedef Kokkos::View<double***, Kokkos::LayoutLeft, device_type> serial_view_t_3d;
	typedef Kokkos::View<double****, Kokkos::LayoutLeft, device_type> serial_view_t_4d;
	typedef Kokkos::View<double*****, Kokkos::LayoutLeft, device_type> serial_view_t_5d;
	typedef Kokkos::View<double******, Kokkos::LayoutLeft, device_type> serial_view_t_6d;
	typedef Kokkos::View<double*, Kokkos::LayoutStride> serial_view_t_1d_s;

//   cout << " a " << endl;
  
	int cmom = nmom * nmom;
	int d1 = nang; // TEST, (timedep == 1 => 		d1 = nang; d2 = nx; d3 = ny; d4 = nz )
	int d2 = nx; // TEST
	int d3 = ny; // TEST
	int d4 = nz; // TEST

//   cout << " b " << endl;
  
  serial_view_t_1d hj( "hj", nang );
  serial_view_t_1d hk( "hk", nang );
  serial_view_t_1d mu( "mu", nang );
	serial_view_t_1d w( "w", nang );
  serial_view_t_6d qim( "qim", nang, nx, ny, nz, noct, ng );
	serial_view_t_3d psii( "psii", nang, ny, nz );
	serial_view_t_3d psij( "psij", nang, ichunk, nz );
	serial_view_t_3d psik( "psik", nang, ichunk, ny );
  serial_view_t_4d qtot( "qtot", cmom, nx, ny, nz );
	serial_view_t_2d ec( "ec", nang, cmom );
  serial_view_t_4d ptr_in( "ptr_in", d1, d2, d3, d4 );
  serial_view_t_4d ptr_out( "ptr_out", d1, d2, d3, d4 );
  serial_view_t_4d dinv( "dinv", nang, nx, ny, nz );
	serial_view_t_3d flux( "flux", nx, ny, nz );
  serial_view_t_4d fluxm( "fluxm", cmom-1, nx, ny, nz );
 	serial_view_t_3d jb_in( "jb_in", nang, ichunk, nz );
  serial_view_t_3d jb_out( "jb_out", nang, ichunk, nz );
	serial_view_t_3d kb_in( "kb_in", nang, ichunk, ny );
	serial_view_t_3d kb_out( "kb_out", nang, ichunk, ny );
	serial_view_t_1d wmu( "wmu", nang );
	serial_view_t_1d weta( "weta", nang );
	serial_view_t_1d wxi( "wxi", nang );
  serial_view_t_3d flkx( "flkx", nx+1, ny, nz );
  serial_view_t_3d flky( "flky", nx, ny+1, nz );
  serial_view_t_3d flkz( "flkz", nx, ny, nz+1 );
  serial_view_t_3d t_xs( "t_xs", nx, ny, nz );

//   cout << " c " << endl;
  
  for (int ii = 0; ii < n_test_iter; ii++) {
    time(&timer_start);
    
    for (int oct = 0; oct < noct; oct++) {
      for (int g = 0; g < ng; g++) {	
        dim3_sweep< device_type, serial_view_t_1d, serial_view_t_2d,
                  serial_view_t_3d, serial_view_t_4d, serial_view_t_5d,
                  serial_view_t_6d, serial_view_t_1d_s >
                ( ichunk, firsty, lasty, firstz, lastz, nnested,
                  nx, hi, hj, hk, ndimen, ny, nz, ndiag, diag,
                  cmom, nang, mu, w, noct,
                  src_opt, ng, qim,
                  fixup,
                  ich, id, d1, d2, d3, d4, jd, kd, jlo, klo, oct, g,
                  jhi, khi, jst, kst, psii, psij, psik, qtot, ec, vdelt,
                  ptr_in, ptr_out, dinv, flux, fluxm, jb_in, jb_out,
                  kb_in, kb_out, wmu, weta, wxi, flkx, flky, flkz, t_xs );
      }
    }
    
    time(&timer_end);
    std::cout << " ii " << ii << " elapsed time " << difftime(timer_end, timer_start) << std::endl;
  }
}

// void run_threads( int nx, int ny, int nz, int ndiag, const vector<diag_c>& diag,
//                   int ndimen, int id, int jd, int kd,
//                   int nang, int nmom, int noct, int ng,
//                   int ich, int ichunk,
//                   int jlo, int klo, int jhi, int khi, int jst, int kst,
//                   bool firsty, bool lasty, bool firstz, bool lastz,
//                   int nnested, int src_opt, int fixup,
//                   double hi, double vdelt )
// {
// #ifdef KOKKOS_HAVE_PTHREAD
// 	typedef Kokkos::Threads device_type;
// 	typedef Kokkos::View<double*, Kokkos::LayoutLeft, device_type> pthread_view_t_1d;
// 	typedef Kokkos::View<double**, Kokkos::LayoutLeft, device_type> pthread_view_t_2d;
// 	typedef Kokkos::View<double***, Kokkos::LayoutLeft, device_type> pthread_view_t_3d;
// 	typedef Kokkos::View<double****, Kokkos::LayoutLeft, device_type> pthread_view_t_4d;
// 	typedef Kokkos::View<double*****, Kokkos::LayoutLeft, device_type> pthread_view_t_5d;
// 	typedef Kokkos::View<double******, Kokkos::LayoutLeft, device_type> pthread_view_t_6d;
// 	typedef Kokkos::View<double*, Kokkos::LayoutStride> pthread_view_t_1d_s;
// 
// 	int cmom = nmom * nmom;
// 	int d1 = nang; // TEST, (timedep == 1 => 		d1 = nang; d2 = nx; d3 = ny; d4 = nz )
// 	int d2 = nx; // TEST
// 	int d3 = ny; // TEST
// 	int d4 = nz; // TEST
// 
//   pthread_view_t_1d hj( "hj", nang );
//   pthread_view_t_1d hk( "hk", nang );
//   pthread_view_t_1d mu( "mu", nang );
// 	pthread_view_t_1d w( "w", nang );
//   pthread_view_t_6d qim( "qim", nang, nx, ny, nz, noct, ng );
// 	pthread_view_t_3d psii( "psii", nang, ny, nz );
// 	pthread_view_t_3d psij( "psij", nang, ichunk, nz );
// 	pthread_view_t_3d psik( "psik", nang, ichunk, ny );
//   pthread_view_t_4d qtot( "qtot", cmom, nx, ny, nz );
// 	pthread_view_t_2d ec( "ec", nang, cmom );
//   pthread_view_t_4d ptr_in( "ptr_in", d1, d2, d3, d4 );
//   pthread_view_t_4d ptr_out( "ptr_out", d1, d2, d3, d4 );
//   pthread_view_t_4d dinv( "dinv", nang, nx, ny, nz );
// 	pthread_view_t_3d flux( "flux", nx, ny, nz );
//   pthread_view_t_4d fluxm( "fluxm", cmom-1, nx, ny, nz );
//  	pthread_view_t_3d jb_in( "jb_in", nang, ichunk, nz );
//   pthread_view_t_3d jb_out( "jb_out", nang, ichunk, nz );
// 	pthread_view_t_3d kb_in( "kb_in", nang, ichunk, ny );
// 	pthread_view_t_3d kb_out( "kb_out", nang, ichunk, ny );
// 	pthread_view_t_1d wmu( "wmu", nang );
// 	pthread_view_t_1d weta( "weta", nang );
// 	pthread_view_t_1d wxi( "wxi", nang );
//   pthread_view_t_3d flkx( "flkx", nx+1, ny, nz );
//   pthread_view_t_3d flky( "flky", nx, ny+1, nz );
//   pthread_view_t_3d flkz( "flkz", nx, ny, nz+1 );
//   pthread_view_t_3d t_xs( "t_xs", nx, ny, nz );
//   			
//   for (int ii = 0; ii < n_test_iter; ii++) {
//     time(&timer_start);
//     for (int oct = 0; oct < noct; oct++) {
//       for (int g = 0; g < ng; g++) {
//         dim3_sweep< device_type, pthread_view_t_1d, pthread_view_t_2d,
//                   pthread_view_t_3d, pthread_view_t_4d, pthread_view_t_5d,
//                   pthread_view_t_6d, pthread_view_t_1d_s >
//                 ( ichunk, firsty, lasty, firstz, lastz, nnested,
//                   nx, hi, hj, hk, ndimen, ny, nz, ndiag, diag,
//                   cmom, nang, mu, w, noct,
//                   src_opt, ng, qim,
//                   fixup,
//                   ich, id, d1, d2, d3, d4, jd, kd, jlo, klo, oct, g,
//                   jhi, khi, jst, kst, psii, psij, psik, qtot, ec, vdelt,
//                   ptr_in, ptr_out, dinv, flux, fluxm, jb_in, jb_out,
//                   kb_in, kb_out, wmu, weta, wxi, flkx, flky, flkz, t_xs );
//       }
//     }
//     time(&timer_end);
//     std::cout << " ii " << ii << " elapsed time " << difftime(timer_end, timer_start) << std::endl;
//   }
// #else
//   return 0;
// #endif
// }

// void run_openmp(  int nx, int ny, int nz, int ndiag, const vector<diag_c>& diag,
//                   int ndimen, int id, int jd, int kd,
//                   int nang, int nmom, int noct, int ng,
//                   int ich, int ichunk,
//                   int jlo, int klo, int jhi, int khi, int jst, int kst,
//                   bool firsty, bool lasty, bool firstz, bool lastz,
//                   int nnested, int src_opt, int fixup,
//                   double hi, double vdelt )
// {
// #ifdef KOKKOS_HAVE_OPENMP
// 	typedef Kokkos::OpenMP device_type;
// 	typedef Kokkos::View<double*, Kokkos::LayoutLeft, device_type> openmp_view_t_1d;
// 	typedef Kokkos::View<double**, Kokkos::LayoutLeft, device_type> openmp_view_t_2d;
// 	typedef Kokkos::View<double***, Kokkos::LayoutLeft, device_type> openmp_view_t_3d;
// 	typedef Kokkos::View<double****, Kokkos::LayoutLeft, device_type> openmp_view_t_4d;
// 	typedef Kokkos::View<double*****, Kokkos::LayoutLeft, device_type> openmp_view_t_5d;
// 	typedef Kokkos::View<double******, Kokkos::LayoutLeft, device_type> openmp_view_t_6d;
// 	typedef Kokkos::View<double*, Kokkos::LayoutStride> openmp_view_t_1d_s;
// 
// 	int cmom = nmom * nmom;
// 	int d1 = nang; // TEST, (timedep == 1 => 		d1 = nang; d2 = nx; d3 = ny; d4 = nz )
// 	int d2 = nx; // TEST
// 	int d3 = ny; // TEST
// 	int d4 = nz; // TEST
// 
//   openmp_view_t_1d hj( "hj", nang );
//   openmp_view_t_1d hk( "hk", nang );
//   openmp_view_t_1d mu( "mu", nang );
// 	openmp_view_t_1d w( "w", nang );
//   openmp_view_t_6d qim( "qim", nang, nx, ny, nz, noct, ng );
// 	openmp_view_t_3d psii( "psii", nang, ny, nz );
// 	openmp_view_t_3d psij( "psij", nang, ichunk, nz );
// 	openmp_view_t_3d psik( "psik", nang, ichunk, ny );
//   openmp_view_t_4d qtot( "qtot", cmom, nx, ny, nz );
// 	openmp_view_t_2d ec( "ec", nang, cmom );
//   openmp_view_t_4d ptr_in( "ptr_in", d1, d2, d3, d4 );
//   openmp_view_t_4d ptr_out( "ptr_out", d1, d2, d3, d4 );
//   openmp_view_t_4d dinv( "dinv", nang, nx, ny, nz );
// 	openmp_view_t_3d flux( "flux", nx, ny, nz );
//   openmp_view_t_4d fluxm( "fluxm", cmom-1, nx, ny, nz );
//  	openmp_view_t_3d jb_in( "jb_in", nang, ichunk, nz );
//   openmp_view_t_3d jb_out( "jb_out", nang, ichunk, nz );
// 	openmp_view_t_3d kb_in( "kb_in", nang, ichunk, ny );
// 	openmp_view_t_3d kb_out( "kb_out", nang, ichunk, ny );
// 	openmp_view_t_1d wmu( "wmu", nang );
// 	openmp_view_t_1d weta( "weta", nang );
// 	openmp_view_t_1d wxi( "wxi", nang );
//   openmp_view_t_3d flkx( "flkx", nx+1, ny, nz );
//   openmp_view_t_3d flky( "flky", nx, ny+1, nz );
//   openmp_view_t_3d flkz( "flkz", nx, ny, nz+1 );
//   openmp_view_t_3d t_xs( "t_xs", nx, ny, nz );
// 
//   for (int ii = 0; ii < n_test_iter; ii++) {
//     time(&timer_start);
//     for (int oct = 0; oct < noct; oct++) {
//       for (int g = 0; g < ng; g++) {
//         dim3_sweep< device_type, openmp_view_t_1d, openmp_view_t_2d,
//                     openmp_view_t_3d, openmp_view_t_4d, openmp_view_t_5d,
//                     openmp_view_t_6d, openmp_view_t_1d_s >
//                   ( ichunk, firsty, lasty, firstz, lastz, nnested,
//                     nx, hi, hj, hk, ndimen, ny, nz, ndiag, diag,
//                     cmom, nang, mu, w, noct,
//                     src_opt, ng, qim,
//                     fixup,
//                     ich, id, d1, d2, d3, d4, jd, kd, jlo, klo, oct, g,
//                     jhi, khi, jst, kst, psii, psij, psik, qtot, ec, vdelt,
//                     ptr_in, ptr_out, dinv, flux, fluxm, jb_in, jb_out,
//                     kb_in, kb_out, wmu, weta, wxi, flkx, flky, flkz, t_xs );
//       }
//     }
//     time(&timer_end);
//     std::cout << " ii " << ii << " elapsed time " << difftime(timer_end, timer_start) << std::endl;
//     
//   }
// #else
//   return 0;
// #endif
// }

// #ifdef KOKKOS_HAVE_CUDA
// extern void run_cuda(void);
// #else
// void run_cuda(void)
// {
// }
// #endif

} // namespace SNAPPY


int main(int argc, char *argv[])
{

  // query the topology of the host
  unsigned team_count = 1;
  unsigned threads_count = 1;

  //avoid unused variable warning
  (void)team_count;

  int nx = 8;
  int ny = 8;
  int nz = 8;
  int ndiag;
 
	int ndimen = 3; // TEST
  int id = 2; // TEST
  int jd = 2; // TEST
  int kd = 2; // TEST

//   int oct = 1; // TEST
//   int g = 12; // TEST
  int ich = 1; // TEST
  int ichunk = 8; // TEST

  int jlo = 0; // TEST
  int klo = 0; // TEST
  int jhi = ny - 1; // TEST
  int khi = nz - 1; // TEST
  int jst = 1; // TEST
  int kst = 1; // TEST

  bool lasty;
  bool firsty;
  bool lastz;
  bool firstz;

  int nnested;
  int src_opt = 3; // TEST
  int fixup = 0; // TEST

  double hi = c0; // TEST	
  double vdelt = c0; // TEST
 
  vector<diag_c> diag;

  SNAPPY::populate_diag( ichunk, ny, nz, diag );
  //system("numactl --show");
  

   if (Kokkos::hwloc::available()) {
     threads_count = Kokkos::hwloc::get_available_numa_count() *
                     Kokkos::hwloc::get_available_cores_per_numa() *
                     Kokkos::hwloc::get_available_threads_per_core();
     std::cout << "Kokkos::hwloc::get_available_numa_count(): " << Kokkos::hwloc::get_available_numa_count() << std::endl;
     std::cout << "Kokkos::hwloc::get_available_cores_per_numa(): " << Kokkos::hwloc::get_available_cores_per_numa() << std::endl;
     std::cout << "Kokkos::hwloc::get_available_threads_per_core(): " << Kokkos::hwloc::get_available_threads_per_core() << std::endl;
   }

  std::cout << "threads_count " << threads_count << std::endl;

  std::cout << "Serial" << std::endl;
  //__itt_resume();
  SNAPPY::run_serial( nx, ny, nz, ndiag, diag, ndimen, id, jd, kd, nang, nmom, noct, ng,
                      ich, ichunk, jlo, klo, jhi, khi, jst, kst,
                      firsty, lasty, firstz, lastz, nnested, src_opt, fixup, hi, vdelt );
  //__itt_pause();

// #ifdef KOKKOS_HAVE_PTHREAD
//   std::cout << "Pthreads" << std::endl;
//   Kokkos::Threads::initialize( threads_count );
//   //system("numactl --show");
// //   __itt_resume();
//   SNAPPY::run_threads(  nx, ny, nz, ndiag, diag, ndimen, id, jd, kd, nang, nmom, noct, ng,
//                         ich, ichunk, jlo, klo, jhi, khi, jst, kst,
//                         firsty, lasty, firstz, lastz, nnested, src_opt, fixup, hi, vdelt );
// //   __itt_pause();
//   Kokkos::Threads::finalize();
// #endif

// #ifdef KOKKOS_HAVE_OPENMP
//   std::cout << "OpenMP" << std::endl;
// 
//   Kokkos::OpenMP::initialize( threads_count );
//   //system("numactl --show");
// //  __itt_pause();
//   SNAPPY::run_openmp( nx, ny, nz, ndiag, diag, ndimen, id, jd, kd, nang, nmom, noct, ng,
//                       ich, ichunk, jlo, klo, jhi, khi, jst, kst,
//                       firsty, lasty, firstz, lastz, nnested, src_opt, fixup, hi, vdelt );
// //  __itt_resume();
//   Kokkos::OpenMP::finalize();
// #endif

}

