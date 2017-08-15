#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <iostream>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef typename Kokkos::DefaultExecutionSpace device_t;

typedef Kokkos::TeamPolicy<device_t> team_policy_t;

typedef Kokkos::View< int*, Kokkos::LayoutLeft, device_t > view_1i_t;

typedef Kokkos::View< double*, Kokkos::LayoutLeft, device_t > view_1d_t;

typedef Kokkos::View< double**, Kokkos::LayoutLeft, device_t > view_2d_t;

typedef Kokkos::View< double***, Kokkos::LayoutLeft, device_t > view_3d_t;

typedef Kokkos::View< double****, Kokkos::LayoutLeft, device_t > view_4d_t;

typedef Kokkos::View< double*****, Kokkos::LayoutLeft, device_t > view_5d_t;

typedef Kokkos::View< double******, Kokkos::LayoutLeft, device_t > view_6d_t;

typedef Kokkos::View< double*******, Kokkos::LayoutLeft, device_t > view_7d_t;

typedef Kokkos::View< double*, Kokkos::LayoutStride, device_t > s_view_1d_t;

typedef Kokkos::View< double**, Kokkos::LayoutStride, device_t > s_view_2d_t;

typedef Kokkos::View< double***, Kokkos::LayoutStride, device_t > s_view_3d_t;

typedef Kokkos::View< double****, Kokkos::LayoutStride, device_t > s_view_4d_t;

void c_kokkos_initialize_with_args( int argc, char *argv[] );
void c_kokkos_initialize( void );
void c_kokkos_finalize( void );
void c_kokkos_daxpy( const int* n, const double* a, const double* x, double* y );
void c_kokkos_allocate_A( const int* m, const int* n, double** raw_A, view_2d_t v_A );


void c_kokkos_allocate_1i( const int* m, int** A,view_1i_t** v_A,const char* a_name );

void c_kokkos_allocate_1d( const int* m, double** A,view_1d_t** v_A,const char* a_name );

void c_kokkos_allocate_2d( const int* m, const int* n, double** A,view_2d_t** v_A,const char* a_name );
void c_kokkos_allocate_3d( const int* m, const int* n, const int* o, double** A,view_3d_t** v_A,const char* a_name );
void c_kokkos_allocate_4d( const int* m, const int* n, const int* o, const int* p,
						   double** A,view_4d_t** v_A,const char* a_name );
void c_kokkos_allocate_5d( const int* m, const int* n, const int* o, const int* p, const int* q,
						   double** A,view_5d_t** v_A,const char* a_name );
void c_kokkos_allocate_6d( const int* m, const int* n, const int* o, const int* p, const int* q, const int* r,
						   double** A,view_6d_t** v_A,const char* a_name );
void c_kokkos_allocate_7d( const int* m, const int* n, const int* o, const int* p, const int* q, const int* r, const int* s,
						   double** A,view_7d_t** v_A,const char* a_name );

void c_kokkos_free_1d( view_1d_t** v_A );
void c_kokkos_free_2d( view_2d_t** v_A );
void c_kokkos_free_4d( view_4d_t** v_A );

#ifdef __cplusplus
}
#endif
