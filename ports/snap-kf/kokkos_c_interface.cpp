#include "kokkos_c_interface.hpp"

#include <vector>

using Kokkos::ALL;
using Kokkos::subview;

void c_kokkos_initialize_with_args( int argc, char *argv[] ) {

  Kokkos::initialize( argc, argv );
  
}

void c_kokkos_initialize() {

  Kokkos::initialize();

}

void c_kokkos_finalize( void ) {

  Kokkos::finalize();

}

void c_kokkos_daxpy( const int* n, const double* a, const double* x, double* y ) {

	typedef typename Kokkos::DefaultExecutionSpace device_t;
	
	size_t nn = *n;
	double aa = *a;

	//Kokkos::View<double*, device_t, Kokkos::MemoryTraits<Kokkos::Unmanaged>> xx( x, *n );
	typedef Kokkos::View< double*, device_t, Kokkos::MemoryTraits< Kokkos::Unmanaged > > view_1d_t;
	typedef Kokkos::View< const double*, device_t, Kokkos::MemoryTraits< Kokkos::Unmanaged > > const_view_1d_t;
	//aview_1d_t xx( x, *n );
	const_view_1d_t xx( x, nn );
	view_1d_t yy( y, nn );
	
	Kokkos::Impl::Timer timer;
	Kokkos::parallel_for( nn, KOKKOS_LAMBDA(const int ii) { yy[ii] = aa * xx[ii] + yy[ii]; }  );
	std::cout << "elapsed time " << timer.seconds() << std::endl;
}

void c_kokkos_allocate_A( const int* m, const int* n, double** raw_A, view_2d_t v_A ) {
//   size_t mm = *m;
//   size_t nn = *n;
  
  view_2d_t A( "A", *m, *n );
  *raw_A = A.ptr_on_device();
    
}

void c_kokkos_allocate_1i( const int* m, int** A,view_1i_t** v_A,const char* a_name ) {
//   size_t mm = *m;
//   size_t nn = *n;
	const int mt = std::max(*m,1);

// 	printf("view_1i_t* %s(%i)\n",a_name, mt );

	char name[32];
	sprintf(name,"%s",a_name);
	*v_A = new view_1i_t(name, mt);
	Kokkos::deep_copy(device_t(),**v_A,0);

//	phii_views.push_back(*phii);
	*A = (*v_A)->ptr_on_device();

}

void c_kokkos_allocate_1d( const int* m, double** A,view_1d_t** v_A,const char* a_name ) {
//   size_t mm = *m;
//   size_t nn = *n;
	const int mt = std::max(*m,1);

// 	printf("view_1d_t* %s(%i)\n",a_name, mt );

	char name[32];
	sprintf(name,"%s",a_name);
	*v_A = (new view_1d_t(name, mt));
	Kokkos::deep_copy(device_t(),**v_A,0.0);

//	phii_views.push_back(*phii);
	*A = (*v_A)->ptr_on_device();

}

void c_kokkos_allocate_2d( const int* m, const int* n, double** A,view_2d_t** v_A,const char* a_name ) {
//   size_t mm = *m;
//   size_t nn = *n;
	const int mt = std::max(*m,1);
	const int nt = std::max(*n,1);

// 	printf("view_2d_t* %s(%i, %i)\n",a_name, mt,nt );



	char name[32];
	sprintf(name,"%s",a_name);
	*v_A = (new view_2d_t(name, mt,nt));
	Kokkos::deep_copy(device_t(),**v_A,0.0);

//	phii_views.push_back(*phii);
	*A = (*v_A)->ptr_on_device();

}

void c_kokkos_allocate_3d( const int* m, const int* n, const int* o, double** A,view_3d_t** v_A,const char* a_name ) {
//   size_t mm = *m;
//   size_t nn = *n;

	const int mt = std::max(*m,1);
	const int nt = std::max(*n,1);
	const int ot = std::max(*o,1);


// 	printf("view_3d_t* %s(%i, %i, %i)\n",a_name, mt,nt,ot );

	char name[32];
	sprintf(name,"%s",a_name);
	*v_A = (new view_3d_t(name, mt,nt,ot ));

	Kokkos::deep_copy(device_t(),**v_A,0.0);

//	phii_views.push_back(*phii);
	*A = (*v_A)->ptr_on_device();

// 	printf("label %s\n",(*v_A)->label().c_str());

}


void c_kokkos_allocate_4d( const int* m, const int* n, const int* o, const int* p,
						   double** A,view_4d_t** v_A,const char* a_name )
{
//   size_t mm = *m;
//   size_t nn = *n;
	const int mt = std::max(*m,1);
	const int nt = std::max(*n,1);
	const int ot = std::max(*o,1);
	const int pt = std::max(*p,1);

// 	printf("view_4d_t* %s(%i, %i, %i, %i)\n",a_name, mt,nt,ot,pt );

	char name[32];
	sprintf(name,"%s",a_name);
	*v_A = (new view_4d_t(name, mt,nt,ot,pt));
	Kokkos::deep_copy(device_t(),**v_A,0.0);

//	phii_views.push_back(*phii);
	*A = (*v_A)->ptr_on_device();

}

void c_kokkos_allocate_5d( const int* m, const int* n, const int* o, const int* p, const int* q,
						   double** A,view_5d_t** v_A,const char* a_name )
{
//   size_t mm = *m;
//   size_t nn = *n;

	const int mt = std::max(*m,1);
	const int nt = std::max(*n,1);
	const int ot = std::max(*o,1);
	const int pt = std::max(*p,1);
	const int qt = std::max(*q,1);
// 	printf("view_5d_t* %s(%i, %i, %i, %i, %i)\n",a_name, mt,nt,ot,pt,qt );



	char name[32];
	sprintf(name,"%s",a_name);
	*v_A = (new view_5d_t(name, mt,nt,ot,pt,qt));
	Kokkos::deep_copy(device_t(),**v_A,0.0);

//	phii_views.push_back(*phii);
	*A = (*v_A)->ptr_on_device();

}
void c_kokkos_allocate_6d( const int* m, const int* n, const int* o, const int* p, const int* q, const int* r,
						   double** A,view_6d_t** v_A,const char* a_name )
{
//   size_t mm = *m;
//   size_t nn = *n;
	const int mt = std::max(*m,1);
	const int nt = std::max(*n,1);
	const int ot = std::max(*o,1);
	const int pt = std::max(*p,1);
	const int qt = std::max(*q,1);
	const int rt = std::max(*r,1);

// 	printf("view_6d_t* %s(%i, %i, %i, %i, %i, %i)\n",a_name, mt,nt,ot,pt,qt,rt );

	char name[32];
	sprintf(name,"%s",a_name);
	*v_A = (new view_6d_t(name, mt,nt,ot,pt,qt,rt ));
	Kokkos::deep_copy(device_t(),**v_A,0.0);

//	phii_views.push_back(*phii);
	*A = (*v_A)->ptr_on_device();

}
void c_kokkos_allocate_7d( const int* m, const int* n, const int* o, const int* p, const int* q, const int* r, const int* s,
						   double** A,view_7d_t** v_A,const char* a_name )
{
//   size_t mm = *m;
//   size_t nn = *n;
	const int mt = std::max(*m,1);
	const int nt = std::max(*n,1);
	const int ot = std::max(*o,1);
	const int pt = std::max(*p,1);
	const int qt = std::max(*q,1);
	const int rt = std::max(*r,1);
	const int st = std::max(*s,1);

// 	printf("view_7d_t* %s(%i, %i, %i, %i, %i, %i, %i)\n",a_name, mt,nt,ot,pt,qt,rt,st );

	char name[32];
	sprintf(name,"%s",a_name);
	*v_A = (new view_7d_t(name, mt,nt,ot,pt,qt,rt,st));
	Kokkos::deep_copy(device_t(),**v_A,0.0);

//	phii_views.push_back(*phii);
	*A = (*v_A)->ptr_on_device();

}

void c_kokkos_free_1d( view_1d_t** v_A )
{
	delete *v_A;
}

void c_kokkos_free_2d( view_2d_t** v_A )
{
	delete *v_A;
}

void c_kokkos_free_4d( view_4d_t** v_A )
{
	delete *v_A;
}
