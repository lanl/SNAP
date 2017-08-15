MODULE kokkos_f_interface
	USE, INTRINSIC :: ISO_C_BINDING

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	INTERFACE
		SUBROUTINE f_kokkos_initialize() &
			BIND(c, NAME='c_kokkos_initialize')
			USE, INTRINSIC :: ISO_C_BINDING
			IMPLICIT NONE
		END SUBROUTINE f_kokkos_initialize
	END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_finalize() &
                        BIND(c, NAME='c_kokkos_finalize')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                END SUBROUTINE f_kokkos_finalize
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_daxpy(n,a,x,y) &
                        BIND(c, NAME='c_kokkos_daxpy')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                        INTEGER (C_INT), INTENT(IN) :: n
                        REAL (C_DOUBLE), INTENT(IN) :: a
                        REAL (C_DOUBLE), INTENT(IN), DIMENSION(*) :: x
                        REAL (C_DOUBLE), INTENT(INOUT), DIMENSION(*) :: y
                END SUBROUTINE f_kokkos_daxpy
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_allocate_A(m,n,c_A,v_A) &
                        BIND(c, NAME='c_kokkos_allocate_A')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                        INTEGER (C_INT), INTENT(IN) :: m
                        INTEGER (C_INT), INTENT(IN) :: n
                        TYPE (C_PTR), INTENT(OUT) :: c_A
                        TYPE (C_PTR), INTENT(OUT) :: v_A
                END SUBROUTINE f_kokkos_allocate_A
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_allocate_1i(m,c_A,v_A,n_A) &
                        BIND(c, NAME='c_kokkos_allocate_1i')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                        INTEGER (C_INT), INTENT(IN) :: m
                        TYPE (C_PTR), INTENT(OUT) :: c_A
                        TYPE (C_PTR), INTENT(OUT) :: v_A
                        character(kind=c_char), INTENT(IN) :: n_A(*)
                END SUBROUTINE f_kokkos_allocate_1i
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_allocate_1d(m,c_A,v_A,n_A) &
                        BIND(c, NAME='c_kokkos_allocate_1d')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                        INTEGER (C_INT), INTENT(IN) :: m
                        TYPE (C_PTR), INTENT(OUT) :: c_A
                        TYPE (C_PTR), INTENT(OUT) :: v_A
                        character(kind=c_char), INTENT(IN) :: n_A(*)
                END SUBROUTINE f_kokkos_allocate_1d
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_allocate_2d(m,n,c_A,v_A,n_A) &
                        BIND(c, NAME='c_kokkos_allocate_2d')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                        INTEGER (C_INT), INTENT(IN) :: m
                        INTEGER (C_INT), INTENT(IN) :: n
                        TYPE (C_PTR), INTENT(OUT) :: c_A
                        TYPE (C_PTR), INTENT(OUT) :: v_A
                        character(kind=c_char), INTENT(IN) :: n_A(*)
                END SUBROUTINE f_kokkos_allocate_2d
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_allocate_3d(m,n,o,c_A,v_A,n_A) &
                        BIND(c, NAME='c_kokkos_allocate_3d')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                        INTEGER (C_INT), INTENT(IN) :: m
                        INTEGER (C_INT), INTENT(IN) :: n
                        INTEGER (C_INT), INTENT(IN) :: o
                        TYPE (C_PTR), INTENT(OUT) :: c_A
                        TYPE (C_PTR), INTENT(OUT) :: v_A
                        character(kind=c_char), INTENT(IN) :: n_A(*)
                END SUBROUTINE f_kokkos_allocate_3d
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_allocate_4d(m,n,o,p,c_A,v_A,n_A) &
                        BIND(c, NAME='c_kokkos_allocate_4d')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                        INTEGER (C_INT), INTENT(IN) :: m
                        INTEGER (C_INT), INTENT(IN) :: n
                        INTEGER (C_INT), INTENT(IN) :: o
                        INTEGER (C_INT), INTENT(IN) :: p
                        TYPE (C_PTR), INTENT(OUT) :: c_A
                        TYPE (C_PTR), INTENT(OUT) :: v_A
                        character(kind=c_char), INTENT(IN) :: n_A(*)
                END SUBROUTINE f_kokkos_allocate_4d
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_allocate_5d(m,n,o,p,q,c_A,v_A,n_A) &
                        BIND(c, NAME='c_kokkos_allocate_5d')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                        INTEGER (C_INT), INTENT(IN) :: m
                        INTEGER (C_INT), INTENT(IN) :: n
                        INTEGER (C_INT), INTENT(IN) :: o
                        INTEGER (C_INT), INTENT(IN) :: p
                        INTEGER (C_INT), INTENT(IN) :: q
                        TYPE (C_PTR), INTENT(OUT) :: c_A
                        TYPE (C_PTR), INTENT(OUT) :: v_A
                        character(kind=c_char), INTENT(IN) :: n_A(*)
                END SUBROUTINE f_kokkos_allocate_5d
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_allocate_6d(m,n,o,p,q,r,c_A,v_A,n_A) &
                        BIND(c, NAME='c_kokkos_allocate_6d')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                        INTEGER (C_INT), INTENT(IN) :: m
                        INTEGER (C_INT), INTENT(IN) :: n
                        INTEGER (C_INT), INTENT(IN) :: o
                        INTEGER (C_INT), INTENT(IN) :: p
                        INTEGER (C_INT), INTENT(IN) :: q
                        INTEGER (C_INT), INTENT(IN) :: r
                        TYPE (C_PTR), INTENT(OUT) :: c_A
                        TYPE (C_PTR), INTENT(OUT) :: v_A
                        character(kind=c_char), INTENT(IN) :: n_A(*)
                END SUBROUTINE f_kokkos_allocate_6d
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_allocate_7d(m,n,o,p,q,r,s,c_A,v_A,n_A) &
                        BIND(c, NAME='c_kokkos_allocate_7d')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                        INTEGER (C_INT), INTENT(IN) :: m
                        INTEGER (C_INT), INTENT(IN) :: n
                        INTEGER (C_INT), INTENT(IN) :: o
                        INTEGER (C_INT), INTENT(IN) :: p
                        INTEGER (C_INT), INTENT(IN) :: q
                        INTEGER (C_INT), INTENT(IN) :: r
                        INTEGER (C_INT), INTENT(IN) :: s
                        TYPE (C_PTR), INTENT(OUT) :: c_A
                        TYPE (C_PTR), INTENT(OUT) :: v_A
                        character(kind=c_char), INTENT(IN) :: n_A(*)
                END SUBROUTINE f_kokkos_allocate_7d
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_free_1d(v_A) &
                        BIND(c, NAME='c_kokkos_free_1d')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                        TYPE (C_PTR), INTENT(IN) :: v_A
                END SUBROUTINE f_kokkos_free_1d
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_free_2d(v_A) &
                        BIND(c, NAME='c_kokkos_free_2d')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                        TYPE (C_PTR), INTENT(IN) :: v_A
                END SUBROUTINE f_kokkos_free_2d
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        INTERFACE
                SUBROUTINE f_kokkos_free_4d(v_A) &
                        BIND(c, NAME='c_kokkos_free_4d')
                        USE, INTRINSIC :: ISO_C_BINDING
                        IMPLICIT NONE
                        TYPE (C_PTR), INTENT(IN) :: v_A
                END SUBROUTINE f_kokkos_free_4d
        END INTERFACE
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        CONTAINS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_initialize()
		USE, INTRINSIC :: ISO_C_BINDING
		IMPLICIT NONE
		CALL f_kokkos_initialize()
	END SUBROUTINE kokkos_initialize
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_finalize()
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                CALL f_kokkos_finalize()
        END SUBROUTINE kokkos_finalize
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_daxpy(n,a,x,y)
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN) :: n
                REAL (C_DOUBLE), INTENT(IN) :: a
                REAL (C_DOUBLE), INTENT(IN), DIMENSION(*) :: x
                REAL (C_DOUBLE), INTENT(INOUT), DIMENSION(*) :: y
                CALL f_kokkos_daxpy(n,a,x,y)
        END SUBROUTINE kokkos_daxpy
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_allocate_A(m,n,f_A,v_A)
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN) :: m
                INTEGER (C_INT), INTENT(IN) :: n
                REAL (C_DOUBLE), POINTER, DIMENSION(:,:), INTENT(OUT) :: f_A
                TYPE (C_PTR), INTENT(OUT) :: v_A
                TYPE (C_PTR) :: c_A
                
                CALL f_kokkos_allocate_A(m,n,c_A,v_A)
                CALL C_F_POINTER(c_A,f_A,SHAPE=[m,n])
                
        END SUBROUTINE kokkos_allocate_A
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_allocate_1i(A,m,v_A,n_A)
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN) :: m
                INTEGER (C_INT), POINTER, DIMENSION(:), INTENT(INOUT) :: A
                TYPE (C_PTR), INTENT(OUT) :: v_A
                CHARACTER(kind=c_char), INTENT(IN) :: n_A(*)
                TYPE (C_PTR) :: c_A

                IF ( (m==0) ) THEN
                    CALL f_kokkos_allocate_1i(1,c_A,v_A,n_A)
                    ALLOCATE( A(0) )
                ELSE
                    CALL f_kokkos_allocate_1i(m,c_A,v_A,n_A)
                    CALL C_F_POINTER(c_A,A,SHAPE=[m])
                END IF
        END SUBROUTINE kokkos_allocate_1i
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_allocate_1d(A,m,v_A,n_A)
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN) :: m
                REAL (C_DOUBLE), POINTER, DIMENSION(:), INTENT(INOUT) :: A
                TYPE (C_PTR), INTENT(OUT) :: v_A
                CHARACTER(kind=c_char), INTENT(IN) :: n_A(*)
                TYPE (C_PTR) :: c_A


                IF ( (m==0) ) THEN
                    CALL f_kokkos_allocate_1d(1,c_A,v_A,n_A)
                    ALLOCATE( A(0) )
                ELSE
                    CALL f_kokkos_allocate_1d(m,c_A,v_A,n_A)
                    CALL C_F_POINTER(c_A,A,SHAPE=[m])
                END IF
        END SUBROUTINE kokkos_allocate_1d
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_allocate_2d(A,m,n,v_A,n_A)
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN) :: m
                INTEGER (C_INT), INTENT(IN) :: n
                REAL (C_DOUBLE), POINTER, DIMENSION(:,:), INTENT(INOUT) :: A
                TYPE (C_PTR), INTENT(OUT) :: v_A
                CHARACTER(kind=c_char), INTENT(IN) :: n_A(*)
                TYPE (C_PTR) :: c_A

                IF ( (m==0).AND.(n==0) ) THEN
                    CALL f_kokkos_allocate_2d(1,1,c_A,v_A,n_A)
                    ALLOCATE( A(0,0) )
                ELSE
                    CALL f_kokkos_allocate_2d(m,n,c_A,v_A,n_A)
                    CALL C_F_POINTER(c_A,A,SHAPE=[m,n])
                END IF
        END SUBROUTINE kokkos_allocate_2d
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_allocate_3d(A,m,n,o,v_A,n_A)
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN) :: m
                INTEGER (C_INT), INTENT(IN) :: n
                INTEGER (C_INT), INTENT(IN) :: o
                REAL (C_DOUBLE), POINTER, DIMENSION(:,:,:), INTENT(INOUT) :: A
                TYPE (C_PTR), INTENT(OUT) :: v_A
                CHARACTER(kind=c_char), INTENT(IN) :: n_A(*)
                TYPE (C_PTR) :: c_A

                IF ( (m==0).AND.(n==0).AND.(o==0) ) THEN
                    CALL f_kokkos_allocate_3d(1,1,1,c_A,v_A,n_A)
                    ALLOCATE( A(0,0,0) )
                ELSE
                    CALL f_kokkos_allocate_3d(m,n,o,c_A,v_A,n_A)
                    CALL C_F_POINTER(c_A,A,SHAPE=[m,n,o])
                END IF
        END SUBROUTINE kokkos_allocate_3d
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_allocate_4d(A,m,n,o,p,v_A,n_A)
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN) :: m
                INTEGER (C_INT), INTENT(IN) :: n
                INTEGER (C_INT), INTENT(IN) :: o
                INTEGER (C_INT), INTENT(IN) :: p
                REAL (C_DOUBLE), POINTER, DIMENSION(:,:,:,:), INTENT(INOUT) :: A
                TYPE (C_PTR), INTENT(OUT) :: v_A
                CHARACTER(kind=c_char), INTENT(IN) :: n_A(*)
                TYPE (C_PTR) :: c_A

                IF ( (m==0).AND.(n==0).AND.(o==0).AND.(p==0) ) THEN
                    CALL f_kokkos_allocate_4d(1,1,1,1,c_A,v_A,n_A)
                    ALLOCATE( A(0,0,0,0) )
                ELSE
                    CALL f_kokkos_allocate_4d(m,n,o,p,c_A,v_A,n_A)
                    CALL C_F_POINTER(c_A,A,SHAPE=[m,n,o,p])
                END IF

        END SUBROUTINE kokkos_allocate_4d
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_allocate_5d(A,m,n,o,p,q,v_A,n_A)
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN) :: m
                INTEGER (C_INT), INTENT(IN) :: n
                INTEGER (C_INT), INTENT(IN) :: o
                INTEGER (C_INT), INTENT(IN) :: p
                INTEGER (C_INT), INTENT(IN) :: q
                REAL (C_DOUBLE), POINTER, DIMENSION(:,:,:,:,:), INTENT(INOUT) :: A
                TYPE (C_PTR), INTENT(OUT) :: v_A
                CHARACTER(kind=c_char), INTENT(IN) :: n_A(*)
                TYPE (C_PTR) :: c_A

                IF ( (m==0).AND.(n==0).AND.(o==0).AND.(p==0).AND.(q==0) ) THEN
                    CALL f_kokkos_allocate_5d(1,1,1,1,1,c_A,v_A,n_A)
                    ALLOCATE( A(0,0,0,0,0) )
                ELSE
                    CALL f_kokkos_allocate_5d(m,n,o,p,q,c_A,v_A,n_A)
                    CALL C_F_POINTER(c_A,A,SHAPE=[m,n,o,p,q])
                END IF

        END SUBROUTINE kokkos_allocate_5d
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_allocate_6d(A,m,n,o,p,q,r,v_A,n_A)
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN) :: m
                INTEGER (C_INT), INTENT(IN) :: n
                INTEGER (C_INT), INTENT(IN) :: o
                INTEGER (C_INT), INTENT(IN) :: p
                INTEGER (C_INT), INTENT(IN) :: q
                INTEGER (C_INT), INTENT(IN) :: r
                REAL (C_DOUBLE), POINTER, DIMENSION(:,:,:,:,:,:), INTENT(INOUT) :: A
                TYPE (C_PTR), INTENT(OUT) :: v_A
                CHARACTER(kind=c_char), INTENT(IN) :: n_A(*)
                TYPE (C_PTR) :: c_A


                IF ( (m==0).AND.(n==0).AND.(o==0).AND.(p==0).AND.(q==0).AND.(r==0) ) THEN
                    CALL f_kokkos_allocate_6d(1,1,1,1,1,1,c_A,v_A,n_A)
                    ALLOCATE( A(0,0,0,0,0,0) )
                ELSE
                    CALL f_kokkos_allocate_6d(m,n,o,p,q,r,c_A,v_A,n_A)
                    CALL C_F_POINTER(c_A,A,SHAPE=[m,n,o,p,q,r])
                END IF
        END SUBROUTINE kokkos_allocate_6d
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_allocate_7d(A,m,n,o,p,q,r,s,v_A,n_A)
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN) :: m
                INTEGER (C_INT), INTENT(IN) :: n
                INTEGER (C_INT), INTENT(IN) :: o
                INTEGER (C_INT), INTENT(IN) :: p
                INTEGER (C_INT), INTENT(IN) :: q
                INTEGER (C_INT), INTENT(IN) :: r
                INTEGER (C_INT), INTENT(IN) :: s
                REAL (C_DOUBLE), POINTER, DIMENSION(:,:,:,:,:,:,:), INTENT(INOUT) :: A
                TYPE (C_PTR), INTENT(OUT) :: v_A
                CHARACTER(kind=c_char), INTENT(IN) :: n_A(*)
                TYPE (C_PTR) :: c_A

                IF ( (m==0).AND.(n==0).AND.(o==0).AND.(p==0).AND.(q==0).AND.(r==0).AND.(s==0) ) THEN
                    CALL f_kokkos_allocate_7d(1,1,1,1,1,1,1,c_A,v_A,n_A)
                    ALLOCATE( A(0,0,0,0,0,0,0) )
                ELSE
                    CALL f_kokkos_allocate_7d(m,n,o,p,q,r,s,c_A,v_A,n_A)
                    CALL C_F_POINTER(c_A,A,SHAPE=[m,n,o,p,q,r,s])
                END IF
        END SUBROUTINE kokkos_allocate_7d
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_free_1d(v_A)
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN) :: v_A
                CALL f_kokkos_free_1d(v_A)
        END SUBROUTINE kokkos_free_1d
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_free_2d(v_A)
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN) :: v_A
                CALL f_kokkos_free_2d(v_A)
        END SUBROUTINE kokkos_free_2d
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        SUBROUTINE kokkos_free_4d(v_A)
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                TYPE (C_PTR), INTENT(IN) :: v_A
                CALL f_kokkos_free_4d(v_A)
        END SUBROUTINE kokkos_free_4d
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
END MODULE kokkos_f_interface
