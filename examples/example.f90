program main
    use iso_c_binding
    use spla
    implicit none
    include 'mpif.h'
    integer :: m, n, k_local, block_size, proc_grid_rows, proc_grid_cols
    integer :: lda, ldb, ldc, spla_status, ierror, world_size
    real(C_DOUBLE) :: alpha, beta
    real(C_DOUBLE), allocatable, target :: A(:, :), B(:, :), C(:, :)
    type(c_ptr) :: ctx = c_null_ptr
    type(c_ptr) :: c_dist = c_null_ptr

    call MPI_INIT(ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, world_size, ierror)

    m = 100
    n = 100
    k_local = 100

    allocate( A(k_local, m) )
    allocate( B(k_local, n) )
    allocate( C(m, n) ) ! Allocate full C for simplicity

    lda = k_local
    ldb = k_local
    ldc = m


    proc_grid_rows = sqrt(real(world_size));
    proc_grid_cols = world_size / proc_grid_rows;

    ! Create context, which holds any resources SPLA will require, allowing reuse between functions
    ! calls. The given processing unit will be used for any computations.
    spla_status = spla_ctx_create(ctx, SPLA_PU_HOST)
    if (spla_status /= SPLA_SUCCESS) error stop

    ! Create matrix distribution for C
    spla_status = spla_mat_dis_create_block_cyclic(c_dist, MPI_COMM_WORLD, 'R', &
                            proc_grid_rows, proc_grid_cols, block_size, block_size)
    if (spla_status /= SPLA_SUCCESS) error stop

    ! Compute parallel stripe-stripe-block matrix multiplication. To describe the stripe distribution
    ! of matrices A and B, only the local k dimension is required.
    spla_status = spla_pdgemm_ssb(m, n, k_local, SPLA_OP_TRANSPOSE, alpha, &
                                  c_loc(A(1,1)), lda, c_loc(B(1,1)), ldb, &
                                  beta, c_loc(C(1,1)), ldc, 0, 0, c_dist, ctx)
    if (spla_status /= SPLA_SUCCESS) error stop

    spla_status = spla_ctx_destroy(ctx);
    spla_status = spla_mat_dis_destroy(c_dist);

    deallocate(A)
    deallocate(B)
    deallocate(C)

    call MPI_FINALIZE(ierror)
end program
