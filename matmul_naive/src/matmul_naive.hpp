#ifndef __MatMUL_NAIVE_HPP__
#define __MatMUL_NAIVE_HPP__

template <typename TT,              // Datatype of the elements of the matrix
          int rows_a,               // Rows of matrix A
          int common,               // Columns of matrix A / rows of matrix B
          int cols_b,               // Columns of matrix B
          int num_matrices         // Number of pairs of matrices to multiply
        >
class MatMulNaive {
public:
    TT *a_ptr;   // Input matrix pointer
    TT *b_ptr;   //
    TT *c_ptr;   //
    int repetitions; // Number of times to write the same matrix to the pipe

  void operator()() const {
    constexpr int kMatsizeA = rows_a * common;
    constexpr int kMatsizeB = cols_b * common;
    constexpr int kMatsizeC = rows_a * cols_b;

#if defined(IS_BSP)
    // When targeting a BSP, we instruct the compiler that this pointer lives on
    // the device.
    // Knowing this, the compiler won't generate hardware to potentially get
    // data from the host.
    sycl::device_ptr<TT> a_ptr_located(a_ptr);
    sycl::device_ptr<TT> b_ptr_located(b_ptr);
    sycl::device_ptr<TT> c_ptr_located(c_ptr);
#else
    // Device pointers are not supported when targeting an FPGA family/part
    TT *a_ptr_located(a_ptr);
    TT *b_ptr_located(b_ptr);
    TT *c_ptr_located(c_ptr);
#endif

  for( int matrix_idx = 0; matrix_idx < num_matrices; matrix_idx++ ) {
    for (int row = 0; row < rows_a; row++) {
      for (int col = 0; col < cols_b; col++) {
        float dot_prod{0};
    #pragma unroll
        for (int k = 0; k < common; k++) {
          dot_prod = intelfpga::fpga_reg(dot_prod) + a_ptr_located[matrix_idx * kMatsizeA + k * rows_a + row] 
                                                   * b_ptr_located[matrix_idx * kMatsizeB + col * common + k];
        }
        c_ptr_located[matrix_idx * kMatsizeC + col * rows_a + row] = dot_prod;
      }
    }
  }


  }       // end of operator
};


#endif