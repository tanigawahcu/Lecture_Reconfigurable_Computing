#include <iomanip>
#include <iostream>
#include <chrono>
#include <vector>

// Fills a matrix with random numbers within the range [l_bound, u_bound).
void FillRand(std::vector<float> &m_matrix, int l_bound, int u_bound,
              int elements) {
  for (int element = 0; element < elements; element++) {
    m_matrix[element] =
        static_cast<float>(rand()) /
            (static_cast<float>((RAND_MAX) / (u_bound - l_bound))) +
        l_bound;
  }
}


// Output a matrix to the screen (assumes column-major format).
void PrintMat(std::vector<float> &m_matrix, int rows, int cols) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      // Copy old state of cout
      std::ios oldState(nullptr);
      oldState.copyfmt(std::cout);

      // Edit the output format of cout
      std::cout << std::fixed << std::setprecision(2);

      // Print the results
      std::cout << std::setw(8) << m_matrix[col * rows + row] << " ";

      // Restore the output format of cout
      std::cout.copyfmt(oldState);
    }
    std::cout << std::endl;
  }
}

// Transpose num_matrices matrices in m_matrix and store the results in
// m_transposed.
void TransposeMat(std::vector<float> &m_matrix,
                  std::vector<float> &m_transposed, int rows, int cols,
                  int num_matrices) {
  int matsize = rows * cols;

  for (int matrix_idx = 0; matrix_idx < num_matrices; matrix_idx++) {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        m_transposed[matrix_idx * matsize + row * cols + col] =
            m_matrix[matrix_idx * matsize + col * rows + row];
      }
    }
  }
}

// Multiply num_matrices pairs of matrices from a_matrix and b_matrix and store
// all the results in c_matrix.
void MatmulRef(std::vector<float> &a_matrix, std::vector<float> &b_matrix,
               std::vector<float> &c_matrix, int rows_a, int common, int cols_b,
               int num_matrices) {
  int matsize_a = rows_a * common;
  int matsize_b = cols_b * common;
  int matsize_c = rows_a * cols_b;

  for (int matrix_idx = 0; matrix_idx < num_matrices; matrix_idx++) {
    for (int col = 0; col < cols_b; col++) {
      for (int row = 0; row < rows_a; row++) {
        float sum = 0;
        for (int k = 0; k < common; k++) {
          sum += a_matrix[matrix_idx * matsize_a + k * rows_a + row] *
                 b_matrix[matrix_idx * matsize_b + col * common + k];
        }
        c_matrix[matrix_idx * matsize_c + col * rows_a + row] = sum;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  // Matrix paramters specified by build system
  constexpr int kRowsA = ROWS_A; // 64
  constexpr int kCommon = COMMON; // 64
  constexpr int kColsB = COLS_B;  // 64

  // Matrix sizes
  constexpr int kMatsizeA = kRowsA * kCommon;
  constexpr int kMatsizeB = kColsB * kCommon;
  constexpr int kMatsizeC = kRowsA * kColsB;

  // Repetitions and number of matrices to measure performance
  int repetitions = 1;
  constexpr int kNumMatrices = 1000;

  // Create arrays to hold the input and output matrices
  std::vector<float> a_matrix(kMatsizeA * kNumMatrices);
  std::vector<float> b_matrix(kMatsizeB * kNumMatrices);
  std::vector<float> c_matrix(kMatsizeC * kNumMatrices);

  // Generate random A and B matrices
  constexpr int kRandMin = 1;
  constexpr int kRandMax = 10;
  srand(1138);
  FillRand(a_matrix, kRandMin, kRandMax, kMatsizeA * kNumMatrices);
  FillRand(b_matrix, kRandMin, kRandMax, kMatsizeB * kNumMatrices);

  std::cout << " Matrix A size: " << kRowsA << " x " << kCommon << std::endl
            << " Matrix B size: " << kCommon << " x " << kColsB << std::endl
            << std::endl;
  std::cout << "Running matrix multiplication of " << kNumMatrices
            << ((kNumMatrices > 1) ? " matrices " : " matrix ") << repetitions
            << " times" << std::endl;

  // Calculate a reference to compare our answer to and store it in c_reference
  // NOTE: since the systolic matrix multiply interprets B as transposed, we
  // need to first transpose b_matrix to b_transposed to use it in the standard
  // MM algorithm
  std::vector<float> b_transposed(kMatsizeB * kNumMatrices);
  std::vector<float> c_reference(kMatsizeC * kNumMatrices);
  TransposeMat(b_matrix, b_transposed, kColsB, kCommon, kNumMatrices);

  auto start_time = std::chrono::system_clock::now();
  MatmulRef(a_matrix, b_transposed, c_reference, kRowsA, kCommon, kColsB,
            kNumMatrices);
  auto end_time = std::chrono::system_clock::now();
  auto duration = end_time - start_time;
  auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  std::cout << "Total duration: " << msec << " ms" << std::endl;
  std::cout << "Throughput    : " << (double)repetitions * kNumMatrices / msec
            << "k matrices/s" << std::endl; 

  // Print A, B, C and reference matrices
  for (int matrix_idx = 0; matrix_idx < kNumMatrices; matrix_idx++) {
    std::cout << std::endl << matrix_idx << std::endl;

    std::cout << std::endl << "Matrix A" << std::endl;
    std::vector<float> a_vector = {
        a_matrix.begin() + matrix_idx * kMatsizeA,
        a_matrix.begin() + (matrix_idx + 1) * kMatsizeA};
    PrintMat(a_vector, kRowsA, kCommon);

    std::cout << std::endl << "Matrix B" << std::endl;
    std::vector<float> b_vector = {
        b_transposed.begin() + matrix_idx * kMatsizeB,
        b_transposed.begin() + (matrix_idx + 1) * kMatsizeB};
    PrintMat(b_vector, kCommon, kColsB);

    std::cout << std::endl << "Matrix C reference" << std::endl;
    std::vector<float> c_ref_vector = {
        c_reference.begin() + matrix_idx * kMatsizeC,
        c_reference.begin() + (matrix_idx + 1) * kMatsizeC};
    PrintMat(c_ref_vector, kRowsA, kColsB);

  }

  return 0;

} // end of main
