#include <algorithm>
#include <random>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "memory/buffer.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "spla/config.h"
#include "spla/spla.hpp"
#include "util/blas_interface.hpp"
#include "util/common_types.hpp"

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include "gpu_util/gpu_runtime_api.hpp"
#include "memory/gpu_allocator.hpp"
#endif

using namespace spla;

static auto convert_op(SplaOperation op) -> ::spla::blas::Operation {
  if (op == SPLA_OP_TRANSPOSE) return ::spla::blas::Operation::TRANS;
  if (op == SPLA_OP_CONJ_TRANSPOSE) return ::spla::blas::Operation::CONJ_TRANS;
  return ::spla::blas::Operation::NONE;
}

template <typename T>
class GemmTest
    : public ::testing::TestWithParam<std::tuple<int, int, int, SplaOperation, SplaOperation>> {
protected:
  using ValueType = T;

  GemmTest()
      : opA_(std::get<3>(GetParam())),
        opB_(std::get<4>(GetParam())),
        m_(std::get<0>(GetParam())),
        n_(std::get<1>(GetParam())),
        k_(std::get<2>(GetParam())),
        lda_((opA_ == SPLA_OP_NONE ? m_ : k_) + 5),
        ldb_((opB_ == SPLA_OP_NONE ? k_ : n_) + 6),
        ldc_(m_ + 7) {
    std::uniform_real_distribution<double> valueDistribution(0.0, 100.0);

    vecA_.resize((opA_ == SPLA_OP_NONE ? k_ : m_) * lda_);
    vecB_.resize((opB_ == SPLA_OP_NONE ? n_ : k_) * ldb_);
    vecC_.resize(ldc_ * n_);

    for (auto& val : vecA_) {
      val = valueDistribution(staticRandGen_);
    }
    for (auto& val : vecB_) {
      val = valueDistribution(staticRandGen_);
    }
    for (auto& val : vecC_) {
      val = valueDistribution(staticRandGen_);
    }
    vecCRef_ = vecC_;  // copy C to compare result
  }

  auto mulitply_host() -> void {
    Context ctx(SPLA_PU_HOST);

    // compute reference by calling blas library directly
    ::spla::blas::gemm(::spla::blas::Order::COL_MAJOR, convert_op(opA_), convert_op(opB_), m_, n_,
                       k_, 2.0, vecA_.data(), lda_, vecB_.data(), ldb_, 3.0, vecCRef_.data(), ldc_);

    // compute with public gemm interface
    gemm(opA_, opB_, m_, n_, k_, 2.0, vecA_.data(), lda_, vecB_.data(), ldb_, 3.0, vecC_.data(),
         ldc_, ctx);

    for (std::size_t i = 0; i < vecC_.size(); ++i) {
      ASSERT_NEAR(std::real(vecC_[i]), std::real(vecCRef_[i]), 1e-6);
      ASSERT_NEAR(std::imag(vecC_[i]), std::imag(vecCRef_[i]), 1e-6);
    }
  }

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
  auto mulitply_gpu() -> void {
    Context ctx(SPLA_PU_GPU);

    // compute reference by calling blas library directly
    ::spla::blas::gemm(::spla::blas::Order::COL_MAJOR, convert_op(opA_), convert_op(opB_), m_, n_,
                       k_, 2.0, vecA_.data(), lda_, vecB_.data(), ldb_, 3.0, vecCRef_.data(), ldc_);

    // compute with public gemm interface
    gemm(opA_, opB_, m_, n_, k_, 2.0, vecA_.data(), lda_, vecB_.data(), ldb_, 3.0, vecC_.data(),
         ldc_, ctx);

    for (std::size_t i = 0; i < vecC_.size(); ++i) {
      ASSERT_NEAR(std::real(vecC_[i]), std::real(vecCRef_[i]), 1e-6);
      ASSERT_NEAR(std::imag(vecC_[i]), std::imag(vecCRef_[i]), 1e-6);
    }
  }

  auto mulitply_gpu_from_gpu() -> void {
    Context ctx(SPLA_PU_GPU);

    // compute reference by calling blas library directly
    ::spla::blas::gemm(::spla::blas::Order::COL_MAJOR, convert_op(opA_), convert_op(opB_), m_, n_,
                       k_, 2.0, vecA_.data(), lda_, vecB_.data(), ldb_, 3.0, vecCRef_.data(), ldc_);

    Buffer<GPUAllocator> gpuBufferA;
    gpuBufferA.resize<T>(vecA_.size());
    Buffer<GPUAllocator> gpuBufferB;
    gpuBufferB.resize<T>(vecB_.size());
    Buffer<GPUAllocator> gpuBufferC;
    gpuBufferC.resize<T>(vecC_.size());

    if (vecA_.size())
      gpu::check_status(gpu::memcpy(static_cast<void*>(gpuBufferA.data<T>()),
                                    static_cast<const void*>(vecA_.data()),
                                    vecA_.size() * sizeof(T), gpu::flag::MemcpyHostToDevice));
    if (vecB_.size())
      gpu::check_status(gpu::memcpy(static_cast<void*>(gpuBufferB.data<T>()),
                                    static_cast<const void*>(vecB_.data()),
                                    vecB_.size() * sizeof(T), gpu::flag::MemcpyHostToDevice));
    if (vecC_.size())
      gpu::check_status(gpu::memcpy(static_cast<void*>(gpuBufferC.data<T>()),
                                    static_cast<const void*>(vecC_.data()),
                                    vecC_.size() * sizeof(T), gpu::flag::MemcpyHostToDevice));

    // compute with public gemm interface
    gemm(opA_, opB_, m_, n_, k_, 2.0, gpuBufferA.empty() ? nullptr : gpuBufferA.data<T>(), lda_,
         gpuBufferA.empty() ? nullptr : gpuBufferB.data<T>(), ldb_, 3.0,
         gpuBufferC.empty() ? nullptr : gpuBufferC.data<T>(), ldc_, ctx);

    if (vecC_.size())
      gpu::check_status(gpu::memcpy(static_cast<void*>(vecC_.data()),
                                    static_cast<const void*>(gpuBufferC.data<T>()),
                                    vecC_.size() * sizeof(T), gpu::flag::MemcpyDeviceToHost));

    for (std::size_t i = 0; i < vecC_.size(); ++i) {
      ASSERT_NEAR(std::real(vecC_[i]), std::real(vecCRef_[i]), 1e-6);
      ASSERT_NEAR(std::imag(vecC_[i]), std::imag(vecCRef_[i]), 1e-6);
    }
  }
#endif

  SplaOperation opA_, opB_;
  int m_, n_, k_, lda_, ldb_, ldc_;
  std::vector<T> vecA_, vecB_, vecC_, vecCRef_;

  static std::mt19937 staticRandGen_;  // must produce same numbers on each rank
};

template <typename T>
std::mt19937 GemmTest<T>::staticRandGen_(42);

typedef GemmTest<double> GemmScalar;

typedef GemmTest<std::complex<double>> GemmComplex;

TEST_P(GemmScalar, Host) {
  try {
    this->mulitply_host();
  } catch (const std::exception& e) {
    ASSERT_TRUE(false);
  }
}

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
TEST_P(GemmScalar, GPU) {
  try {
    this->mulitply_gpu();
  } catch (const std::exception& e) {
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmScalar, GPUFromGPU) {
  try {
    this->mulitply_gpu_from_gpu();
  } catch (const std::exception& e) {
    ASSERT_TRUE(false);
  }
}
#endif

TEST_P(GemmComplex, Host) {
  try {
    this->mulitply_host();
  } catch (const std::exception& e) {
    ASSERT_TRUE(false);
  }
}

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
TEST_P(GemmComplex, GPU) {
  try {
    this->mulitply_gpu();
  } catch (const std::exception& e) {
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmComplex, GPUFromGPU) {
  try {
    this->mulitply_gpu_from_gpu();
  } catch (const std::exception& e) {
    ASSERT_TRUE(false);
  }
}
#endif

static auto param_type_names(
    const ::testing::TestParamInfo<std::tuple<int, int, int, SplaOperation, SplaOperation>>& info)
    -> std::string {
  std::stringstream stream;
  if (std::get<3>(info.param) == SPLA_OP_NONE) stream << "N_";
  if (std::get<3>(info.param) == SPLA_OP_TRANSPOSE) stream << "T_";
  if (std::get<3>(info.param) == SPLA_OP_CONJ_TRANSPOSE) stream << "C_";
  if (std::get<4>(info.param) == SPLA_OP_NONE) stream << "N_";
  if (std::get<4>(info.param) == SPLA_OP_TRANSPOSE) stream << "T_";
  if (std::get<4>(info.param) == SPLA_OP_CONJ_TRANSPOSE) stream << "C_";
  stream << "m_" << std::get<0>(info.param) << "_";
  stream << "n_" << std::get<1>(info.param) << "_";
  stream << "k_" << std::get<2>(info.param);

  return stream.str();
}

INSTANTIATE_TEST_CASE_P(FullGemmTest, GemmScalar,
                        ::testing::Combine(::testing::Values(1, 13, 32, 263),
                                           ::testing::Values(1, 13, 32, 263),
                                           ::testing::Values(1, 13, 32, 263),
                                           ::testing::Values(SPLA_OP_NONE, SPLA_OP_CONJ_TRANSPOSE),
                                           ::testing::Values(SPLA_OP_NONE, SPLA_OP_CONJ_TRANSPOSE)),
                        param_type_names);

INSTANTIATE_TEST_CASE_P(FullGemmTest, GemmComplex,
                        ::testing::Combine(::testing::Values(1, 13, 32, 263),
                                           ::testing::Values(1, 13, 32, 263),
                                           ::testing::Values(1, 13, 32, 263),
                                           ::testing::Values(SPLA_OP_NONE, SPLA_OP_CONJ_TRANSPOSE),
                                           ::testing::Values(SPLA_OP_NONE, SPLA_OP_CONJ_TRANSPOSE)),
                        param_type_names);
