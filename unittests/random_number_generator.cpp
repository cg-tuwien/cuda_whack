

#include "whack/RandomNumberGenerator.h"
#include "whack/Tensor.h"

whack::Tensor<float, 3> compute_random_numbers_with_fixed_seed(bool cuda);

// template <int n_dims>
// torch::Tensor compute_Nd_normal(bool cuda);

// extern template torch::Tensor compute_Nd_normal<2>(bool cuda);
// extern template torch::Tensor compute_Nd_normal<3>(bool cuda);

// TEMPLATE_TEST_CASE("random_number_generator", "", std::true_type, std::false_type)
//{
//     bool use_cuda = TestType::value;
//     auto rnd = compute_random_numbers_with_fixed_seed(use_cuda);
//     //    std::cout << rnd << std::endl;

//    CHECK((rnd == torch::zeros_like(rnd)).sum().cpu().item<int64_t>() < rnd.numel() * 0.001);
//    CHECK(rnd.mean().cpu().item<float>() == Approx(0.0).scale(1).epsilon(0.001));
//    CHECK(rnd.var().cpu().item<float>() == Approx(1.0).scale(1).epsilon(0.01));
//    //    CHECK(rnd.view({-1, 2}).cov().cpu().item<float>() == Approx(1.0).scale(1).epsilon(0.001));
//    auto rnd2 = compute_random_numbers_with_fixed_seed(use_cuda);
//    CHECK((rnd == rnd2).all().cpu().item<bool>() == true);
//}

// TEMPLATE_TEST_CASE("random_number_generator 2d cov", "", std::true_type, std::false_type)
//{
//     bool use_cuda = TestType::value;
//     auto rnd = compute_Nd_normal<2>(use_cuda);
//     //    std::cout << rnd << std::endl;
//     CHECK((rnd == torch::zeros_like(rnd)).sum().cpu().item<int64_t>() < rnd.numel() * 0.001);
//     CHECK(rnd.mean().cpu().item<float>() == Approx(0.0).scale(1).epsilon(0.01));
//     CHECK(rnd.var().cpu().item<float>() == Approx(1.0).scale(1).epsilon(0.01));
//     const auto cov = rnd.view({ -1, 2 }).transpose(0, 1).cov().cpu();
//     CHECK(cov.index({ 0, 0 }).item<float>() == Approx(1.0).scale(1).epsilon(0.005));
//     CHECK(cov.index({ 0, 1 }).item<float>() == Approx(0.0).scale(1).epsilon(0.005));
//     CHECK(cov.index({ 1, 0 }).item<float>() == Approx(0.0).scale(1).epsilon(0.005));
//     CHECK(cov.index({ 1, 1 }).item<float>() == Approx(1.0).scale(1).epsilon(0.005));
//     //    std::cout << rnd.view({-1, 2}).sizes() << std::endl;
//     //    std::cout << rnd.view({-1, 2}).transpose(0, 1).cov() << std::endl;
//     //    CHECK(rnd.view({-1, 2}).cov().cpu().item<float>() == Approx(1.0).scale(1).epsilon(0.001));
//     auto rnd2 = compute_Nd_normal<2>(use_cuda);
//     CHECK((rnd == rnd2).all().cpu().item<bool>() == true);
// }

// TEMPLATE_TEST_CASE("random_number_generator 3d cov", "", std::true_type, std::false_type)
//{
//     bool use_cuda = TestType::value;
//     auto rnd = compute_Nd_normal<3>(use_cuda);
//     //    std::cout << rnd.sizes() << std::endl;
//     CHECK((rnd == torch::zeros_like(rnd)).sum().cpu().item<int64_t>() < rnd.numel() * 0.001);
//     CHECK(rnd.mean().cpu().item<float>() == Approx(0.0).scale(1).epsilon(0.01));
//     CHECK(rnd.var().cpu().item<float>() == Approx(1.0).scale(1).epsilon(0.01));
//     const auto cov = rnd.view({ -1, 3 }).transpose(0, 1).cov().cpu();
//     CHECK(cov.index({ 0, 0 }).item<float>() == Approx(1.0).scale(1).epsilon(0.005));
//     CHECK(cov.index({ 0, 1 }).item<float>() == Approx(0.0).scale(1).epsilon(0.005));
//     CHECK(cov.index({ 0, 2 }).item<float>() == Approx(0.0).scale(1).epsilon(0.005));
//     CHECK(cov.index({ 1, 0 }).item<float>() == Approx(0.0).scale(1).epsilon(0.005));
//     CHECK(cov.index({ 1, 1 }).item<float>() == Approx(1.0).scale(1).epsilon(0.005));
//     CHECK(cov.index({ 1, 2 }).item<float>() == Approx(0.0).scale(1).epsilon(0.005));
//     CHECK(cov.index({ 2, 0 }).item<float>() == Approx(0.0).scale(1).epsilon(0.005));
//     CHECK(cov.index({ 2, 1 }).item<float>() == Approx(0.0).scale(1).epsilon(0.005));
//     CHECK(cov.index({ 2, 2 }).item<float>() == Approx(1.0).scale(1).epsilon(0.005));
//     //    std::cout << rnd.view({-1, 3}).sizes() << std::endl;
//     //    std::cout << cov << std::endl;
//     //    CHECK(rnd.view({-1, 2}).cov().cpu().item<float>() == Approx(1.0).scale(1).epsilon(0.001));
//     auto rnd2 = compute_Nd_normal<3>(use_cuda);
//     CHECK((rnd == rnd2).all().cpu().item<bool>() == true);
// }
