/*****************************************************************************
 * Whack
 * Copyright (C) 2023 Adam Celarek
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#pragma once

#include <ostream>
#include <string_view>

#include "Tensor.h"
#include "TensorView.h"

namespace whack::detail {
// from https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c/64490578#64490578
template <typename T>
constexpr std::string_view type_name();

template <>
constexpr std::string_view type_name<void>()
{
    return "void";
}

using type_name_prober = void;

template <typename T>
constexpr std::string_view wrapped_type_name()
{
#if __cplusplus >= 202002L
    return std::source_location::current().function_name();
#else
#if defined(__clang__) || defined(__GNUC__)
    return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
    return __FUNCSIG__;
#else
#error "Unsupported compiler"
#endif
#endif // __cplusplus >= 202002L
}

constexpr std::size_t wrapped_type_name_prefix_length()
{
    return wrapped_type_name<type_name_prober>()
        .find(type_name<type_name_prober>());
}

constexpr std::size_t wrapped_type_name_suffix_length()
{
    return wrapped_type_name<type_name_prober>().length()
        - wrapped_type_name_prefix_length()
        - type_name<type_name_prober>().length();
}

template <typename T>
constexpr std::string_view type_name()
{
    constexpr auto wrapped_name = detail::wrapped_type_name<T>();
    constexpr auto prefix_length = detail::wrapped_type_name_prefix_length();
    constexpr auto suffix_length = detail::wrapped_type_name_suffix_length();
    constexpr auto type_name_length = wrapped_name.length() - prefix_length - suffix_length;
    return wrapped_name.substr(prefix_length, type_name_length);
}

template <typename T, typename IndexStoreType, typename IndexCalculateType>
void write(std::ostream& os, whack::TensorView<T, 2, IndexStoreType, IndexCalculateType> view)
{
    if (view.size(0) == 0 || view.size(1) == 1)
        return;

    const auto print_row = [&os, &view](unsigned row) {
        os << "{" << view(row, 0);
        for (int j = 1; j < view.size(1); ++j) {
            os << ", " << view(row, j);
        }
        os << "}";
    };
    print_row(0);

    for (int row = 1; row < view.size(0); ++row) {
        os << "," << std::endl;
        print_row(row);
    }
}

template <typename T, typename IndexStoreType, typename IndexCalculateType>
void write(std::ostream& os, whack::TensorView<T, 1, IndexStoreType, IndexCalculateType> view)
{
    if (view.size(0) == 0)
        return;
    os << view(0);
    for (int i = 1; i < view.size(0); ++i) {
        os << ", " << view(i);
    }
}

} // namespace whack::detail

template <typename T, whack::size_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
std::ostream& operator<<(std::ostream& os, const whack::Tensor<T, n_dims, IndexStoreType, IndexCalculateType>& data)
{
    const std::string location = data.location() == whack::Location::Device ? "device" : (data.location() == whack::Location::Host ? "host" : "invalid");
    os << location << " Tensor<" << whack::detail::type_name<T>() << ", " << n_dims << "> = ";
    if (n_dims > 1)
        os << std::endl;
    os << "{";
    const auto host_copy = data.host_copy();
    whack::detail::write(os, host_copy.view());
    os << "}";

    return os;
}
