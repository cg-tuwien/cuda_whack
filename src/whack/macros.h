/****************************************************************************
 *  Copyright (C) 2023 Adam Celarek (github.com/adam-ce, github.com/cg-tuwien)
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 *  of the Software, and to permit persons to whom the Software is furnished to do so,
 *  subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 ****************************************************************************/

#pragma once
#include <cuda_runtime.h>

//#ifdef __CUDACC__
//#warning using nvcc
//#ifdef __CUDA_ARCH__
//#warning device code trajectory
//#else
//#warning nvcc host code trajectory
//#endif
//#else
//#warning non-nvcc code trajectory
//#endif

#ifdef __CUDACC__
#define WHACK_DEVICES __host__ __device__
#else
#define WHACK_DEVICES
#endif

#if defined(NDEBUG)
#if defined(__CUDACC__)
#define WHACK_INLINE __forceinline__
#else
#define WHACK_INLINE inline
#endif
#else
#define WHACK_INLINE
#endif

#define WHACK_DEVICES_INLINE WHACK_DEVICES WHACK_INLINE
