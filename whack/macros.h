/*
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

#ifdef __CUDACC__
#define WHACK_DEVICES __host__ __device__
#else
#define WHACK_DEVICES
#endif

#ifdef NDEBUG
#define WHACK_INLINE __forceinline__
#else
#define WHACK_INLINE
#endif

#define WHACK_DEVICES_INLINE WHACK_DEVICES WHACK_INLINE
