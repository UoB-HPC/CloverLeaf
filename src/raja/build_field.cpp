/*
 Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/.
 */

// @brief  Allocates the data for each mesh chunk
// @author Wayne Gaudin
// @details The data fields for the mesh chunk are allocated based on the mesh
// size.

#include "build_field.h"
#include "context.h"

#include <RAJA/RAJA.hpp>

// Allocate Kokkos Views for the data arrays
void build_field(global_variables &globals) {

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

    tile_type &t = globals.chunk.tiles[tile];

    const size_t xrange = (t.info.t_xmax + 2) - (t.info.t_xmin - 2) + 1;
    const size_t yrange = (t.info.t_ymax + 2) - (t.info.t_ymin - 2) + 1;

    // (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+2)

    //		t.field.density0 = Buffer2D<double>(range<2>(xrange, yrange));
    //		t.field.density1 = Buffer2D<double>(range<2>(xrange, yrange));
    //		t.field.energy0 = Buffer2D<double>(range<2>(xrange, yrange));
    //		t.field.energy1 = Buffer2D<double>(range<2>(xrange, yrange));
    //		t.field.pressure = Buffer2D<double>(range<2>(xrange, yrange));
    //		t.field.viscosity = Buffer2D<double>(range<2>(xrange, yrange));
    //		t.field.soundspeed = Buffer2D<double>(range<2>(xrange, yrange));
    //
    //		// (t_xmin-2:t_xmax+3, t_ymin-2:t_ymax+3)
    //		t.field.xvel0 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
    //		t.field.xvel1 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
    //		t.field.yvel0 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
    //		t.field.yvel1 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
    //
    //		// (t_xmin-2:t_xmax+3, t_ymin-2:t_ymax+2)
    //		t.field.vol_flux_x = Buffer2D<double>(range<2>(xrange + 1, yrange));
    //		t.field.mass_flux_x = Buffer2D<double>(range<2>(xrange + 1, yrange));
    //		// (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+3)
    //		t.field.vol_flux_y = Buffer2D<double>(range<2>(xrange, yrange + 1));
    //		t.field.mass_flux_y = Buffer2D<double>(range<2>(xrange, yrange + 1));
    //
    //		// (t_xmin-2:t_xmax+3, t_ymin-2:t_ymax+3)
    //		t.field.work_array1 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
    //		t.field.work_array2 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
    //		t.field.work_array3 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
    //		t.field.work_array4 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
    //		t.field.work_array5 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
    //		t.field.work_array6 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
    //		t.field.work_array7 = Buffer2D<double>(range<2>(xrange + 1, yrange + 1));
    //
    //		// (t_xmin-2:t_xmax+2)
    //		t.field.cellx = Buffer1D<double>(range<1>(xrange));
    //		t.field.celldx = Buffer1D<double>(range<1>(xrange));
    //		// (t_ymin-2:t_ymax+2)
    //		t.field.celly = Buffer1D<double>(range<1>(yrange));
    //		t.field.celldy = Buffer1D<double>(range<1>(yrange));
    //		// (t_xmin-2:t_xmax+3)
    //		t.field.vertexx = Buffer1D<double>(range<1>(xrange + 1));
    //		t.field.vertexdx = Buffer1D<double>(range<1>(xrange + 1));
    //		// (t_ymin-2:t_ymax+3)
    //		t.field.vertexy = Buffer1D<double>(range<1>(yrange + 1));
    //		t.field.vertexdy = Buffer1D<double>(range<1>(yrange + 1));
    //
    //		// (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+2)
    //		t.field.volume = Buffer2D<double>(range<2>(xrange, yrange));
    //		// (t_xmin-2:t_xmax+3, t_ymin-2:t_ymax+2)
    //		t.field.xarea = Buffer2D<double>(range<2>(xrange + 1, yrange));
    //		// (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+3)
    //		t.field.yarea = Buffer2D<double>(range<2>(xrange, yrange + 1));

    // Zeroing isn't strictly necessary but it ensures physical pages
    // are allocated. This prevents first touch overheads in the main code
    // cycle which can skew timings in the first step

    // Take a reference to the lowest structure, as Kokkos device cannot necessarily chase through the structure.
    field_type &field = t.field;

    //		Kokkos::MDRangePolicy <Kokkos::Rank<2>> loop_bounds_1({0, 0}, {xrange + 1, yrange + 1});

    // Nested loop over (t_ymin-2:t_ymax+3) and (t_xmin-2:t_xmax+3) inclusive
    // clover::par_ranged2(Range2d{0u, 0u, xrange + 1, yrange + 1}, [=] DEVICE_KERNEL(const int i, const int j) {
    const RAJA::TypedRangeSegment<int> row_Range(0,  yrange + 1);
    const RAJA::TypedRangeSegment<int> col_Range(0,  xrange + 1);
    RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range, row_Range),
        [=] RAJA_HOST_DEVICE (const int i, const int j) {
      field.work_array1(i, j) = 0.0;
      field.work_array2(i, j) = 0.0;
      field.work_array3(i, j) = 0.0;
      field.work_array4(i, j) = 0.0;
      field.work_array5(i, j) = 0.0;
      field.work_array6(i, j) = 0.0;
      field.work_array7(i, j) = 0.0;

      field.xvel0(i, j) = 0.0;
      field.xvel1(i, j) = 0.0;
      field.yvel0(i, j) = 0.0;
      field.yvel1(i, j) = 0.0;
    });

    // Nested loop over (t_ymin-2:t_ymax+2) and (t_xmin-2:t_xmax+2) inclusive
    // clover::par_ranged2(Range2d{0u, 0u, xrange, yrange}, [=] DEVICE_KERNEL(const int i, const int j) {
    const RAJA::TypedRangeSegment<int> row_Range1(0,  yrange);
    const RAJA::TypedRangeSegment<int> col_Range1(0,  xrange);
    RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range1, row_Range1),
        [=] RAJA_HOST_DEVICE (const int i, const int j) {
      field.density0(i, j) = 0.0;
      field.density1(i, j) = 0.0;
      field.energy0(i, j) = 0.0;
      field.energy1(i, j) = 0.0;
      field.pressure(i, j) = 0.0;
      field.viscosity(i, j) = 0.0;
      field.soundspeed(i, j) = 0.0;
      field.volume(i, j) = 0.0;
    });

    // Nested loop over (t_ymin-2:t_ymax+2) and (t_xmin-2:t_xmax+3) inclusive
    // clover::par_ranged2(Range2d{0u, 0u, xrange, yrange}, [=] DEVICE_KERNEL(const int i, const int j) {
    RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range1, row_Range1),
        [=] RAJA_HOST_DEVICE (const int i, const int j) {
      field.vol_flux_x(i, j) = 0.0;
      field.mass_flux_x(i, j) = 0.0;
      field.xarea(i, j) = 0.0;
    });

    // Nested loop over (t_ymin-2:t_ymax+3) and (t_xmin-2:t_xmax+2) inclusive
    //clover::par_ranged2(Range2d{0u, 0u, xrange, yrange + 1}, [=] DEVICE_KERNEL(const int i, const int j) {
    const RAJA::TypedRangeSegment<int> row_Range2(0,  yrange + 1);
    const RAJA::TypedRangeSegment<int> col_Range2(0,  xrange);
    RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range2, row_Range2),
        [=] RAJA_HOST_DEVICE (const int i, const int j) {
      field.vol_flux_y(i, j) = 0.0;
      field.mass_flux_y(i, j) = 0.0;
      field.yarea(i, j) = 0.0;
    });

    // (t_xmin-2:t_xmax+2) inclusive
    RAJA::forall<raja_default_policy>(
        RAJA::TypedRangeSegment<int>(0u, xrange), [=] RAJA_HOST_DEVICE (int id ) {
      field.cellx[id] = 0.0;
      field.celldx[id] = 0.0;
    });

    // (t_ymin-2:t_ymax+2) inclusive
    RAJA::forall<raja_default_policy>(
        RAJA::TypedRangeSegment<int>(0u, yrange), [=] RAJA_HOST_DEVICE (int id ) {
      field.celly[id] = 0.0;
      field.celldy[id] = 0.0;
    });

    // (t_xmin-2:t_xmax+3) inclusive
    RAJA::forall<raja_default_policy>(
        RAJA::TypedRangeSegment<int>(0u, xrange+1), [=] RAJA_HOST_DEVICE (int id ) {
      field.vertexx[id] = 0.0;
      field.vertexdx[id] = 0.0;
    });

    // (t_ymin-2:t_ymax+3) inclusive
    RAJA::forall<raja_default_policy>(
        RAJA::TypedRangeSegment<int>(0u, yrange+1), [=] RAJA_HOST_DEVICE (int id ) {
      field.vertexy[id] = 0.0;
      field.vertexdy[id] = 0.0;
    });
  }
}
