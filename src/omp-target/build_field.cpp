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

// Allocate device buffers for the data arrays
void build_field(global_variables &globals) {

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

    tile_type &t = globals.chunk.tiles[tile];
    field_type &field = t.field;

    double *density0 = field.density0.data;
    double *density1 = field.density1.data;
    double *energy0 = field.energy0.data;
    double *energy1 = field.energy1.data;
    double *pressure = field.pressure.data;
    double *viscosity = field.viscosity.data;
    double *soundspeed = field.soundspeed.data;
    double *yvel0 = field.yvel0.data;
    double *yvel1 = field.yvel1.data;
    double *xvel0 = field.xvel0.data;
    double *xvel1 = field.xvel1.data;
    double *vol_flux_x = field.vol_flux_x.data;
    double *vol_flux_y = field.vol_flux_y.data;
    double *mass_flux_x = field.mass_flux_x.data;
    double *mass_flux_y = field.mass_flux_y.data;
    double *work_array1 = field.work_array1.data;
    double *work_array2 = field.work_array2.data;
    double *work_array3 = field.work_array3.data;
    double *work_array4 = field.work_array4.data;
    double *work_array5 = field.work_array5.data;
    double *work_array6 = field.work_array6.data;
    double *work_array7 = field.work_array7.data;
    double *cellx = field.cellx.data;
    double *celldx = field.celldx.data;
    double *celly = field.celly.data;
    double *celldy = field.celldy.data;
    double *vertexx = field.vertexx.data;
    double *vertexdx = field.vertexdx.data;
    double *vertexy = field.vertexy.data;
    double *vertexdy = field.vertexdy.data;
    double *volume = field.volume.data;
    double *xarea = field.xarea.data;
    double *yarea = field.yarea.data;

#pragma omp target enter data map(alloc : density0[ : field.density0.N()]) map(alloc : density1[ : field.density1.N()])                    \
    map(alloc : energy0[ : field.energy0.N()]) map(alloc : energy1[ : field.energy1.N()]) map(alloc : pressure[ : field.pressure.N()])     \
    map(alloc : viscosity[ : field.viscosity.N()]) map(alloc : soundspeed[ : field.soundspeed.N()]) map(alloc : yvel0[ : field.yvel0.N()]) \
    map(alloc : yvel1[ : field.yvel1.N()]) map(alloc : xvel0[ : field.xvel0.N()]) map(alloc : xvel1[ : field.xvel1.N()])                   \
    map(alloc : vol_flux_x[ : field.vol_flux_x.N()]) map(alloc : vol_flux_y[ : field.vol_flux_y.N()])                                      \
    map(alloc : mass_flux_x[ : field.mass_flux_x.N()]) map(alloc : mass_flux_y[ : field.mass_flux_y.N()])                                  \
    map(alloc : work_array1[ : field.work_array1.N()]) map(alloc : work_array2[ : field.work_array2.N()])                                  \
    map(alloc : work_array3[ : field.work_array3.N()]) map(alloc : work_array4[ : field.work_array4.N()])                                  \
    map(alloc : work_array5[ : field.work_array5.N()]) map(alloc : work_array6[ : field.work_array6.N()])                                  \
    map(alloc : work_array7[ : field.work_array7.N()]) map(alloc : cellx[ : field.cellx.N()]) map(alloc : celldx[ : field.celldx.N()])     \
    map(alloc : celly[ : field.celly.N()]) map(alloc : celldy[ : field.celldy.N()]) map(alloc : vertexx[ : field.vertexx.N()])             \
    map(alloc : vertexdx[ : field.vertexdx.N()]) map(alloc : vertexy[ : field.vertexy.N()]) map(alloc : vertexdy[ : field.vertexdy.N()])   \
    map(alloc : volume[ : field.volume.N()]) map(alloc : xarea[ : field.xarea.N()]) map(alloc : yarea[ : field.yarea.N()])

    const int xrange = (t.info.t_xmax + 2) - (t.info.t_xmin - 2) + 1;
    const int yrange = (t.info.t_ymax + 2) - (t.info.t_ymin - 2) + 1;

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

    //		Kokkos::MDRangePolicy <Kokkos::Rank<2>> loop_bounds_1({0, 0}, {xrange + 1, yrange + 1});

    // Nested loop over (t_ymin-2:t_ymax+3) and (t_xmin-2:t_xmax+3) inclusive

    const int vels_wk_stride = field.vels_wk_stride;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(globals.context.use_target)
    for (int j = 0; j < (yrange + 1); j++) {
      for (int i = 0; i < (xrange + 1); i++) {
        work_array1[i + j * vels_wk_stride] = 0.0;
        work_array2[i + j * vels_wk_stride] = 0.0;
        work_array3[i + j * vels_wk_stride] = 0.0;
        work_array4[i + j * vels_wk_stride] = 0.0;
        work_array5[i + j * vels_wk_stride] = 0.0;
        work_array6[i + j * vels_wk_stride] = 0.0;
        work_array7[i + j * vels_wk_stride] = 0.0;
        xvel0[i + j * vels_wk_stride] = 0.0;
        xvel1[i + j * vels_wk_stride] = 0.0;
        yvel0[i + j * vels_wk_stride] = 0.0;
        yvel1[i + j * vels_wk_stride] = 0.0;
      }
    }

    // Nested loop over (t_ymin-2:t_ymax+2) and (t_xmin-2:t_xmax+2) inclusive
    const int base_stride = field.base_stride;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(globals.context.use_target)
    for (int j = 0; j < (yrange); j++) {
      for (int i = 0; i < (xrange); i++) {
        density0[i + j * base_stride] = 0.0;
        density1[i + j * base_stride] = 0.0;
        energy0[i + j * base_stride] = 0.0;
        energy1[i + j * base_stride] = 0.0;
        pressure[i + j * base_stride] = 0.0;
        viscosity[i + j * base_stride] = 0.0;
        soundspeed[i + j * base_stride] = 0.0;
        volume[i + j * base_stride] = 0.0;
      }
    }

    // Nested loop over (t_ymin-2:t_ymax+2) and (t_xmin-2:t_xmax+3) inclusive
    const int flux_x_stride = field.flux_x_stride;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(globals.context.use_target)
    for (int j = 0; j < (yrange); j++) {
      for (int i = 0; i < (xrange); i++) {
        vol_flux_x[i + j * flux_x_stride] = 0.0;
        mass_flux_x[i + j * flux_x_stride] = 0.0;
        xarea[i + j * flux_x_stride] = 0.0;
      }
    }

    // Nested loop over (t_ymin-2:t_ymax+3) and (t_xmin-2:t_xmax+2) inclusive
    const int flux_y_stride = field.flux_y_stride;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(globals.context.use_target)
    for (int j = 0; j < (yrange + 1); j++) {
      for (int i = 0; i < (xrange); i++) {
        vol_flux_y[i + j * flux_y_stride] = 0.0;
        mass_flux_y[i + j * flux_y_stride] = 0.0;
        yarea[i + j * flux_y_stride] = 0.0;
      }
    }

// (t_xmin-2:t_xmax+2) inclusive
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int id = 0; id < (xrange); id++) {
      cellx[id] = 0.0;
      celldx[id] = 0.0;
    }

// (t_ymin-2:t_ymax+2) inclusive
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int id = 0; id < (yrange); id++) {
      celly[id] = 0.0;
      celldy[id] = 0.0;
    }

// (t_xmin-2:t_xmax+3) inclusive
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int id = 0; id < (xrange + 1); id++) {
      vertexx[id] = 0.0;
      vertexdx[id] = 0.0;
    }

// (t_ymin-2:t_ymax+3) inclusive
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int id = 0; id < (yrange + 1); id++) {
      vertexy[id] = 0.0;
      vertexdy[id] = 0.0;
    }
  }
}
