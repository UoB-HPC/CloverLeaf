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

// @brief Driver for chunk initialisation.
// @author Wayne Gaudin
// @details Invokes the user specified chunk initialisation kernel.
// @brief Fortran chunk initialisation kernel.
// @author Wayne Gaudin
// @details Calculates mesh geometry for the mesh chunk based on the mesh size.

#include "initialise_chunk.h"

void initialise_chunk(const int tile, global_variables &globals) {

  double dx = (globals.config.grid.xmax - globals.config.grid.xmin) / (double)(globals.config.grid.x_cells);
  double dy = (globals.config.grid.ymax - globals.config.grid.ymin) / (double)(globals.config.grid.y_cells);

  double xmin = globals.config.grid.xmin + dx * (double)(globals.chunk.tiles[tile].info.t_left - 1);

  double ymin = globals.config.grid.ymin + dy * (double)(globals.chunk.tiles[tile].info.t_bottom - 1);

  size_t x_min = globals.chunk.tiles[tile].info.t_xmin;
  size_t x_max = globals.chunk.tiles[tile].info.t_xmax;
  size_t y_min = globals.chunk.tiles[tile].info.t_ymin;
  size_t y_max = globals.chunk.tiles[tile].info.t_ymax;

  size_t xrange = (x_max + 3) - (x_min - 2) + 1;
  size_t yrange = (y_max + 3) - (y_min - 2) + 1;

  // Take a reference to the lowest structure, as Kokkos device cannot necessarily chase through the structure.
  field_type &field = globals.chunk.tiles[tile].field;

  double *vertexx = field.vertexx.data;
  double *vertexdx = field.vertexdx.data;

#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
  for (int j = 0; j < (xrange); j++) {
    vertexx[j] = xmin + dx * (j - 1 - x_min);
    vertexdx[j] = dx;
  }

  double *vertexy = field.vertexy.data;
  double *vertexdy = field.vertexdy.data;

#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
  for (int k = 0; k < (yrange); k++) {
    vertexy[k] = ymin + dy * (k - 1 - y_min);
    vertexdy[k] = dy;
  }

  size_t xrange1 = (x_max + 2) - (x_min - 2) + 1;
  size_t yrange1 = (y_max + 2) - (y_min - 2) + 1;

  double *cellx = field.cellx.data;
  double *celldx = field.celldx.data;
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
  for (int j = 0; j < (xrange1); j++) {
    cellx[j] = 0.5 * (vertexx[j] + vertexx[j + 1]);
    celldx[j] = dx;
  }

  double *celly = field.celly.data;
  double *celldy = field.celldy.data;
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
  for (int k = 0; k < (yrange1); k++) {
    celly[k] = 0.5 * (vertexy[k] + vertexy[k + 1]);
    celldy[k] = dy;
  }

  size_t base_stride = field.base_stride;
  size_t flux_x_stride = field.flux_x_stride;
  size_t flux_y_stride = field.flux_y_stride;

  double *volume = field.volume.data;
  double *xarea = field.xarea.data;
  double *yarea = field.yarea.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(globals.context.use_target)
  for (int j = 0; j < (yrange1); j++) {
    for (int i = 0; i < (xrange1); i++) {
      volume[i + j * base_stride] = dx * dy;
      xarea[i + j * flux_x_stride] = celldy[j];
      yarea[i + j * flux_y_stride] = celldx[i];
    }
  }
}
