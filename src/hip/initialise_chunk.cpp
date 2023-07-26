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
#include "context.h"

void initialise_chunk(const int tile, global_variables &globals) {

  double dx = (globals.config.grid.xmax - globals.config.grid.xmin) / (double)(globals.config.grid.x_cells);
  double dy = (globals.config.grid.ymax - globals.config.grid.ymin) / (double)(globals.config.grid.y_cells);

  double xmin = globals.config.grid.xmin + dx * (double)(globals.chunk.tiles[tile].info.t_left - 1);

  double ymin = globals.config.grid.ymin + dy * (double)(globals.chunk.tiles[tile].info.t_bottom - 1);

  const int x_min = globals.chunk.tiles[tile].info.t_xmin;
  const int x_max = globals.chunk.tiles[tile].info.t_xmax;
  const int y_min = globals.chunk.tiles[tile].info.t_ymin;
  const int y_max = globals.chunk.tiles[tile].info.t_ymax;

  const size_t xrange = (x_max + 3) - (x_min - 2) + 1;
  const size_t yrange = (y_max + 3) - (y_min - 2) + 1;

  // Take a reference to the lowest structure, as Kokkos device cannot necessarily chase through the structure.
  field_type &field = globals.chunk.tiles[tile].field;

  clover::par_ranged1(Range1d{0u, xrange}, [=] DEVICE_KERNEL(int j) {
    field.vertexx[j] = xmin + dx * (static_cast<int>(j) - 1 - x_min);
    field.vertexdx[j] = dx;
  });

  clover::par_ranged1(Range1d{0u, yrange}, [=] DEVICE_KERNEL(int k) {
    field.vertexy[k] = ymin + dy * (static_cast<int>(k) - 1 - y_min);
    field.vertexdy[k] = dy;
  });

  const size_t xrange1 = (x_max + 2) - (x_min - 2) + 1;
  const size_t yrange1 = (y_max + 2) - (y_min - 2) + 1;

  clover::par_ranged1(Range1d{0u, xrange1}, [=] DEVICE_KERNEL(int j) {
    field.cellx[j] = 0.5 * (field.vertexx[j] + field.vertexx[j + 1]);
    field.celldx[j] = dx;
  });

  clover::par_ranged1(Range1d{0u, yrange1}, [=] DEVICE_KERNEL(int k) {
    field.celly[k] = 0.5 * (field.vertexy[k] + field.vertexy[k + 1]);
    field.celldy[k] = dy;
  });

  clover::par_ranged2(Range2d{0u, 0u, xrange1, yrange1}, [=] DEVICE_KERNEL(const int i, const int j) {
    field.volume(i, j) = dx * dy;
    field.xarea(i, j) = field.celldy[j];
    field.yarea(i, j) = field.celldx[i];
  });
}
