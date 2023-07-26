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

  ////    CALL initialise_chunk_kernel(chunk%tiles(tile)%t_xmin,    &
  //     chunk%tiles(tile)%t_xmax,    &
  //     chunk%tiles(tile)%t_ymin,    &
  //     chunk%tiles(tile)%t_ymax,    &
  //     xmin,ymin,dx,dy,              &
  //     chunk%tiles(tile)%field%vertexx,  &
  //     chunk%tiles(tile)%field%vertexdx, &
  //     chunk%tiles(tile)%field%vertexy,  &
  //     chunk%tiles(tile)%field%vertexdy, &
  //     chunk%tiles(tile)%field%cellx,    &
  //     chunk%tiles(tile)%field%celldx,   &
  //     chunk%tiles(tile)%field%celly,    &
  //     chunk%tiles(tile)%field%celldy,   &
  //     chunk%tiles(tile)%field%volume,   &
  //     chunk%tiles(tile)%field%xarea,    &
  //     chunk%tiles(tile)%field%yarea     )

  const int x_min = globals.chunk.tiles[tile].info.t_xmin;
  const int x_max = globals.chunk.tiles[tile].info.t_xmax;
  const int y_min = globals.chunk.tiles[tile].info.t_ymin;
  const int y_max = globals.chunk.tiles[tile].info.t_ymax;

  const size_t xrange = (x_max + 3) - (x_min - 2) + 1;
  const size_t yrange = (y_max + 3) - (y_min - 2) + 1;

  // Take a reference to the lowest structure, as Kokkos device cannot necessarily chase through the structure.
  field_type &field = globals.chunk.tiles[tile].field;

  clover::execute(globals.context.queue, [&](handler &h) {
    auto vertexx = field.vertexx.access<W>(h);
    auto vertexdx = field.vertexdx.access<W>(h);
    clover::par_ranged<class APPEND_LN(initialise)>(h, {0u, xrange}, [=](id<1> j) {
      vertexx[j] = xmin + dx * (static_cast<int>(j[0]) - 1 - x_min);
      vertexdx[j] = dx;
    });
  });

  clover::execute(globals.context.queue, [&](handler &h) {
    auto vertexy = field.vertexy.access<W>(h);
    auto vertexdy = field.vertexdy.access<W>(h);
    clover::par_ranged<class APPEND_LN(initialise)>(h, {0u, yrange}, [=](id<1> k) {
      vertexy[k] = ymin + dy * (static_cast<int>(k[0]) - 1 - y_min);
      vertexdy[k] = dy;
    });
  });

  const size_t xrange1 = (x_max + 2) - (x_min - 2) + 1;
  const size_t yrange1 = (y_max + 2) - (y_min - 2) + 1;

  clover::execute(globals.context.queue, [&](handler &h) {
    auto cellx = field.cellx.access<W>(h);
    auto celldx = field.celldx.access<W>(h);
    auto vertexx = field.vertexx.access<R>(h);
    clover::par_ranged<class APPEND_LN(initialise)>(h, {0u, xrange1}, [=](id<1> j) {
      cellx[j] = 0.5 * (vertexx[j] + vertexx[j[0] + 1]);
      celldx[j] = dx;
    });
  });

  clover::execute(globals.context.queue, [&](handler &h) {
    auto celly = field.celly.access<W>(h);
    auto celldy = field.celldy.access<W>(h);
    auto vertexy = field.vertexy.access<R>(h);

    clover::par_ranged<class APPEND_LN(initialise)>(h, {0u, yrange1}, [=](id<1> k) {
      celly[k] = 0.5 * (vertexy[k] + vertexy[k[0] + 1]);
      celldy[k] = dy;
    });
  });

  clover::execute(globals.context.queue, [&](handler &h) {
    auto volume = field.volume.access<W>(h);
    auto xarea = field.xarea.access<W>(h);
    auto yarea = field.yarea.access<W>(h);
    auto celldx = field.celldx.access<R>(h);
    auto celldy = field.celldy.access<R>(h);
    clover::par_ranged<class APPEND_LN(initialise)>(h, {0u, 0u, xrange1, yrange1}, [=](id<2> idx) {
      volume[idx] = dx * dy;
      xarea[idx] = celldy[idx[1]];
      yarea[idx] = celldx[idx[0]];
    });
  });
}
