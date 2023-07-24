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

#include "ideal_gas.h"

//  @brief Fortran ideal gas kernel.
//  @author Wayne Gaudin
//  @details Calculates the pressure and sound speed for the mesh chunk using
//  the ideal gas equation of state, with a fixed gamma of 1.4.
void ideal_gas_kernel(int x_min, int x_max, int y_min, int y_max, Kokkos::View<double **> &density, Kokkos::View<double **> &energy,
                      Kokkos::View<double **> &pressure, Kokkos::View<double **> &soundspeed) {

  // DO k=y_min,y_max
  //   DO j=x_min,x_max
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});

  Kokkos::parallel_for(
      "ideal_gas", policy, KOKKOS_LAMBDA(const int j, const int k) {
        double v = 1.0 / density(j, k);
        pressure(j, k) = (1.4 - 1.0) * density(j, k) * energy(j, k);
        double pressurebyenergy = (1.4 - 1.0) * density(j, k);
        double pressurebyvolume = -density(j, k) * pressure(j, k);
        double sound_speed_squared = v * v * (pressure(j, k) * pressurebyenergy - pressurebyvolume);
        soundspeed(j, k) = sqrt(sound_speed_squared);
      });
}

//  @brief Ideal gas kernel driver
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the ideal gas equation of
//  state using the specified time level data.

void ideal_gas(global_variables &globals, const int tile, bool predict) {

  if (!predict) {
    ideal_gas_kernel(globals.chunk.tiles[tile].info.t_xmin, globals.chunk.tiles[tile].info.t_xmax, globals.chunk.tiles[tile].info.t_ymin,
                     globals.chunk.tiles[tile].info.t_ymax, globals.chunk.tiles[tile].field.density0.view,
                     globals.chunk.tiles[tile].field.energy0.view, globals.chunk.tiles[tile].field.pressure.view,
                     globals.chunk.tiles[tile].field.soundspeed.view);
  } else {
    ideal_gas_kernel(globals.chunk.tiles[tile].info.t_xmin, globals.chunk.tiles[tile].info.t_xmax, globals.chunk.tiles[tile].info.t_ymin,
                     globals.chunk.tiles[tile].info.t_ymax, globals.chunk.tiles[tile].field.density1.view,
                     globals.chunk.tiles[tile].field.energy1.view, globals.chunk.tiles[tile].field.pressure.view,
                     globals.chunk.tiles[tile].field.soundspeed.view);
  }
}
