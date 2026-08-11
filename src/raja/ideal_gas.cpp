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
#include "context.h"
#include <cmath>

#include <RAJA/RAJA.hpp>

#define IDX(buffer, x, y) buffer[idx[(x)]][idx[(y)]]

#define SWP(x, y) (x), (y)

int N = 0;

//  @brief Fortran ideal gas kernel.
//  @author Wayne Gaudin
//  @details Calculates the pressure and sound speed for the mesh chunk using
//  the ideal gas equation of state, with a fixed gamma of 1.4.
void ideal_gas_kernel(int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &density, clover::Buffer2D<double> &energy,
                      clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &soundspeed) {

  // std::cout <<" ideal_gas(" << x_min+1 << ","<< y_min+1<< ","<< x_max+2<< ","<< y_max +2  << ")" << std::endl;
  //  DO k=y_min,y_max
  //    DO j=x_min,x_max

  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});

  // clover::par_ranged2(Range2d{x_min + 1, y_min + 1, x_max + 2, y_max + 2}, [=] DEVICE_KERNEL(const int i, const int j) {
  const RAJA::TypedRangeSegment<int> row_Range1(y_min + 1,  y_max + 2);
  const RAJA::TypedRangeSegment<int> col_Range1(x_min + 1,  x_max + 2);
  RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range1, row_Range1),
      [=] RAJA_HOST_DEVICE (const int i, const int j) {
    double v = 1.0 / density(i, j);
    pressure(i, j) = (1.4 - 1.0) * density(i, j) * energy(i, j);
    double pressurebyenergy = (1.4 - 1.0) * density(i, j);
    double pressurebyvolume = -density(i, j) * pressure(i, j);
    double sound_speed_squared = v * v * (pressure(i, j) * pressurebyenergy - pressurebyvolume);
    soundspeed(i, j) = std::sqrt(sound_speed_squared);
  });
}

//  @brief Ideal gas kernel driver
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the ideal gas equation of
//  state using the specified time level data.

void ideal_gas(global_variables &globals, const int tile, bool predict) {

  tile_type &t = globals.chunk.tiles[tile];

  if (!predict) {
    ideal_gas_kernel(t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density0, t.field.energy0, t.field.pressure,
                     t.field.soundspeed);
  } else {
    ideal_gas_kernel(t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density1, t.field.energy1, t.field.pressure,
                     t.field.soundspeed);
  }
}
