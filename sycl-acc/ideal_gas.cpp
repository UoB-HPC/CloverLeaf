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

#define IDX(buffer, x, y) buffer[idx[(x)]][idx[(y)]]

#define SWP(x, y) (x), (y)

int N = 0;

//  @brief Fortran ideal gas kernel.
//  @author Wayne Gaudin
//  @details Calculates the pressure and sound speed for the mesh chunk using
//  the ideal gas equation of state, with a fixed gamma of 1.4.
void ideal_gas_kernel(handler &h, int x_min, int x_max, int y_min, int y_max, clover::Accessor<double, 2, R>::Type density,
                      clover::Accessor<double, 2, R>::Type energy, clover::Accessor<double, 2, RW>::Type pressure,
                      clover::Accessor<double, 2, W>::Type soundspeed) {

  // std::cout <<" ideal_gas(" << x_min+1 << ","<< y_min+1<< ","<< x_max+2<< ","<< y_max +2  << ")" << std::endl;
  //  DO k=y_min,y_max
  //    DO j=x_min,x_max

  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});

  clover::par_ranged<class ideal_gas>(h, {x_min + 1, y_min + 1, x_max + 2, y_max + 2}, [=](id<2> idx) {
    double v = 1.0 / density[idx];
    pressure[idx] = (1.4 - 1.0) * density[idx] * energy[idx];
    double pressurebyenergy = (1.4 - 1.0) * density[idx];
    double pressurebyvolume = -density[idx] * pressure[idx];
    double sound_speed_squared = v * v * (pressure[idx] * pressurebyenergy - pressurebyvolume);
    soundspeed[idx] = sycl::sqrt(sound_speed_squared);
  });
}

//  @brief Ideal gas kernel driver
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the ideal gas equation of
//  state using the specified time level data.

void ideal_gas(global_variables &globals, const int tile, bool predict) {

  tile_type &t = globals.chunk.tiles[tile];

  clover::execute(globals.context.queue, [&](handler &h) {
    if (!predict) {
      ideal_gas_kernel(h, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density0.access<R>(h),
                       t.field.energy0.access<R>(h), t.field.pressure.access<RW>(h), t.field.soundspeed.access<W>(h));
    } else {
      ideal_gas_kernel(h, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density1.access<R>(h),
                       t.field.energy1.access<R>(h), t.field.pressure.access<RW>(h), t.field.soundspeed.access<W>(h));
    }
  });
}
