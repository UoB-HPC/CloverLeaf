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

#include "PdV.h"
#include "comms.h"
#include "context.h"
#include "ideal_gas.h"
#include "report.h"
#include "revert.h"
#include "timer.h"
#include "update_halo.h"
#include <cmath>

//  @brief Fortran PdV kernel.
//  @author Wayne Gaudin
//  @details Calculates the change in energy and density in a cell using the
//  change on cell volume due to the velocity gradients in a cell. The time
//  level of the velocity data depends on whether it is invoked as the
//  predictor or corrector.
void PdV_kernel(bool predict, int x_min, int x_max, int y_min, int y_max, double dt, clover::Buffer2D<double> &xarea,
                clover::Buffer2D<double> &yarea, clover::Buffer2D<double> &volume, clover::Buffer2D<double> &density0,
                clover::Buffer2D<double> &density1, clover::Buffer2D<double> &energy0, clover::Buffer2D<double> &energy1,
                clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &viscosity, clover::Buffer2D<double> &xvel0,
                clover::Buffer2D<double> &xvel1, clover::Buffer2D<double> &yvel0, clover::Buffer2D<double> &yvel1,
                clover::Buffer2D<double> &volume_change) {

  // DO k=y_min,y_max
  //   DO j=x_min,x_max
  clover::Range2d policy(x_min + 1, y_min + 1, x_max + 2, y_max + 2);

  if (predict) {

    clover::par_ranged2(policy, [=] DEVICE_KERNEL(const int i, const int j) {
      double left_flux = (xarea(i, j) * (xvel0(i, j) + xvel0(i + 0, j + 1) + xvel0(i, j) + xvel0(i + 0, j + 1))) * 0.25 * dt * 0.5;
      double right_flux =
          (xarea(i + 1, j + 0) * (xvel0(i + 1, j + 0) + xvel0(i + 1, j + 1) + xvel0(i + 1, j + 0) + xvel0(i + 1, j + 1))) * 0.25 * dt * 0.5;
      double bottom_flux = (yarea(i, j) * (yvel0(i, j) + yvel0(i + 1, j + 0) + yvel0(i, j) + yvel0(i + 1, j + 0))) * 0.25 * dt * 0.5;
      double top_flux =
          (yarea(i + 0, j + 1) * (yvel0(i + 0, j + 1) + yvel0(i + 1, j + 1) + yvel0(i + 0, j + 1) + yvel0(i + 1, j + 1))) * 0.25 * dt * 0.5;
      double total_flux = right_flux - left_flux + top_flux - bottom_flux;
      double volume_change_s = volume(i, j) / (volume(i, j) + total_flux);
      double recip_volume = 1.0 / volume(i, j);
      double energy_change = (pressure(i, j) / density0(i, j) + viscosity(i, j) / density0(i, j)) * total_flux * recip_volume;
      energy1(i, j) = energy0(i, j) - energy_change;
      density1(i, j) = density0(i, j) * volume_change_s;
    });

  } else {

    clover::par_ranged2(policy, [=] DEVICE_KERNEL(const int i, const int j) {
      double left_flux = (xarea(i, j) * (xvel0(i, j) + xvel0(i + 0, j + 1) + xvel1(i, j) + xvel1(i + 0, j + 1))) * 0.25 * dt;
      double right_flux =
          (xarea(i + 1, j + 0) * (xvel0(i + 1, j + 0) + xvel0(i + 1, j + 1) + xvel1(i + 1, j + 0) + xvel1(i + 1, j + 1))) * 0.25 * dt;
      double bottom_flux = (yarea(i, j) * (yvel0(i, j) + yvel0(i + 1, j + 0) + yvel1(i, j) + yvel1(i + 1, j + 0))) * 0.25 * dt;
      double top_flux =
          (yarea(i + 0, j + 1) * (yvel0(i + 0, j + 1) + yvel0(i + 1, j + 1) + yvel1(i + 0, j + 1) + yvel1(i + 1, j + 1))) * 0.25 * dt;
      double total_flux = right_flux - left_flux + top_flux - bottom_flux;
      double volume_change_s = volume(i, j) / (volume(i, j) + total_flux);
      double recip_volume = 1.0 / volume(i, j);
      double energy_change = (pressure(i, j) / density0(i, j) + viscosity(i, j) / density0(i, j)) * total_flux * recip_volume;
      energy1(i, j) = energy0(i, j) - energy_change;
      density1(i, j) = density0(i, j) * volume_change_s;
    });
  }
}

//  @brief Driver for the PdV update.
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the PdV update.
void PdV(global_variables &globals, bool predict) {

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

  globals.error_condition = 0;

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];
    PdV_kernel(predict, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, globals.dt, t.field.xarea, t.field.yarea,
               t.field.volume, t.field.density0, t.field.density1, t.field.energy0, t.field.energy1, t.field.pressure, t.field.viscosity,
               t.field.xvel0, t.field.xvel1, t.field.yvel0, t.field.yvel1, t.field.work_array1);
  }

  clover_check_error(globals.error_condition);
  if (globals.profiler_on) {
    clover::checkError(cudaDeviceSynchronize());
    globals.profiler.PdV += timer() - kernel_time;
  }

  if (globals.error_condition == 1) {
    report_error((char *)"PdV", (char *)"error in PdV");
  }

  if (predict) {
    if (globals.profiler_on) kernel_time = timer();
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      ideal_gas(globals, tile, true);
    }

    if (globals.profiler_on) {
      clover::checkError(cudaDeviceSynchronize());
      globals.profiler.ideal_gas += timer() - kernel_time;
    }

    int fields[NUM_FIELDS];
    for (int &field : fields)
      field = 0;
    fields[field_pressure] = 1;
    update_halo(globals, fields, 1);
  }

  if (predict) {
    if (globals.profiler_on) kernel_time = timer();
    revert(globals);
    if (globals.profiler_on) {
      clover::checkError(cudaDeviceSynchronize());
      globals.profiler.revert += timer() - kernel_time;
    }
  }
}
