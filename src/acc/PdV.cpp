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
void PdV_kernel(bool use_target, bool predict, int x_min, int x_max, int y_min, int y_max, double dt, field_type &field) {

  const int base_stride = field.base_stride;
  const int vels_wk_stride = field.vels_wk_stride;
  const int flux_x_stride = field.flux_x_stride;
  const int flux_y_stride = field.flux_y_stride;

  // DO k=y_min,y_max
  //   DO j=x_min,x_max

  if (predict) {

    double *xarea = field.xarea.data;

    double *yarea = field.yarea.data;
    double *volume = field.volume.data;
    double *density0 = field.density0.data;
    double *density1 = field.density1.data;
    double *energy0 = field.energy0.data;
    double *energy1 = field.energy1.data;
    double *pressure = field.pressure.data;
    double *viscosity = field.viscosity.data;
    double *xvel0 = field.xvel0.data;
    double *yvel0 = field.yvel0.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min + 1); j < (y_max + 2); j++) {
      for (int i = (x_min + 1); i < (x_max + 2); i++) {
        double left_flux = (xarea[i + j * flux_x_stride] * (xvel0[i + j * vels_wk_stride] + xvel0[(i + 0) + (j + 1) * vels_wk_stride] +
                                                            xvel0[i + j * vels_wk_stride] + xvel0[(i + 0) + (j + 1) * vels_wk_stride])) *
                           0.25 * dt * 0.5;
        double right_flux = (xarea[(i + 1) + (j + 0) * flux_x_stride] *
                             (xvel0[(i + 1) + (j + 0) * vels_wk_stride] + xvel0[(i + 1) + (j + 1) * vels_wk_stride] +
                              xvel0[(i + 1) + (j + 0) * vels_wk_stride] + xvel0[(i + 1) + (j + 1) * vels_wk_stride])) *
                            0.25 * dt * 0.5;
        double bottom_flux = (yarea[i + j * flux_y_stride] * (yvel0[i + j * vels_wk_stride] + yvel0[(i + 1) + (j + 0) * vels_wk_stride] +
                                                              yvel0[i + j * vels_wk_stride] + yvel0[(i + 1) + (j + 0) * vels_wk_stride])) *
                             0.25 * dt * 0.5;
        double top_flux = (yarea[(i + 0) + (j + 1) * flux_y_stride] *
                           (yvel0[(i + 0) + (j + 1) * vels_wk_stride] + yvel0[(i + 1) + (j + 1) * vels_wk_stride] +
                            yvel0[(i + 0) + (j + 1) * vels_wk_stride] + yvel0[(i + 1) + (j + 1) * vels_wk_stride])) *
                          0.25 * dt * 0.5;
        double total_flux = right_flux - left_flux + top_flux - bottom_flux;
        double volume_change_s = volume[i + j * base_stride] / (volume[i + j * base_stride] + total_flux);
        double recip_volume = 1.0 / volume[i + j * base_stride];
        double energy_change = (pressure[i + j * base_stride] / density0[i + j * base_stride] +
                                viscosity[i + j * base_stride] / density0[i + j * base_stride]) *
                               total_flux * recip_volume;
        energy1[i + j * base_stride] = energy0[i + j * base_stride] - energy_change;
        density1[i + j * base_stride] = density0[i + j * base_stride] * volume_change_s;
      }
    }

  } else {

    double *xarea = field.xarea.data;
    double *yarea = field.yarea.data;
    double *volume = field.volume.data;
    double *density0 = field.density0.data;
    double *density1 = field.density1.data;
    double *energy0 = field.energy0.data;
    double *energy1 = field.energy1.data;
    double *pressure = field.pressure.data;
    double *viscosity = field.viscosity.data;
    double *xvel0 = field.xvel0.data;
    double *xvel1 = field.xvel1.data;
    double *yvel0 = field.yvel0.data;
    double *yvel1 = field.yvel1.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min + 1); j < (y_max + 2); j++) {
      for (int i = (x_min + 1); i < (x_max + 2); i++) {
        double left_flux = (xarea[i + j * flux_x_stride] * (xvel0[i + j * vels_wk_stride] + xvel0[(i + 0) + (j + 1) * vels_wk_stride] +
                                                            xvel1[i + j * vels_wk_stride] + xvel1[(i + 0) + (j + 1) * vels_wk_stride])) *
                           0.25 * dt;
        double right_flux = (xarea[(i + 1) + (j + 0) * flux_x_stride] *
                             (xvel0[(i + 1) + (j + 0) * vels_wk_stride] + xvel0[(i + 1) + (j + 1) * vels_wk_stride] +
                              xvel1[(i + 1) + (j + 0) * vels_wk_stride] + xvel1[(i + 1) + (j + 1) * vels_wk_stride])) *
                            0.25 * dt;
        double bottom_flux = (yarea[i + j * flux_y_stride] * (yvel0[i + j * vels_wk_stride] + yvel0[(i + 1) + (j + 0) * vels_wk_stride] +
                                                              yvel1[i + j * vels_wk_stride] + yvel1[(i + 1) + (j + 0) * vels_wk_stride])) *
                             0.25 * dt;
        double top_flux = (yarea[(i + 0) + (j + 1) * flux_y_stride] *
                           (yvel0[(i + 0) + (j + 1) * vels_wk_stride] + yvel0[(i + 1) + (j + 1) * vels_wk_stride] +
                            yvel1[(i + 0) + (j + 1) * vels_wk_stride] + yvel1[(i + 1) + (j + 1) * vels_wk_stride])) *
                          0.25 * dt;
        double total_flux = right_flux - left_flux + top_flux - bottom_flux;
        double volume_change_s = volume[i + j * base_stride] / (volume[i + j * base_stride] + total_flux);
        double recip_volume = 1.0 / volume[i + j * base_stride];
        double energy_change = (pressure[i + j * base_stride] / density0[i + j * base_stride] +
                                viscosity[i + j * base_stride] / density0[i + j * base_stride]) *
                               total_flux * recip_volume;
        energy1[i + j * base_stride] = energy0[i + j * base_stride] - energy_change;
        density1[i + j * base_stride] = density0[i + j * base_stride] * volume_change_s;
      }
    }
  }
}

//  @brief Driver for the PdV update.
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the PdV update.
void PdV(global_variables &globals, bool predict) {

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

  globals.error_condition = 0;

#if SYNC_BUFFERS
  globals.hostToDevice();
#endif

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];
    PdV_kernel(globals.context.use_target, predict, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, globals.dt, t.field);
  }

#if SYNC_BUFFERS
  globals.deviceToHost();
#endif

  clover_check_error(globals.error_condition);
  if (globals.profiler_on) globals.profiler.PdV += timer() - kernel_time;

  if (globals.error_condition == 1) {
    report_error((char *)"PdV", (char *)"error in PdV");
  }

  if (predict) {
    if (globals.profiler_on) kernel_time = timer();
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      ideal_gas(globals, tile, true);
    }

    if (globals.profiler_on) globals.profiler.ideal_gas += timer() - kernel_time;

    int fields[NUM_FIELDS];
    for (int &field : fields)
      field = 0;
    fields[field_pressure] = 1;
    update_halo(globals, fields, 1);
  }

  if (predict) {
    if (globals.profiler_on) kernel_time = timer();
    revert(globals);
    if (globals.profiler_on) globals.profiler.revert += timer() - kernel_time;
  }
}
