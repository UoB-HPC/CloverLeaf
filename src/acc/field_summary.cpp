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

#include "field_summary.h"
#include "ideal_gas.h"
#include "report.h"
#include "timer.h"

#include <cmath>
#include <iomanip>

//  @brief Fortran field summary kernel
//  @author Wayne Gaudin
//  @details The total mass, internal energy, kinetic energy and volume weighted
//  pressure for the chunk is calculated.
//  @brief Driver for the field summary kernels
//  @author Wayne Gaudin
//  @details The user specified field summary kernel is invoked here. A summation
//  across all mesh chunks is then performed and the information outputed.
//  If the run is a test problem, the final result is compared with the expected
//  result and the difference output.
//  Note the reference solution is the value returned from an Intel compiler with
//  ieee options set on a single core crun.

void field_summary(global_variables &globals, parallel_ &parallel) {

  clover_report_step_header(globals, parallel);

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    ideal_gas(globals, tile, false);
  }

  if (globals.profiler_on) {
    globals.profiler.ideal_gas += timer() - kernel_time;
    kernel_time = timer();
  }

  double vol = 0.0;
  double mass = 0.0;
  double ie = 0.0;
  double ke = 0.0;
  double press = 0.0;

#if SYNC_BUFFERS
  globals.hostToDevice();
#endif

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];

    int ymax = t.info.t_ymax;
    int ymin = t.info.t_ymin;
    int xmax = t.info.t_xmax;
    int xmin = t.info.t_xmin;
    field_type &field = t.field;

    const size_t base_stride = field.base_stride;
    const size_t vels_wk_stride = field.vels_wk_stride;

    double *volume = field.volume.data;
    double *density0 = field.density0.data;
    double *energy0 = field.energy0.data;
    double *pressure = field.pressure.data;
    double *xvel0 = field.xvel0.data;
    double *yvel0 = field.yvel0.data;

#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target) map(tofrom : vol) map(tofrom : mass)   \
    map(tofrom : ie) map(tofrom : ke) map(tofrom : press) reduction(+ : vol, mass, ie, ke, press)
    for (int idx = 0; idx < ((ymax - ymin + 1) * (xmax - xmin + 1)); idx++) {
      const int j = xmin + 1 + idx % (xmax - xmin + 1);
      const int k = ymin + 1 + idx / (xmax - xmin + 1);
      double vsqrd = 0.0;
      for (int kv = k; kv <= k + 1; ++kv) {
        for (int jv = j; jv <= j + 1; ++jv) {
          vsqrd += 0.25 * (xvel0[(jv) + (kv)*vels_wk_stride] * xvel0[(jv) + (kv)*vels_wk_stride] +
                           yvel0[(jv) + (kv)*vels_wk_stride] * yvel0[(jv) + (kv)*vels_wk_stride]);
        }
      }
      double cell_vol = volume[j + (k)*base_stride];
      double cell_mass = cell_vol * density0[j + (k)*base_stride];
      vol += cell_vol;
      mass += cell_mass;
      ie += cell_mass * energy0[j + (k)*base_stride];
      ke += cell_mass * 0.5 * vsqrd;
      press += cell_vol * pressure[j + (k)*base_stride];
    }
  }

#if SYNC_BUFFERS
  globals.deviceToHost();
#endif

  clover_sum(vol);
  clover_sum(mass);
  clover_sum(ie);
  clover_sum(ke);
  clover_sum(press);

  if (globals.profiler_on) globals.profiler.summary += timer() - kernel_time;

  clover_report_step(globals, parallel, vol, mass, ie, ke, mass);
}
