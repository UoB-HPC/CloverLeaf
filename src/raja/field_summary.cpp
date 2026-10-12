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
#include "context.h"
#include "ideal_gas.h"
#include "report.h"
#include "timer.h"

#include <iomanip>
#include <numeric>

#include <RAJA/RAJA.hpp>

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
    clover::checkError(rajaDeviceSynchronize());
    globals.profiler.ideal_gas += timer() - kernel_time;
    kernel_time = timer();
  }

  double vol = 0.0;
  double mass = 0.0;
  double ie = 0.0;
  double ke = 0.0;
  double press = 0.0;

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];

    int ymax = t.info.t_ymax;
    int ymin = t.info.t_ymin;
    int xmax = t.info.t_xmax;
    int xmin = t.info.t_xmin;
    field_type &field = t.field;

    int range = (ymax - ymin + 1) * (xmax - xmin + 1);

    using RAJA_VALOP_PLUS = RAJA::expt::ValOp<double, RAJA::operators::plus>;
    RAJA::forall<reduce_policy>(RAJA::TypedRangeSegment<int>(0, range),
      RAJA::expt::Reduce<RAJA::operators::plus>(&vol),
      RAJA::expt::Reduce<RAJA::operators::plus>(&mass),
      RAJA::expt::Reduce<RAJA::operators::plus>(&ie),
      RAJA::expt::Reduce<RAJA::operators::plus>(&ke),
      RAJA::expt::Reduce<RAJA::operators::plus>(&press),
      [=] RAJA_HOST_DEVICE (int gid, RAJA_VALOP_PLUS &_vol, RAJA_VALOP_PLUS &_mass,
        RAJA_VALOP_PLUS &_ie, RAJA_VALOP_PLUS &_ke, RAJA_VALOP_PLUS &_press) {

        int v = gid;

        const size_t j = xmin + 1 + v % (xmax - xmin + 1);
        const size_t k = ymin + 1 + v / (xmax - xmin + 1);
        double vsqrd = 0.0;
        for (size_t kv = k; kv <= k + 1; ++kv) {
          for (size_t jv = j; jv <= j + 1; ++jv) {
            vsqrd += 0.25 * (field.xvel0(jv, kv) * field.xvel0(jv, kv) + field.yvel0(jv, kv) * field.yvel0(jv, kv));
          }
        }
        double cell_vol = field.volume(j, k);
        double cell_mass = cell_vol * field.density0(j, k);

        _vol += cell_vol;
        _mass += cell_mass;
        _ie  += cell_mass * field.energy0(j, k);
        _ke += cell_mass * 0.5 * vsqrd;
        _press += cell_vol * field.pressure(j, k);
    });

    if (globals.profiler_on) {
      globals.profiler.summary += timer() - kernel_time;
      kernel_time = timer();
    }
  }

  if (globals.profiler_on) {
    clover::checkError(rajaDeviceSynchronize());
    globals.profiler.summary += timer() - kernel_time;
  }

  clover_report_step(globals, parallel, vol, mass, ie, ke, press);
}
