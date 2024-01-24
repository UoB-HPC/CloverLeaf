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

//  @brief Calculate the minimum timestep for all mesh chunks.
//  @author Wayne Gaudin
//  @details Invokes the kernels needed to calculate the timestep and finds
//  the minimum across all chunks. Checks if the timestep falls below the
//  user specified limitand outputs the timestep information.

#include "timestep.h"

#include "calc_dt.h"
#include "ideal_gas.h"
#include "report.h"
#include "timer.h"
#include "update_halo.h"
#include "viscosity.h"

extern std::ostream g_out;

void timestep(global_variables &globals, parallel_ &parallel) {

  globals.dt = g_big;
  int small = 0;

  int fields[NUM_FIELDS];

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    ideal_gas(globals, tile, false);
  }

  if (globals.profiler_on) globals.profiler.ideal_gas += timer() - kernel_time;

  for (int i = 0; i < NUM_FIELDS; ++i)
    fields[i] = 0;
  fields[field_pressure] = 1;
  fields[field_energy0] = 1;
  fields[field_density0] = 1;
  fields[field_xvel0] = 1;
  fields[field_yvel0] = 1;
  update_halo(globals, fields, 1);

  if (globals.profiler_on) kernel_time = timer();
  viscosity(globals);
  if (globals.profiler_on) globals.profiler.viscosity += timer() - kernel_time;

  for (int i = 0; i < NUM_FIELDS; ++i)
    fields[i] = 0;
  fields[field_viscosity] = 1;
  update_halo(globals, fields, 1);

  if (globals.profiler_on) globals.profiler.kernel_time = timer();

  int jldt{}, kldt{};
  double dtlp{};
  double x_pos{}, y_pos{}, xl_pos{}, yl_pos{};
  std::string dt_control, dtl_control;
  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    calc_dt(globals, tile, dtlp, dtl_control, xl_pos, yl_pos, jldt, kldt);

    if (dtlp <= globals.dt) {
      globals.dt = dtlp;
      dt_control = dtl_control;
      x_pos = xl_pos;
      y_pos = yl_pos;
      globals.jdt = jldt;
      globals.kdt = kldt;
    }
  }

  globals.dt = std::min(std::min(globals.dt, globals.dtold * globals.config.dtrise), globals.config.dtmax);

  //	globals.queue.wait_and_throw();
  clover_min(globals.dt);
  if (globals.profiler_on) {
    globals.profiler.timestep += timer() - globals.profiler.kernel_time;
  }

  if (globals.dt < globals.config.dtmin) small = 1;

  if (parallel.boss) {
    g_out << " Step " << globals.step << " time " << globals.time << " control " << dt_control << " timestep  " << globals.dt << " "
          << globals.jdt << "," << globals.kdt << " x " << x_pos << " y " << y_pos << std::endl;
    std::cout << " Step " << globals.step << " time " << globals.time << " control " << dt_control << " timestep  " << globals.dt << " "
              << globals.jdt << "," << globals.kdt << " x " << x_pos << " y " << y_pos << std::endl;
  }
  if (small == 1) {
    report_error((char *)"timestep", (char *)"small timestep");
  }

  globals.dtold = globals.dt;
}
