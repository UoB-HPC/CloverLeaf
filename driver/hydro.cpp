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

#include "hydro.h"
#include "PdV.h"
#include "accelerate.h"
#include "advection.h"
#include "field_summary.h"
#include "flux_calc.h"
#include "reset_field.h"
#include "shared.h"
#include "timer.h"
#include "timestep.h"
#include "visit.h"

extern std::ostream g_out;

int maxloc(const std::vector<double> &totals, const int len) {
  int loc = -1;
  double max = -1.0;
  for (int i = 0; i < len; ++i) {
    if (totals[i] >= max) {
      loc = i;
      max = totals[i];
    }
  }
  return loc;
}

void hydro(global_variables &globals, parallel_ &parallel) {

  double timerstart = timer();

  bool dump = false;

  if (dump) clover::dump(globals, "dumps/dump_" + std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_0.txt");

  while (true) {

    double step_time = timer();

    globals.step += 1;

    timestep(globals, parallel);
    if (dump) clover::dump(globals, "dumps/dump_" + std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_1_timestep.txt");

    PdV(globals, true);
    if (dump) clover::dump(globals, "dumps/dump_" + std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_2_PdV.txt");

    accelerate(globals);
    if (dump) clover::dump(globals, "dumps/dump_" + std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_3_accelerate.txt");

    PdV(globals, false);
    if (dump) clover::dump(globals, "dumps/dump_" + std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_4_PdV.txt");

    flux_calc(globals);
    if (dump) clover::dump(globals, "dumps/dump_" + std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_5_flux_calc.txt");

    advection(globals);
    if (dump) clover::dump(globals, "dumps/dump_" + std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_6_advection.txt");

    reset_field(globals);
    if (dump) clover::dump(globals, "dumps/dump_" + std::to_string(parallel.task) + "_" + std::to_string(globals.step) + "_7_reset_field.txt");

    globals.advect_x = !globals.advect_x;

    globals.time += globals.dt;
    //		globals.queue.wait_and_throw();

    if (globals.config.summary_frequency != 0) {
      if (globals.step % globals.config.summary_frequency == 0) field_summary(globals, parallel);
    }
    if (globals.config.visit_frequency != 0) {
      if (globals.step % globals.config.visit_frequency == 0) visit(globals, parallel);
    }

    // Sometimes there can be a significant start up cost that appears in the first step.
    // Sometimes it is due to the number of MPI tasks, or OpenCL kernel compilation.
    // On the short test runs, this can skew the results, so should be taken into account
    //  in recorded run times.
    double wall_clock;
    double first_step = 0, second_step = 0;
    if (globals.step == 1) first_step = timer() - step_time;
    if (globals.step == 2) second_step = timer() - step_time;

    if (globals.time + g_small > globals.config.end_time || globals.step >= globals.config.end_step) {

      globals.complete = true;
      field_summary(globals, parallel);
      if (globals.config.visit_frequency != 0) visit(globals, parallel);

      wall_clock = timer() - timerstart;
      if (parallel.boss) {
        g_out << std::endl
              << "Calculation complete" << std::endl
              << "Clover is finishing" << std::endl
              << "Wall clock " << wall_clock << std::endl
              << "First step overhead " << first_step - second_step << std::endl;
        std::cout << "Wall clock " << wall_clock << std::endl << "First step overhead " << first_step - second_step << std::endl;
      }

      std::vector<double> totals(parallel.max_task);
      if (globals.profiler_on) {
        // First we need to find the maximum kernel time for each task. This
        // seems to work better than finding the maximum time for each kernel and
        // adding it up, which always gives over 100%. I think this is because it
        // does not take into account compute overlaps before syncronisations
        // caused by halo exhanges.
        double kernel_total = globals.profiler.timestep + globals.profiler.ideal_gas + globals.profiler.viscosity + globals.profiler.PdV +
                              globals.profiler.revert + globals.profiler.acceleration + globals.profiler.flux +
                              globals.profiler.cell_advection + globals.profiler.mom_advection + globals.profiler.reset +
                              globals.profiler.summary + globals.profiler.visit + globals.profiler.tile_halo_exchange +
                              globals.profiler.self_halo_exchange + globals.profiler.mpi_halo_exchange;
        clover_allgather(kernel_total, totals);

        // So then what I do is use the individual kernel times for the
        // maximum kernel time task for the profile print
        int loc = maxloc(totals, parallel.max_task);
        kernel_total = totals[loc];
        clover_allgather(globals.profiler.timestep, totals);
        globals.profiler.timestep = totals[loc];
        clover_allgather(globals.profiler.ideal_gas, totals);
        globals.profiler.ideal_gas = totals[loc];
        clover_allgather(globals.profiler.viscosity, totals);
        globals.profiler.viscosity = totals[loc];
        clover_allgather(globals.profiler.PdV, totals);
        globals.profiler.PdV = totals[loc];
        clover_allgather(globals.profiler.revert, totals);
        globals.profiler.revert = totals[loc];
        clover_allgather(globals.profiler.acceleration, totals);
        globals.profiler.acceleration = totals[loc];
        clover_allgather(globals.profiler.flux, totals);
        globals.profiler.flux = totals[loc];
        clover_allgather(globals.profiler.cell_advection, totals);
        globals.profiler.cell_advection = totals[loc];
        clover_allgather(globals.profiler.mom_advection, totals);
        globals.profiler.mom_advection = totals[loc];
        clover_allgather(globals.profiler.reset, totals);
        globals.profiler.reset = totals[loc];
        clover_allgather(globals.profiler.tile_halo_exchange, totals);
        globals.profiler.tile_halo_exchange = totals[loc];
        clover_allgather(globals.profiler.self_halo_exchange, totals);
        globals.profiler.self_halo_exchange = totals[loc];
        clover_allgather(globals.profiler.mpi_halo_exchange, totals);
        globals.profiler.mpi_halo_exchange = totals[loc];
        clover_allgather(globals.profiler.summary, totals);
        globals.profiler.summary = totals[loc];
        clover_allgather(globals.profiler.visit, totals);
        globals.profiler.visit = totals[loc];

        if (parallel.boss) {
          g_out << std::endl
                << "Profiler Output                 Time            Percentage" << std::endl
                << "Timestep              :" << globals.profiler.timestep << " " << 100.0 * (globals.profiler.timestep / wall_clock)
                << std::endl
                << "Ideal Gas             :" << globals.profiler.ideal_gas << " " << 100.0 * (globals.profiler.ideal_gas / wall_clock)
                << std::endl
                << "Viscosity             :" << globals.profiler.viscosity << " " << 100.0 * (globals.profiler.viscosity / wall_clock)
                << std::endl
                << "PdV                   :" << globals.profiler.PdV << " " << 100.0 * (globals.profiler.PdV / wall_clock) << std::endl
                << "Revert                :" << globals.profiler.revert << " " << 100.0 * (globals.profiler.revert / wall_clock)
                << std::endl
                << "Acceleration          :" << globals.profiler.acceleration << " " << 100.0 * (globals.profiler.acceleration / wall_clock)
                << std::endl
                << "Fluxes                :" << globals.profiler.flux << " " << 100.0 * (globals.profiler.flux / wall_clock) << std::endl
                << "Cell Advection        :" << globals.profiler.cell_advection << " "
                << 100.0 * (globals.profiler.cell_advection / wall_clock) << std::endl
                << "Momentum Advection    :" << globals.profiler.mom_advection << " "
                << 100.0 * (globals.profiler.mom_advection / wall_clock) << std::endl
                << "Reset                 :" << globals.profiler.reset << " " << 100.0 * (globals.profiler.reset / wall_clock) << std::endl
                << "Summary               :" << globals.profiler.summary << " " << 100.0 * (globals.profiler.summary / wall_clock)
                << std::endl
                << "Visit                 :" << globals.profiler.visit << " " << 100.0 * (globals.profiler.visit / wall_clock) << std::endl
                << "Tile Halo Exchange    :" << globals.profiler.tile_halo_exchange << " "
                << 100.0 * (globals.profiler.tile_halo_exchange / wall_clock) << std::endl
                << "Self Halo Exchange    :" << globals.profiler.self_halo_exchange << " "
                << 100.0 * (globals.profiler.self_halo_exchange / wall_clock) << std::endl
                << "MPI Halo Exchange     :" << globals.profiler.mpi_halo_exchange << " "
                << 100.0 * (globals.profiler.mpi_halo_exchange / wall_clock) << std::endl
                << "Total                 :" << kernel_total << " " << 100.0 * (kernel_total / wall_clock) << std::endl
                << "The Rest              :" << wall_clock - kernel_total << " " << 100.0 * (wall_clock - kernel_total) / wall_clock
                << std::endl
                << std::endl;
        }
      }

      // clover_finalize(); Skipped as just closes the file and calls MPI_Finalize (which is done back in main).

      break;
    }

    if (parallel.boss) {
      wall_clock = timer() - timerstart;
      double step_clock = timer() - step_time;
      g_out << "Wall clock " << wall_clock << std::endl;
      std::cout << "Wall clock " << wall_clock << std::endl;
      double cells = globals.config.grid.x_cells * globals.config.grid.y_cells;
      double rstep = globals.step;
      double grind_time = wall_clock / (rstep * cells);
      double step_grind = step_clock / cells;
      std::cout << "Average time per cell " << grind_time << std::endl;
      g_out << "Average time per cell " << grind_time << std::endl;
      std::cout << "Step time per cell    " << step_grind << std::endl;
      g_out << "Step time per cell    " << step_grind << std::endl;
    }
  }
}
