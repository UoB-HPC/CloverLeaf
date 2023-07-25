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

//  @brief CloverLeaf top level program: Invokes the main cycle
//  @author Wayne Gaudin
//  @details CloverLeaf in a proxy-app that solves the compressible Euler
//  Equations using an explicit finite volume method on a Cartesian grid.
//  The grid is staggered with internal energy, density and pressure at cell
//  centres and velocities on cell vertices.
//
//  A second order predictor-corrector method is used to advance the solution
//  in time during the Lagrangian phase. A second order advective remap is then
//  carried out to return the mesh to an orthogonal state.
//
//  NOTE: that the proxy-app uses uniformly spaced mesh. The actual method will
//  work on a mesh with varying spacing to keep it relevant to it's parent code.
//  For this reason, optimisations should only be carried out on the software
//  that do not change the underlying numerical method. For example, the
//  volume, though constant for all cells, should remain array and not be
//  converted to a scalar.

#include <iostream>

#include "comms.h"
#include "definitions.h"
#include "finalise.h"
#include "hydro.h"
#include "initialise.h"
#include "read_input.h"
#include "report.h"
#include "start.h"
#include "version.h"

// Output file handler
std::ostream g_out(nullptr);

std::ofstream of;

global_variables initialise(parallel_ &parallel, const std::vector<std::string> &args) {
  global_config config;

  auto &&[ctx, run_args] = create_context(!parallel.boss, args);
  config.dumpDir = run_args.dumpDir;

  bool mpi_enabled =
#ifdef NO_MPI
      false;
#else
      true;
#endif

  std::optional<bool> mpi_cuda_aware_header =
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
      true;
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
      false;
#else
          {};
#endif

  std::optional<bool> mpi_cuda_aware_runtime =
#if defined(MPIX_CUDA_AWARE_SUPPORT)
      MPIX_Query_cuda_support() ? true : false;
#else
      {};
#endif
  switch (run_args.staging_buffer) {
    case run_args::staging_buffer::enabled: config.staging_buffer = true; break;
    case run_args::staging_buffer::disable: config.staging_buffer = false; break;
    case run_args::staging_buffer::automatic:
      config.staging_buffer = !(mpi_cuda_aware_header.value_or(false) && mpi_cuda_aware_runtime.value_or(false));
      break;
  }

  if (parallel.boss) {
    std::cout << "MPI: " << (mpi_enabled ? "true" : "false") << std::endl;
    std::cout << " - MPI header device-awareness (CUDA-awareness): "
              << (mpi_cuda_aware_header ? (*mpi_cuda_aware_header ? "true" : "false") : "unknown") << std::endl;
    std::cout << " - MPI runtime device-awareness (CUDA-awareness): "
              << (mpi_cuda_aware_runtime ? (*mpi_cuda_aware_runtime ? "true" : "false") : "unknown") << std::endl;
    std::cout << " - Host-Device halo exchange staging buffer: " << (config.staging_buffer ? "true" : "false") << std::endl;
    report_context(ctx);
  }

  if (parallel.boss) {
    of.open(run_args.outFile.empty() ? "clover.out" : run_args.outFile);
    if (!of.is_open()) report_error((char *)"initialise", (char *)"Error opening clover.out file.");
    g_out.rdbuf(of.rdbuf());
  } else {
    g_out.rdbuf(std::cout.rdbuf());
  }

  if (parallel.boss) {
    g_out << "Clover Version " << g_version << std::endl     //
          << "Task Count " << parallel.max_task << std::endl //
          << std::endl;
    std::cout << "Output file clover.out opened. All output will go there." << std::endl;
  }

  clover_barrier();

  std::ifstream g_in;
  if (parallel.boss) {
    g_out << "Clover will run from the following input:-" << std::endl << std::endl;
    if (!args.empty()) {
      std::cout << "Args:";
      for (const auto &arg : args)
        std::cout << " " << arg;
      std::cout << std::endl;
    }
  }

  if (!run_args.inFile.empty()) {
    if (parallel.boss) std::cout << "Using input: `" << run_args.inFile << "`" << std::endl;
    g_in.open(run_args.inFile);
    if (g_in.fail()) {
      std::cerr << "Unable to open file: `" << run_args.inFile << "`" << std::endl;
      std::exit(1);
    }
  } else {
    if (parallel.boss) std::cout << "No input file specified, using default input" << std::endl;
    std::ofstream out_unit("clover.in");
    out_unit << "*clover" << std::endl
             << " state 1 density=0.2 energy=1.0" << std::endl
             << " state 2 density=1.0 energy=2.5 geometry=rectangle xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0" << std::endl
             << " x_cells=10" << std::endl
             << " y_cells=2" << std::endl
             << " xmin=0.0" << std::endl
             << " ymin=0.0" << std::endl
             << " xmax=10.0" << std::endl
             << " ymax=2.0" << std::endl
             << " initial_timestep=0.04" << std::endl
             << " timestep_rise=1.5" << std::endl
             << " max_timestep=0.04" << std::endl
             << " end_time=3.0" << std::endl
             << " test_problem 1" << std::endl
             << "*endclover" << std::endl;
    out_unit.close();
    g_in.open("clover.in");
  }
  //}

  clover_barrier();
  if (parallel.boss) {
    g_out << std::endl << "Initialising and generating" << std::endl << std::endl;
  }
  read_input(g_in, parallel, config);
  if (run_args.profile) {
    config.profiler_on = *run_args.profile;
  }

  clover_barrier();

  //	globals.step = 0;
  config.number_of_chunks = parallel.max_task;

  auto globals = start(parallel, config, ctx);
  clover_barrier(globals);
  if (parallel.boss) {
    g_out << "Starting the calculation" << std::endl;
  }
  g_in.close();
  return globals;
}

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  parallel_ parallel;

  if (parallel.boss) {
    std::cout << std::endl
              << "Clover Version " << g_version << std::endl //
              << "Task Count " << parallel.max_task << std::endl
              << std::endl;
  }

  global_variables config = initialise(parallel, std::vector<std::string>(argv + 1, argv + argc));
  if (parallel.boss) {
    std::cout << "Launching hydro" << std::endl;
  }
  hydro(config, parallel);
  finalise(config);
  MPI_Finalize();

  if (parallel.boss) {
    std::cout << "Done" << (config.report_test_fail ? ", but test problem FAILED!" : "") << std::endl;
  }
  return config.report_test_fail ? EXIT_FAILURE : EXIT_SUCCESS;
}
