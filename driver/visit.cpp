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

#include "visit.h"
#include "ideal_gas.h"
#include "timer.h"
#include "update_halo.h"
#include "viscosity.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>

//  @brief Generates graphics output files.
//  @author Wayne Gaudin
//  @details The field data over all mesh chunks is written to a .vtk files and
//  the .visit file is written that defines the time for each set of vtk files.
//  The ideal gas and viscosity routines are invoked to make sure this data is
//  up to data with the current energy, density and velocity.

static bool first_call = true;

void visit(global_variables &globals, parallel_ &parallel) {

  std::string name = "clover";
  if (parallel.boss) {

    if (first_call) {

      int nblocks = globals.config.number_of_chunks * globals.config.tiles_per_chunk;
      std::string filename = "clover.visit";
      std::ofstream u;
      u.open(filename);
      u << "!NBLOCKS " << nblocks << std::endl;
      u.close();
      first_call = false;
    }
  }

  double kernel_time{};
  if (globals.profiler_on) kernel_time = timer();
  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    ideal_gas(globals, tile, false);
  }
  if (globals.profiler_on) globals.profiler.ideal_gas += timer() - kernel_time;

  int fields[NUM_FIELDS];
  for (int i = 0; i < NUM_FIELDS; ++i)
    fields[i] = 0;
  fields[field_pressure] = 1;
  fields[field_xvel0] = 1;
  fields[field_yvel0] = 1;
  update_halo(globals, fields, 1);

  if (globals.profiler_on) kernel_time = timer();
  viscosity(globals);
  if (globals.profiler_on) globals.profiler.viscosity += timer() - kernel_time;

  if (parallel.boss) {

    std::string filename = "clover.visit";
    std::ofstream u;
    u.open(filename, std::ios::app);
    for (int c = 0; c < parallel.max_task; ++c) {
      std::stringstream namestream;
      namestream << "." << std::setfill('0') << std::setw(5) << c;
      for (int tile = 1; tile <= globals.config.tiles_per_chunk; ++tile) {
        namestream << "." << std::setfill('0') << std::setw(5) << tile;
        namestream << "." << std::setfill('0') << std::setw(5) << globals.step;
        namestream << ".vtk";
        u << name << namestream.str() << std::endl;
      }
    }
    u.close();
  }

  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    if (globals.chunk.task == parallel.task) {
      int nxc = globals.chunk.tiles[tile].info.t_xmax - globals.chunk.tiles[tile].info.t_xmin + 1;
      int nyc = globals.chunk.tiles[tile].info.t_ymax - globals.chunk.tiles[tile].info.t_ymin + 1;
      int nxv = nxc + 1;
      int nyv = nyc + 1;

      std::stringstream namestream;
      namestream << name;
      namestream << "." << std::setfill('0') << std::setw(5) << parallel.task;
      namestream << "." << std::setfill('0') << std::setw(5) << tile + 1;
      namestream << "." << std::setfill('0') << std::setw(5) << globals.step;
      namestream << ".vtk";
      std::ofstream u;
      u.open(namestream.str());
      u << "# vtk DataFile Version 3.0" << std::endl;
      u << "vtk output" << std::endl;
      u << "ASCII" << std::endl;
      u << "DATASET RECTILINEAR_GRID" << std::endl;
      u << "DIMENSIONS " << nxv << " " << nyv << " 1" << std::endl;
      u << "X_COORDINATES " << nxv << " double" << std::endl;

      if (globals.profiler_on) {
        globals.profiler.visit += timer() - kernel_time;
        kernel_time = timer();
      }

      // JMK: These all copy data back to the host
      auto hm_vertexx = globals.chunk.tiles[tile].field.vertexx.mirrored();
      auto hm_vertexy = globals.chunk.tiles[tile].field.vertexy.mirrored();
      auto hm_density0 = globals.chunk.tiles[tile].field.density0.mirrored2();
      auto hm_energy0 = globals.chunk.tiles[tile].field.energy0.mirrored2();
      auto hm_pressure = globals.chunk.tiles[tile].field.pressure.mirrored2();
      auto hm_viscosity = globals.chunk.tiles[tile].field.viscosity.mirrored2();
      auto hm_xvel0 = globals.chunk.tiles[tile].field.xvel0.mirrored2();
      auto hm_yvel0 = globals.chunk.tiles[tile].field.yvel0.mirrored2();

      if (globals.profiler_on) {
        globals.profiler.device_to_host += timer() - kernel_time;
        kernel_time = timer();
      }

      for (int j = globals.chunk.tiles[tile].info.t_xmin + 1; j <= globals.chunk.tiles[tile].info.t_xmax + 1 + 1; ++j) {
        u << hm_vertexx[j] << std::endl;
      }

      u << "Y_COORDINATES " << nyv << " double" << std::endl;


      for (int k = globals.chunk.tiles[tile].info.t_ymin + 1; k <= globals.chunk.tiles[tile].info.t_ymax + 1 + 1; ++k) {
        u << hm_vertexy[k] << std::endl;
      }

      u << "Z_COORDINATES 1 double" << std::endl;
      u << "0" << std::endl;

      u << "CELL_DATA " << nxc * nyc << std::endl;
      u << "FIELD FieldData 4" << std::endl;
      u << "density 1 " << nxc * nyc << " double" << std::endl;
      for (int k = globals.chunk.tiles[tile].info.t_ymin + 1; k <= globals.chunk.tiles[tile].info.t_ymax + 1; ++k) {
        for (int j = globals.chunk.tiles[tile].info.t_xmin + 1; j <= globals.chunk.tiles[tile].info.t_xmax + 1; ++j) {
          u << std::scientific << std::setprecision(3) << hm_density0(j, k) << std::endl;
        }
      }

      u << "energy 1 " << nxc * nyc << " double" << std::endl;
      for (int k = globals.chunk.tiles[tile].info.t_ymin + 1; k <= globals.chunk.tiles[tile].info.t_ymax + 1; ++k) {
        for (int j = globals.chunk.tiles[tile].info.t_xmin + 1; j <= globals.chunk.tiles[tile].info.t_xmax + 1; ++j) {
          u << std::scientific << std::setprecision(3) << hm_energy0(j, k) << std::endl;
        }
      }

      u << "pressure 1 " << nxc * nyc << " double" << std::endl;
      for (int k = globals.chunk.tiles[tile].info.t_ymin + 1; k <= globals.chunk.tiles[tile].info.t_ymax + 1; ++k) {
        for (int j = globals.chunk.tiles[tile].info.t_xmin + 1; j <= globals.chunk.tiles[tile].info.t_xmax + 1; ++j) {
          u << std::scientific << std::setprecision(3) << hm_pressure(j, k) << std::endl;
        }
      }

      u << "viscosity 1 " << nxc * nyc << " double" << std::endl;
      for (int k = globals.chunk.tiles[tile].info.t_ymin + 1; k <= globals.chunk.tiles[tile].info.t_ymax + 1; ++k) {
        for (int j = globals.chunk.tiles[tile].info.t_xmin + 1; j <= globals.chunk.tiles[tile].info.t_xmax + 1; ++j) {
          double temp = (std::fabs(hm_viscosity(j, k)) > 0.00000001) ? hm_viscosity(j, k) : 0.0;
          u << std::scientific << std::setprecision(3) << temp << std::endl;
        }
      }

      u << "POINT_DATA " << nxv * nyv << std::endl;
      u << "FIELD FieldData 2" << std::endl;
      u << "x_vel 1 " << nxv * nyv << " double" << std::endl;
      for (int k = globals.chunk.tiles[tile].info.t_ymin + 1; k <= globals.chunk.tiles[tile].info.t_ymax + 1 + 1; ++k) {
        for (int j = globals.chunk.tiles[tile].info.t_xmin + 1; j <= globals.chunk.tiles[tile].info.t_xmax + 1 + 1; ++j) {
          double temp = (std::fabs(hm_xvel0(j, k)) > 0.00000001) ? hm_xvel0(j, k) : 0.0;
          u << std::scientific << std::setprecision(3) << temp << std::endl;
        }
      }
      u << "y_vel 1 " << nxv * nyv << " double" << std::endl;
      for (int k = globals.chunk.tiles[tile].info.t_ymin + 1; k <= globals.chunk.tiles[tile].info.t_ymax + 1 + 1; ++k) {
        for (int j = globals.chunk.tiles[tile].info.t_xmin + 1; j <= globals.chunk.tiles[tile].info.t_xmax + 1 + 1; ++j) {
          double temp = (std::fabs(hm_yvel0(j, k)) > 0.00000001) ? hm_yvel0(j, k) : 0.0;
          u << std::scientific << std::setprecision(3) << temp << std::endl;
        }
      }
      u.close();
    }
  }
  if (globals.profiler_on) globals.profiler.visit += timer() - kernel_time;
}
