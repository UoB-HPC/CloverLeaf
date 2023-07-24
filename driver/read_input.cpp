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

// @brief Reads the user input
// @author Wayne Gaudin
// @details Reads and parses the user input from the processed file and sets
// the variables used in the generation phase. Default values are also set
// here.

#include "read_input.h"

#include "report.h"

#include <cstring>
#include <iostream>
#include <iterator>
#include <sstream>

extern std::ostream g_out;

void read_input(std::ifstream &g_in, parallel_ &parallel, global_config &globals) {

  globals.test_problem = 0;

  int state_max = 0;

  globals.grid.xmin = 0.0;
  globals.grid.ymin = 0.0;
  globals.grid.xmax = 0.0;
  globals.grid.ymax = 0.0;

  globals.grid.x_cells = 10;
  globals.grid.y_cells = 10;

  globals.end_time = 10.0;
  globals.end_step = g_ibig;
  //	globals.complete = false;

  globals.visit_frequency = 0;
  globals.summary_frequency = 10;

  globals.tiles_per_chunk = 1;

  globals.dtinit = 0.1;
  globals.dtmax = 1.0;
  globals.dtmin = 0.0000001;
  globals.dtrise = 1.5;
  globals.dtc_safe = 0.7;
  globals.dtu_safe = 0.5;
  globals.dtv_safe = 0.5;
  globals.dtdiv_safe = 0.7;

  //	globals.profiler_on = false;
  //	globals.profiler.timestep = 0.0;
  //	globals.profiler.acceleration = 0.0;
  //	globals.profiler.PdV = 0.0;
  //	globals.profiler.cell_advection = 0.0;
  //	globals.profiler.mom_advection = 0.0;
  //	globals.profiler.viscosity = 0.0;
  //	globals.profiler.ideal_gas = 0.0;
  //	globals.profiler.visit = 0.0;
  //	globals.profiler.summary = 0.0;
  //	globals.profiler.reset = 0.0;
  //	globals.profiler.revert = 0.0;
  //	globals.profiler.flux = 0.0;
  //	globals.profiler.tile_halo_exchange = 0.0;
  //	globals.profiler.self_halo_exchange = 0.0;
  //	globals.profiler.mpi_halo_exchange = 0.0;

  if (parallel.boss) {
    g_out << "Reading input file" << std::endl << std::endl;
  }

  std::string line;

  // Count the number of "state ..." lines in the input file
  while (true) {
    std::getline(g_in, line);
    if (g_in.eof()) break;
    if (line.empty()) continue;

    // Break on spaces
    std::istringstream iss(line);
    std::vector<std::string> words((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
    if (words[0] == "state") {
      state_max = std::max(state_max, std::stoi(words[1]));
    }
  }

  globals.number_of_states = state_max;

  if (globals.number_of_states < 1) report_error((char *)"read_input", (char *)"No states defined.");

  globals.states = std::vector<state_type>(globals.number_of_states);
  for (int s = 0; s < globals.number_of_states; ++s) {
    globals.states[s].defined = false;
    globals.states[s].energy = 0.0;
    globals.states[s].density = 0.0;
    globals.states[s].xvel = 0.0;
    globals.states[s].yvel = 0.0;
  }

  // Rewind input file
  g_in.clear();
  g_in.seekg(0);

  while (true) {
    std::getline(g_in, line);
    if (g_in.eof()) break;
    if (line.empty()) continue;

    // Split line on spaces and =
    std::vector<std::string> words;
    char *c_line = new char[line.size() + 1];
    std::strcpy(c_line, line.c_str());
    for (char *w = std::strtok(c_line, " ="); w != nullptr; w = std::strtok(nullptr, " =")) {
      words.emplace_back(w);
    }

    // Set options based on keywords
    if (words[0].empty()) break;

    if (words[0] == "initial_timestep") {
      globals.dtinit = std::stod(words[1]);
      if (parallel.boss) g_out << " initial_timestep " << globals.dtinit << std::endl;
    } else if (words[0] == "max_timestep") {
      globals.dtmax = std::stod(words[1]);
      if (parallel.boss) g_out << " max_timestep " << globals.dtmax << std::endl;
    } else if (words[0] == "timestep_rise") {
      globals.dtrise = std::stod(words[1]);
      if (parallel.boss) g_out << " timestep_rise " << globals.dtrise << std::endl;
    } else if (words[0] == "end_time") {
      globals.end_time = std::stod(words[1]);
      if (parallel.boss) g_out << " end_time " << globals.end_time << std::endl;
    } else if (words[0] == "end_step") {
      globals.end_step = std::stoi(words[1]);
      if (parallel.boss) g_out << " end_step " << globals.end_step << std::endl;
    } else if (words[0] == "xmin") {
      globals.grid.xmin = std::stod(words[1]);
      if (parallel.boss) g_out << " xmin " << globals.grid.xmin << std::endl;
    } else if (words[0] == "xmax") {
      globals.grid.xmax = std::stod(words[1]);
      if (parallel.boss) g_out << " xmax " << globals.grid.xmax << std::endl;
    } else if (words[0] == "ymin") {
      globals.grid.ymin = std::stod(words[1]);
      if (parallel.boss) g_out << " ymin " << globals.grid.ymin << std::endl;
    } else if (words[0] == "ymax") {
      globals.grid.ymax = std::stod(words[1]);
      if (parallel.boss) g_out << " ymax " << globals.grid.ymax << std::endl;
    } else if (words[0] == "x_cells") {
      globals.grid.x_cells = std::stoi(words[1]);
      if (parallel.boss) g_out << " x_cells " << globals.grid.x_cells << std::endl;
    } else if (words[0] == "y_cells") {
      globals.grid.y_cells = std::stoi(words[1]);
      if (parallel.boss) g_out << " y_cells " << globals.grid.y_cells << std::endl;
    } else if (words[0] == "visit_frequency") {
      globals.visit_frequency = std::stoi(words[1]);
      if (parallel.boss) g_out << " visit_frequency " << globals.visit_frequency << std::endl;
    } else if (words[0] == "summary_frequency") {
      globals.summary_frequency = std::stoi(words[1]);
      if (parallel.boss) g_out << " summary_frequency " << globals.summary_frequency << std::endl;
    } else if (words[0] == "tiles_per_chunk") {
      globals.tiles_per_chunk = std::stoi(words[1]);
      if (parallel.boss) g_out << " tiles_per_chunk " << globals.tiles_per_chunk << std::endl;
    } else if (words[0] == "tiles_per_problem") {
      globals.tiles_per_chunk = std::stoi(words[1]) / parallel.max_task;
      if (parallel.boss) g_out << " tiles_per_chunk " << globals.tiles_per_chunk << std::endl;
    } else if (words[0] == "profiler_on") {
      globals.profiler_on = true;
      if (parallel.boss) g_out << " Profiler on" << std::endl;
    } else if (words[0] == "test_problem") {
      globals.test_problem = std::stoi(words[1]);
      if (parallel.boss) g_out << " test_problem " << globals.test_problem;
    } else if (words[0] == "state") {
      int state = std::stoi(words[1]) - 1;

      if (parallel.boss) g_out << "Reading specification for state " << state + 1 << std::endl << std::endl;
      if (globals.states[state].defined) report_error((char *)"read_input", (char *)"State defined twice.");

      globals.states[state].defined = true;
      for (size_t iw = 2; iw < words.size(); ++iw) {
        std::string w = words[iw];
        if (w.empty()) break;

        if (w == "xvel") {
          w = words[++iw];
          globals.states[state].xvel = std::stod(w);
          if (parallel.boss) g_out << " xvel " << globals.states[state].xvel << std::endl;
        } else if (w == "yvel") {
          w = words[++iw];
          globals.states[state].yvel = std::stod(w);
          if (parallel.boss) g_out << " yvel " << globals.states[state].yvel << std::endl;
        } else if (w == "xmin") {
          w = words[++iw];
          globals.states[state].xmin = std::stod(w);
          if (parallel.boss) g_out << " xmin " << globals.states[state].xmin << std::endl;
        } else if (w == "ymin") {
          w = words[++iw];
          globals.states[state].ymin = std::stod(w);
          if (parallel.boss) g_out << " ymin " << globals.states[state].ymin << std::endl;
        } else if (w == "xmax") {
          w = words[++iw];
          globals.states[state].xmax = std::stod(w);
          if (parallel.boss) g_out << " xmax " << globals.states[state].xmax << std::endl;
        } else if (w == "ymax") {
          w = words[++iw];
          globals.states[state].ymax = std::stod(w);
          if (parallel.boss) g_out << " ymax " << globals.states[state].ymax << std::endl;
        } else if (w == "radius") {
          w = words[++iw];
          globals.states[state].radius = std::stod(w);
          if (parallel.boss) g_out << " radius " << globals.states[state].radius << std::endl;
        } else if (w == "density") {
          w = words[++iw];
          globals.states[state].density = std::stod(w);
          if (parallel.boss) g_out << " density " << globals.states[state].density << std::endl;
        } else if (w == "energy") {
          w = words[++iw];
          globals.states[state].energy = std::stod(w);
          if (parallel.boss) g_out << " energy " << globals.states[state].energy << std::endl;
        } else if (w == "geometry") {
          w = words[++iw];
          if (w == "rectangle") {
            globals.states[state].geometry = geometry_type::g_rect;
            if (parallel.boss) g_out << " state geometry rectangular" << std::endl;
          } else if (w == "circle") {
            globals.states[state].geometry = geometry_type::g_circ;
            if (parallel.boss) g_out << " state geometry circular" << std::endl;
          } else if (w == "point") {
            globals.states[state].geometry = geometry_type::g_point;
            if (parallel.boss) g_out << " state geometry point" << std::endl;
          }
        }
      }
    }
    if (parallel.boss) g_out << std::endl;
  }

  if (parallel.boss) {
    g_out << std::endl << std::endl << "Input read finished." << std::endl << std::endl;
  }

  // If a state boundary falls exactly on a cell boundary then round off can
  // cause the state to be put one cell further that expected. This is compiler
  // /system dependent. To avoid this, a state boundary is reduced/increased by a 100th
  // of a cell width so it lies well with in the intended cell.
  // Because a cell is either full or empty of a specified state, this small
  // modification to the state extents does not change the answers.
  double dx, dy;
  dx = (globals.grid.xmax - globals.grid.xmin) / (float)globals.grid.x_cells;
  dy = (globals.grid.ymax - globals.grid.ymin) / (float)globals.grid.y_cells;
  for (int n = 1; n < globals.number_of_states; ++n) {
    globals.states[n].xmin += dx / 100.0;
    globals.states[n].ymin += dy / 100.0;
    globals.states[n].xmax -= dx / 100.0;
    globals.states[n].ymax -= dy / 100.0;
  }
}
