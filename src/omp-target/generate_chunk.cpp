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

//  @brief Mesh chunk generation driver
//  @author Wayne Gaudin
//  @details Invoked the users specified chunk generator.
//  @brief Mesh chunk generation driver
//  @author Wayne Gaudin
//  @details Invoked the users specified chunk generator.

#include "generate_chunk.h"
#include <cmath>

void generate_chunk(const int tile, global_variables &globals) {

  // Need to copy the host array of state input data into a device array
  clover::Buffer1D<double> state_density_buffer(globals.context, globals.config.number_of_states);
  clover::Buffer1D<double> state_energy_buffer(globals.context, globals.config.number_of_states);
  clover::Buffer1D<double> state_xvel_buffer(globals.context, globals.config.number_of_states);
  clover::Buffer1D<double> state_yvel_buffer(globals.context, globals.config.number_of_states);
  clover::Buffer1D<double> state_xmin_buffer(globals.context, globals.config.number_of_states);
  clover::Buffer1D<double> state_xmax_buffer(globals.context, globals.config.number_of_states);
  clover::Buffer1D<double> state_ymin_buffer(globals.context, globals.config.number_of_states);
  clover::Buffer1D<double> state_ymax_buffer(globals.context, globals.config.number_of_states);
  clover::Buffer1D<double> state_radius_buffer(globals.context, globals.config.number_of_states);
  clover::Buffer1D<int> state_geometry_buffer(globals.context, globals.config.number_of_states);

  // Copy the data to the new views
  for (int state = 0; state < globals.config.number_of_states; ++state) {
    state_density_buffer[state] = globals.config.states[state].density;
    state_energy_buffer[state] = globals.config.states[state].energy;
    state_xvel_buffer[state] = globals.config.states[state].xvel;
    state_yvel_buffer[state] = globals.config.states[state].yvel;
    state_xmin_buffer[state] = globals.config.states[state].xmin;
    state_xmax_buffer[state] = globals.config.states[state].xmax;
    state_ymin_buffer[state] = globals.config.states[state].ymin;
    state_ymax_buffer[state] = globals.config.states[state].ymax;
    state_radius_buffer[state] = globals.config.states[state].radius;
    state_geometry_buffer[state] = globals.config.states[state].geometry;
  }

  // Kokkos::deep_copy (TO, FROM)

  const int x_min = globals.chunk.tiles[tile].info.t_xmin;
  const int x_max = globals.chunk.tiles[tile].info.t_xmax;
  const int y_min = globals.chunk.tiles[tile].info.t_ymin;
  const int y_max = globals.chunk.tiles[tile].info.t_ymax;

  int xrange = (x_max + 2) - (x_min - 2) + 1;
  int yrange = (y_max + 2) - (y_min - 2) + 1;

  // Take a reference to the lowest structure, as Kokkos device cannot necessarily chase through the structure.

  field_type &field = globals.chunk.tiles[tile].field;

  const double state_energy_0 = state_energy_buffer[0];
  const double state_density_0 = state_density_buffer[0];
  const double state_xvel_0 = state_xvel_buffer[0];
  const double state_yvel_0 = state_yvel_buffer[0];

  const int base_stride = field.base_stride;
  const int vels_wk_stride = field.vels_wk_stride;

  // State 1 is always the background state
  double *energy0 = field.energy0.data;
  double *density0 = field.density0.data;
  double *xvel0 = field.xvel0.data;
  double *yvel0 = field.yvel0.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(globals.context.use_target)
  for (int j = 0; j < (yrange); j++) {
    for (int i = 0; i < (xrange); i++) {
      energy0[i + j * base_stride] = state_energy_0;
      density0[i + j * base_stride] = state_density_0;
      xvel0[i + j * vels_wk_stride] = state_xvel_0;
      yvel0[i + j * vels_wk_stride] = state_yvel_0;
    }
  }

  for (int state = 1; state < globals.config.number_of_states; ++state) {

    double *cellx = field.cellx.data;
    double *celly = field.celly.data;

    double *vertexx = field.vertexx.data;
    double *vertexy = field.vertexy.data;

    const double *state_density = state_density_buffer.data;
    const double *state_energy = state_energy_buffer.data;
    const double *state_xvel = state_xvel_buffer.data;
    const double *state_yvel = state_yvel_buffer.data;
    const double *state_xmin = state_xmin_buffer.data;
    const double *state_xmax = state_xmax_buffer.data;
    const double *state_ymin = state_ymin_buffer.data;
    const double *state_ymax = state_ymax_buffer.data;
    const double *state_radius = state_radius_buffer.data;
    const int *state_geometry = state_geometry_buffer.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(globals.context.use_target)                            \
    map(to : state_density[ : state_density_buffer.N()]) map(to : state_energy[ : state_energy_buffer.N()])                                \
    map(to : state_xvel[ : state_xvel_buffer.N()]) map(to : state_yvel[ : state_yvel_buffer.N()])                                          \
    map(to : state_xmin[ : state_xmin_buffer.N()]) map(to : state_xmax[ : state_xmax_buffer.N()])                                          \
    map(to : state_ymin[ : state_ymin_buffer.N()]) map(to : state_ymax[ : state_ymax_buffer.N()])                                          \
    map(to : state_radius[ : state_radius_buffer.N()]) map(to : state_geometry[ : state_geometry_buffer.N()])
    for (int j = 0; j < (yrange); j++) {
      for (int i = 0; i < (xrange); i++) {
        double x_cent = state_xmin[state];
        double y_cent = state_ymin[state];
        if (state_geometry[state] == g_rect) {
          if (vertexx[i + 1] >= state_xmin[state] && vertexx[i] < state_xmax[state]) {
            if (vertexy[j + 1] >= state_ymin[state] && vertexy[j] < state_ymax[state]) {
              energy0[i + j * base_stride] = state_energy[state];
              density0[i + j * base_stride] = state_density[state];
              for (int kt = j; kt <= j + 1; ++kt) {
                for (int jt = i; jt <= i + 1; ++jt) {
                  xvel0[jt + kt * vels_wk_stride] = state_xvel[state];
                  yvel0[jt + kt * vels_wk_stride] = state_yvel[state];
                }
              }
            }
          }
        } else if (state_geometry[state] == g_circ) {
          double radius = sqrt((cellx[i] - x_cent) * (cellx[i] - x_cent) + (celly[j] - y_cent) * (celly[j] - y_cent));
          if (radius <= state_radius[state]) {
            energy0[i + j * base_stride] = state_energy[state];
            density0[i + j * base_stride] = state_density[state];
            for (int kt = j; kt <= j + 1; ++kt) {
              for (int jt = i; jt <= i + 1; ++jt) {
                xvel0[jt + kt * vels_wk_stride] = state_xvel[state];
                yvel0[jt + kt * vels_wk_stride] = state_yvel[state];
              }
            }
          }
        } else if (state_geometry[state] == g_point) {
          if (vertexx[i] == x_cent && vertexy[j] == y_cent) {
            energy0[i + j * base_stride] = state_energy[state];
            density0[i + j * base_stride] = state_density[state];
            for (int kt = j; kt <= j + 1; ++kt) {
              for (int jt = i; jt <= i + 1; ++jt) {
                xvel0[jt + kt * vels_wk_stride] = state_xvel[state];
                yvel0[jt + kt * vels_wk_stride] = state_yvel[state];
              }
            }
          }
        }
      }
    }
  }
}
