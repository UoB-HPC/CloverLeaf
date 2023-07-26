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
#include "context.h"

void generate_chunk(const int tile, global_variables &globals) {

  // Need to copy the host array of state input data into a device array
  std::vector<double> state_density_vec(globals.config.number_of_states);
  std::vector<double> state_energy_vec(globals.config.number_of_states);
  std::vector<double> state_xvel_vec(globals.config.number_of_states);
  std::vector<double> state_yvel_vec(globals.config.number_of_states);
  std::vector<double> state_xmin_vec(globals.config.number_of_states);
  std::vector<double> state_xmax_vec(globals.config.number_of_states);
  std::vector<double> state_ymin_vec(globals.config.number_of_states);
  std::vector<double> state_ymax_vec(globals.config.number_of_states);
  std::vector<double> state_radius_vec(globals.config.number_of_states);
  std::vector<int> state_geometry_vec(globals.config.number_of_states);

  // Copy the data to the new views
  for (int state = 0; state < globals.config.number_of_states; ++state) {
    state_density_vec[state] = globals.config.states[state].density;
    state_energy_vec[state] = globals.config.states[state].energy;
    state_xvel_vec[state] = globals.config.states[state].xvel;
    state_yvel_vec[state] = globals.config.states[state].yvel;
    state_xmin_vec[state] = globals.config.states[state].xmin;
    state_xmax_vec[state] = globals.config.states[state].xmax;
    state_ymin_vec[state] = globals.config.states[state].ymin;
    state_ymax_vec[state] = globals.config.states[state].ymax;
    state_radius_vec[state] = globals.config.states[state].radius;
    state_geometry_vec[state] = globals.config.states[state].geometry;
  }

  // Create host mirrors of this to copy on the host
  clover::Buffer1D<double> hm_state_density(state_density_vec.begin(), state_density_vec.end());
  clover::Buffer1D<double> hm_state_energy(state_energy_vec.begin(), state_energy_vec.end());
  clover::Buffer1D<double> hm_state_xvel(state_xvel_vec.begin(), state_xvel_vec.end());
  clover::Buffer1D<double> hm_state_yvel(state_yvel_vec.begin(), state_yvel_vec.end());
  clover::Buffer1D<double> hm_state_xmin(state_xmin_vec.begin(), state_xmin_vec.end());
  clover::Buffer1D<double> hm_state_xmax(state_xmax_vec.begin(), state_xmax_vec.end());
  clover::Buffer1D<double> hm_state_ymin(state_ymin_vec.begin(), state_ymin_vec.end());
  clover::Buffer1D<double> hm_state_ymax(state_ymax_vec.begin(), state_ymax_vec.end());
  clover::Buffer1D<double> hm_state_radius(state_radius_vec.begin(), state_radius_vec.end());
  clover::Buffer1D<int> hm_state_geometry(state_geometry_vec.begin(), state_geometry_vec.end());

  // Kokkos::deep_copy (TO, FROM)

  const int x_min = globals.chunk.tiles[tile].info.t_xmin;
  const int x_max = globals.chunk.tiles[tile].info.t_xmax;
  const int y_min = globals.chunk.tiles[tile].info.t_ymin;
  const int y_max = globals.chunk.tiles[tile].info.t_ymax;

  size_t xrange = (x_max + 2) - (x_min - 2) + 1;
  size_t yrange = (y_max + 2) - (y_min - 2) + 1;

  // Take a reference to the lowest structure, as Kokkos device cannot necessarily chase through the structure.

  clover::Range2d xyrange_policy(0u, 0u, xrange, yrange);

  field_type &field = globals.chunk.tiles[tile].field;

  clover::execute(globals.context.queue, [&](handler &h) {
    auto density0 = field.density0.access<W>(h);
    auto xvel0 = field.xvel0.access<W>(h);
    auto yvel0 = field.yvel0.access<W>(h);
    auto energy0 = field.energy0.access<W>(h);

    auto state_density = hm_state_density.access<R>(h);
    auto state_energy = hm_state_energy.access<R>(h);
    auto state_xvel = hm_state_xvel.access<R>(h);
    auto state_yvel = hm_state_yvel.access<R>(h);
    // State 1 is always the background state
    clover::par_ranged<class generate_chunk_1>(h, xyrange_policy, [=](id<2> idx) {
      energy0[idx] = state_energy[0];
      density0[idx] = state_density[0];
      xvel0[idx] = state_xvel[0];
      yvel0[idx] = state_yvel[0];
    });
  });

  for (int state = 1; state < globals.config.number_of_states; ++state) {
    clover::execute(globals.context.queue, [&](handler &h) {
      auto density0 = field.density0.access<W>(h);
      auto xvel0 = field.xvel0.access<W>(h);
      auto yvel0 = field.yvel0.access<W>(h);
      auto energy0 = field.energy0.access<W>(h);

      auto state_density = hm_state_density.access<R>(h);
      auto state_energy = hm_state_energy.access<R>(h);
      auto state_xvel = hm_state_xvel.access<R>(h);
      auto state_yvel = hm_state_yvel.access<R>(h);

      auto state_xmin = hm_state_xmin.access<R>(h);
      auto state_xmax = hm_state_xmax.access<R>(h);
      auto state_ymin = hm_state_ymin.access<R>(h);
      auto state_ymax = hm_state_ymax.access<R>(h);
      auto state_radius = hm_state_radius.access<R>(h);
      auto state_geometry = hm_state_geometry.access<R>(h);

      auto cellx = field.cellx.access<RW>(h);
      auto celly = field.celly.access<RW>(h);

      auto vertexx = field.vertexx.access<RW>(h);
      auto vertexy = field.vertexy.access<RW>(h);

      clover::par_ranged<class generate_chunk_2>(h, xyrange_policy, [=](id<2> idx) {
        const int j = idx.get(0);
        const int k = idx.get(1);

        double x_cent = state_xmin[state];
        double y_cent = state_ymin[state];

        if (state_geometry[state] == g_rect) {
          if (vertexx[j + 1] >= state_xmin[state] && vertexx[j] < state_xmax[state]) {
            if (vertexy[k + 1] >= state_ymin[state] && vertexy[k] < state_ymax[state]) {
              energy0[idx] = state_energy[state];
              density0[idx] = state_density[state];
              for (int kt = k; kt <= k + 1; ++kt) {
                for (int jt = j; jt <= j + 1; ++jt) {
                  xvel0[jt][kt] = state_xvel[state];
                  yvel0[jt][kt] = state_yvel[state];
                }
              }
            }
          }
        } else if (state_geometry[state] == g_circ) {
          double radius = sycl::sqrt((cellx[j] - x_cent) * (cellx[j] - x_cent) + (celly[k] - y_cent) * (celly[k] - y_cent));
          if (radius <= state_radius[state]) {
            energy0[idx] = state_energy[state];
            density0[idx] = state_density[state];
            for (int kt = k; kt <= k + 1; ++kt) {
              for (int jt = j; jt <= j + 1; ++jt) {
                xvel0[jt][kt] = state_xvel[state];
                yvel0[jt][kt] = state_yvel[state];
              }
            }
          }
        } else if (state_geometry[state] == g_point) {
          if (vertexx[j] == x_cent && vertexy[k] == y_cent) {
            energy0[idx] = state_energy[state];
            density0[idx] = state_density[state];
            for (int kt = k; kt <= k + 1; ++kt) {
              for (int jt = j; jt <= j + 1; ++jt) {
                xvel0[jt][kt] = state_xvel[state];
                yvel0[jt][kt] = state_yvel[state];
              }
            }
          }
        }
      });
    });
  }
}
