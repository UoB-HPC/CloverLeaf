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

void generate_chunk(const int tile, global_variables &globals) {

  // Need to copy the host array of state input data into a device array
  // XXX This probably doesn't need ViewAllocateWithoutInitializing because of how small `globals.number_of_states` is.
  Kokkos::View<double *> state_density("state_density", globals.config.number_of_states);
  Kokkos::View<double *> state_energy("state_energy", globals.config.number_of_states);
  Kokkos::View<double *> state_xvel("state_xvel", globals.config.number_of_states);
  Kokkos::View<double *> state_yvel("state_yvel", globals.config.number_of_states);
  Kokkos::View<double *> state_xmin("state_xmin", globals.config.number_of_states);
  Kokkos::View<double *> state_xmax("state_xmax", globals.config.number_of_states);
  Kokkos::View<double *> state_ymin("state_ymin", globals.config.number_of_states);
  Kokkos::View<double *> state_ymax("state_ymax", globals.config.number_of_states);
  Kokkos::View<double *> state_radius("state_radius", globals.config.number_of_states);
  Kokkos::View<int *> state_geometry("state_geometry", globals.config.number_of_states);

  // Create host mirrors of this to copy on the host
  typename Kokkos::View<double *>::HostMirror hm_state_density = Kokkos::create_mirror_view(state_density);
  typename Kokkos::View<double *>::HostMirror hm_state_energy = Kokkos::create_mirror_view(state_energy);
  typename Kokkos::View<double *>::HostMirror hm_state_xvel = Kokkos::create_mirror_view(state_xvel);
  typename Kokkos::View<double *>::HostMirror hm_state_yvel = Kokkos::create_mirror_view(state_yvel);
  typename Kokkos::View<double *>::HostMirror hm_state_xmin = Kokkos::create_mirror_view(state_xmin);
  typename Kokkos::View<double *>::HostMirror hm_state_xmax = Kokkos::create_mirror_view(state_xmax);
  typename Kokkos::View<double *>::HostMirror hm_state_ymin = Kokkos::create_mirror_view(state_ymin);
  typename Kokkos::View<double *>::HostMirror hm_state_ymax = Kokkos::create_mirror_view(state_ymax);
  typename Kokkos::View<double *>::HostMirror hm_state_radius = Kokkos::create_mirror_view(state_radius);
  typename Kokkos::View<int *>::HostMirror hm_state_geometry = Kokkos::create_mirror_view(state_geometry);

  // Copy the data to the new views
  for (int state = 0; state < globals.config.number_of_states; ++state) {
    hm_state_density(state) = globals.config.states[state].density;
    hm_state_energy(state) = globals.config.states[state].energy;
    hm_state_xvel(state) = globals.config.states[state].xvel;
    hm_state_yvel(state) = globals.config.states[state].yvel;
    hm_state_xmin(state) = globals.config.states[state].xmin;
    hm_state_xmax(state) = globals.config.states[state].xmax;
    hm_state_ymin(state) = globals.config.states[state].ymin;
    hm_state_ymax(state) = globals.config.states[state].ymax;
    hm_state_radius(state) = globals.config.states[state].radius;
    hm_state_geometry(state) = globals.config.states[state].geometry;
  }

  Kokkos::deep_copy(state_density, hm_state_density);
  Kokkos::deep_copy(state_energy, hm_state_energy);
  Kokkos::deep_copy(state_xvel, hm_state_xvel);
  Kokkos::deep_copy(state_yvel, hm_state_yvel);
  Kokkos::deep_copy(state_xmin, hm_state_xmin);
  Kokkos::deep_copy(state_xmax, hm_state_xmax);
  Kokkos::deep_copy(state_ymin, hm_state_ymin);
  Kokkos::deep_copy(state_ymax, hm_state_ymax);
  Kokkos::deep_copy(state_radius, hm_state_radius);
  Kokkos::deep_copy(state_geometry, hm_state_geometry);

  const int x_min = globals.chunk.tiles[tile].info.t_xmin;
  const int x_max = globals.chunk.tiles[tile].info.t_xmax;
  const int y_min = globals.chunk.tiles[tile].info.t_ymin;
  const int y_max = globals.chunk.tiles[tile].info.t_ymax;

  size_t xrange = (x_max + 2) - (x_min - 2) + 1;
  size_t yrange = (y_max + 2) - (y_min - 2) + 1;

  // Take a reference to the lowest structure, as Kokkos device cannot necessarily chase through the structure.
  field_type &field = globals.chunk.tiles[tile].field;

  Kokkos::MDRangePolicy<Kokkos::Rank<2>> xyrange_policy({0, 0}, {xrange, yrange});

  // State 1 is always the background state
  Kokkos::parallel_for(
      xyrange_policy, KOKKOS_LAMBDA(const int j, const int k) {
        field.energy0.view(j, k) = state_energy(0);
        field.density0.view(j, k) = state_density(0);
        field.xvel0.view(j, k) = state_xvel(0);
        field.yvel0.view(j, k) = state_yvel(0);
      });

  for (int state = 1; state < globals.config.number_of_states; ++state) {
    Kokkos::parallel_for(
        xyrange_policy, KOKKOS_LAMBDA(const int j, const int k) {
          double x_cent = state_xmin(state);
          double y_cent = state_ymin(state);

          if (state_geometry(state) == g_rect) {
            if (field.vertexx.view(j + 1) >= state_xmin(state) && field.vertexx.view(j) < state_xmax(state)) {
              if (field.vertexy.view(k + 1) >= state_ymin(state) && field.vertexy.view(k) < state_ymax(state)) {
                field.energy0.view(j, k) = state_energy(state);
                field.density0.view(j, k) = state_density(state);
                for (int kt = k; kt <= k + 1; ++kt) {
                  for (int jt = j; jt <= j + 1; ++jt) {
                    field.xvel0.view(jt, kt) = state_xvel(state);
                    field.yvel0.view(jt, kt) = state_yvel(state);
                  }
                }
              }
            }
          } else if (state_geometry(state) == g_circ) {
            double radius = sqrt((field.cellx.view(j) - x_cent) * (field.cellx.view(j) - x_cent) +
                                 (field.celly.view(k) - y_cent) * (field.celly.view(k) - y_cent));
            if (radius <= state_radius(state)) {
              field.energy0.view(j, k) = state_energy(state);
              field.density0.view(j, k) = state_density(state);
              for (int kt = k; kt <= k + 1; ++kt) {
                for (int jt = j; jt <= j + 1; ++jt) {
                  field.xvel0.view(jt, kt) = state_xvel(state);
                  field.yvel0.view(jt, kt) = state_yvel(state);
                }
              }
            }
          } else if (state_geometry(state) == g_point) {
            if (field.vertexx.view(j) == x_cent && field.vertexy.view(k) == y_cent) {
              field.energy0.view(j, k) = state_energy(state);
              field.density0.view(j, k) = state_density(state);
              for (int kt = k; kt <= k + 1; ++kt) {
                for (int jt = j; jt <= j + 1; ++jt) {
                  field.xvel0.view(jt, kt) = state_xvel(state);
                  field.yvel0.view(jt, kt) = state_yvel(state);
                }
              }
            }
          }
        });
  }
}
