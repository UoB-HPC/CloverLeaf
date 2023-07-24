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

#pragma once

#include "context.h"
#include "definitions.h"

void update_tile_halo_l_kernel(global_variables &globals, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &density0,
                               clover::Buffer2D<double> &energy0, clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &viscosity,
                               clover::Buffer2D<double> &soundspeed, clover::Buffer2D<double> &density1, clover::Buffer2D<double> &energy1,
                               clover::Buffer2D<double> &xvel0, clover::Buffer2D<double> &yvel0, clover::Buffer2D<double> &xvel1,
                               clover::Buffer2D<double> &yvel1, clover::Buffer2D<double> &vol_flux_x, clover::Buffer2D<double> &vol_flux_y,
                               clover::Buffer2D<double> &mass_flux_x, clover::Buffer2D<double> &mass_flux_y, int left_xmin, int left_xmax,
                               int left_ymin, int left_ymax, clover::Buffer2D<double> &left_density0,
                               clover::Buffer2D<double> &left_energy0, clover::Buffer2D<double> &left_pressure,
                               clover::Buffer2D<double> &left_viscosity, clover::Buffer2D<double> &left_soundspeed,
                               clover::Buffer2D<double> &left_density1, clover::Buffer2D<double> &left_energy1,
                               clover::Buffer2D<double> &left_xvel0, clover::Buffer2D<double> &left_yvel0,
                               clover::Buffer2D<double> &left_xvel1, clover::Buffer2D<double> &left_yvel1,
                               clover::Buffer2D<double> &left_vol_flux_x, clover::Buffer2D<double> &left_vol_flux_y,
                               clover::Buffer2D<double> &left_mass_flux_x, clover::Buffer2D<double> &left_mass_flux_y,
                               const int fields[NUM_FIELDS], int depth);

void update_tile_halo_r_kernel(global_variables &globals, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &density0,
                               clover::Buffer2D<double> &energy0, clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &viscosity,
                               clover::Buffer2D<double> &soundspeed, clover::Buffer2D<double> &density1, clover::Buffer2D<double> &energy1,
                               clover::Buffer2D<double> &xvel0, clover::Buffer2D<double> &yvel0, clover::Buffer2D<double> &xvel1,
                               clover::Buffer2D<double> &yvel1, clover::Buffer2D<double> &vol_flux_x, clover::Buffer2D<double> &vol_flux_y,
                               clover::Buffer2D<double> &mass_flux_x, clover::Buffer2D<double> &mass_flux_y, int right_xmin, int right_xmax,
                               int right_ymin, int right_ymax, clover::Buffer2D<double> &right_density0,
                               clover::Buffer2D<double> &right_energy0, clover::Buffer2D<double> &right_pressure,
                               clover::Buffer2D<double> &right_viscosity, clover::Buffer2D<double> &right_soundspeed,
                               clover::Buffer2D<double> &right_density1, clover::Buffer2D<double> &right_energy1,
                               clover::Buffer2D<double> &right_xvel0, clover::Buffer2D<double> &right_yvel0,
                               clover::Buffer2D<double> &right_xvel1, clover::Buffer2D<double> &right_yvel1,
                               clover::Buffer2D<double> &right_vol_flux_x, clover::Buffer2D<double> &right_vol_flux_y,
                               clover::Buffer2D<double> &right_mass_flux_x, clover::Buffer2D<double> &right_mass_flux_y,
                               const int fields[NUM_FIELDS], int depth);

void update_tile_halo_t_kernel(global_variables &globals, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &density0,
                               clover::Buffer2D<double> &energy0, clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &viscosity,
                               clover::Buffer2D<double> &soundspeed, clover::Buffer2D<double> &density1, clover::Buffer2D<double> &energy1,
                               clover::Buffer2D<double> &xvel0, clover::Buffer2D<double> &yvel0, clover::Buffer2D<double> &xvel1,
                               clover::Buffer2D<double> &yvel1, clover::Buffer2D<double> &vol_flux_x, clover::Buffer2D<double> &vol_flux_y,
                               clover::Buffer2D<double> &mass_flux_x, clover::Buffer2D<double> &mass_flux_y, int top_xmin, int top_xmax,
                               int top_ymin, int top_ymax, clover::Buffer2D<double> &top_density0, clover::Buffer2D<double> &top_energy0,
                               clover::Buffer2D<double> &top_pressure, clover::Buffer2D<double> &top_viscosity,
                               clover::Buffer2D<double> &top_soundspeed, clover::Buffer2D<double> &top_density1,
                               clover::Buffer2D<double> &top_energy1, clover::Buffer2D<double> &top_xvel0,
                               clover::Buffer2D<double> &top_yvel0, clover::Buffer2D<double> &top_xvel1,
                               clover::Buffer2D<double> &top_yvel1, clover::Buffer2D<double> &top_vol_flux_x,
                               clover::Buffer2D<double> &top_vol_flux_y, clover::Buffer2D<double> &top_mass_flux_x,
                               clover::Buffer2D<double> &top_mass_flux_y, const int fields[NUM_FIELDS], int depth);

void update_tile_halo_b_kernel(global_variables &globals, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &density0,
                               clover::Buffer2D<double> &energy0, clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &viscosity,
                               clover::Buffer2D<double> &soundspeed, clover::Buffer2D<double> &density1, clover::Buffer2D<double> &energy1,
                               clover::Buffer2D<double> &xvel0, clover::Buffer2D<double> &yvel0, clover::Buffer2D<double> &xvel1,
                               clover::Buffer2D<double> &yvel1, clover::Buffer2D<double> &vol_flux_x, clover::Buffer2D<double> &vol_flux_y,
                               clover::Buffer2D<double> &mass_flux_x, clover::Buffer2D<double> &mass_flux_y, int bottom_xmin,
                               int bottom_xmax, int bottom_ymin, int bottom_ymax, clover::Buffer2D<double> &bottom_density0,
                               clover::Buffer2D<double> &bottom_energy0, clover::Buffer2D<double> &bottom_pressure,
                               clover::Buffer2D<double> &bottom_viscosity, clover::Buffer2D<double> &bottom_soundspeed,
                               clover::Buffer2D<double> &bottom_density1, clover::Buffer2D<double> &bottom_energy1,
                               clover::Buffer2D<double> &bottom_xvel0, clover::Buffer2D<double> &bottom_yvel0,
                               clover::Buffer2D<double> &bottom_xvel1, clover::Buffer2D<double> &bottom_yvel1,
                               clover::Buffer2D<double> &bottom_vol_flux_x, clover::Buffer2D<double> &bottom_vol_flux_y,
                               clover::Buffer2D<double> &bottom_mass_flux_x, clover::Buffer2D<double> &bottom_mass_flux_y,
                               const int fields[NUM_FIELDS], int depth);
