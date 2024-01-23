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

#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#define g_ibig (640000)
#define g_small (1.0e-16)
#define g_big (1.0e+21)
#define NUM_FIELDS (15)

#define PP_CAT(a, b) PP_CAT_I(a, b)
#define PP_CAT_I(a, b) PP_CAT_II(~, a##b)
#define PP_CAT_II(p, res) res

#define APPEND_LN(base) PP_CAT(base, __LINE__) // appends line number to the given base name

namespace clover {
template <typename T> struct Buffer2D;
template <typename T> struct Buffer1D;
// template <typename T> struct StagingBuffer1D;

struct chunk_context;
struct context;

// template <typename T> Buffer1D<T> alloc(size_t x);
// template <typename T> Buffer2D<T> alloc(size_t x, size_t y);

} // namespace clover

enum geometry_type { g_rect = 1, g_circ = 2, g_point = 3 };
// In the Fortran version these are 1,2,3,4,-1, but they are used directly to index an array in this version
enum chunk_neighbour_type { chunk_left = 0, chunk_right = 1, chunk_bottom = 2, chunk_top = 3, external_face = -1 };
enum tile_neighbour_type { tile_left = 0, tile_right = 1, tile_bottom = 3, tile_top = 3, external_tile = -1 };

// Again, start at 0 as used for indexing an array of length NUM_FIELDS
enum field_parameter {
  field_density0 = 0,
  field_density1 = 1,
  field_energy0 = 2,
  field_energy1 = 3,
  field_pressure = 4,
  field_viscosity = 5,
  field_soundspeed = 6,
  field_xvel0 = 7,
  field_xvel1 = 8,
  field_yvel0 = 9,
  field_yvel1 = 10,
  field_vol_flux_x = 11,
  field_vol_flux_y = 12,
  field_mass_flux_x = 13,
  field_mass_flux_y = 14
};

enum data_parameter { cell_data = 1, vertex_data = 2, x_face_data = 3, y_face_data = 4 };
enum dir_parameter { g_xdir = 1, g_ydir = 2 };

struct state_type {

  bool defined;

  double density;
  double energy;
  double xvel;
  double yvel;

  geometry_type geometry;

  double xmin;
  double ymin;
  double xmax;
  double ymax;
  double radius;
};

struct grid_type {

  double xmin;
  double ymin;
  double xmax;
  double ymax;

  int x_cells;
  int y_cells;
};

struct profiler_type {
  double host_to_device = 0.0;
  double device_to_host = 0.0;
  double timestep = 0.0;
  double acceleration = 0.0;
  double PdV = 0.0;
  double cell_advection = 0.0;
  double mom_advection = 0.0;
  double viscosity = 0.0;
  double ideal_gas = 0.0;
  double visit = 0.0;
  double summary = 0.0;
  double reset = 0.0;
  double revert = 0.0;
  double flux = 0.0;
  double tile_halo_exchange = 0.0;
  double self_halo_exchange = 0.0;
  double mpi_halo_exchange = 0.0;
};

struct field_type {

  clover::Buffer2D<double> density0;
  clover::Buffer2D<double> density1;
  clover::Buffer2D<double> energy0;
  clover::Buffer2D<double> energy1;
  clover::Buffer2D<double> pressure;
  clover::Buffer2D<double> viscosity;
  clover::Buffer2D<double> soundspeed;
  clover::Buffer2D<double> xvel0, xvel1;
  clover::Buffer2D<double> yvel0, yvel1;
  clover::Buffer2D<double> vol_flux_x, mass_flux_x;
  clover::Buffer2D<double> vol_flux_y, mass_flux_y;

  clover::Buffer2D<double> work_array1; // node_flux, stepbymass, volume_change, pre_vol
  clover::Buffer2D<double> work_array2; // node_mass_post, post_vol
  clover::Buffer2D<double> work_array3; // node_mass_pre,pre_mass
  clover::Buffer2D<double> work_array4; // advec_vel, post_mass
  clover::Buffer2D<double> work_array5; // mom_flux, advec_vol
  clover::Buffer2D<double> work_array6; // pre_vol, post_ener
  clover::Buffer2D<double> work_array7; // post_vol, ener_flux

  clover::Buffer1D<double> cellx;
  clover::Buffer1D<double> celldx;
  clover::Buffer1D<double> celly;
  clover::Buffer1D<double> celldy;
  clover::Buffer1D<double> vertexx;
  clover::Buffer1D<double> vertexdx;
  clover::Buffer1D<double> vertexy;
  clover::Buffer1D<double> vertexdy;

  clover::Buffer2D<double> volume;
  clover::Buffer2D<double> xarea;
  clover::Buffer2D<double> yarea;

  size_t base_stride;
  size_t vels_wk_stride;
  size_t flux_x_stride, flux_y_stride;

  explicit field_type(size_t xrange, size_t yrange, clover::context &ctx);
};

struct tile_info {

  std::array<int, 4> tile_neighbours;
  std::array<int, 4> external_tile_mask;
  int t_xmin, t_xmax, t_ymin, t_ymax;
  int t_left, t_right, t_bottom, t_top;
};

struct tile_type {

  tile_info info;
  field_type field;

  explicit tile_type(const tile_info &info, clover::context &ctx)
      : info(info),
        // (t_xmin-2:t_xmax+2, t_ymin-2:t_ymax+2)
        // XXX see build_field()
        field((info.t_xmax + 2) - (info.t_xmin - 2) + 1, (info.t_ymax + 2) - (info.t_ymin - 2) + 1, ctx) {}
};

struct chunk_type {

  clover::chunk_context context;
  //  chunk_context

  // MPI Buffers in device memory

  // MPI Buffers in host memory - to be created with Kokkos::create_mirror_view() and Kokkos::deep_copy()
  //	std::vector<double > hm_left_rcv_buffer, hm_right_rcv_buffer, hm_bottom_rcv_buffer, hm_top_rcv_buffer;
  //	std::vector<double > hm_left_snd_buffer, hm_right_snd_buffer, hm_bottom_snd_buffer, hm_top_snd_buffer;
  const std::array<int, 4> chunk_neighbours; // Chunks, not tasks, so we can overload in the future

  const int task; // MPI task
  const int x_min;
  const int y_min;
  const int x_max;
  const int y_max;

  const int left, right, bottom, top;
  const int left_boundary, right_boundary, bottom_boundary, top_boundary;

  //  clover::Buffer1D<double> left_rcv_buffer, right_rcv_buffer, bottom_rcv_buffer, top_rcv_buffer;
  //  clover::Buffer1D<double> left_snd_buffer, right_snd_buffer, bottom_snd_buffer, top_snd_buffer;

  std::vector<tile_type> tiles;

  chunk_type(const std::array<int, 4> &chunkNeighbours,                                //
             int task,                                                                 //
             int xMin, int yMin, int xMax, int yMax,                                   //
             int left, int right, int bottom, int top,                                 //
             int leftBoundary, int rightBoundary, int bottomBoundary, int topBoundary, //
             int tiles_per_chunk);
};

// Collection of globally defined variables
struct global_config {
  std::string dumpDir;
  bool staging_buffer;
  std::vector<state_type> states;
  int number_of_states;
  int tiles_per_chunk;
  int test_problem;
  bool profiler_on;
  double end_time;
  int end_step;

  double dtinit;
  double dtmin;
  double dtmax;
  double dtrise;
  double dtu_safe;
  double dtv_safe;
  double dtc_safe;
  double dtdiv_safe;
  double dtc;
  double dtu;
  double dtv;
  double dtdiv;

  int visit_frequency;
  int summary_frequency;
  int number_of_chunks;

  grid_type grid;
};

struct global_variables {
  const global_config config;
  clover::context context;
  chunk_type chunk;

  int error_condition{};

  int step{};
  bool advect_x = true;
  double time{};

  double dt{};
  double dtold{};

  bool complete = false;
  bool report_test_fail = false;
  int jdt{}, kdt{};

  bool profiler_on = false; // Internal code profiler to make comparisons across systems easier
  profiler_type profiler{};

  global_variables(const global_config &config, clover::context queue, chunk_type chunk);
};
