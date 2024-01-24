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

#include "field_summary.h"
#include "context.h"
#include "ideal_gas.h"
#include "report.h"
#include "timer.h"

#include <iomanip>
#include <numeric>

//  @brief Fortran field summary kernel
//  @author Wayne Gaudin
//  @details The total mass, internal energy, kinetic energy and volume weighted
//  pressure for the chunk is calculated.
//  @brief Driver for the field summary kernels
//  @author Wayne Gaudin
//  @details The user specified field summary kernel is invoked here. A summation
//  across all mesh chunks is then performed and the information outputed.
//  If the run is a test problem, the final result is compared with the expected
//  result and the difference output.
//  Note the reference solution is the value returned from an Intel compiler with
//  ieee options set on a single core crun.

void field_summary(global_variables &globals, parallel_ &parallel) {

  clover_report_step_header(globals, parallel);

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    ideal_gas(globals, tile, false);
  }

  if (globals.profiler_on) {
    clover::checkError(cudaDeviceSynchronize());
    globals.profiler.ideal_gas += timer() - kernel_time;
    kernel_time = timer();
  }

  double vol = 0.0;
  double mass = 0.0;
  double ie = 0.0;
  double ke = 0.0;
  double press = 0.0;

  const int BLOCK = 256;
  clover::Buffer1D<double> vol_buffer(globals.context, BLOCK);
  clover::Buffer1D<double> mass_buffer(globals.context, BLOCK);
  clover::Buffer1D<double> ie_buffer(globals.context, BLOCK);
  clover::Buffer1D<double> ke_buffer(globals.context, BLOCK);
  clover::Buffer1D<double> press_buffer(globals.context, BLOCK);

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];

    int ymax = t.info.t_ymax;
    int ymin = t.info.t_ymin;
    int xmax = t.info.t_xmax;
    int xmin = t.info.t_xmin;
    field_type &field = t.field;

    int range = (ymax - ymin + 1) * (xmax - xmin + 1);
    clover::par_reduce<BLOCK, BLOCK>([=] __device__(int gid) {
      __shared__ double vol[BLOCK];
      __shared__ double mass[BLOCK];
      __shared__ double ie[BLOCK];
      __shared__ double ke[BLOCK];
      __shared__ double press[BLOCK];
      vol[threadIdx.x] = 0.0;
      mass[threadIdx.x] = 0.0;
      ie[threadIdx.x] = 0.0;
      ke[threadIdx.x] = 0.0;
      press[threadIdx.x] = 0.0;

      for (int v = gid; v < range; v += blockDim.x * gridDim.x) {
        const size_t j = xmin + 1 + v % (xmax - xmin + 1);
        const size_t k = ymin + 1 + v / (xmax - xmin + 1);
        double vsqrd = 0.0;
        for (size_t kv = k; kv <= k + 1; ++kv) {
          for (size_t jv = j; jv <= j + 1; ++jv) {
            vsqrd += 0.25 * (field.xvel0(jv, kv) * field.xvel0(jv, kv) + field.yvel0(jv, kv) * field.yvel0(jv, kv));
          }
        }
        double cell_vol = field.volume(j, k);
        double cell_mass = cell_vol * field.density0(j, k);

        vol[threadIdx.x] += cell_vol;
        mass[threadIdx.x] += cell_mass;
        ie[threadIdx.x] += cell_mass * field.energy0(j, k);
        ke[threadIdx.x] += cell_mass * 0.5 * vsqrd;
        press[threadIdx.x] += cell_vol * field.pressure(j, k);
      }

      clover::reduce<double, BLOCK / 2>::run(vol, vol_buffer.data, [](auto l, auto r) { return l + r; });
      clover::reduce<double, BLOCK / 2>::run(mass, mass_buffer.data, [](auto l, auto r) { return l + r; });
      clover::reduce<double, BLOCK / 2>::run(ie, ie_buffer.data, [](auto l, auto r) { return l + r; });
      clover::reduce<double, BLOCK / 2>::run(ke, ke_buffer.data, [](auto l, auto r) { return l + r; });
      clover::reduce<double, BLOCK / 2>::run(press, press_buffer.data, [](auto l, auto r) { return l + r; });
    });

    if (globals.profiler_on) {
      globals.profiler.summary += timer() - kernel_time;
      kernel_time = timer();
    }

    // JMK: Copies data back to host
    auto vol_host = vol_buffer.mirrored();
    auto mass_host = mass_buffer.mirrored();
    auto ie_host = ie_buffer.mirrored();
    auto ke_host = ke_buffer.mirrored();
    auto press_host = press_buffer.mirrored();

    if (globals.profiler_on) {
      globals.profiler.device_to_host += timer() - kernel_time;
      kernel_time = timer();
    }

    vol = std::reduce(vol_host.begin(), vol_host.end(), vol, std::plus<>());
    mass = std::reduce(mass_host.begin(), mass_host.end(), mass, std::plus<>());
    ie = std::reduce(ie_host.begin(), ie_host.end(), ie, std::plus<>());
    ke = std::reduce(ke_host.begin(), ke_host.end(), ke, std::plus<>());
    press = std::reduce(press_host.begin(), press_host.end(), press, std::plus<>());
  }
  vol_buffer.release();
  mass_buffer.release();
  ie_buffer.release();
  ke_buffer.release();
  press_buffer.release();

  //    summary s;
  //
  //    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
  //      tile_type &t = globals.chunk.tiles[tile];
  //
  //      int ymax = t.info.t_ymax;
  //      int ymin = t.info.t_ymin;
  //      int xmax = t.info.t_xmax;
  //      int xmin = t.info.t_xmin;
  //      field_type &field = t.field;
  //
  //      auto r = clover::range<size_t>(0, (ymax - ymin + 1) * (xmax - xmin + 1));
  //      s = std::transform_reduce(r.begin(), r.end(), s, std::plus<>(), [=](size_t idx) {
  //        const size_t j = xmin + 1 + idx % (xmax - xmin + 1);
  //        const size_t k = ymin + 1 + idx / (xmax - xmin + 1);
  //        double vsqrd = 0.0;
  //        for (size_t kv = k; kv <= k + 1; ++kv) {
  //          for (size_t jv = j; jv <= j + 1; ++jv) {
  //            vsqrd += 0.25 * (field.xvel0(jv, kv) * field.xvel0(jv, kv) + field.yvel0(jv, kv) * field.yvel0(jv, kv));
  //          }
  //        }
  //        double cell_vol = field.volume(j, k);
  //        double cell_mass = cell_vol * field.density0(j, k);
  //
  //        return summary{.vol = cell_vol,
  //                       .mass = cell_mass,
  //                       .ie = cell_mass * field.energy0(j, k),
  //                       .ke = cell_mass * 0.5 * vsqrd,
  //                       .press = cell_vol * field.pressure(j, k)};
  //      });
  //    }
  //
  //    auto [vol, mass, ie, ke, press] = s;

  clover_sum(vol);
  clover_sum(mass);
  clover_sum(ie);
  clover_sum(ke);
  clover_sum(press);

  if (globals.profiler_on) {
    clover::checkError(cudaDeviceSynchronize());
    globals.profiler.summary += timer() - kernel_time;
  }

  clover_report_step(globals, parallel, vol, mass, ie, ke, mass);
}
