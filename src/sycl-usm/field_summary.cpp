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
    globals.profiler.ideal_gas += timer() - kernel_time;
    kernel_time = timer();
  }

  struct summary {
    double vol = 0.0;
    double mass = 0.0;
    double ie = 0.0;
    double ke = 0.0;
    double press = 0.0;
    summary operator+(const summary &s) const {
      return {
          vol + s.vol, mass + s.mass, ie + s.ie, ke + s.ke, press + s.press,
      };
    }
  };

  clover::Buffer1D<summary> summaryResults(globals.context, 1);
  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];

    int ymax = t.info.t_ymax;
    int ymin = t.info.t_ymin;
    int xmax = t.info.t_xmax;
    int xmin = t.info.t_xmin;
    auto &field = t.field;

    size_t sizeX = xmax - xmin + 1;
    size_t sizeY = ymax - ymin + 1;

    globals.context.queue
        .submit([&](sycl::handler &cgh) {
#if defined(__HIPSYCL__) || defined(__OPENSYCL__)
          cgh.parallel_for(                                          //
              sycl::range<1>((ymax - ymin + 1) * (xmax - xmin + 1)), //
              sycl::reduction(summaryResults.data, {}, sycl::plus<summary>()),
              [=](sycl::id<1> idx, auto &acc) {
                const size_t j = xmin + 1 + idx[0] % (xmax - xmin + 1);
                const size_t k = ymin + 1 + idx[0] / (xmax - xmin + 1);
#else
          size_t maxThreadPerBlock = 256;
          size_t localX = std::ceil(double(sizeX) / double(maxThreadPerBlock));
          size_t localY = std::ceil(double(sizeY) / double(maxThreadPerBlock));

          auto uniformLocalX = sizeX % localX == 0 ? localX : sizeX + (localX - sizeX % localX);
          auto uniformLocalY = sizeY % localY == 0 ? localY : sizeY + (localY - sizeY % localY);
          uniformLocalX = uniformLocalX >= sizeX ? 1 : uniformLocalX;
          uniformLocalY = uniformLocalY >= sizeY ? 1 : uniformLocalY;

          cgh.parallel_for( 
              sycl::nd_range<2>(sycl::range<2>(sizeX, sizeY), sycl::range<2>(uniformLocalX, uniformLocalY)),
              sycl::reduction(summaryResults.data, {}, sycl::plus<>(),
                sycl::property::reduction::initialize_to_identity()),
              [=](sycl::nd_item<2> idx, auto &acc) {
                const auto global_idx = idx.get_global_id();
                const size_t j = xmin + 1 + global_idx[0];
                const size_t k = ymin + 1 + global_idx[1];
#endif
                double vsqrd = 0.0;
                for (size_t kv = k; kv <= k + 1; ++kv) {
                  for (size_t jv = j; jv <= j + 1; ++jv) {
                    vsqrd += 0.25 * (field.xvel0(jv, kv) * field.xvel0(jv, kv) + field.yvel0(jv, kv) * field.yvel0(jv, kv));
                  }
                }
                double cell_vol = field.volume(j, k);
                double cell_mass = cell_vol * field.density0(j, k);

                acc += summary{.vol = cell_vol,
                               .mass = cell_mass,
                               .ie = cell_mass * field.energy0(j, k),
                               .ke = cell_mass * 0.5 * vsqrd,
                               .press = cell_vol * field.pressure(j, k)};
              });
        })
        .wait_and_throw();
  }
  globals.context.queue.wait_and_throw();

  summary* h_summary = (summary*) malloc(sizeof(summary));
  globals.context.queue.memcpy(h_summary, summaryResults.data, sizeof(summary)).wait_and_throw();
  auto [vol, mass, ie, ke, press] = h_summary[0];

  std::free(h_summary);
  clover::free(globals.context.queue, summaryResults);

  clover_sum(vol);
  clover_sum(mass);
  clover_sum(ie);
  clover_sum(ke);
  clover_sum(press);

  if (globals.profiler_on) globals.profiler.summary += timer() - kernel_time;

  clover_report_step(globals, parallel, vol, mass, ie, ke, mass);
}
