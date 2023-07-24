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
#include "sycl_reduction.hpp"
#include "timer.h"
#include "report.h"

#include <iomanip>

// #define USE_SYCL2020_REDUCTION


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

struct summary {
  double vol = 0.0, mass = 0.0, ie = 0.0, ke = 0.0, press = 0.0;
  summary operator+(const summary &s) const {
    return {
        vol + s.vol, mass + s.mass, ie + s.ie, ke + s.ke, press + s.press,
    };
  }
};

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

  summary total{};
  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];
    int ymax = t.info.t_ymax;
    int ymin = t.info.t_ymin;
    int xmax = t.info.t_xmax;
    int xmin = t.info.t_xmin;
#ifdef USE_SYCL2020_REDUCTION
    clover::Buffer<summary, 1> summaryResults(1);
    globals.context
        .submit([&](sycl::handler &h) {
          auto xvel0_ = t.field.xvel0.access<R>(h);
          auto yvel0_ = t.field.yvel0.access<R>(h);
          auto volume_ = t.field.volume.access<R>(h);
          auto density0_ = t.field.density0.access<R>(h);
          auto energy0_ = t.field.energy0.access<R>(h);
          auto pressure_ = t.field.pressure.access<R>(h);
          auto mm = summaryResults.access<W>(h);
          h.parallel_for(                                            //
              sycl::range<1>((ymax - ymin + 1) * (xmax - xmin + 1)), //
              sycl::reduction(summaryResults.buffer, h, {}, sycl::plus<>(),
                              sycl::property::reduction::initialize_to_identity()), //
              [=](sycl::id<1> idx, auto &acc) {
                const size_t j = xmin + 1 + idx[0] % (xmax - xmin + 1);
                const size_t k = ymin + 1 + idx[0] / (xmax - xmin + 1);

                double vsqrd = 0.0;
                for (size_t kv = k; kv <= k + 1; ++kv) {
                  for (size_t jv = j; jv <= j + 1; ++jv) {
                    vsqrd += 0.25 * (xvel0_[jv][kv] * xvel0_[jv][kv] + yvel0_[jv][kv] * yvel0_[jv][kv]);
                  }
                }
                double cell_vol = volume_[j][k];
                double cell_mass = cell_vol * density0_[j][k];

                acc += summary{.vol = cell_vol,
                               .mass = cell_mass,
                               .ie = cell_mass * energy0_[j][k],
                               .ke = cell_mass * 0.5 * vsqrd,
                               .press = cell_vol * pressure_[j][k]};
              });
        })
        .wait_and_throw();
    total = total + summaryResults.access()[0];
#else
    clover::Range1d policy(0, (ymax - ymin + 1) * (xmax - xmin + 1));
    struct captures {
      clover::Accessor<double, 2, R>::Type volume;
      clover::Accessor<double, 2, R>::Type density0;
      clover::Accessor<double, 2, R>::Type energy0;
      clover::Accessor<double, 2, R>::Type pressure;
      clover::Accessor<double, 2, R>::Type xvel0;
      clover::Accessor<double, 2, R>::Type yvel0;
    };
    using Reducer = clover::local_reducer<summary, summary, captures>;
    clover::Buffer1D<summary> result(globals.context, policy.size);
    clover::par_reduce_1d<class field_summary, summary>(
        globals.context.queue, policy,
        [=](handler &h, size_t &size) mutable {
          return Reducer(h, size,
                         {t.field.volume.access<R>(h), t.field.density0.access<R>(h), t.field.energy0.access<R>(h),
                          t.field.pressure.access<R>(h), t.field.xvel0.access<R>(h), t.field.yvel0.access<R>(h)},
                         result.buffer);
        },
        [](const Reducer &ctx, id<1> lidx) { ctx.local[lidx] = {}; },
        [ymin, xmax, xmin](const Reducer &ctx, id<1> lidx, id<1> idx) {
          const size_t j = xmin + 1 + idx[0] % (xmax - xmin + 1);
          const size_t k = ymin + 1 + idx[0] / (xmax - xmin + 1);

          double vsqrd = 0.0;
          for (size_t kv = k; kv <= k + 1; ++kv) {
            for (size_t jv = j; jv <= j + 1; ++jv) {
              vsqrd += 0.25 * (ctx.actual.xvel0[jv][kv] * ctx.actual.xvel0[jv][kv] + ctx.actual.yvel0[jv][kv] * ctx.actual.yvel0[jv][kv]);
            }
          }
          double cell_vol = ctx.actual.volume[j][k];
          double cell_mass = cell_vol * ctx.actual.density0[j][k];

          ctx.local[lidx].vol += cell_vol;
          ctx.local[lidx].mass += cell_mass;
          ctx.local[lidx].ie += cell_mass * ctx.actual.energy0[j][k];
          ctx.local[lidx].ke += cell_mass * 0.5 * vsqrd;
          ctx.local[lidx].press += cell_vol * ctx.actual.pressure[j][k];
        },
        [](const Reducer &ctx, id<1> idx, id<1> idy) {
          ctx.local[idx].vol += ctx.local[idy].vol;
          ctx.local[idx].mass += ctx.local[idy].mass;
          ctx.local[idx].ie += ctx.local[idy].ie;
          ctx.local[idx].ke += ctx.local[idy].ke;
          ctx.local[idx].press += ctx.local[idy].press;
        },
        [](const Reducer &ctx, size_t group, id<1> idx) { ctx.result[group] = ctx.local[idx]; });
    total = total + result.access()[0];
#endif
  }
  globals.context.queue.wait_and_throw();
  auto [vol, mass, ie, ke, press] = total;

  clover_sum(vol);
  clover_sum(mass);
  clover_sum(ie);
  clover_sum(ke);
  clover_sum(press);
  if (globals.profiler_on) globals.profiler.summary += timer() - kernel_time;

  clover_report_step(globals, parallel, vol, mass, ie, ke, mass);
}
