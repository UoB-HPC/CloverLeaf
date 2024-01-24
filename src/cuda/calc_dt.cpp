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

#include "calc_dt.h"
#include "context.h"
#include "../../driver/timer.h"
#include <cmath>
#include <numeric>
#include <string>

//  @brief Fortran timestep kernel
//  @author Wayne Gaudin
//  @details Calculates the minimum timestep on the mesh chunk based on the CFL
//  condition, the velocity gradient and the velocity divergence. A safety
//  factor is used to ensure numerical stability.

__device__ inline double SUM(double a, double b) { return a + b; }

void calc_dt_kernel(global_variables &globals, int x_min, int x_max, int y_min, int y_max, double dtmin, double dtc_safe, double dtu_safe,
                    double dtv_safe, double dtdiv_safe, clover::Buffer2D<double> &xarea, clover::Buffer2D<double> &yarea,
                    clover::Buffer1D<double> &cellx, clover::Buffer1D<double> &celly, clover::Buffer1D<double> &celldx,
                    clover::Buffer1D<double> &celldy, clover::Buffer2D<double> &volume, clover::Buffer2D<double> &density0,
                    clover::Buffer2D<double> &energy0, clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &viscosity_a,
                    clover::Buffer2D<double> &soundspeed, clover::Buffer2D<double> &xvel0, clover::Buffer2D<double> &yvel0,
                    double &dt_min_val, int &dtl_control, double &xl_pos, double &yl_pos, int &jldt, int &kldt, int &small) {

  small = 0;
  dt_min_val = g_big;
  double jk_control = 1.1;

  // DO k=y_min,y_max
  //   DO j=x_min,x_max
  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});

  auto policy = clover::Range2d(x_min + 1, y_min + 1, x_max + 2, y_max + 2);

  int xStart = x_min + 1, xEnd = x_max + 2;
  int yStart = y_min + 1, yEnd = y_max + 2;
  int sizeX = xEnd - xStart;
  // y = y_min + 1  |  y_max + 2

  const int BLOCK = 256;
  clover::Buffer1D<double> dt_min_val_buffer(globals.context, BLOCK);
  int range = (xEnd - xStart) * (yEnd - yStart);
  clover::par_reduce<BLOCK, BLOCK>([=] __device__(int gid) {
    __shared__ double mins[BLOCK];
    mins[threadIdx.x] = g_big;
    for (int v = gid; v < range; v += blockDim.x * gridDim.x) {
      const auto i = xStart + (v % sizeX);
      const auto j = yStart + (v / sizeX);

      double dsx = celldx[i];
      double dsy = celldy[j];

      double cc = soundspeed(i, j) * soundspeed(i, j);
      cc = cc + 2.0 * viscosity_a(i, j) / density0(i, j);
      cc = std::fmax(std::sqrt(cc), g_small);
      double dtct = dtc_safe * std::fmin(dsx, dsy) / cc;
      double div = 0.0;
      double dv1 = (xvel0(i, j) + xvel0(i + 0, j + 1)) * xarea(i, j);
      double dv2 = (xvel0(i + 1, j + 0) + xvel0(i + 1, j + 1)) * xarea(i + 1, j + 0);
      div = div + dv2 - dv1;
      double dtut = dtu_safe * 2.0 * volume(i, j) / std::fmax(std::fmax(std::fabs(dv1), std::fabs(dv2)), g_small * volume(i, j));
      dv1 = (yvel0(i, j) + yvel0(i + 1, j + 0)) * yarea(i, j);
      dv2 = (yvel0(i + 0, j + 1) + yvel0(i + 1, j + 1)) * yarea(i + 0, j + 1);
      div = div + dv2 - dv1;
      double dtvt = dtv_safe * 2.0 * volume(i, j) / std::fmax(std::fmax(std::fabs(dv1), std::fabs(dv2)), g_small * volume(i, j));
      div = div / (2.0 * volume(i, j));
      double dtdivt;
      if (div < -g_small) {
        dtdivt = dtdiv_safe * (-1.0 / div);
      } else {
        dtdivt = g_big;
      }
      mins[threadIdx.x] = std::fmin(dtct, std::fmin(dtut, std::fmin(dtvt, std::fmin(dtdivt, mins[threadIdx.x]))));
    }
    clover::reduce<double, BLOCK / 2>::run(mins, dt_min_val_buffer.data, [](auto l, auto r) { return std::fmin(l, r); });
  });

  // JMK: Copies data back from device to host
  if (globals.profiler_on) {
    globals.profiler.timestep += timer() - globals.profiler.kernel_time;
    globals.profiler.kernel_time = timer();
  }

  auto dt_min_val_host = dt_min_val_buffer.mirrored();

  if (globals.profiler_on) {
    globals.profiler.device_to_host += timer() - globals.profiler.kernel_time;
    globals.profiler.kernel_time = timer();
  }

  dt_min_val_buffer.release();
  dt_min_val = std::reduce(dt_min_val_host.begin(), dt_min_val_host.end(), g_big, [](auto l, auto r) { return std::fmin(l, r); });

  //  auto r = clover::range<int>(0, (xEnd - xStart) * (yEnd - yStart));
  //  dt_min_val = std::transform_reduce(  r.begin(), r.end(), g_big, [](auto l, auto r) { return std::fmin(l, r); },
  //      [=](int v) {
  //        const auto i = xStart + (v % sizeX);
  //        const auto j = yStart + (v / sizeX);
  //
  //        double dsx = celldx[i];
  //        double dsy = celldy[j];
  //
  //        double cc = soundspeed(i, j) * soundspeed(i, j);
  //        cc = cc + 2.0 * viscosity_a(i, j) / density0(i, j);
  //        cc = std::fmax(std::sqrt(cc), g_small);
  //        double dtct = dtc_safe * std::fmin(dsx, dsy) / cc;
  //        double div = 0.0;
  //        double dv1 = (xvel0(i, j) + xvel0(i + 0, j + 1)) * xarea(i, j);
  //        double dv2 = (xvel0(i + 1, j + 0) + xvel0(i + 1, j + 1)) * xarea(i + 1, j + 0);
  //        div = div + dv2 - dv1;
  //        double dtut = dtu_safe * 2.0 * volume(i, j) / std::fmax(std::fmax(std::fabs(dv1), std::fabs(dv2)), g_small * volume(i, j));
  //        dv1 = (yvel0(i, j) + yvel0(i + 1, j + 0)) * yarea(i, j);
  //        dv2 = (yvel0(i + 0, j + 1) + yvel0(i + 1, j + 1)) * yarea(i + 0, j + 1);
  //        div = div + dv2 - dv1;
  //        double dtvt = dtv_safe * 2.0 * volume(i, j) / std::fmax(std::fmax(std::fabs(dv1), std::fabs(dv2)), g_small * volume(i, j));
  //        div = div / (2.0 * volume(i, j));
  //        double dtdivt;
  //        if (div < -g_small) {
  //          dtdivt = dtdiv_safe * (-1.0 / div);
  //        } else {
  //          dtdivt = g_big;
  //        }
  //        double mins = std::fmin(dtct, std::fmin(dtut, std::fmin(dtvt, std::fmin(dtdivt, g_big))));
  //        return mins;
  //        //		dt_min_val = std::fmin(mins, dt_min_val);
  //      });

  dtl_control = static_cast<int>(10.01 * (jk_control - static_cast<int>(jk_control)));
  jk_control = jk_control - (jk_control - (int)(jk_control));
  jldt = ((int)jk_control) % x_max;
  kldt = static_cast<int>(1.f + (jk_control / x_max));

  if (dt_min_val < dtmin) small = 1;

  if (small != 0) {

    std::cout << "Timestep information:" << std::endl
              << "j, k                 : " << jldt << " " << kldt << std::endl
              << "x, y                 : " << cellx[jldt] << " " << celly[kldt] << std::endl
              << "timestep : " << dt_min_val << std::endl
              << "Cell velocities;" << std::endl
              << xvel0(jldt, kldt) << " " << yvel0(jldt, kldt) << std::endl
              << xvel0(jldt + 1, kldt) << " " << yvel0(jldt + 1, kldt) << std::endl
              << xvel0(jldt + 1, kldt + 1) << " " << yvel0(jldt + 1, kldt + 1) << std::endl
              << xvel0(jldt, kldt + 1) << " " << yvel0(jldt, kldt + 1) << std::endl
              << "density, energy, pressure, soundspeed " << std::endl
              << density0(jldt, kldt) << " " << energy0(jldt, kldt) << " " << pressure(jldt, kldt) << " " << soundspeed(jldt, kldt)
              << std::endl;
  }
}

//  @brief Driver for the timestep kernels
//  @author Wayne Gaudin
//  @details Invokes the user specified timestep kernel.
void calc_dt(global_variables &globals, int tile, double &local_dt, std::string &local_control, double &xl_pos, double &yl_pos, int &jldt,
             int &kldt) {

  local_dt = g_big;

  int l_control;
  int small = 0;

  tile_type &t = globals.chunk.tiles[tile];
  calc_dt_kernel(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, globals.config.dtmin, globals.config.dtc_safe,
                 globals.config.dtu_safe, globals.config.dtv_safe, globals.config.dtdiv_safe, t.field.xarea, t.field.yarea, t.field.cellx,
                 t.field.celly, t.field.celldx, t.field.celldy, t.field.volume, t.field.density0, t.field.energy0, t.field.pressure,
                 t.field.viscosity, t.field.soundspeed, t.field.xvel0, t.field.yvel0, local_dt, l_control, xl_pos, yl_pos, jldt, kldt,
                 small);

  if (l_control == 1) local_control = "sound";
  if (l_control == 2) local_control = "xvel";
  if (l_control == 3) local_control = "yvel";
  if (l_control == 4) local_control = "div";
}
