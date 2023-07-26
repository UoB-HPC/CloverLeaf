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
#include <string>

#include <cmath>

//  @brief Fortran timestep kernel
//  @author Wayne Gaudin
//  @details Calculates the minimum timestep on the mesh chunk based on the CFL
//  condition, the velocity gradient and the velocity divergence. A safety
//  factor is used to ensure numerical stability.

void calc_dt_kernel(bool use_target, int x_min, int x_max, int y_min, int y_max, double dtmin, double dtc_safe, double dtu_safe,
                    double dtv_safe, double dtdiv_safe, field_type &field, double &dt_min_val, int &dtl_control, double &xl_pos,
                    double &yl_pos, int &jldt, int &kldt, int &small) {

  small = 0;
  dt_min_val = g_big;
  double jk_control = 1.1;

  // DO k=y_min,y_max
  //   DO j=x_min,x_max
  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});

  const int flux_x_stride = field.flux_x_stride;
  const int flux_y_stride = field.flux_y_stride;

  const int base_stride = field.base_stride;
  const int vels_wk_stride = field.vels_wk_stride;

  double *xarea = field.xarea.data;
  double *yarea = field.yarea.data;
  double *celldx = field.celldx.data;
  double *celldy = field.celldy.data;
  double *volume = field.volume.data;
  double *density0 = field.density0.data;
  double *viscosity = field.viscosity.data;
  double *soundspeed = field.soundspeed.data;
  double *xvel0 = field.xvel0.data;
  double *yvel0 = field.yvel0.data;

  // XXX See https://forums.developer.nvidia.com/t/nvc-f-0000-internal-compiler-error-unhandled-size-for-preparing-max-constant/221740
  double dt_min_val0 = dt_min_val;
#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target) map(tofrom : dt_min_val)                   \
    reduction(min : dt_min_val0)
  for (int j = (y_min + 1); j < (y_max + 2); j++) {
    for (int i = (x_min + 1); i < (x_max + 2); i++) {
      double dsx = celldx[i];
      double dsy = celldy[j];
      double cc = soundspeed[i + j * base_stride] * soundspeed[i + j * base_stride];
      cc = cc + 2.0 * viscosity[i + j * base_stride] / density0[i + j * base_stride];
      cc = fmax(sqrt(cc), g_small);
      double dtct = dtc_safe * fmin(dsx, dsy) / cc;
      double div = 0.0;
      double dv1 = (xvel0[i + j * vels_wk_stride] + xvel0[(i + 0) + (j + 1) * vels_wk_stride]) * xarea[i + j * flux_x_stride];
      double dv2 = (xvel0[(i + 1) + (j + 0) * vels_wk_stride] + xvel0[(i + 1) + (j + 1) * vels_wk_stride]) *
                   xarea[(i + 1) + (j + 0) * flux_x_stride];
      div = div + dv2 - dv1;
      double dtut = dtu_safe * 2.0 * volume[i + j * base_stride] / fmax(fmax(fabs(dv1), fabs(dv2)), g_small * volume[i + j * base_stride]);
      dv1 = (yvel0[i + j * vels_wk_stride] + yvel0[(i + 1) + (j + 0) * vels_wk_stride]) * yarea[i + j * flux_y_stride];
      dv2 = (yvel0[(i + 0) + (j + 1) * vels_wk_stride] + yvel0[(i + 1) + (j + 1) * vels_wk_stride]) *
            yarea[(i + 0) + (j + 1) * flux_y_stride];
      div = div + dv2 - dv1;
      double dtvt = dtv_safe * 2.0 * volume[i + j * base_stride] / fmax(fmax(fabs(dv1), fabs(dv2)), g_small * volume[i + j * base_stride]);
      div = div / (2.0 * volume[i + j * base_stride]);
      double dtdivt;
      if (div < -g_small) {
        dtdivt = dtdiv_safe * (-1.0 / div);
      } else {
        dtdivt = g_big;
      }
      double mins = fmin(dtct, fmin(dtut, fmin(dtvt, fmin(dtdivt, g_big))));
      dt_min_val0 = fmin(mins, dt_min_val0);
    }
  }
  dt_min_val = dt_min_val0;

  dtl_control = static_cast<int>(10.01 * (jk_control - static_cast<int>(jk_control)));
  jk_control = jk_control - (jk_control - (int)(jk_control));
  jldt = ((int)jk_control) % x_max;
  kldt = static_cast<int>(1.f + (jk_control / x_max));

  if (dt_min_val < dtmin) small = 1;

  if (small != 0) {

    auto &cellx_acc = field.cellx;
    auto &celly_acc = field.celly;
    auto &density0_acc = field.density0;
    auto &energy0_acc = field.energy0;
    auto &pressure_acc = field.pressure;
    auto &soundspeed_acc = field.soundspeed;
    auto &xvel0_acc = field.xvel0;
    auto &yvel0_acc = field.yvel0;

    std::cout << "Timestep information:" << std::endl
              << "j, k                 : " << jldt << " " << kldt << std::endl
              << "x, y                 : " << cellx_acc[jldt] << " " << celly_acc[kldt] << std::endl
              << "timestep : " << dt_min_val << std::endl
              << "Cell velocities;" << std::endl
              << xvel0_acc(jldt, kldt) << " " << yvel0_acc(jldt, kldt) << std::endl
              << xvel0_acc(jldt + 1, kldt) << " " << yvel0_acc(jldt + 1, kldt) << std::endl
              << xvel0_acc(jldt + 1, kldt + 1) << " " << yvel0_acc(jldt + 1, kldt + 1) << std::endl
              << xvel0_acc(jldt, kldt + 1) << " " << yvel0_acc(jldt, kldt + 1) << std::endl
              << "density, energy, pressure, soundspeed " << std::endl
              << density0_acc(jldt, kldt) << " " << energy0_acc(jldt, kldt) << " " << pressure_acc(jldt, kldt) << " "
              << soundspeed_acc(jldt, kldt) << std::endl;
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

#if SYNC_BUFFERS
  globals.hostToDevice();
#endif

  tile_type &t = globals.chunk.tiles[tile];
  calc_dt_kernel(globals.context.use_target, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, globals.config.dtmin,
                 globals.config.dtc_safe, globals.config.dtu_safe, globals.config.dtv_safe, globals.config.dtdiv_safe, t.field, local_dt,
                 l_control, xl_pos, yl_pos, jldt, kldt, small);

#if SYNC_BUFFERS
  globals.deviceToHost();
#endif

  if (l_control == 1) local_control = "sound";
  if (l_control == 2) local_control = "xvel";
  if (l_control == 3) local_control = "yvel";
  if (l_control == 4) local_control = "div";
}
