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

//  @brief Fortran timestep kernel
//  @author Wayne Gaudin
//  @details Calculates the minimum timestep on the mesh chunk based on the CFL
//  condition, the velocity gradient and the velocity divergence. A safety
//  factor is used to ensure numerical stability.
void calc_dt_kernel(int x_min, int x_max, int y_min, int y_max, double dtmin, double dtc_safe, double dtu_safe, double dtv_safe,
                    double dtdiv_safe, Kokkos::View<double **> &xarea, Kokkos::View<double **> &yarea, Kokkos::View<double *> &cellx,
                    Kokkos::View<double *> &celly, Kokkos::View<double *> &celldx, Kokkos::View<double *> &celldy,
                    Kokkos::View<double **> &volume, Kokkos::View<double **> &density0, Kokkos::View<double **> &energy0,
                    Kokkos::View<double **> &pressure, Kokkos::View<double **> &viscosity_a, Kokkos::View<double **> &soundspeed,
                    Kokkos::View<double **> &xvel0, Kokkos::View<double **> &yvel0, Kokkos::View<double **> &dt_min, double &dt_min_val,
                    int &dtl_control, double &xl_pos, double &yl_pos, int &jldt, int &kldt, int &small) {

  small = 0;
  dt_min_val = g_big;
  double jk_control = 1.1;

  // DO k=y_min,y_max
  //   DO j=x_min,x_max
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});
  Kokkos::parallel_reduce(
      "calc_dt", policy,
      KOKKOS_LAMBDA(const int j, const int k, double &dt_min_val) {
        double dsx = celldx(j);
        double dsy = celldy(k);

        double cc = soundspeed(j, k) * soundspeed(j, k);
        cc = cc + 2.0 * viscosity_a(j, k) / density0(j, k);
        cc = Kokkos::max(sqrt(cc), g_small);

        double dtct = dtc_safe * Kokkos::min(dsx, dsy) / cc;

        double div = 0.0;

        double dv1 = (xvel0(j, k) + xvel0(j, k + 1)) * xarea(j, k);
        double dv2 = (xvel0(j + 1, k) + xvel0(j + 1, k + 1)) * xarea(j + 1, k);

        div = div + dv2 - dv1;

        double dtut = dtu_safe * 2.0 * volume(j, k) / Kokkos::max(Kokkos::max(fabs(dv1), fabs(dv2)), g_small * volume(j, k));

        dv1 = (yvel0(j, k) + yvel0(j + 1, k)) * yarea(j, k);
        dv2 = (yvel0(j, k + 1) + yvel0(j + 1, k + 1)) * yarea(j, k + 1);

        div = div + dv2 - dv1;

        double dtvt = dtv_safe * 2.0 * volume(j, k) / Kokkos::max(Kokkos::max(fabs(dv1), fabs(dv2)), g_small * volume(j, k));

        div = div / (2.0 * volume(j, k));

        double dtdivt;
        if (div < -g_small) {
          dtdivt = dtdiv_safe * (-1.0 / div);
        } else {
          dtdivt = g_big;
        }

        dt_min_val = Kokkos::min(dt_min_val, dtct);
        dt_min_val = Kokkos::min(dt_min_val, dtut);
        dt_min_val = Kokkos::min(dt_min_val, dtvt);
        dt_min_val = Kokkos::min(dt_min_val, dtdivt);
      },
      Kokkos::Min<double>(dt_min_val));

  //  Extract the mimimum timestep information
  dtl_control = 10.01 * (jk_control - (int)(jk_control));
  jk_control = jk_control - (jk_control - (int)(jk_control));
  jldt = ((int)jk_control) % x_max;
  kldt = 1 + (jk_control / x_max);
  // TODO: cannot do this with GPU memory directly
  // xl_pos = cellx(jldt+1); // Offset by 1 because of Fortran halos in original code
  // yl_pos = celly(kldt+1);

  if (dt_min_val < dtmin) small = 1;

  if (small != 0) {
    std::cout << "Timestep information:" << std::endl
              << "j, k                 : " << jldt << " " << kldt << std::endl
              << "x, y                 : " << cellx(jldt) << " " << celly(kldt) << std::endl
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

  calc_dt_kernel(globals.chunk.tiles[tile].info.t_xmin, globals.chunk.tiles[tile].info.t_xmax, globals.chunk.tiles[tile].info.t_ymin,
                 globals.chunk.tiles[tile].info.t_ymax, globals.config.dtmin, globals.config.dtc_safe, globals.config.dtu_safe,
                 globals.config.dtv_safe, globals.config.dtdiv_safe, globals.chunk.tiles[tile].field.xarea.view,
                 globals.chunk.tiles[tile].field.yarea.view, globals.chunk.tiles[tile].field.cellx.view,
                 globals.chunk.tiles[tile].field.celly.view, globals.chunk.tiles[tile].field.celldx.view,
                 globals.chunk.tiles[tile].field.celldy.view, globals.chunk.tiles[tile].field.volume.view,
                 globals.chunk.tiles[tile].field.density0.view, globals.chunk.tiles[tile].field.energy0.view,
                 globals.chunk.tiles[tile].field.pressure.view, globals.chunk.tiles[tile].field.viscosity.view,
                 globals.chunk.tiles[tile].field.soundspeed.view, globals.chunk.tiles[tile].field.xvel0.view,
                 globals.chunk.tiles[tile].field.yvel0.view, globals.chunk.tiles[tile].field.work_array1.view, local_dt, l_control, xl_pos,
                 yl_pos, jldt, kldt, small);

  if (l_control == 1) local_control = "sound";
  if (l_control == 2) local_control = "xvel";
  if (l_control == 3) local_control = "yvel";
  if (l_control == 4) local_control = "div";
}
