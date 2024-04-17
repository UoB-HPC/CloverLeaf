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

//  @brief Fortran timestep kernel
//  @author Wayne Gaudin
//  @details Calculates the minimum timestep on the mesh chunk based on the CFL
//  condition, the velocity gradient and the velocity divergence. A safety
//  factor is used to ensure numerical stability.

#define SPLIT

void calc_dt_kernel(clover::context &ctx, int x_min, int x_max, int y_min, int y_max, double dtmin, double dtc_safe, double dtu_safe,
                    double dtv_safe, double dtdiv_safe, clover::Buffer2D<double> xarea, clover::Buffer2D<double> yarea,
                    clover::Buffer1D<double> cellx, clover::Buffer1D<double> celly, clover::Buffer1D<double> celldx,
                    clover::Buffer1D<double> celldy, clover::Buffer2D<double> volume, clover::Buffer2D<double> density0,
                    clover::Buffer2D<double> energy0, clover::Buffer2D<double> pressure, clover::Buffer2D<double> viscosity_a,
                    clover::Buffer2D<double> soundspeed, clover::Buffer2D<double> xvel0, clover::Buffer2D<double> yvel0, double &dt_min_val,
                    int &dtl_control, double &xl_pos, double &yl_pos, int &jldt, int &kldt, int &small) {

  small = 0;
  dt_min_val = g_big;
  double jk_control = 1.1;

  // DO k=y_min,y_max
  //   DO j=x_min,x_max
  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});

  int xStart = x_min + 1, xEnd = x_max + 2;
  int yStart = y_min + 1, yEnd = y_max + 2;
  int sizeX = xEnd - xStart;
  int sizeY = yEnd - yStart;
  // y = y_min + 1  |  y_max + 2

  clover::Buffer1D<double> minResults(ctx, 1);

  ctx.queue
      .submit([&](sycl::handler &cgh) {
#if defined(__HIPSYCL__) || defined(__OPENSYCL__)
        cgh.parallel_for(sycl::range<1>((xEnd - xStart) * (yEnd - yStart)),
                         sycl::reduction(minResults.data, dt_min_val, sycl::minimum<double>()), [=](sycl::id<1> idx, auto &acc) {
                           const auto i = xStart + (idx[0] % sizeX);
                           const auto j = yStart + (idx[0] / sizeX);
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
            sycl::reduction(minResults.data, dt_min_val, sycl::minimum<double>(), sycl::property::reduction::initialize_to_identity()),
            [=](sycl::nd_item<2> idx, auto &acc) {
              const auto global_idx = idx.get_global_id();
              const auto i = xStart + global_idx[0];
              const auto j = yStart + global_idx[1];
#endif
              double dsx = celldx[i];
              double dsy = celldy[j];
              double cc = soundspeed(i, j) * soundspeed(i, j);
              cc = cc + 2.0 * viscosity_a(i, j) / density0(i, j);
              cc = sycl::fmax(sycl::sqrt(cc), g_small);
              double dtct = dtc_safe * sycl::fmin(dsx, dsy) / cc;
              double div = 0.0;
              double dv1 = (xvel0(i, j) + xvel0(i + 0, j + 1)) * xarea(i, j);
              double dv2 = (xvel0(i + 1, j + 0) + xvel0(i + 1, j + 1)) * xarea(i + 1, j + 0);
              div = div + dv2 - dv1;
              double dtut = dtu_safe * 2.0 * volume(i, j) /
                            sycl::fmax(sycl::fmax(sycl::fabs(dv1), sycl::fabs(dv2)), g_small * volume(i, j));

              dv1 = (yvel0(i, j) + yvel0(i + 1, j + 0)) * yarea(i, j);
              dv2 = (yvel0(i + 0, j + 1) + yvel0(i + 1, j + 1)) * yarea(i + 0, j + 1);
              div = div + dv2 - dv1;
              double dtvt = dtv_safe * 2.0 * volume(i, j) /
                            sycl::fmax(sycl::fmax(sycl::fabs(dv1), sycl::fabs(dv2)), g_small * volume(i, j));
              div = div / (2.0 * volume(i, j));
              double dtdivt;
              if (div < -g_small) {
                dtdivt = dtdiv_safe * (-1.0 / div);
              } else {
                dtdivt = g_big;
              }
              acc.combine(sycl::fmin(dtct, sycl::fmin(dtut, sycl::fmin(dtvt, sycl::fmin(dtdivt, g_big)))));
            });
      })
      .wait_and_throw();
  ctx.queue.wait_and_throw();

  double *h_minRes = (double *)malloc(sizeof(double));
  ctx.queue.memcpy(h_minRes, minResults.data, sizeof(double)).wait_and_throw();
  dt_min_val = h_minRes[0];
  std::free(h_minRes);
  clover::free(ctx.queue, minResults);

  dtl_control = static_cast<int>(10.01 * (jk_control - static_cast<int>(jk_control)));
  jk_control = jk_control - (jk_control - (int)(jk_control));
  jldt = ((int)jk_control) % x_max;
  kldt = static_cast<int>(1.f + (jk_control / x_max));

  if (dt_min_val < dtmin) small = 1;

  if (small != 0) {
    double *h_cellx = (double *)malloc(sizeof(double));
    double *h_celly = (double *)malloc(sizeof(double));
    double *h_xvel0 = (double *)malloc(4 * sizeof(double));
    double *h_yvel0 = (double *)malloc(4 * sizeof(double));
    double *h_density0 = (double *)malloc(sizeof(double));
    double *h_energy0 = (double *)malloc(sizeof(double));
    double *h_pressure = (double *)malloc(sizeof(double));
    double *h_soundspeed = (double *)malloc(sizeof(double));
    ctx.queue
        .submit([&](sycl::handler &cgh) {
          cgh.single_task([=]() {
            h_cellx[0] = cellx[jldt];
            h_celly[0] = celly[kldt];
            int s = 0;
            for (int i = 0; i < 2; i++) {
              for (int j = 0; j < 2; j++) {
                h_xvel0[s] = xvel0(jldt + i, kldt + j);
                h_yvel0[s] = yvel0(jldt + i, kldt + j);
                ++s;
              }
            }
            h_density0[0] = density0(jldt, kldt);
            h_energy0[0] = energy0(jldt, kldt);
            h_pressure[0] = pressure(jldt, kldt);
            h_soundspeed[0] = soundspeed(jldt, kldt);
          });
        })
        .wait_and_throw();

    std::cout << "Timestep information:" << std::endl
              << "j, k                 : " << jldt << " " << kldt << std::endl
              << "x, y                 : " << h_cellx[0] << " " << h_celly[0] << std::endl
              << "timestep : " << dt_min_val << std::endl
              << "Cell velocities;" << std::endl
              << h_xvel0[0] << " " << h_yvel0[0] << std::endl
              << h_xvel0[2] << " " << h_yvel0[2] << std::endl
              << h_xvel0[3] << " " << h_yvel0[3] << std::endl
              << h_xvel0[1] << " " << h_yvel0[1] << std::endl
              << "density, energy, pressure, soundspeed " << std::endl
              << h_density0[0] << " " << h_energy0[0] << " " << h_pressure[0] << " " << h_soundspeed[0] << std::endl;

    std::free(h_cellx);
    std::free(h_celly);
    std::free(h_xvel0);
    std::free(h_yvel0);
    std::free(h_density0);
    std::free(h_energy0);
    std::free(h_pressure);
    std::free(h_soundspeed);
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
  calc_dt_kernel(globals.context, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, globals.config.dtmin, globals.config.dtc_safe,
                 globals.config.dtu_safe, globals.config.dtv_safe, globals.config.dtdiv_safe, t.field.xarea, t.field.yarea, t.field.cellx,
                 t.field.celly, t.field.celldx, t.field.celldy, t.field.volume, t.field.density0, t.field.energy0, t.field.pressure,
                 t.field.viscosity, t.field.soundspeed, t.field.xvel0, t.field.yvel0, local_dt, l_control, xl_pos, yl_pos, jldt, kldt,
                 small);

  if (l_control == 1) local_control = "sound";
  if (l_control == 2) local_control = "xvel";
  if (l_control == 3) local_control = "yvel";
  if (l_control == 4) local_control = "div";
}
