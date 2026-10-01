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

#include "advec_cell.h"
#include "context.h"
#include <cmath>

#include <RAJA/RAJA.hpp>

//  @brief Fortran cell advection kernel.
//  @author Wayne Gaudin
//  @details Performs a second order advective remap using van-Leer limiting
//  with directional splitting.
void advec_cell_kernel(int x_min, int x_max, int y_min, int y_max, int dir, int sweep_number, clover::Buffer1D<double> &vertexdx,
                       clover::Buffer1D<double> &vertexdy, clover::Buffer2D<double> &volume, clover::Buffer2D<double> &density1,
                       clover::Buffer2D<double> &energy1, clover::Buffer2D<double> &mass_flux_x, clover::Buffer2D<double> &vol_flux_x,
                       clover::Buffer2D<double> &mass_flux_y, clover::Buffer2D<double> &vol_flux_y, clover::Buffer2D<double> &pre_vol,
                       clover::Buffer2D<double> &post_vol, clover::Buffer2D<double> &pre_mass, clover::Buffer2D<double> &post_mass,
                       clover::Buffer2D<double> &advec_vol, clover::Buffer2D<double> &post_ener, clover::Buffer2D<double> &ener_flux) {

  const double one_by_six = 1.0 / 6.0;

  if (dir == g_xdir) {

    // DO k=y_min-2,y_max+2
    //   DO j=x_min-2,x_max+2

//    const clover::Range2d policy(x_min - 2 + 1, y_min - 2 + 1, x_max + 2 + 2, y_max + 2 + 2);
    const RAJA::TypedRangeSegment<int> row_Range(y_min - 2 + 1,  y_max + 2 + 2);
    const RAJA::TypedRangeSegment<int> col_Range(x_min - 2 + 1,  x_max + 2 + 2);

    if (sweep_number == 1) {

      RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range, row_Range),
        [=] RAJA_HOST_DEVICE (const int i, const int j) {
        pre_vol(i, j) = volume(i, j) + (vol_flux_x(i + 1, j + 0) - vol_flux_x(i, j) + vol_flux_y(i + 0, j + 1) - vol_flux_y(i, j));
        post_vol(i, j) = pre_vol(i, j) - (vol_flux_x(i + 1, j + 0) - vol_flux_x(i, j));
      });

    } else {
      RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range, row_Range),
        [=] RAJA_HOST_DEVICE (const int i, const int j) {
        pre_vol(i, j) = volume(i, j) + vol_flux_x(i + 1, j + 0) - vol_flux_x(i, j);
        post_vol(i, j) = volume(i, j);
      });
    }

    // DO k=y_min,y_max
    //   DO j=x_min,x_max+2
    const RAJA::TypedRangeSegment<int> row_Range1(y_min + 1,  y_max + 2);
    const RAJA::TypedRangeSegment<int> col_Range1(x_min + 1,  x_max + 2 + 2);

    RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range1, row_Range1),
        [=] RAJA_HOST_DEVICE (const int x, const int y) {
      int upwind, donor, downwind, dif;
      double sigmat, sigma3, sigma4, sigmav,  sigmam, diffuw, diffdw, limiter, wind;

      const int j = x;
      const int k = y;

      if (vol_flux_x(x, y) > 0.0) {
        upwind = j - 2;
        donor = j - 1;
        downwind = j;
        dif = donor;
      } else {
      #if defined(RAJA_TARGET_CPU)
        upwind = std::min(j + 1, x_max + 2);
      #else
        upwind = min(j + 1, x_max + 2); // XXX can't do std::min because CUDA
      #endif
        donor = j;
        downwind = j - 1;
        dif = upwind;
      }

      sigmat = std::fabs(vol_flux_x(x, y)) / pre_vol(donor, k);
      sigma3 = (1.0 + sigmat) * (vertexdx[j] / vertexdx[dif]);
      sigma4 = 2.0 - sigmat;

      sigmav = sigmat;

      diffuw = density1(donor, k) - density1(upwind, k);
      diffdw = density1(downwind, k) - density1(donor, k);
      wind = 1.0;
      if (diffdw <= 0.0) wind = -1.0;
      if (diffuw * diffdw > 0.0) {
        limiter = (1.0 - sigmav) * wind *
                  std::fmin(std::fmin(std::fabs(diffuw), std::fabs(diffdw)),
                            one_by_six * (sigma3 * std::fabs(diffuw) + sigma4 * std::fabs(diffdw)));
      } else {
        limiter = 0.0;
      }
      mass_flux_x(x, y) = vol_flux_x(x, y) * (density1(donor, k) + limiter);

      sigmam = std::fabs(mass_flux_x(x, y)) / (density1(donor, k) * pre_vol(donor, k));
      diffuw = energy1(donor, k) - energy1(upwind, k);
      diffdw = energy1(downwind, k) - energy1(donor, k);
      wind = 1.0;
      if (diffdw <= 0.0) wind = -1.0;
      if (diffuw * diffdw > 0.0) {
        limiter = (1.0 - sigmam) * wind *
                  std::fmin(std::fmin(std::fabs(diffuw), std::fabs(diffdw)),
                            one_by_six * (sigma3 * std::fabs(diffuw) + sigma4 * std::fabs(diffdw)));
      } else {
        limiter = 0.0;
      }

      ener_flux(x, y) = mass_flux_x(x, y) * (energy1(donor, k) + limiter);
    });

    // DO k=y_min,y_max
    //   DO j=x_min,x_max
    const RAJA::TypedRangeSegment<int> row_Range123(y_min + 1,  y_max + 2);
    const RAJA::TypedRangeSegment<int> col_Range123(x_min + 1,  x_max + 2);

    RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range123, row_Range123),
        [=] RAJA_HOST_DEVICE (const int i, const int j) {
      double pre_mass_s = density1(i, j) * pre_vol(i, j);
      double post_mass_s = pre_mass_s + mass_flux_x(i, j) - mass_flux_x(i + 1, j + 0);
      double post_ener_s = (energy1(i, j) * pre_mass_s + ener_flux(i, j) - ener_flux(i + 1, j + 0)) / post_mass_s;
      double advec_vol_s = pre_vol(i, j) + vol_flux_x(i, j) - vol_flux_x(i + 1, j + 0);
      density1(i, j) = post_mass_s / advec_vol_s;
      energy1(i, j) = post_ener_s;
    });

  } else if (dir == g_ydir) {

    // DO k=y_min-2,y_max+2
    //   DO j=x_min-2,x_max+2
//    clover::Range2d policy(x_min - 2 + 1, y_min - 2 + 1, x_max + 2 + 2, y_max + 2 + 2);
    const RAJA::TypedRangeSegment<int> row_Range1(y_min - 2 + 1,  y_max + 2 + 2);
    const RAJA::TypedRangeSegment<int> col_Range1(x_min - 2 + 1,  x_max + 2 + 2);
 
    if (sweep_number == 1) {

//      clover::par_ranged2(policy, [=] DEVICE_KERNEL(const int i, const int j) {
      RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range1, row_Range1),
        [=] RAJA_HOST_DEVICE (const int i, const int j) {
        pre_vol(i, j) = volume(i, j) + (vol_flux_y(i + 0, j + 1) - vol_flux_y(i, j) + vol_flux_x(i + 1, j + 0) - vol_flux_x(i, j));
        post_vol(i, j) = pre_vol(i, j) - (vol_flux_y(i + 0, j + 1) - vol_flux_y(i, j));
      });

    } else {
//      clover::par_ranged2(policy, [=] DEVICE_KERNEL(const int i, const int j) {
      RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range1, row_Range1),
        [=] RAJA_HOST_DEVICE (const int i, const int j) {
        pre_vol(i, j) = volume(i, j) + vol_flux_y(i + 0, j + 1) - vol_flux_y(i, j);
        post_vol(i, j) = volume(i, j);
      });
    }

    // DO k=y_min,y_max+2
    //   DO j=x_min,x_max
    // clover::par_ranged2(Range2d{x_min + 1, y_min + 1, x_max + 2, y_max + 2 + 2}, [=] DEVICE_KERNEL(const int x, const int y) {
    const RAJA::TypedRangeSegment<int> row_Range2(y_min + 1,  y_max + 2 + 2);
    const RAJA::TypedRangeSegment<int> col_Range2(x_min + 1,  x_max  + 2);

    RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range2, row_Range2),
        [=] RAJA_HOST_DEVICE (const int x, const int y) {
      int upwind, donor, downwind, dif;
      double sigmat, sigma3, sigma4, sigmav,  sigmam, diffuw, diffdw, limiter, wind;

      const int j = x;
      const int k = y;

      if (vol_flux_y(x, y) > 0.0) {
        upwind = k - 2;
        donor = k - 1;
        downwind = k;
        dif = donor;
      } else {
      #if defined(RAJA_TARGET_CPU)
        upwind = std::min(k + 1, y_max + 2); // XXX can't do std::min because CUDA
      #else
        upwind = min(k + 1, y_max + 2); // XXX can't do std::min because CUDA
      #endif
        donor = k;
        downwind = k - 1;
        dif = upwind;
      }

      sigmat = std::fabs(vol_flux_y(x, y)) / pre_vol(j, donor);
      sigma3 = (1.0 + sigmat) * (vertexdy[k] / vertexdy[dif]);
      sigma4 = 2.0 - sigmat;

      sigmav = sigmat;

      diffuw = density1(j, donor) - density1(j, upwind);
      diffdw = density1(j, downwind) - density1(j, donor);
      wind = 1.0;
      if (diffdw <= 0.0) wind = -1.0;
      if (diffuw * diffdw > 0.0) {
        limiter = (1.0 - sigmav) * wind *
                  std::fmin(std::fmin(std::fabs(diffuw), std::fabs(diffdw)),
                            one_by_six * (sigma3 * std::fabs(diffuw) + sigma4 * std::fabs(diffdw)));
      } else {
        limiter = 0.0;
      }
      mass_flux_y(x, y) = vol_flux_y(x, y) * (density1(j, donor) + limiter);

      sigmam = std::fabs(mass_flux_y(x, y)) / (density1(j, donor) * pre_vol(j, donor));
      diffuw = energy1(j, donor) - energy1(j, upwind);
      diffdw = energy1(j, downwind) - energy1(j, donor);
      wind = 1.0;
      if (diffdw <= 0.0) wind = -1.0;
      if (diffuw * diffdw > 0.0) {
        limiter = (1.0 - sigmam) * wind *
                  std::fmin(std::fmin(std::fabs(diffuw), std::fabs(diffdw)),
                            one_by_six * (sigma3 * std::fabs(diffuw) + sigma4 * std::fabs(diffdw)));
      } else {
        limiter = 0.0;
      }
      ener_flux(x, y) = mass_flux_y(x, y) * (energy1(j, donor) + limiter);
    });

    // DO k=y_min,y_max
    //   DO j=x_min,x_max
    const RAJA::TypedRangeSegment<int> row_Range3(y_min + 1,  y_max + 2);
    const RAJA::TypedRangeSegment<int> col_Range3(x_min + 1,  x_max + 2);
//    clover::par_ranged2(Range2d{x_min + 1, y_min + 1, x_max + 2, y_max + 2}, [=] DEVICE_KERNEL(const int i, const int j) {
     RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range3, row_Range3),
        [=] RAJA_HOST_DEVICE (const int i, const int j) {
      double pre_mass_s = density1(i, j) * pre_vol(i, j);
      double post_mass_s = pre_mass_s + mass_flux_y(i, j) - mass_flux_y(i + 0, j + 1);
      double post_ener_s = (energy1(i, j) * pre_mass_s + ener_flux(i, j) - ener_flux(i + 0, j + 1)) / post_mass_s;
      double advec_vol_s = pre_vol(i, j) + vol_flux_y(i, j) - vol_flux_y(i + 0, j + 1);
      density1(i, j) = post_mass_s / advec_vol_s;
      energy1(i, j) = post_ener_s;
    });
  }
}

//  @brief Cell centred advection driver.
//  @author Wayne Gaudin
//  @details Invokes the user selected advection kernel.
void advec_cell_driver(global_variables &globals, int tile, int sweep_number, int direction) {

  tile_type &t = globals.chunk.tiles[tile];
  advec_cell_kernel(t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, direction, sweep_number, t.field.vertexdx, t.field.vertexdy,
                    t.field.volume, t.field.density1, t.field.energy1, t.field.mass_flux_x, t.field.vol_flux_x, t.field.mass_flux_y,
                    t.field.vol_flux_y, t.field.work_array1, t.field.work_array2, t.field.work_array3, t.field.work_array4,
                    t.field.work_array5, t.field.work_array6, t.field.work_array7);
}
