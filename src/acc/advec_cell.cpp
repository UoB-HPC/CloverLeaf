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
#include <cmath>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

//  @brief Fortran cell advection kernel.
//  @author Wayne Gaudin
//  @details Performs a second order advective remap using van-Leer limiting
//  with directional splitting.
void advec_cell_kernel(bool use_target, int x_min, int x_max, int y_min, int y_max, int dir, int sweep_number, field_type &field) {

  const double one_by_six = 1.0 / 6.0;

  const size_t base_stride = field.base_stride;
  const size_t vels_wk_stride = field.vels_wk_stride;
  const size_t flux_x_stride = field.flux_x_stride;
  const size_t flux_y_stride = field.flux_y_stride;

  if (dir == g_xdir) {

    // DO k=y_min-2,y_max+2
    //   DO j=x_min-2,x_max+2

    if (sweep_number == 1) {

      double *volume = field.volume.data;
      double *vol_flux_x = field.vol_flux_x.data;
      double *vol_flux_y = field.vol_flux_y.data;
      double *pre_vol = field.work_array1.data;
      double *post_vol = field.work_array2.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
        for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
          pre_vol[i + j * vels_wk_stride] =
              volume[i + j * base_stride] + (vol_flux_x[(i + 1) + (j + 0) * flux_x_stride] - vol_flux_x[i + j * flux_x_stride] +
                                             vol_flux_y[(i + 0) + (j + 1) * flux_y_stride] - vol_flux_y[i + j * flux_y_stride]);
          post_vol[i + j * vels_wk_stride] =
              pre_vol[i + j * vels_wk_stride] - (vol_flux_x[(i + 1) + (j + 0) * flux_x_stride] - vol_flux_x[i + j * flux_x_stride]);
        }
      }

    } else {

      double *volume = field.volume.data;
      double *vol_flux_x = field.vol_flux_x.data;
      double *pre_vol = field.work_array1.data;
      double *post_vol = field.work_array2.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
        for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
          pre_vol[i + j * vels_wk_stride] =
              volume[i + j * base_stride] + vol_flux_x[(i + 1) + (j + 0) * flux_x_stride] - vol_flux_x[i + j * flux_x_stride];
          post_vol[i + j * vels_wk_stride] = volume[i + j * base_stride];
        }
      }
    }

    // DO k=y_min,y_max
    //   DO j=x_min,x_max+2
    double *vertexdx = field.vertexdx.data;
    double *density1 = field.density1.data;
    double *energy1 = field.energy1.data;
    double *mass_flux_x = field.mass_flux_x.data;
    double *vol_flux_x = field.vol_flux_x.data;
    double *pre_vol = field.work_array1.data;
    double *ener_flux = field.work_array7.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min + 1); j < (y_max + 2); j++) {
      for (int i = (x_min + 1); i < (x_max + 2 + 2); i++)
        ({
          int upwind, donor, downwind, dif;
          double sigmat, sigma3, sigma4, sigmav, sigmam, diffuw, diffdw, limiter, wind;
          if (vol_flux_x[i + j * flux_x_stride] > 0.0) {
            upwind = i - 2;
            donor = i - 1;
            downwind = i;
            dif = donor;
          } else {
            upwind = MIN(i + 1, x_max + 2);
            donor = i;
            downwind = i - 1;
            dif = upwind;
          }
          sigmat = fabs(vol_flux_x[i + j * flux_x_stride]) / pre_vol[donor + j * vels_wk_stride];
          sigma3 = (1.0 + sigmat) * (vertexdx[i] / vertexdx[dif]);
          sigma4 = 2.0 - sigmat;
          //					sigma = sigmat;
          sigmav = sigmat;
          diffuw = density1[donor + j * base_stride] - density1[upwind + j * base_stride];
          diffdw = density1[downwind + j * base_stride] - density1[donor + j * base_stride];
          wind = 1.0;
          if (diffdw <= 0.0) wind = -1.0;
          if (diffuw * diffdw > 0.0) {
            limiter = (1.0 - sigmav) * wind *
                      fmin(fmin(fabs(diffuw), fabs(diffdw)), one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
          } else {
            limiter = 0.0;
          }
          mass_flux_x[i + j * flux_x_stride] = vol_flux_x[i + j * flux_x_stride] * (density1[donor + j * base_stride] + limiter);
          sigmam = fabs(mass_flux_x[i + j * flux_x_stride]) / (density1[donor + j * base_stride] * pre_vol[donor + j * vels_wk_stride]);
          diffuw = energy1[donor + j * base_stride] - energy1[upwind + j * base_stride];
          diffdw = energy1[downwind + j * base_stride] - energy1[donor + j * base_stride];
          wind = 1.0;
          if (diffdw <= 0.0) wind = -1.0;
          if (diffuw * diffdw > 0.0) {
            limiter = (1.0 - sigmam) * wind *
                      fmin(fmin(fabs(diffuw), fabs(diffdw)), one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
          } else {
            limiter = 0.0;
          }
          ener_flux[i + j * vels_wk_stride] = mass_flux_x[i + j * flux_x_stride] * (energy1[donor + j * base_stride] + limiter);
        });
    }

    // DO k=y_min,y_max
    //   DO j=x_min,x_max

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min + 1); j < (y_max + 2); j++) {
      for (int i = (x_min + 1); i < (x_max + 2); i++) {
        double pre_mass_s = density1[i + j * base_stride] * pre_vol[i + j * vels_wk_stride];
        double post_mass_s = pre_mass_s + mass_flux_x[i + j * flux_x_stride] - mass_flux_x[(i + 1) + (j + 0) * flux_x_stride];
        double post_ener_s = (energy1[i + j * base_stride] * pre_mass_s + ener_flux[i + j * vels_wk_stride] -
                              ener_flux[(i + 1) + (j + 0) * vels_wk_stride]) /
                             post_mass_s;
        double advec_vol_s =
            pre_vol[i + j * vels_wk_stride] + vol_flux_x[i + j * flux_x_stride] - vol_flux_x[(i + 1) + (j + 0) * flux_x_stride];
        density1[i + j * base_stride] = post_mass_s / advec_vol_s;
        energy1[i + j * base_stride] = post_ener_s;
      }
    }

  } else if (dir == g_ydir) {

    // DO k=y_min-2,y_max+2
    //   DO j=x_min-2,x_max+2

    if (sweep_number == 1) {

      double *volume = field.volume.data;
      double *vol_flux_x = field.vol_flux_x.data;
      double *vol_flux_y = field.vol_flux_y.data;
      double *pre_vol = field.work_array1.data;
      double *post_vol = field.work_array2.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
        for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
          pre_vol[i + j * vels_wk_stride] =
              volume[i + j * base_stride] + (vol_flux_y[(i + 0) + (j + 1) * flux_y_stride] - vol_flux_y[i + j * flux_y_stride] +
                                             vol_flux_x[(i + 1) + (j + 0) * flux_x_stride] - vol_flux_x[i + j * flux_x_stride]);
          post_vol[i + j * vels_wk_stride] =
              pre_vol[i + j * vels_wk_stride] - (vol_flux_y[(i + 0) + (j + 1) * flux_y_stride] - vol_flux_y[i + j * flux_y_stride]);
        }
      }

    } else {

      double *volume = field.volume.data;
      double *vol_flux_y = field.vol_flux_y.data;
      double *pre_vol = field.work_array1.data;
      double *post_vol = field.work_array2.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
        for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
          pre_vol[i + j * vels_wk_stride] =
              volume[i + j * base_stride] + vol_flux_y[(i + 0) + (j + 1) * flux_y_stride] - vol_flux_y[i + j * flux_y_stride];
          post_vol[i + j * vels_wk_stride] = volume[i + j * base_stride];
        }
      }
    }

    // DO k=y_min,y_max+2
    //   DO j=x_min,x_max
    double *vertexdy = field.vertexdy.data;
    double *density1 = field.density1.data;
    double *energy1 = field.energy1.data;
    double *mass_flux_y = field.mass_flux_y.data;
    double *vol_flux_y = field.vol_flux_y.data;
    double *pre_vol = field.work_array1.data;
    double *ener_flux = field.work_array7.data;
#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min + 1); j < (y_max + 2 + 2); j++) {
      for (int i = (x_min + 1); i < (x_max + 2); i++)
        ({
          int upwind, donor, downwind, dif;
          double sigmat, sigma3, sigma4, sigmav, sigmam, diffuw, diffdw, limiter, wind;
          if (vol_flux_y[i + j * flux_y_stride] > 0.0) {
            upwind = j - 2;
            donor = j - 1;
            downwind = j;
            dif = donor;
          } else {
            upwind = MIN(j + 1, y_max + 2);
            donor = j;
            downwind = j - 1;
            dif = upwind;
          }
          sigmat = fabs(vol_flux_y[i + j * flux_y_stride]) / pre_vol[i + donor * vels_wk_stride];
          sigma3 = (1.0 + sigmat) * (vertexdy[j] / vertexdy[dif]);
          sigma4 = 2.0 - sigmat;
          //					sigma = sigmat;
          sigmav = sigmat;
          diffuw = density1[i + donor * base_stride] - density1[i + upwind * base_stride];
          diffdw = density1[i + downwind * base_stride] - density1[i + donor * base_stride];
          wind = 1.0;
          if (diffdw <= 0.0) wind = -1.0;
          if (diffuw * diffdw > 0.0) {
            limiter = (1.0 - sigmav) * wind *
                      fmin(fmin(fabs(diffuw), fabs(diffdw)), one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
          } else {
            limiter = 0.0;
          }
          mass_flux_y[i + j * flux_y_stride] = vol_flux_y[i + j * flux_y_stride] * (density1[i + donor * base_stride] + limiter);
          sigmam = fabs(mass_flux_y[i + j * flux_y_stride]) / (density1[i + donor * base_stride] * pre_vol[i + donor * vels_wk_stride]);
          diffuw = energy1[i + donor * base_stride] - energy1[i + upwind * base_stride];
          diffdw = energy1[i + downwind * base_stride] - energy1[i + donor * base_stride];
          wind = 1.0;
          if (diffdw <= 0.0) wind = -1.0;
          if (diffuw * diffdw > 0.0) {
            limiter = (1.0 - sigmam) * wind *
                      fmin(fmin(fabs(diffuw), fabs(diffdw)), one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
          } else {
            limiter = 0.0;
          }
          ener_flux[i + j * vels_wk_stride] = mass_flux_y[i + j * flux_y_stride] * (energy1[i + donor * base_stride] + limiter);
        });
    }

    // DO k=y_min,y_max
    //   DO j=x_min,x_max

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min + 1); j < (y_max + 2); j++) {
      for (int i = (x_min + 1); i < (x_max + 2); i++) {
        double pre_mass_s = density1[i + j * base_stride] * pre_vol[i + j * vels_wk_stride];
        double post_mass_s = pre_mass_s + mass_flux_y[i + j * flux_y_stride] - mass_flux_y[(i + 0) + (j + 1) * flux_y_stride];
        double post_ener_s = (energy1[i + j * base_stride] * pre_mass_s + ener_flux[i + j * vels_wk_stride] -
                              ener_flux[(i + 0) + (j + 1) * vels_wk_stride]) /
                             post_mass_s;
        double advec_vol_s =
            pre_vol[i + j * vels_wk_stride] + vol_flux_y[i + j * flux_y_stride] - vol_flux_y[(i + 0) + (j + 1) * flux_y_stride];
        density1[i + j * base_stride] = post_mass_s / advec_vol_s;
        energy1[i + j * base_stride] = post_ener_s;
      }
    }
  }
}

//  @brief Cell centred advection driver.
//  @author Wayne Gaudin
//  @details Invokes the user selected advection kernel.
void advec_cell_driver(global_variables &globals, int tile, int sweep_number, int direction) {

#if SYNC_BUFFERS
  globals.hostToDevice();
#endif

  tile_type &t = globals.chunk.tiles[tile];
  advec_cell_kernel(globals.context.use_target, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, direction, sweep_number,
                    t.field);

#if SYNC_BUFFERS
  globals.deviceToHost();
#endif
}
