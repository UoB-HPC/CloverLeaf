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

#include "advec_mom.h"
#include <cmath>

//  @brief Fortran momentum advection kernel
//  @author Wayne Gaudin
//  @details Performs a second order advective remap on the vertex momentum
//  using van-Leer limiting and directional splitting.
//  Note that although pre_vol is only set and not used in the update, please
//  leave it in the method.
void advec_mom_kernel(bool use_target, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &vel1_buffer, field_type &field,
                      int which_vel, int sweep_number, int direction) {

  int mom_sweep = direction + 2 * (sweep_number - 1);

  // DO k=y_min-2,y_max+2
  //   DO j=x_min-2,x_max+2

  const int base_stride = field.base_stride;
  const int vels_wk_stride = field.vels_wk_stride;
  const int flux_x_stride = field.flux_x_stride;
  const int flux_y_stride = field.flux_y_stride;

  if (mom_sweep == 1) { // x 1

    double *vol_flux_y = field.vol_flux_y.data;
    double *vol_flux_x = field.vol_flux_x.data;
    double *volume = field.volume.data;
    double *pre_vol = field.work_array5.data;
    double *post_vol = field.work_array6.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
      for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
        post_vol[i + j * vels_wk_stride] =
            volume[i + j * base_stride] + vol_flux_y[(i + 0) + (j + 1) * flux_y_stride] - vol_flux_y[i + j * flux_y_stride];
        pre_vol[i + j * vels_wk_stride] =
            post_vol[i + j * vels_wk_stride] + vol_flux_x[(i + 1) + (j + 0) * flux_x_stride] - vol_flux_x[i + j * flux_x_stride];
      }
    }
  } else if (mom_sweep == 2) { // y 1

    double *vol_flux_y = field.vol_flux_y.data;
    double *vol_flux_x = field.vol_flux_x.data;
    double *volume = field.volume.data;
    double *pre_vol = field.work_array5.data;
    double *post_vol = field.work_array6.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
      for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
        post_vol[i + j * vels_wk_stride] =
            volume[i + j * base_stride] + vol_flux_x[(i + 1) + (j + 0) * flux_x_stride] - vol_flux_x[i + j * flux_x_stride];
        pre_vol[i + j * vels_wk_stride] =
            post_vol[i + j * vels_wk_stride] + vol_flux_y[(i + 0) + (j + 1) * flux_y_stride] - vol_flux_y[i + j * flux_y_stride];
      }
    }
  } else if (mom_sweep == 3) { // x 2

    double *vol_flux_y = field.vol_flux_y.data;
    double *volume = field.volume.data;
    double *pre_vol = field.work_array5.data;
    double *post_vol = field.work_array6.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
      for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
        post_vol[i + j * vels_wk_stride] = volume[i + j * base_stride];
        pre_vol[i + j * vels_wk_stride] =
            post_vol[i + j * vels_wk_stride] + vol_flux_y[(i + 0) + (j + 1) * flux_y_stride] - vol_flux_y[i + j * flux_y_stride];
      }
    }
  } else if (mom_sweep == 4) { // y 2

    double *vol_flux_x = field.vol_flux_x.data;
    double *volume = field.volume.data;
    double *pre_vol = field.work_array5.data;
    double *post_vol = field.work_array6.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
      for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
        post_vol[i + j * vels_wk_stride] = volume[i + j * base_stride];
        pre_vol[i + j * vels_wk_stride] =
            post_vol[i + j * vels_wk_stride] + vol_flux_x[(i + 1) + (j + 0) * flux_x_stride] - vol_flux_x[i + j * flux_x_stride];
      }
    }
  }

  if (direction == 1) {
    if (which_vel == 1) {
      // DO k=y_min,y_max+1
      //   DO j=x_min-2,x_max+2

      double *mass_flux_x = field.mass_flux_x.data;
      double *node_flux = field.work_array1.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = (y_min + 1); j < (y_max + 1 + 2); j++) {
        for (int i = (x_min - 2 + 1); i < (x_max + 2 + 2); i++) {
          node_flux[i + j * vels_wk_stride] =
              0.25 * (mass_flux_x[(i + 0) + (j - 1) * flux_x_stride] + mass_flux_x[i + j * flux_x_stride] +
                      mass_flux_x[(i + 1) + (j - 1) * flux_x_stride] + mass_flux_x[(i + 1) + (j + 0) * flux_x_stride]);
        }
      }

      // DO k=y_min,y_max+1
      //   DO j=x_min-1,x_max+2

      double *density1 = field.density1.data;
      double *node_mass_post = field.work_array2.data;
      double *node_mass_pre = field.work_array3.data;
      double *post_vol = field.work_array6.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = (y_min + 1); j < (y_max + 1 + 2); j++) {
        for (int i = (x_min - 1 + 1); i < (x_max + 2 + 2); i++) {
          node_mass_post[i + j * vels_wk_stride] =
              0.25 * (density1[(i + 0) + (j - 1) * base_stride] * post_vol[(i + 0) + (j - 1) * vels_wk_stride] +
                      density1[i + j * base_stride] * post_vol[i + j * vels_wk_stride] +
                      density1[(i - 1) + (j - 1) * base_stride] * post_vol[(i - 1) + (j - 1) * vels_wk_stride] +
                      density1[(i - 1) + (j + 0) * base_stride] * post_vol[(i - 1) + (j + 0) * vels_wk_stride]);
          node_mass_pre[i + j * vels_wk_stride] =
              node_mass_post[i + j * vels_wk_stride] - node_flux[(i - 1) + (j + 0) * vels_wk_stride] + node_flux[i + j * vels_wk_stride];
        }
      }
    }

    // DO k=y_min,y_max+1
    //  DO j=x_min-1,x_max+1

    const size_t vel1_sizex = vel1_buffer.nX();
    double *vel1 = vel1_buffer.data;
    double *node_flux = field.work_array1.data;
    double *node_mass_pre = field.work_array3.data;
    double *mom_flux = field.work_array4.data;
    double *celldx = field.celldx.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min + 1); j < (y_max + 1 + 2); j++) {
      for (int i = (x_min - 1 + 1); i < (x_max + 1 + 2); i++)
        ({
          int upwind, donor, downwind, dif;
          double sigma, width, limiter, vdiffuw, vdiffdw, auw, adw, wind, advec_vel_s;
          if (node_flux[i + j * vels_wk_stride] < 0.0) {
            upwind = i + 2;
            donor = i + 1;
            downwind = i;
            dif = donor;
          } else {
            upwind = i - 1;
            donor = i;
            downwind = i + 1;
            dif = upwind;
          }
          sigma = fabs(node_flux[i + j * vels_wk_stride]) / (node_mass_pre[donor + j * vels_wk_stride]);
          width = celldx[i];
          vdiffuw = vel1[donor + j * vel1_sizex] - vel1[upwind + j * vel1_sizex];
          vdiffdw = vel1[downwind + j * vel1_sizex] - vel1[donor + j * vel1_sizex];
          limiter = 0.0;
          if (vdiffuw * vdiffdw > 0.0) {
            auw = fabs(vdiffuw);
            adw = fabs(vdiffdw);
            wind = 1.0;
            if (vdiffdw <= 0.0) wind = -1.0;
            limiter = wind * fmin(fmin(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / celldx[dif]) / 6.0, auw), adw);
          }
          advec_vel_s = vel1[donor + j * vel1_sizex] + (1.0 - sigma) * limiter;
          mom_flux[i + j * vels_wk_stride] = advec_vel_s * node_flux[i + j * vels_wk_stride];
        });
    }

    // DO k=y_min,y_max+1
    //   DO j=x_min,x_max+1

    double *node_mass_post = field.work_array2.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min + 1); j < (y_max + 1 + 2); j++) {
      for (int i = (x_min + 1); i < (x_max + 1 + 2); i++) {
        vel1[i + j * vel1_sizex] = (vel1[i + j * vel1_sizex] * node_mass_pre[i + j * vels_wk_stride] +
                                    mom_flux[(i - 1) + (j + 0) * vels_wk_stride] - mom_flux[i + j * vels_wk_stride]) /
                                   node_mass_post[i + j * vels_wk_stride];
      }
    }
  } else if (direction == 2) {
    if (which_vel == 1) {
      // DO k=y_min-2,y_max+2
      //   DO j=x_min,x_max+1

      double *node_flux = field.work_array1.data;
      double *mass_flux_y = field.mass_flux_y.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = (y_min - 2 + 1); j < (y_max + 2 + 2); j++) {
        for (int i = (x_min + 1); i < (x_max + 1 + 2); i++) {
          node_flux[i + j * vels_wk_stride] =
              0.25 * (mass_flux_y[(i - 1) + (j + 0) * flux_y_stride] + mass_flux_y[i + j * flux_y_stride] +
                      mass_flux_y[(i - 1) + (j + 1) * flux_y_stride] + mass_flux_y[(i + 0) + (j + 1) * flux_y_stride]);
        }
      }

      // DO k=y_min-1,y_max+2
      //   DO j=x_min,x_max+1
      double *density1 = field.density1.data;
      double *node_mass_post = field.work_array2.data;
      double *node_mass_pre = field.work_array3.data;
      double *post_vol = field.work_array6.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = (y_min - 1 + 1); j < (y_max + 2 + 2); j++) {
        for (int i = (x_min + 1); i < (x_max + 1 + 2); i++) {
          node_mass_post[i + j * vels_wk_stride] =
              0.25 * (density1[(i + 0) + (j - 1) * base_stride] * post_vol[(i + 0) + (j - 1) * vels_wk_stride] +
                      density1[i + j * base_stride] * post_vol[i + j * vels_wk_stride] +
                      density1[(i - 1) + (j - 1) * base_stride] * post_vol[(i - 1) + (j - 1) * vels_wk_stride] +
                      density1[(i - 1) + (j + 0) * base_stride] * post_vol[(i - 1) + (j + 0) * vels_wk_stride]);
          node_mass_pre[i + j * vels_wk_stride] =
              node_mass_post[i + j * vels_wk_stride] - node_flux[(i + 0) + (j - 1) * vels_wk_stride] + node_flux[i + j * vels_wk_stride];
        }
      }
    }

    // DO k=y_min-1,y_max+1
    //   DO j=x_min,x_max+1

    const size_t vel1_sizex = vel1_buffer.nX();
    double *vel1 = vel1_buffer.data;
    double *node_flux = field.work_array1.data;
    double *node_mass_pre = field.work_array3.data;
    double *mom_flux = field.work_array4.data;
    double *celldy = field.celldy.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min - 1 + 1); j < (y_max + 1 + 2); j++) {
      for (int i = (x_min + 1); i < (x_max + 1 + 2); i++)
        ({
          int upwind, donor, downwind, dif;
          double sigma, width, limiter, vdiffuw, vdiffdw, auw, adw, wind, advec_vel_s;
          if (node_flux[i + j * vels_wk_stride] < 0.0) {
            upwind = j + 2;
            donor = j + 1;
            downwind = j;
            dif = donor;
          } else {
            upwind = j - 1;
            donor = j;
            downwind = j + 1;
            dif = upwind;
          }
          sigma = fabs(node_flux[i + j * vels_wk_stride]) / (node_mass_pre[i + donor * vels_wk_stride]);
          width = celldy[j];
          vdiffuw = vel1[i + donor * vel1_sizex] - vel1[i + upwind * vel1_sizex];
          vdiffdw = vel1[i + downwind * vel1_sizex] - vel1[i + donor * vel1_sizex];
          limiter = 0.0;
          if (vdiffuw * vdiffdw > 0.0) {
            auw = fabs(vdiffuw);
            adw = fabs(vdiffdw);
            wind = 1.0;
            if (vdiffdw <= 0.0) wind = -1.0;
            limiter = wind * fmin(fmin(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / celldy[dif]) / 6.0, auw), adw);
          }
          advec_vel_s = vel1[i + donor * vel1_sizex] + (1.0 - sigma) * limiter;
          mom_flux[i + j * vels_wk_stride] = advec_vel_s * node_flux[i + j * vels_wk_stride];
        });
    }

    // DO k=y_min,y_max+1
    //   DO j=x_min,x_max+1

    double *node_mass_post = field.work_array2.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_min + 1); j < (y_max + 1 + 2); j++) {
      for (int i = (x_min + 1); i < (x_max + 1 + 2); i++) {
        vel1[i + j * vel1_sizex] = (vel1[i + j * vel1_sizex] * node_mass_pre[i + j * vels_wk_stride] +
                                    mom_flux[(i + 0) + (j - 1) * vels_wk_stride] - mom_flux[i + j * vels_wk_stride]) /
                                   node_mass_post[i + j * vels_wk_stride];
      }
    }
  }
}

//  @brief Momentum advection driver
//  @author Wayne Gaudin
//  @details Invokes the user specified momentum advection kernel.
void advec_mom_driver(global_variables &globals, int tile, int which_vel, int direction, int sweep_number) {

#if SYNC_BUFFERS
  globals.hostToDevice();
#endif

  tile_type &t = globals.chunk.tiles[tile];
  if (which_vel == 1) {
    advec_mom_kernel(globals.context.use_target, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel1, t.field,
                     which_vel, sweep_number, direction);
  } else {
    advec_mom_kernel(globals.context.use_target, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel1, t.field,
                     which_vel, sweep_number, direction);
  }

#if SYNC_BUFFERS
  globals.deviceToHost();
#endif
}
