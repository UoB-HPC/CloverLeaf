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
#include "context.h"

//  @brief Fortran momentum advection kernel
//  @author Wayne Gaudin
//  @details Performs a second order advective remap on the vertex momentum
//  using van-Leer limiting and directional splitting.
//  Note that although pre_vol is only set and not used in the update, please
//  leave it in the method.
void advec_mom_kernel(queue &q, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &vel1_buffer,
                      clover::Buffer2D<double> &mass_flux_x_buffer, clover::Buffer2D<double> &vol_flux_x_buffer,
                      clover::Buffer2D<double> &mass_flux_y_buffer, clover::Buffer2D<double> &vol_flux_y_buffer,
                      clover::Buffer2D<double> &volume_buffer, clover::Buffer2D<double> &density1_buffer,
                      clover::Buffer2D<double> &node_flux_buffer, clover::Buffer2D<double> &node_mass_post_buffer,
                      clover::Buffer2D<double> &node_mass_pre_buffer, clover::Buffer2D<double> &mom_flux_buffer,
                      clover::Buffer2D<double> &pre_vol_buffer, clover::Buffer2D<double> &post_vol_buffer,
                      clover::Buffer1D<double> &celldx_buffer, clover::Buffer1D<double> &celldy_buffer, int which_vel, int sweep_number,
                      int direction) {

  int mom_sweep = direction + 2 * (sweep_number - 1);

  // DO k=y_min-2,y_max+2
  //   DO j=x_min-2,x_max+2

  clover::Range2d policy(x_min - 2 + 1, y_min - 2 + 1, x_max + 2 + 2, y_max + 2 + 2);

  if (mom_sweep == 1) { // x 1
    clover::execute(q, [&](handler &h) {
      auto vol_flux_y = vol_flux_y_buffer.access<R>(h);
      auto vol_flux_x = vol_flux_x_buffer.access<R>(h);
      auto volume = volume_buffer.access<R>(h);
      auto pre_vol = pre_vol_buffer.access<W>(h);
      auto post_vol = post_vol_buffer.access<RW>(h);
      clover::par_ranged<class APPEND_LN(advec_mom_x1)>(h, policy, [=](id<2> idx) {
        post_vol[idx] = volume[idx] + vol_flux_y[clover::offset(idx, 0, 1)] - vol_flux_y[idx];
        pre_vol[idx] = post_vol[idx] + vol_flux_x[clover::offset(idx, 1, 0)] - vol_flux_x[idx];
      });
    });
  } else if (mom_sweep == 2) { // y 1
    clover::execute(q, [&](handler &h) {
      auto vol_flux_y = vol_flux_y_buffer.access<R>(h);
      auto vol_flux_x = vol_flux_x_buffer.access<R>(h);
      auto volume = volume_buffer.access<R>(h);
      auto pre_vol = pre_vol_buffer.access<W>(h);
      auto post_vol = post_vol_buffer.access<RW>(h);
      clover::par_ranged<class APPEND_LN(advec_mom_y1)>(h, policy, [=](id<2> idx) {
        post_vol[idx] = volume[idx] + vol_flux_x[clover::offset(idx, 1, 0)] - vol_flux_x[idx];
        pre_vol[idx] = post_vol[idx] + vol_flux_y[clover::offset(idx, 0, 1)] - vol_flux_y[idx];
      });
    });
  } else if (mom_sweep == 3) { // x 2
    clover::execute(q, [&](handler &h) {
      auto vol_flux_y = vol_flux_y_buffer.access<R>(h);
      auto volume = volume_buffer.access<R>(h);
      auto pre_vol = pre_vol_buffer.access<W>(h);
      auto post_vol = post_vol_buffer.access<RW>(h);
      clover::par_ranged<class APPEND_LN(advec_mom_x1)>(h, policy, [=](id<2> idx) {
        post_vol[idx] = volume[idx];
        pre_vol[idx] = post_vol[idx] + vol_flux_y[clover::offset(idx, 0, 1)] - vol_flux_y[idx];
      });
    });
  } else if (mom_sweep == 4) { // y 2
    clover::execute(q, [&](handler &h) {
      auto vol_flux_x = vol_flux_x_buffer.access<R>(h);
      auto volume = volume_buffer.access<R>(h);
      auto pre_vol = pre_vol_buffer.access<W>(h);
      auto post_vol = post_vol_buffer.access<RW>(h);
      clover::par_ranged<class APPEND_LN(advec_mom_y1)>(h, policy, [=](id<2> idx) {
        post_vol[idx] = volume[idx];
        pre_vol[idx] = post_vol[idx] + vol_flux_x[clover::offset(idx, 1, 0)] - vol_flux_x[idx];
      });
    });
  }

  if (direction == 1) {
    if (which_vel == 1) {
      // DO k=y_min,y_max+1
      //   DO j=x_min-2,x_max+2

      clover::execute(q, [&](handler &h) {
        auto mass_flux_x = mass_flux_x_buffer.access<R>(h);
        auto node_flux = node_flux_buffer.access<RW>(h);
        clover::par_ranged<class advec_mom_dir1_vel1_node_flux>(
            h, {x_min - 2 + 1, y_min + 1, x_max + 2 + 2, y_max + 1 + 2}, [=](id<2> idx) {
              // Find staggered mesh mass fluxes, nodal masses and volumes.
              node_flux[idx] = 0.25 * (mass_flux_x[clover::offset(idx, 0, -1)] + mass_flux_x[idx] +
                                       mass_flux_x[clover::offset(idx, 1, -1)] + mass_flux_x[clover::offset(idx, 1, 0)]);
            });
      });

      // DO k=y_min,y_max+1
      //   DO j=x_min-1,x_max+2

      clover::execute(q, [&](handler &h) {
        auto density1 = density1_buffer.access<R>(h);
        auto node_flux = node_flux_buffer.access<R>(h);
        auto node_mass_post = node_mass_post_buffer.access<RW>(h);
        auto node_mass_pre = node_mass_pre_buffer.access<RW>(h);
        auto post_vol = post_vol_buffer.access<R>(h);
        clover::par_ranged<class advec_mom_dir1_vel1_node_mass_pre>(
            h, {x_min - 1 + 1, y_min + 1, x_max + 2 + 2, y_max + 1 + 2}, [=](id<2> idx) {
              // Staggered cell mass post advection
              node_mass_post[idx] =
                  0.25 * (density1[clover::offset(idx, 0, -1)] * post_vol[clover::offset(idx, 0, -1)] + density1[idx] * post_vol[idx] +
                          density1[clover::offset(idx, -1, -1)] * post_vol[clover::offset(idx, -1, -1)] +
                          density1[clover::offset(idx, -1, 0)] * post_vol[clover::offset(idx, -1, 0)]);
              node_mass_pre[idx] = node_mass_post[idx] - node_flux[clover::offset(idx, -1, 0)] + node_flux[idx];
            });
      });
    }

    // DO k=y_min,y_max+1
    //  DO j=x_min-1,x_max+1

    clover::execute(q, [&](handler &h) {
      auto vel1 = vel1_buffer.access<R>(h);
      auto node_flux = node_flux_buffer.access<R>(h);
      auto node_mass_pre = node_mass_pre_buffer.access<R>(h);
      auto mom_flux = mom_flux_buffer.access<RW>(h);
      auto celldx = celldx_buffer.access<R>(h);
      clover::par_ranged<class advec_mom_dir1_mom_flux>(h, {x_min - 1 + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](id<2> idx) {
        int upwind, donor, downwind, dif;
        double sigma, width, limiter, vdiffuw, vdiffdw, auw, adw, wind, advec_vel_s;

        const int j = idx.get(0);
        const int k = idx.get(1);

        if (node_flux[idx] < 0.0) {
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

        sigma = sycl::fabs(node_flux[idx]) / (node_mass_pre[donor][k]);
        width = celldx[j];
        vdiffuw = vel1[donor][k] - vel1[upwind][k];
        vdiffdw = vel1[downwind][k] - vel1[donor][k];
        limiter = 0.0;
        if (vdiffuw * vdiffdw > 0.0) {
          auw = sycl::fabs(vdiffuw);
          adw = sycl::fabs(vdiffdw);
          wind = 1.0;
          if (vdiffdw <= 0.0) wind = -1.0;
          limiter =
              wind * sycl::fmin(sycl::fmin(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / celldx[dif]) / 6.0, auw), adw);
        }
        advec_vel_s = vel1[donor][k] + (1.0 - sigma) * limiter;
        mom_flux[idx] = advec_vel_s * node_flux[idx];
      });
    });

    // DO k=y_min,y_max+1
    //   DO j=x_min,x_max+1

    clover::execute(q, [&](handler &h) {
      auto vel1 = vel1_buffer.access<RW>(h);
      auto node_mass_post = node_mass_post_buffer.access<R>(h);
      auto node_mass_pre = node_mass_pre_buffer.access<R>(h);
      auto mom_flux = mom_flux_buffer.access<R>(h);
      clover::par_ranged<class advec_mom_dir1_vel1>(h, {x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](id<2> idx) {
        vel1[idx] = (vel1[idx] * node_mass_pre[idx] + mom_flux[clover::offset(idx, -1, 0)] - mom_flux[idx]) / node_mass_post[idx];
      });
    });
  } else if (direction == 2) {
    if (which_vel == 1) {
      // DO k=y_min-2,y_max+2
      //   DO j=x_min,x_max+1

      clover::execute(q, [&](handler &h) {
        auto node_flux = node_flux_buffer.access<RW>(h);
        auto mass_flux_y = mass_flux_y_buffer.access<R>(h);
        clover::par_ranged<class advec_mom_dir2_vel1_node_flux>(
            h, {x_min + 1, y_min - 2 + 1, x_max + 1 + 2, y_max + 2 + 2}, [=](id<2> idx) {
              // Find staggered mesh mass fluxes and nodal masses and volumes.
              node_flux[idx] = 0.25 * (mass_flux_y[clover::offset(idx, -1, 0)] + mass_flux_y[idx] +
                                       mass_flux_y[clover::offset(idx, -1, 1)] + mass_flux_y[clover::offset(idx, 0, 1)]);
            });
      });

      // DO k=y_min-1,y_max+2
      //   DO j=x_min,x_max+1

      clover::execute(q, [&](handler &h) {
        auto density1 = density1_buffer.access<R>(h);
        auto node_flux = node_flux_buffer.access<R>(h);
        auto node_mass_post = node_mass_post_buffer.access<RW>(h);
        auto node_mass_pre = node_mass_pre_buffer.access<RW>(h);
        auto post_vol = post_vol_buffer.access<R>(h);
        clover::par_ranged<class advec_mom_dir2_vel1_node_mass_pre>(
            h, {x_min + 1, y_min - 1 + 1, x_max + 1 + 2, y_max + 2 + 2}, [=](id<2> idx) {
              node_mass_post[idx] =
                  0.25 * (density1[clover::offset(idx, 0, -1)] * post_vol[clover::offset(idx, 0, -1)] + density1[idx] * post_vol[idx] +
                          density1[clover::offset(idx, -1, -1)] * post_vol[clover::offset(idx, -1, -1)] +
                          density1[clover::offset(idx, -1, 0)] * post_vol[clover::offset(idx, -1, 0)]);
              node_mass_pre[idx] = node_mass_post[idx] - node_flux[clover::offset(idx, 0, -1)] + node_flux[idx];
            });
      });
    }

    // DO k=y_min-1,y_max+1
    //   DO j=x_min,x_max+1

    clover::execute(q, [&](handler &h) {
      auto vel1 = vel1_buffer.access<R>(h);
      auto node_flux = node_flux_buffer.access<R>(h);
      auto node_mass_pre = node_mass_pre_buffer.access<R>(h);
      auto mom_flux = mom_flux_buffer.access<RW>(h);
      auto celldy = celldy_buffer.access<R>(h);
      clover::par_ranged<class advec_mom_dir2_mom_flux>(h, {x_min + 1, y_min - 1 + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](id<2> idx) {
        int upwind, donor, downwind, dif;
        double sigma, width, limiter, vdiffuw, vdiffdw, auw, adw, wind, advec_vel_s;

        const int j = idx.get(0);
        const int k = idx.get(1);

        if (node_flux[idx] < 0.0) {
          upwind = k + 2;
          donor = k + 1;
          downwind = k;
          dif = donor;
        } else {
          upwind = k - 1;
          donor = k;
          downwind = k + 1;
          dif = upwind;
        }

        sigma = sycl::fabs(node_flux[idx]) / (node_mass_pre[j][donor]);
        width = celldy[k];
        vdiffuw = vel1[j][donor] - vel1[j][upwind];
        vdiffdw = vel1[j][downwind] - vel1[j][donor];
        limiter = 0.0;
        if (vdiffuw * vdiffdw > 0.0) {
          auw = sycl::fabs(vdiffuw);
          adw = sycl::fabs(vdiffdw);
          wind = 1.0;
          if (vdiffdw <= 0.0) wind = -1.0;
          limiter =
              wind * sycl::fmin(sycl::fmin(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / celldy[dif]) / 6.0, auw), adw);
        }
        advec_vel_s = vel1[j][donor] + (1.0 - sigma) * limiter;
        mom_flux[idx] = advec_vel_s * node_flux[idx];
      });
    });

    // DO k=y_min,y_max+1
    //   DO j=x_min,x_max+1

    clover::execute(q, [&](handler &h) {
      auto vel1 = vel1_buffer.access<RW>(h);
      auto node_mass_post = node_mass_post_buffer.access<R>(h);
      auto node_mass_pre = node_mass_pre_buffer.access<R>(h);
      auto mom_flux = mom_flux_buffer.access<R>(h);
      clover::par_ranged<class advec_mom_dir2_vel1>(h, {x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](id<2> idx) {
        vel1[idx] = (vel1[idx] * node_mass_pre[idx] + mom_flux[clover::offset(idx, 0, -1)] - mom_flux[idx]) / node_mass_post[idx];
      });
    });
  }
}

//  @brief Momentum advection driver
//  @author Wayne Gaudin
//  @details Invokes the user specified momentum advection kernel.
void advec_mom_driver(global_variables &globals, int tile, int which_vel, int direction, int sweep_number) {

  tile_type &t = globals.chunk.tiles[tile];
  if (which_vel == 1) {
    advec_mom_kernel(globals.context.queue, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax,
                     t.field.xvel1,       //
                     t.field.mass_flux_x, //
                     t.field.vol_flux_x,  //
                     t.field.mass_flux_y, //
                     t.field.vol_flux_y,  //
                     t.field.volume,      //
                     t.field.density1,    //
                     t.field.work_array1, //
                     t.field.work_array2, //
                     t.field.work_array3, //
                     t.field.work_array4, //
                     t.field.work_array5, //
                     t.field.work_array6, //
                     t.field.celldx,      //
                     t.field.celldy,      //
                     which_vel, sweep_number, direction);
  } else {
    advec_mom_kernel(globals.context.queue, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax,
                     t.field.yvel1,       //
                     t.field.mass_flux_x, //
                     t.field.vol_flux_x,  //
                     t.field.mass_flux_y, //
                     t.field.vol_flux_y,  //
                     t.field.volume,      //
                     t.field.density1,    //
                     t.field.work_array1, //
                     t.field.work_array2, //
                     t.field.work_array3, //
                     t.field.work_array4, //
                     t.field.work_array5, //
                     t.field.work_array6, //
                     t.field.celldx,      //
                     t.field.celldy,      //
                     which_vel, sweep_number, direction);
  }
}
