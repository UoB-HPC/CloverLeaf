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

//  @brief Fortran cell advection kernel.
//  @author Wayne Gaudin
//  @details Performs a second order advective remap using van-Leer limiting
//  with directional splitting.
void advec_cell_kernel(queue &queue, int x_min, int x_max, int y_min, int y_max, int dir, int sweep_number,
                       clover::Buffer1D<double> &vertexdx_buffer, clover::Buffer1D<double> &vertexdy_buffer,
                       clover::Buffer2D<double> &volume_buffer, clover::Buffer2D<double> &density1_buffer,
                       clover::Buffer2D<double> &energy1_buffer, clover::Buffer2D<double> &mass_flux_x_buffer,
                       clover::Buffer2D<double> &vol_flux_x_buffer, clover::Buffer2D<double> &mass_flux_y_buffer,
                       clover::Buffer2D<double> &vol_flux_y_buffer, clover::Buffer2D<double> &pre_vol_buffer,
                       clover::Buffer2D<double> &post_vol_buffer, clover::Buffer2D<double> &pre_mass_buffer,
                       clover::Buffer2D<double> &post_mass_buffer, clover::Buffer2D<double> &advec_vol_buffer,
                       clover::Buffer2D<double> &post_ener_buffer, clover::Buffer2D<double> &ener_flux_buffer) {

  const double one_by_six = 1.0 / 6.0;

  if (dir == g_xdir) {

    // DO k=y_min-2,y_max+2
    //   DO j=x_min-2,x_max+2

    const clover::Range2d policy(x_min - 2 + 1, y_min - 2 + 1, x_max + 2 + 2, y_max + 2 + 2);

    if (sweep_number == 1) {
      clover::execute(queue, [&](handler &h) {
        auto volume = volume_buffer.access<R>(h);
        auto vol_flux_x = vol_flux_x_buffer.access<R>(h);
        auto vol_flux_y = vol_flux_y_buffer.access<R>(h);
        auto pre_vol = pre_vol_buffer.access<RW>(h);
        auto post_vol = post_vol_buffer.access<W>(h);
        clover::par_ranged<class advec_cell_xdir_seq1>(h, policy, [=](id<2> idx) {
          pre_vol[idx] = volume[idx] + (vol_flux_x[clover::offset(idx, 1, 0)] - vol_flux_x[idx] + vol_flux_y[clover::offset(idx, 0, 1)] -
                                        vol_flux_y[idx]);
          post_vol[idx] = pre_vol[idx] - (vol_flux_x[clover::offset(idx, 1, 0)] - vol_flux_x[idx]);
        });
      });

    } else {
      clover::execute(queue, [&](handler &h) {
        auto volume = volume_buffer.access<R>(h);
        auto vol_flux_x = vol_flux_x_buffer.access<R>(h);
        auto pre_vol = pre_vol_buffer.access<W>(h);
        auto post_vol = post_vol_buffer.access<W>(h);
        clover::par_ranged<class advec_cell_xdir_sne1>(h, policy, [=](id<2> idx) {
          pre_vol[idx] = volume[idx] + vol_flux_x[clover::offset(idx, 1, 0)] - vol_flux_x[idx];
          post_vol[idx] = volume[idx];
        });
      });
    }
    clover::execute(queue, [&](handler &h) {
      auto vertexdx = vertexdx_buffer.access<R>(h);
      auto density1 = density1_buffer.access<R>(h);
      auto energy1 = energy1_buffer.access<R>(h);
      auto mass_flux_x = mass_flux_x_buffer.access<RW>(h);
      auto vol_flux_x = vol_flux_x_buffer.access<R>(h);
      auto pre_vol = pre_vol_buffer.access<R>(h);
      auto ener_flux = ener_flux_buffer.access<W>(h);
      // DO k=y_min,y_max
      //   DO j=x_min,x_max+2
      clover::par_ranged<class advec_cell_xdir_ener_flux>(h, {x_min + 1, y_min + 1, x_max + 2 + 2, y_max + 2}, [=](id<2> idx) {
        int upwind, donor, downwind, dif;
        double sigmat, sigma3, sigma4, sigmav, sigmam, diffuw, diffdw, limiter, wind;

        const int j = idx.get(0);
        const int k = idx.get(1);

        if (vol_flux_x[idx] > 0.0) {
          upwind = j - 2;
          donor = j - 1;
          downwind = j;
          dif = donor;
        } else {
          upwind = sycl::min(j + 1, x_max + 2);
          donor = j;
          downwind = j - 1;
          dif = upwind;
        }

        sigmat = sycl::fabs(vol_flux_x[idx]) / pre_vol[donor][k];
        sigma3 = (1.0 + sigmat) * (vertexdx[j] / vertexdx[dif]);
        sigma4 = 2.0 - sigmat;

        sigmav = sigmat;

        diffuw = density1[donor][k] - density1[upwind][k];
        diffdw = density1[downwind][k] - density1[donor][k];
        wind = 1.0;
        if (diffdw <= 0.0) wind = -1.0;
        if (diffuw * diffdw > 0.0) {
          limiter = (1.0 - sigmav) * wind *
                    sycl::fmin(sycl::fmin(sycl::fabs(diffuw), sycl::fabs(diffdw)),
                               one_by_six * (sigma3 * sycl::fabs(diffuw) + sigma4 * sycl::fabs(diffdw)));
        } else {
          limiter = 0.0;
        }
        mass_flux_x[idx] = vol_flux_x[idx] * (density1[donor][k] + limiter);

        sigmam = sycl::fabs(mass_flux_x[idx]) / (density1[donor][k] * pre_vol[donor][k]);
        diffuw = energy1[donor][k] - energy1[upwind][k];
        diffdw = energy1[downwind][k] - energy1[donor][k];
        wind = 1.0;
        if (diffdw <= 0.0) wind = -1.0;
        if (diffuw * diffdw > 0.0) {
          limiter = (1.0 - sigmam) * wind *
                    sycl::fmin(sycl::fmin(sycl::fabs(diffuw), sycl::fabs(diffdw)),
                               one_by_six * (sigma3 * sycl::fabs(diffuw) + sigma4 * sycl::fabs(diffdw)));
        } else {
          limiter = 0.0;
        }

        ener_flux[idx] = mass_flux_x[idx] * (energy1[donor][k] + limiter);
      });
    });
    clover::execute(queue, [&](handler &h) {
      auto density1 = density1_buffer.access<RW>(h);
      auto energy1 = energy1_buffer.access<RW>(h);
      auto mass_flux_x = mass_flux_x_buffer.access<R>(h);
      auto vol_flux_x = vol_flux_x_buffer.access<R>(h);
      auto pre_vol = pre_vol_buffer.access<R>(h);
      auto ener_flux = ener_flux_buffer.access<R>(h);
      // DO k=y_min,y_max
      //   DO j=x_min,x_max

      clover::par_ranged<class advec_cell_xdir_d1e1>(h, {x_min + 1, y_min + 1, x_max + 2, y_max + 2}, [=](id<2> idx) {
        double pre_mass_s = density1[idx] * pre_vol[idx];
        double post_mass_s = pre_mass_s + mass_flux_x[idx] - mass_flux_x[clover::offset(idx, 1, 0)];
        double post_ener_s = (energy1[idx] * pre_mass_s + ener_flux[idx] - ener_flux[clover::offset(idx, 1, 0)]) / post_mass_s;
        double advec_vol_s = pre_vol[idx] + vol_flux_x[idx] - vol_flux_x[clover::offset(idx, 1, 0)];
        density1[idx] = post_mass_s / advec_vol_s;
        energy1[idx] = post_ener_s;
      });
    });
  } else if (dir == g_ydir) {

    // DO k=y_min-2,y_max+2
    //   DO j=x_min-2,x_max+2
    clover::Range2d policy(x_min - 2 + 1, y_min - 2 + 1, x_max + 2 + 2, y_max + 2 + 2);

    if (sweep_number == 1) {
      clover::execute(queue, [&](handler &h) {
        auto volume = volume_buffer.access<R>(h);
        auto vol_flux_x = vol_flux_x_buffer.access<R>(h);
        auto vol_flux_y = vol_flux_y_buffer.access<R>(h);
        auto pre_vol = pre_vol_buffer.access<RW>(h);
        auto post_vol = post_vol_buffer.access<W>(h);
        clover::par_ranged<class APPEND_LN(advec_cell_ydir_s1)>(h, policy, [=](id<2> idx) {
          pre_vol[idx] = volume[idx] + (vol_flux_y[clover::offset(idx, 0, 1)] - vol_flux_y[idx] + vol_flux_x[clover::offset(idx, 1, 0)] -
                                        vol_flux_x[idx]);
          post_vol[idx] = pre_vol[idx] - (vol_flux_y[clover::offset(idx, 0, 1)] - vol_flux_y[idx]);
        });
      });
    } else {
      clover::execute(queue, [&](handler &h) {
        auto volume = volume_buffer.access<R>(h);
        auto vol_flux_y = vol_flux_y_buffer.access<R>(h);
        auto pre_vol = pre_vol_buffer.access<RW>(h);
        auto post_vol = post_vol_buffer.access<W>(h);
        clover::par_ranged<class APPEND_LN(advec_cell_ydir_s1)>(h, policy, [=](id<2> idx) {
          pre_vol[idx] = volume[idx] + vol_flux_y[clover::offset(idx, 0, 1)] - vol_flux_y[idx];
          post_vol[idx] = volume[idx];
        });
      });
    }

    clover::execute(queue, [&](handler &h) {
      auto vertexdy = vertexdy_buffer.access<R>(h);
      auto density1 = density1_buffer.access<R>(h);
      auto energy1 = energy1_buffer.access<R>(h);
      auto mass_flux_y = mass_flux_y_buffer.access<RW>(h);
      auto vol_flux_y = vol_flux_y_buffer.access<R>(h);
      auto pre_vol = pre_vol_buffer.access<R>(h);
      auto ener_flux = ener_flux_buffer.access<RW>(h);
      // DO k=y_min,y_max+2
      //   DO j=x_min,x_max
      clover::par_ranged<class advec_cell_ydir_ener_flux>(h, {x_min + 1, y_min + 1, x_max + 2, y_max + 2 + 2}, [=](id<2> idx) {
        int upwind, donor, downwind, dif;
        double sigmat, sigma3, sigma4, sigmav, sigmam, diffuw, diffdw, limiter, wind;

        const int j = idx.get(0);
        const int k = idx.get(1);

        if (vol_flux_y[idx] > 0.0) {
          upwind = k - 2;
          donor = k - 1;
          downwind = k;
          dif = donor;
        } else {
          upwind = sycl::min(k + 1, y_max + 2);
          donor = k;
          downwind = k - 1;
          dif = upwind;
        }

        sigmat = sycl::fabs(vol_flux_y[idx]) / pre_vol[j][donor];
        sigma3 = (1.0 + sigmat) * (vertexdy[k] / vertexdy[dif]);
        sigma4 = 2.0 - sigmat;

        sigmav = sigmat;

        diffuw = density1[j][donor] - density1[j][upwind];
        diffdw = density1[j][downwind] - density1[j][donor];
        wind = 1.0;
        if (diffdw <= 0.0) wind = -1.0;
        if (diffuw * diffdw > 0.0) {
          limiter = (1.0 - sigmav) * wind *
                    sycl::fmin(sycl::fmin(sycl::fabs(diffuw), sycl::fabs(diffdw)),
                               one_by_six * (sigma3 * sycl::fabs(diffuw) + sigma4 * sycl::fabs(diffdw)));
        } else {
          limiter = 0.0;
        }
        mass_flux_y[idx] = vol_flux_y[idx] * (density1[j][donor] + limiter);

        sigmam = sycl::fabs(mass_flux_y[idx]) / (density1[j][donor] * pre_vol[j][donor]);
        diffuw = energy1[j][donor] - energy1[j][upwind];
        diffdw = energy1[j][downwind] - energy1[j][donor];
        wind = 1.0;
        if (diffdw <= 0.0) wind = -1.0;
        if (diffuw * diffdw > 0.0) {
          limiter = (1.0 - sigmam) * wind *
                    sycl::fmin(sycl::fmin(sycl::fabs(diffuw), sycl::fabs(diffdw)),
                               one_by_six * (sigma3 * sycl::fabs(diffuw) + sigma4 * sycl::fabs(diffdw)));
        } else {
          limiter = 0.0;
        }
        ener_flux[idx] = mass_flux_y[idx] * (energy1[j][donor] + limiter);
      });
    });

    clover::execute(queue, [&](handler &h) {
      auto density1 = density1_buffer.access<RW>(h);
      auto energy1 = energy1_buffer.access<RW>(h);
      auto mass_flux_y = mass_flux_y_buffer.access<R>(h);
      auto vol_flux_y = vol_flux_y_buffer.access<R>(h);
      auto pre_vol = pre_vol_buffer.access<R>(h);
      auto ener_flux = ener_flux_buffer.access<R>(h);
      // DO k=y_min,y_max
      //   DO j=x_min,x_max
      clover::par_ranged<class advec_cell_ydir_e1d1>(h, {x_min + 1, y_min + 1, x_max + 2, y_max + 2}, [=](id<2> idx) {
        double pre_mass_s = density1[idx] * pre_vol[idx];
        double post_mass_s = pre_mass_s + mass_flux_y[idx] - mass_flux_y[clover::offset(idx, 0, 1)];
        double post_ener_s = (energy1[idx] * pre_mass_s + ener_flux[idx] - ener_flux[clover::offset(idx, 0, 1)]) / post_mass_s;
        double advec_vol_s = pre_vol[idx] + vol_flux_y[idx] - vol_flux_y[clover::offset(idx, 0, 1)];
        density1[idx] = post_mass_s / advec_vol_s;
        energy1[idx] = post_ener_s;
      });
    });
  }
}

//  @brief Cell centred advection driver.
//  @author Wayne Gaudin
//  @details Invokes the user selected advection kernel.
void advec_cell_driver(global_variables &globals, int tile, int sweep_number, int direction) {

  tile_type &t = globals.chunk.tiles[tile];
  advec_cell_kernel(globals.context.queue, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, direction, sweep_number,
                    t.field.vertexdx, t.field.vertexdy, t.field.volume, t.field.density1, t.field.energy1, t.field.mass_flux_x,
                    t.field.vol_flux_x, t.field.mass_flux_y, t.field.vol_flux_y, t.field.work_array1, t.field.work_array2,
                    t.field.work_array3, t.field.work_array4, t.field.work_array5, t.field.work_array6, t.field.work_array7);
}
