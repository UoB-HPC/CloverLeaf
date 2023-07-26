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

//  @brief Fortran momentum advection kernel
//  @author Wayne Gaudin
//  @details Performs a second order advective remap on the vertex momentum
//  using van-Leer limiting and directional splitting.
//  Note that although pre_vol is only set and not used in the update, please
//  leave it in the method.
void advec_mom_kernel(int x_min, int x_max, int y_min, int y_max, Kokkos::View<double **> &vel1, Kokkos::View<double **> &mass_flux_x,
                      Kokkos::View<double **> &vol_flux_x, Kokkos::View<double **> &mass_flux_y, Kokkos::View<double **> &vol_flux_y,
                      Kokkos::View<double **> &volume, Kokkos::View<double **> &density1, Kokkos::View<double **> &node_flux,
                      Kokkos::View<double **> &node_mass_post, Kokkos::View<double **> &node_mass_pre, Kokkos::View<double **> &mom_flux,
                      Kokkos::View<double **> &pre_vol, Kokkos::View<double **> &post_vol, Kokkos::View<double *> &celldx,
                      Kokkos::View<double *> &celldy, int which_vel, int sweep_number, int direction) {

  int mom_sweep = direction + 2 * (sweep_number - 1);

  // DO k=y_min-2,y_max+2
  //   DO j=x_min-2,x_max+2
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min - 2 + 1, y_min - 2 + 1}, {x_max + 2 + 2, y_max + 2 + 2});

  if (mom_sweep == 1) { // x 1
    Kokkos::parallel_for(
        "advec_mom x1", policy, KOKKOS_LAMBDA(const int j, const int k) {
          post_vol(j, k) = volume(j, k) + vol_flux_y(j, k + 1) - vol_flux_y(j, k);
          pre_vol(j, k) = post_vol(j, k) + vol_flux_x(j + 1, k) - vol_flux_x(j, k);
        });
  } else if (mom_sweep == 2) { // y 1
    Kokkos::parallel_for(
        "advec_mom y1", policy, KOKKOS_LAMBDA(const int j, const int k) {
          post_vol(j, k) = volume(j, k) + vol_flux_x(j + 1, k) - vol_flux_x(j, k);
          pre_vol(j, k) = post_vol(j, k) + vol_flux_y(j, k + 1) - vol_flux_y(j, k);
        });
  } else if (mom_sweep == 3) { // x 2
    Kokkos::parallel_for(
        "advec_mom x1", policy, KOKKOS_LAMBDA(const int j, const int k) {
          post_vol(j, k) = volume(j, k);
          pre_vol(j, k) = post_vol(j, k) + vol_flux_y(j, k + 1) - vol_flux_y(j, k);
        });
  } else if (mom_sweep == 4) { // y 2
    Kokkos::parallel_for(
        "advec_mom y1", policy, KOKKOS_LAMBDA(const int j, const int k) {
          post_vol(j, k) = volume(j, k);
          pre_vol(j, k) = post_vol(j, k) + vol_flux_x(j + 1, k) - vol_flux_x(j, k);
        });
  }

  if (direction == 1) {
    if (which_vel == 1) {
      // DO k=y_min,y_max+1
      //   DO j=x_min-2,x_max+2
      Kokkos::parallel_for(
          "advec_mom dir1, vel1, node_flux",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({x_min - 2 + 1, y_min + 1}, {x_max + 2 + 2, y_max + 1 + 2}),
          KOKKOS_LAMBDA(const int j, const int k) {
            // Find staggered mesh mass fluxes, nodal masses and volumes.
            node_flux(j, k) = 0.25 * (mass_flux_x(j, k - 1) + mass_flux_x(j, k) + mass_flux_x(j + 1, k - 1) + mass_flux_x(j + 1, k));
          });

      // DO k=y_min,y_max+1
      //   DO j=x_min-1,x_max+2
      Kokkos::parallel_for(
          "advec_mom dir1, vel1, node_mass_pre",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({x_min - 1 + 1, y_min + 1}, {x_max + 2 + 2, y_max + 1 + 2}),
          KOKKOS_LAMBDA(const int j, const int k) {
            // Staggered cell mass post advection
            node_mass_post(j, k) = 0.25 * (density1(j, k - 1) * post_vol(j, k - 1) + density1(j, k) * post_vol(j, k) +
                                           density1(j - 1, k - 1) * post_vol(j - 1, k - 1) + density1(j - 1, k) * post_vol(j - 1, k));
            node_mass_pre(j, k) = node_mass_post(j, k) - node_flux(j - 1, k) + node_flux(j, k);
          });
    }

    // DO k=y_min,y_max+1
    //  DO j=x_min-1,x_max+1
    Kokkos::parallel_for(
        "advec_mom dir1, mom_flux", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({x_min - 1 + 1, y_min + 1}, {x_max + 1 + 2, y_max + 1 + 2}),
        KOKKOS_LAMBDA(const int j, const int k) {
          int upwind, donor, downwind, dif;
          double sigma, width, limiter, vdiffuw, vdiffdw, auw, adw, wind, advec_vel_s;

          if (node_flux(j, k) < 0.0) {
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

          sigma = fabs(node_flux(j, k)) / (node_mass_pre(donor, k));
          width = celldx(j);
          vdiffuw = vel1(donor, k) - vel1(upwind, k);
          vdiffdw = vel1(downwind, k) - vel1(donor, k);
          limiter = 0.0;
          if (vdiffuw * vdiffdw > 0.0) {
            auw = fabs(vdiffuw);
            adw = fabs(vdiffdw);
            wind = 1.0;
            if (vdiffdw <= 0.0) wind = -1.0;
            limiter =
                wind * Kokkos::min(Kokkos::min(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / celldx(dif)) / 6.0, auw), adw);
          }
          advec_vel_s = vel1(donor, k) + (1.0 - sigma) * limiter;
          mom_flux(j, k) = advec_vel_s * node_flux(j, k);
        });

    // DO k=y_min,y_max+1
    //   DO j=x_min,x_max+1
    Kokkos::parallel_for(
        "advec_mom dir1, vel1", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({x_min + 1, y_min + 1}, {x_max + 1 + 2, y_max + 1 + 2}),
        KOKKOS_LAMBDA(const int j, const int k) {
          vel1(j, k) = (vel1(j, k) * node_mass_pre(j, k) + mom_flux(j - 1, k) - mom_flux(j, k)) / node_mass_post(j, k);
        });
  } else if (direction == 2) {
    if (which_vel == 1) {
      // DO k=y_min-2,y_max+2
      //   DO j=x_min,x_max+1
      Kokkos::parallel_for(
          "advec_mom dir2, vel1, node_flux",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({x_min + 1, y_min - 2 + 1}, {x_max + 1 + 2, y_max + 2 + 2}),
          KOKKOS_LAMBDA(const int j, const int k) {
            // Find staggered mesh mass fluxes and nodal masses and volumes.
            node_flux(j, k) = 0.25 * (mass_flux_y(j - 1, k) + mass_flux_y(j, k) + mass_flux_y(j - 1, k + 1) + mass_flux_y(j, k + 1));
          });

      // DO k=y_min-1,y_max+2
      //   DO j=x_min,x_max+1
      Kokkos::parallel_for(
          "advec_mom dir2, vel1, node_mass_pre",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({x_min + 1, y_min - 1 + 1}, {x_max + 1 + 2, y_max + 2 + 2}),
          KOKKOS_LAMBDA(const int j, const int k) {
            node_mass_post(j, k) = 0.25 * (density1(j, k - 1) * post_vol(j, k - 1) + density1(j, k) * post_vol(j, k) +
                                           density1(j - 1, k - 1) * post_vol(j - 1, k - 1) + density1(j - 1, k) * post_vol(j - 1, k));
            node_mass_pre(j, k) = node_mass_post(j, k) - node_flux(j, k - 1) + node_flux(j, k);
          });
    }

    // DO k=y_min-1,y_max+1
    //   DO j=x_min,x_max+1
    Kokkos::parallel_for(
        "advec_mom dir2, mom_flux", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({x_min + 1, y_min - 1 + 1}, {x_max + 1 + 2, y_max + 1 + 2}),
        KOKKOS_LAMBDA(const int j, const int k) {
          int upwind, donor, downwind, dif;
          double sigma, width, limiter, vdiffuw, vdiffdw, auw, adw, wind, advec_vel_s;

          if (node_flux(j, k) < 0.0) {
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

          sigma = fabs(node_flux(j, k)) / (node_mass_pre(j, donor));
          width = celldy(k);
          vdiffuw = vel1(j, donor) - vel1(j, upwind);
          vdiffdw = vel1(j, downwind) - vel1(j, donor);
          limiter = 0.0;
          if (vdiffuw * vdiffdw > 0.0) {
            auw = fabs(vdiffuw);
            adw = fabs(vdiffdw);
            wind = 1.0;
            if (vdiffdw <= 0.0) wind = -1.0;
            limiter =
                wind * Kokkos::min(Kokkos::min(width * ((2.0 - sigma) * adw / width + (1.0 + sigma) * auw / celldy(dif)) / 6.0, auw), adw);
          }
          advec_vel_s = vel1(j, donor) + (1.0 - sigma) * limiter;
          mom_flux(j, k) = advec_vel_s * node_flux(j, k);
        });

    // DO k=y_min,y_max+1
    //   DO j=x_min,x_max+1
    Kokkos::parallel_for(
        "advec_mom dir2, vel1", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({x_min + 1, y_min + 1}, {x_max + 1 + 2, y_max + 1 + 2}),
        KOKKOS_LAMBDA(const int j, const int k) {
          vel1(j, k) = (vel1(j, k) * node_mass_pre(j, k) + mom_flux(j, k - 1) - mom_flux(j, k)) / node_mass_post(j, k);
        });
  }
}

//  @brief Momentum advection driver
//  @author Wayne Gaudin
//  @details Invokes the user specified momentum advection kernel.
void advec_mom_driver(global_variables &globals, int tile, int which_vel, int direction, int sweep_number) {

  if (which_vel == 1) {
    advec_mom_kernel(globals.chunk.tiles[tile].info.t_xmin, globals.chunk.tiles[tile].info.t_xmax, globals.chunk.tiles[tile].info.t_ymin,
                     globals.chunk.tiles[tile].info.t_ymax, globals.chunk.tiles[tile].field.xvel1.view,
                     globals.chunk.tiles[tile].field.mass_flux_x.view, globals.chunk.tiles[tile].field.vol_flux_x.view,
                     globals.chunk.tiles[tile].field.mass_flux_y.view, globals.chunk.tiles[tile].field.vol_flux_y.view,
                     globals.chunk.tiles[tile].field.volume.view, globals.chunk.tiles[tile].field.density1.view,
                     globals.chunk.tiles[tile].field.work_array1.view, globals.chunk.tiles[tile].field.work_array2.view,
                     globals.chunk.tiles[tile].field.work_array3.view, globals.chunk.tiles[tile].field.work_array4.view,
                     globals.chunk.tiles[tile].field.work_array5.view, globals.chunk.tiles[tile].field.work_array6.view,
                     globals.chunk.tiles[tile].field.celldx.view, globals.chunk.tiles[tile].field.celldy.view, which_vel, sweep_number,
                     direction);
  } else {
    advec_mom_kernel(globals.chunk.tiles[tile].info.t_xmin, globals.chunk.tiles[tile].info.t_xmax, globals.chunk.tiles[tile].info.t_ymin,
                     globals.chunk.tiles[tile].info.t_ymax, globals.chunk.tiles[tile].field.yvel1.view,
                     globals.chunk.tiles[tile].field.mass_flux_x.view, globals.chunk.tiles[tile].field.vol_flux_x.view,
                     globals.chunk.tiles[tile].field.mass_flux_y.view, globals.chunk.tiles[tile].field.vol_flux_y.view,
                     globals.chunk.tiles[tile].field.volume.view, globals.chunk.tiles[tile].field.density1.view,
                     globals.chunk.tiles[tile].field.work_array1.view, globals.chunk.tiles[tile].field.work_array2.view,
                     globals.chunk.tiles[tile].field.work_array3.view, globals.chunk.tiles[tile].field.work_array4.view,
                     globals.chunk.tiles[tile].field.work_array5.view, globals.chunk.tiles[tile].field.work_array6.view,
                     globals.chunk.tiles[tile].field.celldx.view, globals.chunk.tiles[tile].field.celldy.view, which_vel, sweep_number,
                     direction);
  }
}
