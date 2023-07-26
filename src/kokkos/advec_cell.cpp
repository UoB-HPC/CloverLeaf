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

//  @brief Fortran cell advection kernel.
//  @author Wayne Gaudin
//  @details Performs a second order advective remap using van-Leer limiting
//  with directional splitting.
void advec_cell_kernel(int x_min, int x_max, int y_min, int y_max, int dir, int sweep_number, Kokkos::View<double *> &vertexdx,
                       Kokkos::View<double *> &vertexdy, Kokkos::View<double **> &volume, Kokkos::View<double **> &density1,
                       Kokkos::View<double **> &energy1, Kokkos::View<double **> &mass_flux_x, Kokkos::View<double **> &vol_flux_x,
                       Kokkos::View<double **> &mass_flux_y, Kokkos::View<double **> &vol_flux_y, Kokkos::View<double **> &pre_vol,
                       Kokkos::View<double **> &post_vol, Kokkos::View<double **> &pre_mass, Kokkos::View<double **> &post_mass,
                       Kokkos::View<double **> &advec_vol, Kokkos::View<double **> &post_ener, Kokkos::View<double **> &ener_flux) {

  const double one_by_six = 1.0 / 6.0;

  if (dir == g_xdir) {

    // DO k=y_min-2,y_max+2
    //   DO j=x_min-2,x_max+2
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min - 2 + 1, y_min - 2 + 1}, {x_max + 2 + 2, y_max + 2 + 2});

    if (sweep_number == 1) {
      Kokkos::parallel_for(
          "advec_cell xdir sweep_number=1", policy, KOKKOS_LAMBDA(const int j, const int k) {
            pre_vol(j, k) = volume(j, k) + (vol_flux_x(j + 1, k) - vol_flux_x(j, k) + vol_flux_y(j, k + 1) - vol_flux_y(j, k));
            post_vol(j, k) = pre_vol(j, k) - (vol_flux_x(j + 1, k) - vol_flux_x(j, k));
          });
    } else {
      Kokkos::parallel_for(
          "advec_cell xdir sweep_number!=1", policy, KOKKOS_LAMBDA(const int j, const int k) {
            pre_vol(j, k) = volume(j, k) + vol_flux_x(j + 1, k) - vol_flux_x(j, k);
            post_vol(j, k) = volume(j, k);
          });
    }

    // DO k=y_min,y_max
    //   DO j=x_min,x_max+2
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy_x2({x_min + 1, y_min + 1}, {x_max + 2 + 2, y_max + 2});
    Kokkos::parallel_for(
        "advec_cell xdir ener_flux", policy_x2, KOKKOS_LAMBDA(const int j, const int k) {
          int upwind, donor, downwind, dif;
          double sigmat, sigma3, sigma4, sigmav, sigmam, diffuw, diffdw, limiter, wind;

          if (vol_flux_x(j, k) > 0.0) {
            upwind = j - 2;
            donor = j - 1;
            downwind = j;
            dif = donor;
          } else {
            upwind = Kokkos::min(j + 1, x_max + 2);
            donor = j;
            downwind = j - 1;
            dif = upwind;
          }

          sigmat = fabs(vol_flux_x(j, k)) / pre_vol(donor, k);
          sigma3 = (1.0 + sigmat) * (vertexdx(j) / vertexdx(dif));
          sigma4 = 2.0 - sigmat;

          sigmav = sigmat;

          diffuw = density1(donor, k) - density1(upwind, k);
          diffdw = density1(downwind, k) - density1(donor, k);
          wind = 1.0;
          if (diffdw <= 0.0) wind = -1.0;
          if (diffuw * diffdw > 0.0) {
            limiter = (1.0 - sigmav) * wind *
                      Kokkos::min(Kokkos::min(fabs(diffuw), fabs(diffdw)), one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
          } else {
            limiter = 0.0;
          }
          mass_flux_x(j, k) = vol_flux_x(j, k) * (density1(donor, k) + limiter);

          sigmam = fabs(mass_flux_x(j, k)) / (density1(donor, k) * pre_vol(donor, k));
          diffuw = energy1(donor, k) - energy1(upwind, k);
          diffdw = energy1(downwind, k) - energy1(donor, k);
          wind = 1.0;
          if (diffdw <= 0.0) wind = -1.0;
          if (diffuw * diffdw > 0.0) {
            limiter = (1.0 - sigmam) * wind *
                      Kokkos::min(Kokkos::min(fabs(diffuw), fabs(diffdw)), one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
          } else {
            limiter = 0.0;
          }

          ener_flux(j, k) = mass_flux_x(j, k) * (energy1(donor, k) + limiter);
        });

    // DO k=y_min,y_max
    //   DO j=x_min,x_max
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy_xy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});
    Kokkos::parallel_for(
        "advec_cell xdir density1,energy1", policy_xy, KOKKOS_LAMBDA(const int j, const int k) {
          double pre_mass_s = density1(j, k) * pre_vol(j, k);
          double post_mass_s = pre_mass_s + mass_flux_x(j, k) - mass_flux_x(j + 1, k);
          double post_ener_s = (energy1(j, k) * pre_mass_s + ener_flux(j, k) - ener_flux(j + 1, k)) / post_mass_s;
          double advec_vol_s = pre_vol(j, k) + vol_flux_x(j, k) - vol_flux_x(j + 1, k);
          density1(j, k) = post_mass_s / advec_vol_s;
          energy1(j, k) = post_ener_s;
        });
  }

  else if (dir == g_ydir) {

    // DO k=y_min-2,y_max+2
    //   DO j=x_min-2,x_max+2
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min - 2 + 1, y_min - 2 + 1}, {x_max + 2 + 2, y_max + 2 + 2});

    if (sweep_number == 1) {
      Kokkos::parallel_for(
          "advec_cell ydir sweep_number=1", policy, KOKKOS_LAMBDA(const int j, const int k) {
            pre_vol(j, k) = volume(j, k) + (vol_flux_y(j, k + 1) - vol_flux_y(j, k) + vol_flux_x(j + 1, k) - vol_flux_x(j, k));
            post_vol(j, k) = pre_vol(j, k) - (vol_flux_y(j, k + 1) - vol_flux_y(j, k));
          });
    } else {
      Kokkos::parallel_for(
          "advec_cell ydir sweep_number!=1", policy, KOKKOS_LAMBDA(const int j, const int k) {
            pre_vol(j, k) = volume(j, k) + vol_flux_y(j, k + 1) - vol_flux_y(j, k);
            post_vol(j, k) = volume(j, k);
          });
    }

    // DO k=y_min,y_max+2
    //   DO j=x_min,x_max
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy_y2({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2 + 2});
    Kokkos::parallel_for(
        "advec_cell ydir ener_flux", policy_y2, KOKKOS_LAMBDA(const int j, const int k) {
          int upwind, donor, downwind, dif;
          double sigmat, sigma3, sigma4, sigmav, sigmam, diffuw, diffdw, limiter, wind;

          if (vol_flux_y(j, k) > 0.0) {
            upwind = k - 2;
            donor = k - 1;
            downwind = k;
            dif = donor;
          } else {
            upwind = Kokkos::min(k + 1, y_max + 2);
            donor = k;
            downwind = k - 1;
            dif = upwind;
          }

          sigmat = fabs(vol_flux_y(j, k)) / pre_vol(j, donor);
          sigma3 = (1.0 + sigmat) * (vertexdy(k) / vertexdy(dif));
          sigma4 = 2.0 - sigmat;

          sigmav = sigmat;

          diffuw = density1(j, donor) - density1(j, upwind);
          diffdw = density1(j, downwind) - density1(j, donor);
          wind = 1.0;
          if (diffdw <= 0.0) wind = -1.0;
          if (diffuw * diffdw > 0.0) {
            limiter = (1.0 - sigmav) * wind *
                      Kokkos::min(Kokkos::min(fabs(diffuw), fabs(diffdw)), one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
          } else {
            limiter = 0.0;
          }
          mass_flux_y(j, k) = vol_flux_y(j, k) * (density1(j, donor) + limiter);

          sigmam = fabs(mass_flux_y(j, k)) / (density1(j, donor) * pre_vol(j, donor));
          diffuw = energy1(j, donor) - energy1(j, upwind);
          diffdw = energy1(j, downwind) - energy1(j, donor);
          wind = 1.0;
          if (diffdw <= 0.0) wind = -1.0;
          if (diffuw * diffdw > 0.0) {
            limiter = (1.0 - sigmam) * wind *
                      Kokkos::min(Kokkos::min(fabs(diffuw), fabs(diffdw)), one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
          } else {
            limiter = 0.0;
          }
          ener_flux(j, k) = mass_flux_y(j, k) * (energy1(j, donor) + limiter);
        });

    // DO k=y_min,y_max
    //   DO j=x_min,x_max
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy_xy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});
    Kokkos::parallel_for(
        "advec_cell ydir density1,energy1", policy_xy, KOKKOS_LAMBDA(const int j, const int k) {
          double pre_mass_s = density1(j, k) * pre_vol(j, k);
          double post_mass_s = pre_mass_s + mass_flux_y(j, k) - mass_flux_y(j, k + 1);
          double post_ener_s = (energy1(j, k) * pre_mass_s + ener_flux(j, k) - ener_flux(j, k + 1)) / post_mass_s;
          double advec_vol_s = pre_vol(j, k) + vol_flux_y(j, k) - vol_flux_y(j, k + 1);
          density1(j, k) = post_mass_s / advec_vol_s;
          energy1(j, k) = post_ener_s;
        });
  }
}

//  @brief Cell centred advection driver.
//  @author Wayne Gaudin
//  @details Invokes the user selected advection kernel.
void advec_cell_driver(global_variables &globals, int tile, int sweep_number, int direction) {

  advec_cell_kernel(globals.chunk.tiles[tile].info.t_xmin, globals.chunk.tiles[tile].info.t_xmax, globals.chunk.tiles[tile].info.t_ymin,
                    globals.chunk.tiles[tile].info.t_ymax, direction, sweep_number, globals.chunk.tiles[tile].field.vertexdx.view,
                    globals.chunk.tiles[tile].field.vertexdy.view, globals.chunk.tiles[tile].field.volume.view,
                    globals.chunk.tiles[tile].field.density1.view, globals.chunk.tiles[tile].field.energy1.view,
                    globals.chunk.tiles[tile].field.mass_flux_x.view, globals.chunk.tiles[tile].field.vol_flux_x.view,
                    globals.chunk.tiles[tile].field.mass_flux_y.view, globals.chunk.tiles[tile].field.vol_flux_y.view,
                    globals.chunk.tiles[tile].field.work_array1.view, globals.chunk.tiles[tile].field.work_array2.view,
                    globals.chunk.tiles[tile].field.work_array3.view, globals.chunk.tiles[tile].field.work_array4.view,
                    globals.chunk.tiles[tile].field.work_array5.view, globals.chunk.tiles[tile].field.work_array6.view,
                    globals.chunk.tiles[tile].field.work_array7.view);
}
