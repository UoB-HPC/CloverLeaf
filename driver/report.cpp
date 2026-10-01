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

//  @brief Controls error reporting
//  @author Wayne Gaudin
//  @details Outputs error messages and aborts the calculation.

#include <cmath>
#include <iomanip>
#include <iostream>

#include "comms.h"
#include "report.h"

void report_error(char *location, char *error) {

  std::cerr << std::endl
            << " Error from " << location << ":" << std::endl
            << error << std::endl
            << " CLOVER is terminating." << std::endl
            << std::endl;

  g_out << std::endl
        << "Error from " << location << ":" << std::endl
        << error << std::endl
        << "CLOVER is terminating." << std::endl
        << std::endl;

  clover_abort();
}

void clover_report_step_header(global_variables &globals, parallel_ &parallel) {
  if (parallel.boss) {
    g_out << std::endl
          << "Time " << globals.time << std::endl
          << "                "
          << "Volume          "
          << "Mass            "
          << "Density         "
          << "Pressure        "
          << "Internal Energy "
          << "Kinetic Energy  "
          << "Total Energy    " << std::endl;
  }
}

void clover_report_step(global_variables &globals, parallel_ &parallel, //
                        double vol, double mass, double ie, double ke, double press, double invalid) {
  if (parallel.boss) {
    if (globals.step == 0) {
      globals.initial_mass = mass;
      globals.initial_energy = ie + ke;
      globals.initial_volume = (globals.config.grid.xmax - globals.config.grid.xmin) *
                               (globals.config.grid.ymax - globals.config.grid.ymin);
    }

    constexpr double conservation_tolerance = 1.0e-6; // one PPM
    const bool finite = std::isfinite(vol) && std::isfinite(mass) && std::isfinite(ie) && std::isfinite(ke) &&
                        std::isfinite(ie + ke) && std::isfinite(press);
    const bool conserved = globals.initial_mass > 0.0 && globals.initial_volume > 0.0 && mass > 0.0 && vol > 0.0 &&
                           std::fabs(vol - globals.initial_volume) <= conservation_tolerance * globals.initial_volume;
    if (!finite || !conserved || invalid != 0.0 || ie < 0.0 || ke < 0.0 || press < 0.0) {
      if (!globals.report_invariant_fail) {
        for (auto *out : {&std::cout, &g_out}) {
          *out << " Invariant checks FAILED at step " << globals.step
               << ": finite summaries=" << finite << ", invalid cells=" << invalid
               << ", mass=" << mass << " (initial " << globals.initial_mass
               << "), volume=" << vol << " (initial " << globals.initial_volume << ")" << std::endl;
        }
      }
      globals.report_invariant_fail = true;
    }
    auto formatting = g_out.flags();
    g_out << " step: " << globals.step << std::scientific << std::setw(15) << vol << std::scientific << std::setw(15) << mass
          << std::scientific << std::setw(15) << mass / vol << std::scientific << std::setw(15) << press / vol << std::scientific
          << std::setw(15) << ie << std::scientific << std::setw(15) << ke << std::scientific << std::setw(15) << ie + ke << std::endl
          << std::endl;
    g_out.flags(formatting);
  }
  if (globals.complete) {
    if (parallel.boss) {
      for (auto *out : {&std::cout, &g_out}) {
        *out << " Invariant checks " << (globals.report_invariant_fail ? "FAILED" : "PASSED") << std::endl;
        *out << " Invalid cells at final summary: " << invalid << std::endl;
        if (globals.initial_mass > 0.0) {
          *out << " Relative mass change: " << (mass - globals.initial_mass) / globals.initial_mass << std::endl;
        }
        *out << " Total energy change: " << (ie + ke) - globals.initial_energy << std::endl;
        if (globals.initial_energy > 0.0 && std::isfinite(globals.initial_energy)) {
          *out << " Relative total energy change: " << ((ie + ke) - globals.initial_energy) / globals.initial_energy << std::endl;
        }
      }
      if (globals.config.test_problem != 0) {
        double qa_diff{};
        if (globals.config.test_problem == 1) {
          qa_diff = std::fabs((100.0 * (ke / 1.82280367310258)) - 100.0);
        } else if (globals.config.test_problem == 2) { // bm * 87
          qa_diff = std::fabs((100.0 * (ke / 1.19316898756307)) - 100.0);
        } else if (globals.config.test_problem == 3) { // bm * 2955
          qa_diff = std::fabs((100.0 * (ke / 2.58984003503994)) - 100.0);
        } else if (globals.config.test_problem == 4) { // bm16 @ 87
          qa_diff = std::fabs((100.0 * (ke / 0.307475452287895)) - 100.0);
        } else if (globals.config.test_problem == 5) { // bm15 * 2955
          qa_diff = std::fabs((100.0 * (ke / 4.85350315783719)) - 100.0);
        } else if (globals.config.test_problem == 487) { // bm4 @ 87
          qa_diff = std::fabs((100.0 * (ke / 6.088288e-01)) - 100.0);
        } else if (globals.config.test_problem == 287) { // bm2 @ 87
          qa_diff = std::fabs((100.0 * (ke / 6.062609e-01)) - 100.0);
        } else if (globals.config.test_problem == 168) { // bm16 @ 8
          qa_diff = std::fabs((100.0 * (ke / 2.465082e-02)) - 100.0);
        } else {
          qa_diff = 100;
          std::cout << " WARNING: Unknown test problem " << globals.config.test_problem << ", validation will fail" << std::endl;
          g_out << " WARNING: Unknown test problem " << globals.config.test_problem << ", validation will fail" << std::endl;
        }

        std::cout << " Test problem " << globals.config.test_problem << " is within " << qa_diff << "% of the expected solution"
                  << std::endl;
        g_out << "Test problem " << globals.config.test_problem << " is within " << qa_diff << "% of the expected solution" << std::endl;
        if (std::isfinite(qa_diff) && qa_diff < 0.001) {
          std::cout << " This test is considered PASSED" << std::endl;
          g_out << "This test is considered PASSED" << std::endl;
          globals.report_test_fail = false;
        } else {
          std::cout << " This test is considered NOT PASSED" << std::endl;
          g_out << "This test is considered NOT PASSED" << std::endl;
          globals.report_test_fail = true;
        }
      } else {
        std::cout << " Solution check SKIPPED: no test_problem specified" << std::endl;
        g_out << "Solution check SKIPPED: no test_problem specified" << std::endl;
      }
    }
    // Propagate the reference-check result so every MPI rank returns failure.
    int test_fail = (globals.report_test_fail || globals.report_invariant_fail) ? 1 : 0;
    clover_check_error(test_fail);
    globals.report_test_fail = test_fail != 0;
  }
}
