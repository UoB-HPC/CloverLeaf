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

  std::cout << std::endl
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
                        double vol, double mass, double ie, double ke, double press) {
  if (parallel.boss) {
    auto formatting = g_out.flags();
    g_out << " step: " << globals.step << std::scientific << std::setw(15) << vol << std::scientific << std::setw(15) << mass
          << std::scientific << std::setw(15) << mass / vol << std::scientific << std::setw(15) << press / vol << std::scientific
          << std::setw(15) << ie << std::scientific << std::setw(15) << ke << std::scientific << std::setw(15) << ie + ke << std::endl
          << std::endl;
    g_out.flags(formatting);
  }
  if (globals.complete) {
    if (parallel.boss) {
      if (globals.config.test_problem >= 1) {
        double qa_diff{};
        if (globals.config.test_problem == 1) qa_diff = std::fabs((100.0 * (ke / 1.82280367310258)) - 100.0);
        if (globals.config.test_problem == 2) qa_diff = std::fabs((100.0 * (ke / 1.19316898756307)) - 100.0);
        if (globals.config.test_problem == 3) qa_diff = std::fabs((100.0 * (ke / 2.58984003503994)) - 100.0);
        if (globals.config.test_problem == 4) qa_diff = std::fabs((100.0 * (ke / 0.307475452287895)) - 100.0);
        if (globals.config.test_problem == 5) qa_diff = std::fabs((100.0 * (ke / 4.85350315783719)) - 100.0);
        std::cout << " Test problem " << globals.config.test_problem << " is within " << qa_diff << "% of the expected solution"
                  << std::endl;
        g_out << "Test problem " << globals.config.test_problem << " is within " << qa_diff << "% of the expected solution" << std::endl;
        if (!std::isnan(qa_diff) && qa_diff < 0.001) {
          std::cout << " This test is considered PASSED" << std::endl;
          g_out << "This test is considered PASSED" << std::endl;
          globals.report_test_fail = false;
        } else {
          std::cout << " This test is considered NOT PASSED" << std::endl;
          g_out << "This test is considered NOT PASSED" << std::endl;
          globals.report_test_fail = true;
        }
      }
    }
  }
}
