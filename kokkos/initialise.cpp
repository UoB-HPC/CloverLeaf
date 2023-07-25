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

//  @brief Top level initialisation routine
//  @author Wayne Gaudin
//  @details Checks for the user input and either invokes the input reader or
//  switches to the internal test problem. It processes the input and strips
//  comments before writing a final input file.
//  It then calls the start routine.

#include "initialise.h"
#include "read_input.h"
#include "report.h"
#include "start.h"
#include "version.h"

#include <fstream>

std::pair<clover::context, run_args> create_context(bool silent, const std::vector<std::string> &args) {
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize();
  }
  auto [_, parsed] = list_and_parse<std::string>(
      silent, {typeid(Kokkos::DefaultExecutionSpace).name()}, [](auto &d) { return d; }, args);
  return {clover::context{}, parsed};
}

void report_context(const clover::context &) {
  std::cout << "Using Kokkos " << (KOKKOS_VERSION / 10000) << "." << (KOKKOS_VERSION / 100 % 100) << "." << (KOKKOS_VERSION % 100)
            << std::endl;
  std::cout << " - Backend: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
}
