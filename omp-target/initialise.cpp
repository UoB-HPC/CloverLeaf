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

#include <algorithm>
#include <omp.h>
#include <sstream>
#include <string>

std::pair<clover::context, std::string> create_context(const std::vector<std::string> &args) {

  using device = std::pair<std::string, int>;
  auto num_devices = omp_get_num_devices();
  std::vector<device> devices(num_devices);
  for (size_t i = 0; i < devices.size(); ++i)
    devices[i] = {"target device (target:true) #" + std::to_string(i), i};

#ifdef USE_COND_TARGET
  devices.emplace_back("host device (target:false)", -1);
#endif

  auto parsed = list_and_parse<device>(
      devices, [](const auto &d) { return d.first; }, args);

  if (parsed.device.second != -1) {
    omp_set_default_device(parsed.device.second);
  }

  return {clover::context{.use_target = parsed.device.second != -1}, parsed.file};
}

void report_context(const clover::context &ctx) {
  std::cout << "Using OpenMP (target:" << (ctx.use_target ? "true" : "false") << ", #" << omp_get_default_device() << ")" << std::endl;
}
