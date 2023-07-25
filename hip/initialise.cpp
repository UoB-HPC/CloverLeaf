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

#include <fstream>

#include "initialise.h"
#include "start.h"

std::pair<clover::context, run_args> create_context(bool silent, const std::vector<std::string> &args) {
  struct Device {
    int id{};
    std::string name{};
  };
  int count = 0;
  clover::checkError(hipGetDeviceCount(&count));
  std::vector<Device> devices(count);
  for (int i = 0; i < count; ++i) {
    hipDeviceProp_t props{};
    clover::checkError(hipGetDeviceProperties(&props, i));
    devices[i] = {i, std::string(props.name) + " (" +                                        //
                         std::to_string(props.totalGlobalMem / 1024 / 1024) + "MB;" +        //
                         "sm_" + std::to_string(props.major) + std::to_string(props.minor) + //
                         ")"};
  }
  auto [device, parsed] = list_and_parse<Device>(
      silent, devices, [](auto &d) { return d.name; }, args);
  clover::checkError(hipSetDevice(device.id));
  return {clover::context{}, parsed};
}

void report_context(const clover::context &) {
  int device = -1;
  clover::checkError(hipGetDevice(&device));
  hipDeviceProp_t props{};
  clover::checkError(hipGetDeviceProperties(&props, device));

  std::cout << "Using HIP:" << std::endl;
  std::cout << " - Device: " //
            << props.name << " (" << (props.totalGlobalMem / 1024 / 1024) << "MB;"
            << "gfx" << props.gcnArch << ")" << std::endl;
  std::cout << " - HIP managed memory: "
            <<
#ifdef CLOVER_MANAGED_ALLOC
      "true"
#else
      "false"
#endif
            << std::endl;
  std::cout << " - HIP per-kernel synchronisation: "
            <<
#ifdef CLOVER_SYNC_ALL_KERNELS
      "true"
#else
      "false"
#endif
            << std::endl;
}
