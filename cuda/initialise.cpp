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

model create_context(bool silent, const std::vector<std::string> &args) {
  struct Device {
    int id{};
    std::string name{};
  };
  int count = 0;
  clover::checkError(cudaGetDeviceCount(&count));
  std::vector<Device> devices(count);
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp props{};
    clover::checkError(cudaGetDeviceProperties(&props, i));
    devices[i] = {i, std::string(props.name) + " (" +                                        //
                         std::to_string(props.totalGlobalMem / 1024 / 1024) + "MB;" +        //
                         "sm_" + std::to_string(props.major) + std::to_string(props.minor) + //
                         ")"};
  }
  auto [device, parsed] = list_and_parse<Device>(
      silent, devices, [](auto &d) { return d.name; }, args);
  clover::checkError(cudaSetDevice(device.id));
  return model{clover::context{}, "CUDA", true, parsed};
}

void report_context(const clover::context &) {
  int device = -1;
  clover::checkError(cudaGetDevice(&device));
  cudaDeviceProp props{};
  clover::checkError(cudaGetDeviceProperties(&props, device));
  std::cout << " - Device: " //
            << props.name << " (" << (props.totalGlobalMem / 1024 / 1024) << "MB;"
            << "sm_" << props.major << props.minor << ")" << std::endl;
  std::cout << " - CUDA managed memory: "
            <<
#ifdef CLOVER_MANAGED_ALLOC
      "true"
#else
      "false"
#endif
            << std::endl;
  std::cout << " - CUDA per-kernel synchronisation: "
            <<
#ifdef CLOVER_SYNC_ALL_KERNELS
      "true"
#else
      "false"
#endif
            << std::endl;
}
