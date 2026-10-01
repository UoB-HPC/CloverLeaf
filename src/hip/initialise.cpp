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
#include <hip/hip_version.h>

#include "initialise.h"
#include "start.h"

static std::string device_arch(const hipDeviceProp_t &props) {
  // XXX gcnArchName was added in ROCm 3.6; gcnArch was removed in ROCm 6.0.
#if HIP_VERSION_MAJOR > 3 || (HIP_VERSION_MAJOR == 3 && HIP_VERSION_MINOR >= 6)
  return props.gcnArchName;
#else
  return "gfx" + std::to_string(props.gcnArch);
#endif
}

model create_context(bool silent, const std::vector<std::string> &args) {
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
                         device_arch(props) +                                              //
                         ")"};
  }
  auto [device, parsed] = list_and_parse<Device>(
      silent, devices, [](auto &d) { return d.name; }, args);
  clover::checkError(hipSetDevice(device.id));
  return model{clover::context{}, "HIP", true, parsed};
}

void report_context(const clover::context &) {
  int device = -1;
  clover::checkError(hipGetDevice(&device));
  hipDeviceProp_t props{};
  clover::checkError(hipGetDeviceProperties(&props, device));
  std::cout << " - Device: "
            << yaml_quote(std::string(props.name) + " (" + std::to_string(props.totalGlobalMem / 1024 / 1024) +
                          "MB;" + device_arch(props) + ")") << std::endl;
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
