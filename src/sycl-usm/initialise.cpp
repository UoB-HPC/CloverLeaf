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
#include <sstream>
#include <string>

model create_context(bool silent, const std::vector<std::string> &args) {
  const auto &devices = sycl::device::get_devices();
  auto [device, parsed] = list_and_parse<sycl::device>(
      silent, devices, [](const auto &d) { return d.template get_info<sycl::info::device::name>(); }, args);
  auto handler = [](const sycl::exception_list &exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cout << "[SYCL] Async exception:\n" << e.what() << std::endl;
      }
    }
  };
  auto offload = device.get_info<sycl::info::device::device_type>() == sycl::info::device_type::gpu;
  return {clover::context{sycl::queue(device, handler, {})}, "SYCL (accessors)", offload, parsed};
}

static std::string deviceName(sycl::info::device_type type) {
  switch (type) {
    case sycl::info::device_type::cpu: return "cpu";
    case sycl::info::device_type::gpu: return "gpu";
    case sycl::info::device_type::accelerator: return "accelerator";
    case sycl::info::device_type::custom: return "custom";
    case sycl::info::device_type::automatic: return "automatic";
    case sycl::info::device_type::host: return "host";
    case sycl::info::device_type::all: return "all";
    default: return "(unknown: " + std::to_string(static_cast<unsigned int>(type)) + ")";
  }
}

void report_context(const clover::context &ctx) {
#if RANGE2D_MODE == RANGE2D_NORMAL
  std::cout << " - Indexing: RANGE2D_NORMAL" << std::endl;
#elif RANGE2D_MODE == RANGE2D_LINEAR
  std::cout << " - Indexing: RANGE2D_LINEAR" << std::endl;
#elif RANGE2D_MODE == RANGE2D_ROUND
  std::cout << " - Indexing: RANGE2D_ROUND" << std::endl;
#else
  #error "Unsupported RANGE2D_MODE"
#endif
  std::cout << " - SYCL device: " << ctx.queue.get_device().get_info<sycl::info::device::name>() << "\n"
            << "   - Type    : " << deviceName(ctx.queue.get_device().get_info<sycl::info::device::device_type>()) << "\n"
            << "   - Version : " << ctx.queue.get_device().get_info<sycl::info::device::version>() << "\n"
            << "   - Vendor  : " << ctx.queue.get_device().get_info<sycl::info::device::vendor>() << "\n"
            << "   - Driver  : " << ctx.queue.get_device().get_info<sycl::info::device::driver_version>() << std::endl;
}
