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

#pragma once

#include "comms.h"
#include "definitions.h"
#include <functional>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct run_args {
  enum class staging_buffer { enabled, disable, automatic };
  std::string dumpDir;
  std::string inFile;
  std::string outFile;
  staging_buffer staging_buffer;
  std::optional<bool> profile;
};

struct model {
  clover::context context;
  std::string name;
  bool offload;
  run_args args;
};

template <typename T>
std::pair<T, run_args> list_and_parse(bool silent, const std::vector<T> &devices,              //
                                      const std::function<std::string(const T &)> &deviceName, //
                                      const std::vector<std::string> &args                     //
) {

  const auto printHelp = [&]() {
    if (silent) return;
    std::cout << std::endl;
    std::cout //
        << "Usage: cloverleaf [OPTIONS]\n\n"
        << "Options:\n"
        << "  -h  --help                             Print this message\n"
        << "      --list                             List available devices with index and exit\n"
        << "      --device           <INDEX|NAME>    Use device at INDEX from output of --list or substring match iff INDEX is not an id\n"
        << "      --file,--in              <FILE>    Custom clover.in file FILE (defaults to clover.in if unspecified)\n"
        << "      --out                    <FILE>    Custom clover.out file FILE (defaults to clover.out if unspecified)\n"
        << "      --dump                    <DIR>    Dumps all field data in ASCII to ./DIR for debugging, DIR is created if missing\n"
        << "      --profile                          Enables kernel profiling, this takes precedence over the profiler_on in clover.in\n"
        << "      --staging-buffer <true|false|auto> If true, use a host staging buffer for device-host MPI halo exchange.\n"
           "                                         If false, use device pointers directly for MPI halo exchange.\n"
        << "                                         Defaults to auto which elides the buffer if a device-aware (i.e CUDA-aware) is used.\n"
        << "                                         This option is no-op for CPU-only models.\n"
        << "                                         Setting this to false on an MPI that is not device-aware may cause a segfault.\n"
        << std::endl;
  };

  const auto readParam = [&](size_t &current, const std::string &emptyMessage, auto map) {
    if (current + 1 < args.size()) {
        map(args[current + 1]);
        current++;
    } else {
      std::cerr << emptyMessage << std::endl;
      printHelp();
      std::exit(EXIT_FAILURE);
    }
  };

  const auto listAll = [&]() {
    if (silent) return;
    std::cout << "Devices:" << std::endl;
    for (size_t j = 0; j < devices.size(); ++j)
      std::cout << " " << j << ": " << deviceName(devices[j]) << std::endl;
  };

  T device = std::move(devices[0]);
  auto config = run_args{"", "clover.in", "clover.out", run_args::staging_buffer::automatic, {}};
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];
    if (arg == "--help" || arg == "-h") {
      printHelp();
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--dump") {
      readParam(i, "--dump specified but no dir was given", [&config](const auto &param) { config.dumpDir = param; });
    } else if (arg == "--list") {
      listAll();
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--profile") {
      config.profile = true;
    } else if (arg == "--device") {
      readParam(i, "--device specified but no size was given", [&](const auto &param) {
        try {
          device = devices.at(std::stoul(param));
        } catch (const std::exception &e) {
          if (!silent) {
            std::cout << "# Unable to parse/select device index `" << param << "`:" << e.what() << std::endl;
            std::cout << "# Attempting to match device with substring  `" << param << "`" << std::endl;
          }

          auto matching = std::find_if(devices.begin(), devices.end(),
                                       [&](const T &device) { return deviceName(device).find(param) != std::string::npos; });
          if (matching != devices.end()) {
            device = *matching;
            if (!silent) {
              std::cout << "# Using first device matching substring `" << param << "`" << std::endl;
            }
          } else if (devices.size() == 1)
            std::cerr << "# No matching device but there's only one device, will be using that anyway" << std::endl;
          else {
            std::cerr << "No matching devices, all devices:" << std::endl;
            listAll();
            std::exit(EXIT_FAILURE);
          }
        }
      });
    } else if (arg == "--in" || arg == "--file") {
      readParam(i, "--in,--file specified but no path was given", [&config](const auto &param) { config.inFile = param; });
    } else if (arg == "--out") {
      readParam(i, "--out specified but no path was given", [&config](const auto &param) { config.outFile = param; });
    } else if (arg == "--staging-buffer") {
      readParam(i, "--staging-buffer specified but no option given, expecting <true|false|auto>", [&config](const auto &param) {
        if (param == "true") {
          config.staging_buffer = run_args::staging_buffer::enabled;
        } else if (param == "false") {
          config.staging_buffer = run_args::staging_buffer::disable;
        } else if (param == "auto") {
          config.staging_buffer = run_args::staging_buffer::automatic;
        } else {
          std::cerr << "Illegal --staging-buffer option:" << param << std::endl;
          std::exit(EXIT_FAILURE);
        }
      });
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      printHelp();
      std::exit(EXIT_FAILURE);
    }
  }
  listAll();
  return {device, config};
}

model create_context(bool silent, const std::vector<std::string> &args);

void report_context(const clover::context &ctx);

// std::unique_ptr<global_variables> initialise(parallel_ &parallel, const std::vector<std::string> &args);
