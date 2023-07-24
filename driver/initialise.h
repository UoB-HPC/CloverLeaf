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
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

template <typename T> struct run_args {
  std::string file;
  T device;
};

template <typename T>
run_args<T> list_and_parse(const std::vector<T> &devices,                           //
                       const std::function<std::string(const T &)> &deviceName, //
                       const std::vector<std::string> &args                     //
) {

  const auto printHelp = [](const std::string &name) {
    std::cout << std::endl;
    std::cout
        << "Usage: " << name << " [OPTIONS]\n\n"
        << "Options:\n"
        << "  -h  --help                  Print the message\n"
        << "      --list                  List available devices with index and exit\n"
        << "      --device <INDEX|NAME>   Select device at INDEX from output of --list or a substring match iff INDEX is not an integer\n"
        << "      --file   <FILE>         Custom clover.in file FILE (defaults to clover.in if unspecified)\n"
        << std::endl;
  };

  const auto readParam = [&](size_t current, const std::string &emptyMessage, auto map) {
    if (current + 1 < args.size()) {
      return map(args[current + 1]);
    } else {
      std::cerr << emptyMessage << std::endl;
      printHelp(args[0]);
      std::exit(EXIT_FAILURE);
    }
  };

  const auto listAll = [&](){
    for (size_t j = 0; j < devices.size(); ++j)
      std::cout << std::setw(3) << j << ". " << deviceName(devices[j]) << std::endl;
  };

  auto config = run_args<T>{"clover.in", std::move(devices[0])};
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &arg = args[i];

    if (arg == "--help" || arg == "-h") {
      printHelp(args[0]);
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--list") {
      listAll();
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--device") {
      listAll();
      readParam(i, "--device specified but no size was given", [&](const auto &param) {
        try {
          config.device = devices.at(std::stoul(param));
        } catch (const std::exception &e) {
          std::cout << "Unable to parse/select device index `" << param << "`:" << e.what() << std::endl;
          std::cout << "Attempting to match device with substring  `" << param << "`" << std::endl;
          auto matching = std::find_if(devices.begin(), devices.end(),
                                       [&](const T &device) { return deviceName(device).find(param) != std::string::npos; });
          if (matching != devices.end()) {
            config.device = *matching;
            std::cout << "Using first device matching substring `" << param << "`" << std::endl;
          } else if (devices.size() == 1)
            std::cerr << "No matching device but there's only one device, will be using that anyway" << std::endl;
          else {
            std::cerr << "No matching devices" << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }
      });
    } else if (arg == "--file") {
      readParam(i, "--file specified but no file was given", [&config](const auto &param) { config.file = param; });
    }
  }
  return config;
}

std::pair<clover::context, std::string> create_context(const std::vector<std::string> &args);

void report_context(const clover::context &ctx);

// std::unique_ptr<global_variables> initialise(parallel_ &parallel, const std::vector<std::string> &args);
