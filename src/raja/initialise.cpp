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

#include "context.h"
#include "initialise.h"
#include "start.h"

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include "umpire/TypedAllocator.hpp"

std::string clover::CLResourceManager::umpireResources[clover::CLResourceManager::UmpireResourceType::RSEND] = {"HOST", "DEVICE", "UM"};
std::string clover::CLResourceManager::CloverLeafPools[clover::CLResourceManager::UmpireResourceType::RSEND] = {"CloverLeaf-HOST", "CloverLeaf-DEVICE", "CloverLeaf-UM"};
int clover::CLResourceManager::allocator_ids[clover::CLResourceManager::UmpireResourceType::RSEND] = { -1, -1, -1};

#if defined(RAJA_TARGET_GPU)
model create_context(bool silent, const std::vector<std::string> &args) {
  // initialise Umpire Allocators
  for (int Resource = 0; Resource < clover::CLResourceManager::UmpireResourceType::RSEND; Resource++){
    // Previous call initialized allocator
    if ( clover::CLResourceManager::allocator_ids[Resource] >= 0 )
      continue;

    // Pick umpire resource
    auto alloc_name = clover::CLResourceManager::CloverLeafPools[Resource];
    std::cout << "Alloc Name is" << alloc_name << "\n";
    // .. and map it to cloverleaf resource name
    auto& rm = umpire::ResourceManager::getInstance();
    auto alloc_resource = rm.makeAllocator<umpire::strategy::QuickPool, true>(
      alloc_name, rm.getAllocator(clover::CLResourceManager::umpireResources[Resource]));
    std::cout << "Setting up Resource: " <<
      clover::CLResourceManager::umpireResources[Resource].c_str() << " " << Resource << "\n";
    clover::CLResourceManager::allocator_ids[Resource] = alloc_resource.getId();
  }

  struct Device {
    int id{};
    std::string name{};
  };
  int count = 0;
  clover::checkError(rajaGetDeviceCount(&count));
  std::vector<Device> devices(count);
  for (int i = 0; i < count; ++i) {
    rajaDeviceProp props{};
    clover::checkError(rajaGetDeviceProperties(&props, i));
    devices[i] = {i, std::string(props.name) + " (" +                                        //
                         std::to_string(props.totalGlobalMem / 1024 / 1024) + "MB;" +        //
                         "sm_" + std::to_string(props.major) + std::to_string(props.minor) + //
                         ")"};
  }
  auto [device, parsed] = list_and_parse<Device>(
      silent, devices, [](auto &d) { return d.name; }, args);
  clover::checkError(rajaSetDevice(device.id));
  return model{clover::context{}, "CUDA", true, parsed};
}
#else
model create_context(bool silent, const std::vector<std::string> &args) {
  auto alloc_name = clover::CLResourceManager::CloverLeafPools[0];
  std::cout << "Alloc Name is" << alloc_name << "\n";
  
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc_resource = rm.makeAllocator<umpire::strategy::QuickPool, true>(
      alloc_name, rm.getAllocator(clover::CLResourceManager::umpireResources[0]));

  std::cout << "Setting up Resource: " <<
    clover::CLResourceManager::umpireResources[0].c_str() << " " << 0 << "\n";
    clover::CLResourceManager::allocator_ids[0] = alloc_resource.getId();
  
  auto [_, parsed] = list_and_parse<std::string>(
      silent, {"Host CPU"}, [](const auto &d) { return d; }, args);
  return {clover::context{}, "OpenMP (CPU)", false, parsed};
}
#endif

#if defined(RAJA_TARGET_GPU)
void report_context(const clover::context &) {
  int device = -1;
  clover::checkError(rajaGetDevice(&device));
  rajaDeviceProp props{};
  clover::checkError(rajaGetDeviceProperties(&props, device));
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
#else
void report_context(const clover::context &) {}
#endif
