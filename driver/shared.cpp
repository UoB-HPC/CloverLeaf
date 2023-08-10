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

#include <cstring>
#include <functional>
#include <sys/stat.h>
#include <sys/types.h>

#include "definitions.h"
#include "shared.h"

namespace clover {

std::ostream &operator<<(std::ostream &os, const Range1d &d) {
  os << "Range1d{"
     << " X[" << d.from << "->" << d.to << " (" << d.size << ")]"
     << "}";
  return os;
}
std::ostream &operator<<(std::ostream &os, const Range2d &d) {
  os << "Range2d{"
     << " X[" << d.fromX << "->" << d.toX << " (" << d.sizeX << ")]"
     << " Y[" << d.fromY << "->" << d.toY << " (" << d.sizeY << ")]"
     << "}";
  return os;
}
} // namespace clover

// typedef std::chrono::time_point<std::chrono::system_clock> timepoint;
// static inline timepoint mark() { return std::chrono::system_clock::now(); }
// static inline double elapsedMs(timepoint start) {
//   timepoint end = mark();
//   return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0;
// }

// writes content of the provided stream to file with name
static void record(const std::string &name, const std::function<void(std::ofstream &)> &f) {
  std::ios_base::sync_with_stdio(false);
  std::ofstream out;
  out.open(name, std::ofstream::out | std::ofstream::trunc);
  std::vector<char> buffer(1024 * 1024);
  out.rdbuf()->pubsetbuf(buffer.data(), buffer.size());
  f(out);
  out.close();
}

// formats and then dumps content of 1d double buffer to stream
static void show(std::ostream &out, const std::string &name, clover::Buffer1D<double> &buffer) {
  auto view = buffer.mirrored();
  out << name << "(" << 1 << ") [" << buffer.extent<0>() << "]\n";
  if (std::all_of(view.begin(), view.end(), [](auto x) { return x == 0.0; })) {
    out << "\t(0.0)";
  } else {
    out << "\t";
    for (double i : view)
      out << i << ", ";
  }
  out << "\n";
}

// formats and then dumps content of 2d double buffer to stream
static void show(std::ostream &out, const std::string &name, clover::Buffer2D<double> &buffer) {
  auto view = buffer.mirrored2();
  out << name << "(" << 2 << ") [" << buffer.extent<0>() << "x" << buffer.extent<1>() << "]\n";
  out << "\t";
  if (std::all_of(view.actual.begin(), view.actual.end(), [](auto x) { return x == 0.0; })) {
    out << "\t(0.0)";
  } else {
    for (size_t i = 0; i < buffer.extent<0>(); ++i) {
      for (size_t j = 0; j < buffer.extent<1>(); ++j)
        out << view(i,j) << ", ";
      out << "\t\n";
    }
  }
  out << "\n";
}

// dumps all content to file; for debugging only
void clover::dump(global_variables &g, const std::string &filename) {
  if (g.config.dumpDir.empty()) return;
  std::cout << "Dumping globals to " << filename << std::endl;

  const auto dir = g.config.dumpDir + "/";
  struct stat info {};
  if (stat(dir.c_str(), &info) != 0) {
    std::cout << "Creating " << dir << " for field dump" << std::endl;
    if (errno = 0; mkdir(dir.c_str(), 0777) != 0) {
      std::cerr << "Cannot create " << dir << ": " << std::strerror(errno) << ", skipping field dump" << std::endl;
      return;
    }
  } else if (info.st_mode & S_IFDIR) {
    // dir exists, just write into it
  } else {
    std::cerr << dir << " already exists and is not a directory, skipping field dump" << std::endl;
    return;
  }

  record(dir + filename, [&](std::ostream &out) {
    out << "Dump(tileCount = " << g.chunk.tiles.size() << ")\n";

    out << "error_condition" << '=' << g.error_condition << "\n";

    out << "step" << '=' << g.step << "\n";
    out << "advect_x" << '=' << g.advect_x << "\n";
    out << "time" << '=' << g.time << "\n";

    out << "dt" << '=' << g.dt << "\n";
    out << "dtold" << '=' << g.dtold << "\n";

    out << "complete" << '=' << g.complete << "\n";
    out << "jdt" << '=' << g.jdt << "\n";
    out << "kdt" << '=' << g.kdt << "\n";

    for (size_t i = 0; i < g.chunk.tiles.size(); ++i) {
      auto &fs = g.chunk.tiles[i].field;
      out << "\tTile[ " << i << "]:\n";

      tile_info &info = g.chunk.tiles[i].info;
      for (int l = 0; l < 4; ++l) {
        out << "info.tile_neighbours[i]" << '=' << info.tile_neighbours[i] << "\n";
        out << "info.external_tile_mask[i]" << '=' << info.external_tile_mask[i] << "\n";
      }

      out << "info.t_xmin" << '=' << info.t_xmin << "\n";
      out << "info.t_xmax" << '=' << info.t_xmax << "\n";
      out << "info.t_ymin" << '=' << info.t_ymin << "\n";
      out << "info.t_ymax" << '=' << info.t_ymax << "\n";
      out << "info.t_left" << '=' << info.t_left << "\n";
      out << "info.t_right" << '=' << info.t_right << "\n";
      out << "info.t_bottom" << '=' << info.t_bottom << "\n";
      out << "info.t_top" << '=' << info.t_top << "\n";

      show(out, "density0", fs.density0);
      show(out, "density1", fs.density1);
      show(out, "energy0", fs.energy0);
      show(out, "energy1", fs.energy1);
      show(out, "pressure", fs.pressure);
      show(out, "viscosity", fs.viscosity);
      show(out, "soundspeed", fs.soundspeed);
      show(out, "xvel0", fs.xvel0);
      show(out, "xvel1", fs.xvel1);
      show(out, "yvel0", fs.yvel0);
      show(out, "yvel1", fs.yvel1);
      show(out, "vol_flux_x", fs.vol_flux_x);
      show(out, "vol_flux_y", fs.vol_flux_y);
      show(out, "mass_flux_x", fs.mass_flux_x);
      show(out, "mass_flux_y", fs.mass_flux_y);

      show(out, "work_array1", fs.work_array1); // node_flux, stepbymass, volume_change, pre_vol
      show(out, "work_array2", fs.work_array2); // node_mass_post, post_vol
      show(out, "work_array3", fs.work_array3); // node_mass_pre,pre_mass
      show(out, "work_array4", fs.work_array4); // advec_vel, post_mass
      show(out, "work_array5", fs.work_array5); // mom_flux, advec_vol
      show(out, "work_array6", fs.work_array6); // pre_vol, post_ener
      show(out, "work_array7", fs.work_array7); // post_vol, ener_flux

      show(out, "cellx", fs.cellx);
      show(out, "celldx", fs.celldx);
      show(out, "celly", fs.celly);
      show(out, "celldy", fs.celldy);
      show(out, "vertexx", fs.vertexx);
      show(out, "vertexdx", fs.vertexdx);
      show(out, "vertexy", fs.vertexy);
      show(out, "vertexdy", fs.vertexdy);

      show(out, "volume", fs.volume);
      show(out, "xarea", fs.xarea);
      show(out, "yarea", fs.yarea);
      out << std::flush;
    }
  });
}
