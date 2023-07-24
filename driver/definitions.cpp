
#include "definitions.h"

field_type::field_type(const size_t xrange, const size_t yrange, clover::context &ctx)
    //                                               //    --timestep--              --pdV--                   --pdV--
    : density0(ctx, xrange, yrange),        // [ idg           _ ]    [ pdv idg revert ]  accel  [ pdv idg revert ]      |          reset
      density1(ctx, xrange, yrange),        // [ idg           _ ]    [ pdv idg revert ]         [ pdv idg revert ]      | adv_mom  reset
      energy0(ctx, xrange, yrange),         // [ idg viscosity _ ]    [ pdv idg revert ]         [ pdv idg revert ]      |          reset
      energy1(ctx, xrange, yrange),         // [ idg           _ ]    [ pdv idg revert ]         [ pdv idg revert ]      |          reset
      pressure(ctx, xrange, yrange),        // [ idg viscosity _ ]    [ pdv idg        ]  accel  [ pdv idg        ]      |
      viscosity(ctx, xrange, yrange),       // [               _ ]    [ pdv            ]  accel  [ pdv            ]      |
      soundspeed(ctx, xrange, yrange),      // [ idg           _ ]    [     idg        ]         [     idg        ]      |
      xvel0(ctx, xrange + 1, yrange + 1),   // [     viscosity _ ]    [ pdv            ]  accel  [ pdv            ] flux | reset
      xvel1(ctx, xrange + 1, yrange + 1),   // [     viscosity _ ]    [ pdv            ]  accel  [ pdv            ] flux | adv_mom  reset
      yvel0(ctx, xrange + 1, yrange + 1),   // [               _ ]    [ pdv            ]  accel  [ pdv            ] flux | reset
      yvel1(ctx, xrange + 1, yrange + 1),   // [               _ ]    [ pdv            ]  accel  [ pdv            ] flux | adv_mom  reset
      vol_flux_x(ctx, xrange + 1, yrange),  // [               _ ]    [                ]         [                ] flux | adv_mom
      mass_flux_x(ctx, xrange + 1, yrange), // [               _ ]    [                ]         [                ]      | adv_mom
      vol_flux_y(ctx, xrange, yrange + 1),  // [               _ ]    [                ]         [                ] flux | adv_mom
      mass_flux_y(ctx, xrange, yrange + 1), // [               _ ]    [                ]         [                ]      | adv_mom
      work_array1(ctx, xrange + 1, yrange + 1), // [               _ ]    [ pdv            ]         [ pdv            ]      | adv_mom
      work_array2(ctx, xrange + 1, yrange + 1), // [               _ ]    [                ]         [                ]      | adv_mom
      work_array3(ctx, xrange + 1, yrange + 1), // [               _ ]    [                ]         [                ]      | adv_mom
      work_array4(ctx, xrange + 1, yrange + 1), // [               _ ]    [                ]         [                ]      | adv_mom
      work_array5(ctx, xrange + 1, yrange + 1), // [               _ ]    [                ]         [                ]      | adv_mom
      work_array6(ctx, xrange + 1, yrange + 1), // [               _ ]    [                ]         [                ]      | adv_mom
      work_array7(ctx, xrange + 1, yrange + 1), // [               _ ]    [                ]         [                ]      |
      cellx(ctx, xrange),                       // [               _ ]    [                ]         [                ]      |
      celldx(ctx, xrange),                      // [     viscosity _ ]    [                ]         [                ]      | adv_mom
      celly(ctx, yrange),                       // [               _ ]    [                ]         [                ]      |
      celldy(ctx, yrange),                      // [     viscosity _ ]    [                ]         [                ]      | adv_mom
      vertexx(ctx, xrange + 1),                 // [               _ ]    [                ]         [                ]      |
      vertexdx(ctx, xrange + 1),                // [               _ ]    [                ]         [                ]      |
      vertexy(ctx, yrange + 1),                 // [               _ ]    [                ]         [                ]      |
      vertexdy(ctx, yrange + 1),                // [               _ ]    [                ]         [                ]      |
      volume(ctx, xrange, yrange),              // [               _ ]    [ pdv            ]  accel  [ pdv            ]      | adv_mom
      xarea(ctx, xrange + 1, yrange),           // [               _ ]    [ pdv            ]  accel  [ pdv            ] flux |
      yarea(ctx, xrange, yrange + 1),           // [               _ ]    [ pdv            ]  accel  [ pdv            ] flux |
      base_stride(xrange), vels_wk_stride(xrange + 1), flux_x_stride(xrange + 1), flux_y_stride(xrange)

{}

chunk_type::chunk_type(const std::array<int, 4> &chunkNeighbours, const int task, //
                       const int xMin, const int yMin, const int xMax,
                       const int yMax,                                                                                   //
                       const int left, const int right, const int bottom, const int top,                                 //
                       const int leftBoundary, const int rightBoundary, const int bottomBoundary, const int topBoundary, //
                       const int tiles_per_chunk)
    : chunk_neighbours(chunkNeighbours), task(task),      //
      x_min(xMin), y_min(yMin), x_max(xMax), y_max(yMax), //
      left(left), right(right), bottom(bottom), top(top), //
      left_boundary(leftBoundary), right_boundary(rightBoundary), bottom_boundary(bottomBoundary), top_boundary(topBoundary) {}
global_variables::global_variables(const global_config &config, clover::context queue, chunk_type chunk)
    : config(config), context(std::move(queue)), chunk(std::move(chunk)), dt(config.dtinit), dtold(config.dtinit),
      profiler_on(config.profiler_on) {}
