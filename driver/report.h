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
#include <cmath>

extern std::ostream g_out;

void report_error(char *location, char *error);

void clover_report_step_header(global_variables &globals, parallel_ &parallel);

void clover_report_step(global_variables &globals, parallel_ &parallel, //
                        double vol, double mass, double ie, double ke, double press);
