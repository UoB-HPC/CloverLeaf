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

#include <fstream>

#include "initialise.h"
#include "start.h"

std::pair<clover::context, std::string> create_context(const std::vector<std::string> &args) {

  auto parsed = list_and_parse<std::string>(
      {"(default device)"}, [](auto &d) { return d; }, args);
  return {clover::context{}, parsed.file};
}

void report_context(const clover::context &) { std::cout << "Using C++ PSTL (std-indices)"; }
