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

#include "mpi_shim.h"
#include <cstdio>

#ifdef NO_MPI

int MPI_Init(int *, char ***) { return MPI_SUCCESS; }
int MPI_Comm_rank(MPI_Comm, int *rank) {
  *rank = 0;
  return MPI_SUCCESS;
}
int MPI_Comm_size(MPI_Comm, int *size) {
  *size = 1;
  return MPI_SUCCESS;
}
int MPI_Abort(MPI_Comm, int errorcode) {
  std::exit(errorcode);
  return MPI_SUCCESS;
}
int MPI_Finalize() { return MPI_SUCCESS; }

int MPI_Barrier(MPI_Comm) {
  // XXX no-op, correct for 1 rank only
  return MPI_SUCCESS;
}
int MPI_Allgather(const void *, int, MPI_Datatype, void *, int, MPI_Datatype, MPI_Comm) {
  // XXX no-op, correct for 1 rank only
  return MPI_SUCCESS;
}
int MPI_Reduce(const void *, void *, int, MPI_Datatype, MPI_Op, int, MPI_Comm) {
  // XXX no-op, correct for 1 rank only
  return MPI_SUCCESS;
}
int MPI_Allreduce(const void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm) {
  // XXX no-op, correct for 1 rank only
  return MPI_SUCCESS;
}
int MPI_Waitall(int, MPI_Request[], MPI_Status[]) {
  // XXX no-op, correct for 1 rank only
  return MPI_SUCCESS;
}

int MPI_Isend(const void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *) {
  fprintf(stderr, "MPI disabled, stub: %s\n", __func__);
  std::abort();
  return MPI_ERR_COMM;
}
int MPI_Irecv(void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *) {
  fprintf(stderr, "MPI disabled, stub: %s\n", __func__);
  std::abort();
  return MPI_ERR_COMM;
}

#endif
