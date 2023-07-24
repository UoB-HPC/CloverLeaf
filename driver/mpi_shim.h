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

#include <cstdlib>
#ifdef NO_MPI

  #define MPI_SUCCESS (0)
  #define MPI_ERR_COMM (1)
  #define MPI_ERR_COUNT (2)
  #define MPI_ERR_TYPE (3)
  #define MPI_ERR_BUFFER (4)

  #define MPI_INT (0)
  #define MPI_DOUBLE (0)
  #define MPI_SUM (0)
  #define MPI_MIN (0)
  #define MPI_MAX (0)
  #define MPI_STATUS_IGNORE (0)

  #define MPI_COMM_WORLD (0)

using MPI_Comm = int;
using MPI_Request = int;
using MPI_Datatype = int;
using MPI_Op = int;
using MPI_Status = int;

int MPI_Init(int *argc, char ***argv);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Abort(MPI_Comm comm, int errorcode);
int MPI_Barrier(MPI_Comm comm);
int MPI_Finalize();

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm);
int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);

#endif