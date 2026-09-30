#!/usr/bin/env bash

# XXX Use a shell to preserve MPI's inherited descriptors, including PMI_FD
set -u
# XXX AdaptiveCpp ranks must not replace each other's JIT cache files
export ACPP_APPDB_DIR="${ACPP_APPDB_DIR}/rank-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}"
"$1" --file "$2" --out "$3"
result=$?
if [ "$#" -eq 3 ]; then
    exit "$result"
fi
if [ "$result" -ne "$4" ]; then
    printf 'Rank exit status: %s; expected %s\n' "$result" "$4" >&2
    exit 1
fi
