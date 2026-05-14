"""Protobuf service definitions for the SAGA coordinator.

The `.proto` file lives at ``saga_coordinator.proto`` in this directory.
Generated stubs (``saga_coordinator_pb2.py`` and ``saga_coordinator_pb2_grpc.py``)
are produced by:

    python -m grpc_tools.protoc -I src/saga/serving/distributed/proto \
        --python_out=src/saga/serving/distributed/proto \
        --grpc_python_out=src/saga/serving/distributed/proto \
        src/saga/serving/distributed/proto/saga_coordinator.proto

The stubs are intentionally not checked in; ``make proto`` regenerates them.
"""
