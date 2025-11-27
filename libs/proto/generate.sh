#!/bin/bash
# Protobuf code generation script for all languages

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$SCRIPT_DIR"
PYTHON_OUT="$SCRIPT_DIR/generated/python"
TS_OUT="$SCRIPT_DIR/generated/typescript"

echo "Generating protobuf code..."
echo "Proto directory: $PROTO_DIR"

# Create output directories
mkdir -p "$PYTHON_OUT" "$TS_OUT"

# Generate Python code
echo ""
echo "Generating Python code..."
python3 -m grpc_tools.protoc \
  -I"$PROTO_DIR" \
  --python_out="$PYTHON_OUT" \
  --grpc_python_out="$PYTHON_OUT" \
  "$PROTO_DIR"/*.proto

# Create __init__.py for Python package
touch "$PYTHON_OUT/__init__.py"

echo "Python code generated in: $PYTHON_OUT"

# Generate TypeScript code (if ts-proto is available)
if command -v protoc &> /dev/null; then
  echo ""
  echo "Generating TypeScript code..."

  # Check if ts-proto is installed (check local node_modules first, then root)
  TS_PROTO_PLUGIN=""
  if [ -f "$SCRIPT_DIR/node_modules/.bin/protoc-gen-ts_proto" ]; then
    TS_PROTO_PLUGIN="$SCRIPT_DIR/node_modules/.bin/protoc-gen-ts_proto"
  elif [ -f "$SCRIPT_DIR/../../node_modules/.bin/protoc-gen-ts_proto" ]; then
    TS_PROTO_PLUGIN="$SCRIPT_DIR/../../node_modules/.bin/protoc-gen-ts_proto"
  fi

  if [ -n "$TS_PROTO_PLUGIN" ]; then
    protoc \
      -I"$PROTO_DIR" \
      --plugin="$TS_PROTO_PLUGIN" \
      --ts_proto_out="$TS_OUT" \
      --ts_proto_opt=outputServices=grpc-js \
      --ts_proto_opt=esModuleInterop=true \
      "$PROTO_DIR"/*.proto

    echo "TypeScript code generated in: $TS_OUT"
  else
    echo "Warning: ts-proto not found. Run 'npm install' in libs/proto/ or root directory."
    echo "Skipping TypeScript generation."
  fi
else
  echo "Warning: protoc not found. Skipping TypeScript generation."
fi

echo ""
echo "Code generation complete!"
