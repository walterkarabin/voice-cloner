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

# Fix imports in generated Python files to use absolute imports
echo "Fixing Python import paths..."
python3 << EOF
import os
import re

python_dir = "$PYTHON_OUT"
for filename in os.listdir(python_dir):
    if filename.endswith('_pb2.py') or filename.endswith('_pb2_grpc.py'):
        filepath = os.path.join(python_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()

        # Replace relative imports with absolute imports
        # e.g., "import common_pb2" -> "from libs.proto.generated.python import common_pb2"
        # and "import audio_pb2" -> "from libs.proto.generated.python import audio_pb2"
        content = re.sub(
            r'^import ([a-z_]+_pb2(?:_grpc)?) as ([a-z_]+__pb2)',
            r'from libs.proto.generated.python import \1 as \2',
            content,
            flags=re.MULTILINE
        )

        with open(filepath, 'w') as f:
            f.write(content)

print("Import paths fixed in generated Python files")
EOF

echo "Python code generated in: $PYTHON_OUT"

# Generate TypeScript code (if ts-proto is available)
# Check if protoc is available (either as command or via Python)
PROTOC_CMD=""
if command -v protoc &> /dev/null; then
  PROTOC_CMD="protoc"
elif python3 -m grpc_tools.protoc --version &> /dev/null; then
  PROTOC_CMD="python3 -m grpc_tools.protoc"
fi

if [ -n "$PROTOC_CMD" ]; then
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
    $PROTOC_CMD \
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
