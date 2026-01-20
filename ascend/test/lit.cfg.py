import os
import lit.formats

# Minimal lit configuration for Ascend-specific tests.
# Tools are taken from the local build tree; FileCheck is expected on PATH
# (e.g., your llvm-build/bin).

config.name = "TRITON_ASCEND"
config.test_format = lit.formats.ShTest(not getattr(config, "use_lit_shell", False))
config.suffixes = [".mlir", ".ll"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root

# Add built Triton tools to PATH if present.
repo_root = os.path.abspath(os.path.join(config.test_source_root, "..", ".."))
tools_dir = os.path.join(repo_root, "python", "build", "cmake.linux-x86_64-cpython-3.10", "bin")
path_parts = []
if os.path.isdir(tools_dir):
    path_parts.append(tools_dir)
# Preserve existing PATH so user-provided FileCheck/llvm-lit are found.
path_parts.append(os.environ.get("PATH", ""))
config.environment["PATH"] = os.pathsep.join(path_parts)

# Enable FileCheck var-scope semantics like MLIR tests.
config.environment["FILECHECK_OPTS"] = "--enable-var-scope"
