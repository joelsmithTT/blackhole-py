"""Single-instruction MMIO stores and loads, via a compiled helper.

The problem this solves: ctypes stores go through a memcpy; on riscv64 a u32
write was getting turned into 4x sb instructions in a situation where a single
sw instruction is needed for correctness. On x86-64 it's a single mov.

The helper is built on first use and cached next to the source, keyed by a hash
of that source and by the machine, so a stale object is never picked up and a
shared filesystem cannot hand one architecture another's build.
"""
import ctypes, hashlib, os, platform, subprocess, tempfile
from pathlib import Path

SOURCE = Path(__file__).with_name("mmio.c")
_library = None


def _cache_paths():
  digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
  name = f"mmio-{digest}-{platform.machine()}.so"
  # Beside the source when that is writable, which keeps the build with the
  # checkout it belongs to; otherwise anywhere, because a wrong answer is worse
  # than a rebuild.
  return (SOURCE.parent / "__pycache__" / name, Path(tempfile.gettempdir()) / name)


def _build(path):
  compiler = os.environ.get("CC", "cc")
  path.parent.mkdir(parents=True, exist_ok=True)
  # Build to a unique name and rename, so two processes racing here cannot hand
  # each other a half-written object to dlopen.
  handle, staging = tempfile.mkstemp(dir=path.parent, suffix=".so")
  os.close(handle)
  try:
    subprocess.run(
      [compiler, "-O2", "-fPIC", "-shared", "-o", staging, str(SOURCE)],
      check=True, capture_output=True,
    )
    os.replace(staging, path)
  except BaseException:
    Path(staging).unlink(missing_ok=True)
    raise


def _load():
  global _library
  if _library is not None: return _library
  failures = []
  for path in _cache_paths():
    try:
      if not path.exists(): _build(path)
      library = ctypes.CDLL(str(path))
      break
    except (OSError, subprocess.CalledProcessError) as failure:
      detail = getattr(failure, "stderr", b"") or b""
      failures.append(f"{path}: {failure}{detail.decode(errors='replace')}")
  else:
    raise RuntimeError(
      "could not build the MMIO helper from " + str(SOURCE) + ":\n  "
      + "\n  ".join(failures)
      + "\nblackhole-py needs a C compiler for this one file, because ctypes "
        "cannot promise the width of a device store: on riscv64 its 32-bit store "
        "is four byte stores, and a Tensix register keeps only the last one. Set "
        "CC to a working compiler."
    )
  library.bhpy_store32.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
  library.bhpy_store32.restype = None
  library.bhpy_load32.argtypes = [ctypes.c_void_p]
  library.bhpy_load32.restype = ctypes.c_uint32
  _library = library
  return library


def store32(address: int, value: int): _load().bhpy_store32(address, value)

def load32(address: int) -> int: return _load().bhpy_load32(address)
