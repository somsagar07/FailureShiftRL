"""Persistent JSON-lines Vev process with an isolated Transformers runtime."""
from __future__ import annotations

import base64
import binascii
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import sys


ROOT = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 32 * 1024 * 1024
MAX_LINE_CHARS = 16 * 1024 * 1024


def decode_image(encoded):
    from PIL import Image
    if not isinstance(encoded, str) or len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise ValueError("Vev input must be a base64 PNG of at most 8 MiB.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Vev input image is not valid base64 PNG data.") from error
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Vev input PNG must contain between 1 byte and 8 MiB.")
    with Image.open(io.BytesIO(raw)) as image:
        if (image.format != "PNG" or image.width < 1 or image.height < 1
                or image.width * image.height > MAX_IMAGE_PIXELS
                or max(image.size) > 32768):
            raise ValueError("Vev accepts PNG images up to 32 megapixels and 32768 pixels per dimension.")
        image.load()
        return image.convert("RGB")


def _backend(selection, device, batch_size):
    runtime = ROOT / ".vev-runtime"
    if not (runtime / ".ready.json").is_file():
        raise RuntimeError("The Vev runtime is not installed. Run .venv/bin/python -E setup_vev.py and retry.")
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from vev_backend import VevBackend
    return VevBackend(selection, device, batch_size=batch_size)


def serve(input_stream, output_stream, backend_factory=None):
    backend_factory = backend_factory or _backend
    backend = None
    while True:
        line = input_stream.readline(MAX_LINE_CHARS + 1)
        if not line:
            break
        request = response = questions = image = answers = None
        close = False
        try:
            if len(line) > MAX_LINE_CHARS:
                while line and not line.endswith("\n"):
                    line = input_stream.readline(MAX_LINE_CHARS + 1)
                raise ValueError("Vev worker request exceeds the 16 MiB line limit.")
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Vev worker commands must be JSON objects.")
            command = request.get("command")
            with redirect_stdout(sys.stderr):
                if command == "load":
                    if backend is not None:
                        raise RuntimeError("This Vev worker already has a loaded model; start a new worker to change models.")
                    backend = backend_factory(request.get("selection"), request.get("device"), request.get("batch_size", 4))
                    response = {"ok": True, "identity": backend.identity}
                elif command == "decide":
                    if backend is None:
                        raise RuntimeError("Load a Vev model before requesting a decision.")
                    from vev_backend import validate_rows
                    questions = validate_rows(request.get("questions"))
                    image = decode_image(request.get("image"))
                    try:
                        answers = backend.decide(image, questions)
                    finally:
                        image.close()
                    response = {"ok": True, "answers": answers}
                elif command == "close":
                    response, close = {"ok": True}, True
                else:
                    raise ValueError("Unknown Vev worker command; use load, decide, or close.")
        except Exception as error:
            response = {"ok": False, "error": str(error), "error_type": type(error).__name__}
        output_stream.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
        output_stream.flush()
        # No decoded image, encoded payload, or response remains on the next
        # blocking read. The persistent object owns model weights only.
        request = response = line = questions = image = answers = None
        if close:
            break


def _parent_lifetime():
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    if sys.platform.startswith("linux"):
        import ctypes
        parent = os.getppid()
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), "Could not bind Vev worker lifetime to its parent.")
        if parent == 1 or os.getppid() != parent:
            raise SystemExit(0)


def main():
    # Duplicate the protocol pipe, then redirect fd 1 as well as Python stdout:
    # library C-extension prints and progress bars cannot corrupt JSON replies.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    _parent_lifetime()
    try:
        serve(sys.stdin, protocol)
    finally:
        protocol.close()


if __name__ == "__main__":
    main()
