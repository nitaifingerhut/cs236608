import ctypes
import io
import json
import os
import sys
import tempfile
from loguru import logger

from argparse import Namespace
from contextlib import contextmanager


try:
    libc = ctypes.CDLL(None)
    c_stdout = ctypes.c_void_p.in_dll(libc, "stdout")
except ValueError:
    from ctypes import *

    libc = CDLL("libc.dylib")
    c_stdout = c_void_p.in_dll(libc, "__stdoutp")


@contextmanager
def stdout_redirector():
    """context manager that redirect output from stdout and output from C code."""

    stream = io.BytesIO()
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        libc.fflush(c_stdout)
        sys.stdout.close()
        os.dup2(to_fd, original_stdout_fd)
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, "wb"))

    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        tfile = tempfile.TemporaryFile(mode="w+b")
        _redirect_stdout(tfile.fileno())
        yield
        _redirect_stdout(saved_stdout_fd)
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)


def dump_opts_to_json(opts: Namespace, path: str):
    params = vars(opts)
    with open(path, "w") as f:
        logger.info(f'dumping json opts: {path}')
        json.dump(params, f, indent=4)
