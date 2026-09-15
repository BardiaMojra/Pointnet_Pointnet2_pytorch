# -*- coding: utf-8 -*-
"""platform_guard.py -- platform check every dlo/ torch script runs before anything else.
Platform ID = u<Ubuntu major>_<x86|a64>, as ~/git/bashrc/platform.sh computes it (stdlib only)."""
import os
import sys


def platform_id():
    osr = {}
    try:
        with open("/etc/os-release") as f:
            for line in f:
                k, _, v = line.strip().partition("=")
                osr[k] = v.strip('"')
    except OSError:
        pass
    os_id = "u" if osr.get("ID") == "ubuntu" else osr.get("ID", "linux")
    arch = {"x86_64": "x86", "aarch64": "a64"}.get(os.uname()[4], os.uname()[4])
    return "%s%s_%s" % (os_id, osr.get("VERSION_ID", "").split(".")[0], arch)


def require(platforms, script):
    """Exit 3 unless this machine is one of platforms."""
    if platform_id() not in platforms:
        sys.stderr.write("refusing to run %s: written for %s, but %s is %s\n" % (
            os.path.basename(script), " ".join(platforms), os.uname()[1], platform_id()))
        sys.exit(3)
