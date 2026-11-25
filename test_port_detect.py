"""Port auto-detection tests for rover_server.py.

Regression test for Nov 24, 2026: after a reboot the two USB adapters
swapped between /dev/ttyUSB0 and /dev/ttyUSB1, so the TF-Luna reader
(hardcoded to ttyUSB0) read the LD19's stream as garbage and the LD19
finder skipped the port the sensor was actually on — both sensors
went offline. Both readers now find their sensor by its data
signature, so the port assignment no longer matters.

Run: python3 test_port_detect.py
"""
import sys
import types
import glob as glob_mod

sys.path.insert(0, "/home/hatch/workspace/pi-monitoring-rover")

# --- Fake serial module (this VM has no pyserial) -------------------------
STREAMS = {}  # port -> bytes, set per test case


class FakeSerial:
    def __init__(self, port, baud, timeout=0.2):
        if port not in STREAMS:
            raise Exception(f"no such port {port}")
        self._stream = STREAMS[port]
        self._pos = 0

    def read(self, n):
        if self._pos >= len(self._stream):
            return b""
        chunk = self._stream[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def close(self):
        pass


fake_serial_mod = types.SimpleNamespace(Serial=FakeSerial)
sys.modules["serial"] = fake_serial_mod

import rover_server as rs

FAILURES = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f" ({extra})" if extra and not cond else ""))
    if not cond:
        FAILURES.append(name)


# --- Fake serial streams ---------------------------------------------------
def tf_frame(dist_cm, strength=1000):
    f = bytearray([0x59, 0x59, dist_cm & 0xFF, (dist_cm >> 8) & 0xFF,
                   strength & 0xFF, (strength >> 8) & 0xFF, 0, 0])
    f.append(sum(f) & 0xFF)
    return bytes(f)


def ld19_packet(start_deg=0.0, dist_mm=1000):
    pkt = bytearray(47)
    pkt[0] = 0x54
    pkt[1] = 0x2C
    speed = 3600  # 10 Hz * 360 deg/s
    pkt[2] = speed & 0xFF
    pkt[3] = (speed >> 8) & 0xFF
    s = int(start_deg * 100)
    pkt[4] = s & 0xFF
    pkt[5] = (s >> 8) & 0xFF
    for k in range(12):
        off = 6 + k * 3
        pkt[off] = dist_mm & 0xFF
        pkt[off + 1] = (dist_mm >> 8) & 0xFF
        pkt[off + 2] = 200  # intensity
    e = int((start_deg + 10.0) * 100)
    pkt[42] = e & 0xFF
    pkt[43] = (e >> 8) & 0xFF
    pkt[44] = 0
    pkt[45] = 0
    pkt[46] = rs.ld19_crc8(bytes(pkt[:46]))
    return bytes(pkt)


TF_STREAM = tf_frame(150) * 20          # 20 valid TF-Luna frames
LD19_STREAM = ld19_packet(0.0) * 5      # 5 valid LD19 packets


def run_case(name, streams, want_tf, want_ld19):
    """streams: {port: bytes}. Install fakes, run both finders."""
    global STREAMS
    STREAMS = dict(streams)
    orig_glob = glob_mod.glob
    glob_mod.glob = lambda pattern: sorted(STREAMS.keys())
    try:
        rs.LIDAR["port"] = None
        rs.LD19["port"] = None
        got_tf = rs.lidar_find_port()
        rs.LIDAR["port"] = None
        rs.LD19["port"] = None
        got_ld19 = rs.ld19_find_port()
    finally:
        glob_mod.glob = orig_glob
    check(f"{name}: TF-Luna found on {want_tf}",
          got_tf == want_tf, f"got {got_tf}")
    check(f"{name}: LD19 found on {want_ld19}",
          got_ld19 == want_ld19, f"got {got_ld19}")


# --- Cases -----------------------------------------------------------------
# Swapped (the Nov 24 failure): LD19 on ttyUSB0, TF-Luna on ttyUSB1.
run_case("swapped ports",
         {"/dev/ttyUSB0": LD19_STREAM, "/dev/ttyUSB1": TF_STREAM},
         "/dev/ttyUSB1", "/dev/ttyUSB0")

# Normal: TF-Luna on ttyUSB0, LD19 on ttyUSB1.
run_case("normal ports",
         {"/dev/ttyUSB0": TF_STREAM, "/dev/ttyUSB1": LD19_STREAM},
         "/dev/ttyUSB0", "/dev/ttyUSB1")

# Only the TF-Luna plugged in: LD19 finder finds nothing, TF-Luna
# finder still works.
run_case("TF-Luna only",
         {"/dev/ttyUSB0": TF_STREAM},
         "/dev/ttyUSB0", None)

# Only the LD19 plugged in.
run_case("LD19 only",
         {"/dev/ttyUSB1": LD19_STREAM},
         None, "/dev/ttyUSB1")

# Nothing plugged in.
run_case("nothing plugged in", {}, None, None)

# Garbage on a port (wrong baud / other device) is ignored.
run_case("garbage port ignored",
         {"/dev/ttyUSB0": b"\x00\xff\xab" * 100,
          "/dev/ttyUSB1": TF_STREAM, "/dev/ttyUSB2": LD19_STREAM},
         "/dev/ttyUSB1", "/dev/ttyUSB2")

print()
if FAILURES:
    print(f"FAILURES: {len(FAILURES)}")
    for f in FAILURES:
        print(" -", f)
    sys.exit(1)
print("FAILURES: none")
