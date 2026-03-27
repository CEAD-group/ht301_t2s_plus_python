# InfiRay Thermal Camera Python Ecosystem

## This Fork

**CEAD-group/ht301_t2s_plus_python** is a fork of [diminDDL/IR-Py-Thermal](https://github.com/diminDDL/IR-Py-Thermal), which itself is a fork of [stawel/ht301_hacklib](https://github.com/stawel/ht301_hacklib).

This fork adds:
- Frame retry logic for incomplete USB frames
- Startup stabilization (wait for sensor warmup)
- Hot pixel detection alongside dead pixel detection
- Persistent calibration save/load
- OpenCV coordinate fixes
- Reduced dependencies (no scikit-image needed)

## Lineage

```
netman69/inficam (GitLab)          <-- C++ reference implementation (decompiled from Android SDK)
    |
stawel/ht301_hacklib               <-- Original Python port (86 stars, 56 forks)
    |
    +-- diminDDL/IR-Py-Thermal     <-- Most active fork (175 stars). Adds T2S+ v2 raw mode,
    |       |                          dead pixel correction, lock-in thermography
    |       |
    |       +-- CEAD-group/ht301_t2s_plus_python  <-- This repo
    |
    +-- cmair/ht301_hacklib        <-- Early T2S+ support (deprecated)
    +-- sumpster/ht301_viewer      <-- T3S support, high-temp toggle
    +-- lamnguyenvu98/xtherm-python <-- Alternative port from inficam, T2 camera
```

## Key Repositories

| Repository | Stars | Focus | Status |
|-----------|-------|-------|--------|
| [stawel/ht301_hacklib](https://github.com/stawel/ht301_hacklib) | 86 | Original HT-301/T3S/T2S+ library | Active (upstream) |
| [diminDDL/IR-Py-Thermal](https://github.com/diminDDL/IR-Py-Thermal) | 175 | T2S+ v2 raw mode, NUC, dead pixels | Active |
| [MCMH2000/OpenHD_HT301_Driver](https://github.com/MCMH2000/OpenHD_HT301_Driver) | 18 | HT-301 for FPV/OpenHD | Persistent noise calibration |
| [mcguire-steve/ht301_ircam](https://github.com/mcguire-steve/ht301_ircam) | 11 | ROS node (C++) | HT-301 only |
| [LeoDJ/P2Pro-Viewer](https://github.com/LeoDJ/P2Pro-Viewer) | - | InfiRay P2Pro (phone camera) | Different protocol |
| [netman69/inficam](https://gitlab.com/netman69/inficam) | - | C++ reference (Android SDK decompile) | Definitive reference |

## Supported Cameras

All cameras use InfiRay's UVC protocol, communicating via `CAP_PROP_ZOOM` as a command channel.

| Model | Resolution | Notes |
|-------|-----------|-------|
| HT-301 | 384x288 | Original supported model |
| T3S | 384x288 | Same protocol as HT-301 |
| T2S+ v1 | 256x192 | FPGA does NUC in hardware |
| T2S+ v2 (A2) | 256x192 | FPGA present but no processing; needs `camera_raw=True` |
| HT-201 | 240x180 | Lower resolution variant |
| 640px models | 640x512 | Higher resolution |

## Temperature Calculation

Based on radiometric model from [Budzier & Gerlach (2017)](https://www.mdpi.com/1424-8220/17/8/1718):

1. Atmospheric transmittance from humidity, air temperature, distance
2. Sensor-specific calibration polynomial (cal_00..cal_05 from camera metadata)
3. 16384-entry lookup table: raw uint16 → temperature in Celsius
4. Distance correction and user offset

### Known Calibration Issues

- **High-temp mode (450°C)**: Correction coefficients `m=1.17, b=-40.9` are empirically fitted, not from SDK
- **T2S+ v2 shutter temperature**: Register is unreliable due to sensor placement; consider hardcoding ambient temp
- **Startup drift**: Sensor can show inverted/parabolic response for several minutes after power-on

## External References

- [Hacking the T2S+ — Dmytro Engineering](https://dmytroengineering.com/content/projects/t2s-plus-thermal-camera-hacking) — Deep technical blog post on T2S+ reverse engineering
- [Hackaday T2S+ Teardown (2025)](https://hackaday.com/2025/05/23/tearing-down-and-hacking-the-t2s-thermal-camera/) — Confirms FPGA present in v2 but not processing
- [EEVBlog TS2+ A2 Thread](https://www.eevblog.com/forum/thermal-imaging/infiray-ts2_a2-camera-its-metadata-and-the-ht301_hacklib/) — Community reports on A2 variant metadata differences
- [Budzier & Gerlach (2017), MDPI Sensors](https://www.mdpi.com/1424-8220/17/8/1718) — Radiometric model paper referenced in code
