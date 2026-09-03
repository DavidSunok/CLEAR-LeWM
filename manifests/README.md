# Reference manifests

The current canonical suite is CLEAR-LeWM v0.9:

```text
manifests/v0.9/{pusht,cube,reacher,tworoom}/\
  {moderate,strict}-seed{0,1,42}-n100.json
```

Moderate is the minimally corrected LeWM-compatible protocol. Strict is the
tighter task-semantic protocol. Every JSON embeds the complete protocol,
dataset fingerprint, fixed pair IDs, policy seed, and selection statistics.

v0.8 and v0.5 remain checked in for historical reproduction. They restore
recorded dynamic velocity and must not be relabeled as v0.9 results. The
unversioned task directories preserve earlier compatibility artifacts.
