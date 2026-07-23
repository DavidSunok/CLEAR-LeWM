# Reference manifests

The current canonical suite is CLEAR-LeWM v0.5:

```text
manifests/v0.5/{pusht,cube,reacher,tworoom}/\
  {moderate,strict}-seed{0,1,42}-n100.json
```

Moderate is the minimally corrected LeWM-compatible protocol. Strict is the
tighter task-semantic protocol. Every JSON embeds the complete protocol,
dataset fingerprint, fixed pair IDs, policy seed, and selection statistics.

The unversioned task directories and `results/reference/` preserve earlier
compatibility artifacts. Prior suites remain available from their Git release
tags; they are not part of the v0.5 result table.
