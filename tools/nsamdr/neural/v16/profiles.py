"""Constants retained for the negative-evidence boundary-profile oracle audit.

The explicit BoundaryProfileNet formulation was rejected by held-out evidence and
is not part of the active V16.2 model.  This module intentionally contains no
learned network; the audit remains reproducible as a record of the rejected path.
"""

PROFILE_OFFSETS = (-8, -6, -4, -2, 0, 2, 4, 6, 8)
PHYSICAL_CHANNELS = 8
