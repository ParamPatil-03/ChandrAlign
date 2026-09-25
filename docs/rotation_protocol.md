# Protocol: extreme-rotation handling, MATCH-12 (frozen before running)

PS: "Viewpoint variation ... Objects appear shifted, scaled, **rotated** ...". Research doc failure
mode #6 (extreme rotation: orbit-track / pushbroom differences) -> "ASIFT-style simulation or
rotation-invariant descriptors". Our registrations always have a geometric prior (label corners)
that removes rotation, so this is for a pair WITHOUT a usable prior (the generic case).

Known before measuring: SIFT is rotation-invariant by design (dominant orientation), so FEATURES'
done_when ("a 60-deg pair that plain SIFT fails on") is probably mis-stated; the matchers we
actually route to (LoFTR family: eloftr, minima-loftr) are NOT rotation-invariant (SE2-LoFTR,
arXiv:2204.10144; Steerers, arXiv:2312.02152). Retrained equivariant models would add weights and
licences; the matcher-agnostic fix is an image-level rotation search.

## Method (`matching/rotation.py: rotation_search`)
Match at 0 deg; if the robust estimate has >= 40 inliers, return it unchanged (early exit). Else
rotate the source (canvas expanded, nothing cropped) by 30, 60, ..., 330 deg, match each, keep the
angle with the most robust-estimate inliers (ties: smaller |angle|), and map its source points
back to the original frame. Cost: 1 match when the prior was right, 12 when it was not.

## Test
Six real cross-sensor pairs: TMC-2 -> SELENE TC fine frames of the six FLAT windows (N03/N09;
flat so the per-window affine is a clean truth, ~0.3 px), from `--dump-points`. Source: a 900 px
centre crop rotated about its centre by theta, central 640 x 640 kept; reference: 960 px centre
crop, unrotated. theta = 0, 15, ..., 345 (24) -> 144 cases per matcher. Truth: rotation composed
with the window's affine. A case succeeds if a transform is returned with >= 20 inliers and its
error vs truth is <= 1.5 px at the crop centre and <= 3 px at all four crop corners.
Matchers: eloftr (routed for TMC-2 -> TC), minima-loftr (its fallback), sift; each plain and with
the search.

## Decision
Adopt (`matching.rotation_search`, used where no geometric prior exists) if: (a) eloftr + search
succeeds on >= 95% of the 144 cases; (b) every 0-deg case with the search is identical to plain
(early exit, no regression); (c) sift + search succeeds on at least as many cases as plain sift.
Reported, not ruled: plain matchers' success by angle (does plain SIFT fail at 60 deg?), runtime.
