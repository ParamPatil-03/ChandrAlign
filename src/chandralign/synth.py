"""Synthetic lunar scene generator with exactly-known ground truth.

Feature PREC-06 (sub-pixel self-verification), and the input source for the
control gates CHECK-01..CHECK-04. This lives in src/ rather than tests/ because
it is not test scaffolding: the null / noise / perturbation gates are pipeline
stages that run on every benchmark execution, and PREC-06 is a shipped feature.

WHY RENDER FROM A HEIGHT FIELD INSTEAD OF FAKING IT
---------------------------------------------------
The headline challenge in SIH26166 is Sun-angle variation. The cheap way to
simulate that is a brightness or gamma shift, but under a gamma shift the
shadows stay exactly where they are, so a matcher that is merely brightness-
robust passes a test it ought to fail. What actually breaks feature matching on
the Moon is that shadows move and invert as the Sun moves: a crater lit from the
east and the same crater lit from the west are near photometric negatives.

So we build terrain (craters as bowls with raised rims) and render it with a
Lunar-Lambert reflectance model plus ray-marched cast shadows. Changing
sun_az_deg then moves the shadows correctly, for free, and the DEM we rendered
from is exactly the DEM the geometry-guided filter (ALIGN-03) needs as input.

All imagery produced here is SYNTHETIC. Any metric measured on it must be
tagged Metrics.source="synthetic" (rule H5), never "measured".
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .contracts import GeometryLayers, ImagePlane, SceneMeta

LUNAR_LAMBERT_L = 0.8      # limb-darkening parameter, ~0.7-1.0 for the Moon
AMBIENT = 0.04             # scattered light inside a cast shadow; never pure zero
DEPTH_TO_DIAMETER = 0.20   # fresh simple craters sit near d/D ~ 0.2
RIM_HEIGHT_FRAC = 0.04     # rim height as a fraction of crater radius


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------
def _fractal_terrain(shape, rng, gsd_m, rms_slope=0.06, octaves=5,
                     min_wavelength_px=4):
    """Rolling background relief, parameterised by SLOPE rather than height.

    Amplitude is derived from each octave's wavelength instead of being a fixed
    number of metres, because what the renderer and the matcher actually care
    about is terrain gradient, not absolute relief. A fixed amplitude is scale-
    dependent nonsense: 12 m of relief across a 16 m wavelength is a 37-degree
    ramp, and summing five such octaves produced an 83-degree, 79%-shadowed
    scene in an earlier version of this file. Tying amplitude to wavelength
    makes each octave contribute about `rms_slope` of gradient regardless of
    GSD or tile size, so the terrain stays lunar at any resolution.
    """
    h, w = shape
    out = np.zeros(shape, np.float32)
    for o in range(octaves):
        lam_px = max(min_wavelength_px, 2 ** (octaves - o))
        # a * (2*pi/lambda) ~= rms_slope  =>  a = rms_slope * lambda / (2*pi)
        amp_m = rms_slope * (lam_px * gsd_m) / (2.0 * np.pi)
        coarse = rng.standard_normal((h // lam_px + 2, w // lam_px + 2)).astype(np.float32)
        out += amp_m * cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


def lunar_dem(shape=(512, 512), gsd_m=1.0, n_craters=60, seed=0,
              rms_slope=0.06, r_min_px=4.0, r_max_px=60.0):
    """A synthetic lunar height field, in metres.

    Crater radii follow an inverse-cube size-frequency law (many small, few
    large), roughly how real crater populations behave, which conveniently
    produces the repetitive self-similar terrain that failure mode #11 is about.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    dem = _fractal_terrain(shape, rng, gsd_m, rms_slope=rms_slope)

    # Inverse-transform sampling for p(r) ~ r^-3.
    u = rng.random(n_craters)
    radii = (r_min_px ** -2 + u * (r_max_px ** -2 - r_min_px ** -2)) ** -0.5

    yy_full, xx_full = np.mgrid[0:h, 0:w].astype(np.float32)
    for r_px in radii:
        cy = rng.uniform(0, h)
        cx = rng.uniform(0, w)
        pad = int(r_px * 1.6) + 2          # a crater only touches its neighbourhood
        y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad)
        x0, x1 = max(0, int(cx) - pad), min(w, int(cx) + pad)
        if y1 <= y0 or x1 <= x0:
            continue
        d = np.hypot(yy_full[y0:y1, x0:x1] - cy, xx_full[y0:y1, x0:x1] - cx)
        uu = d / r_px
        depth_m = DEPTH_TO_DIAMETER * r_px * gsd_m
        rim_m = RIM_HEIGHT_FRAC * r_px * gsd_m
        bowl = np.where(uu < 1.0, -depth_m * (1.0 - uu ** 2), 0.0)
        rim = rim_m * np.exp(-(((uu - 1.0) / 0.18) ** 2))
        dem[y0:y1, x0:x1] += (bowl + rim).astype(np.float32)

    return dem.astype(np.float32)


# ---------------------------------------------------------------------------
# Illumination
# ---------------------------------------------------------------------------
def _sun_vector(sun_az_deg, sun_el_deg):
    """Unit vector from the surface toward the Sun.

    Convention: x = column (east), y = row (increasing downward, i.e. south),
    z = up. Azimuth is clockwise from north, so north is -y.
    """
    az = np.deg2rad(sun_az_deg)
    el = np.deg2rad(sun_el_deg)
    return np.array([np.cos(el) * np.sin(az),
                     -np.cos(el) * np.cos(az),
                     np.sin(el)], np.float32)


def _shift(arr, dy, dx):
    """Shift with edge replication; np.roll alone would wrap and fake shadows."""
    pad = max(abs(dy), abs(dx)) + 1
    p = np.pad(arr, pad, mode="edge")
    h, w = arr.shape
    return p[pad + dy:pad + dy + h, pad + dx:pad + dx + w]


def cast_shadow_mask(dem, gsd_m, sun_az_deg, sun_el_deg, max_steps=96):
    """Ray-march toward the Sun; a pixel is shadowed if terrain blocks the path.

    This is what makes the harness honest: shadows genuinely move and invert
    with sun_az_deg, instead of staying put as they would under a gamma shift.
    """
    if sun_el_deg >= 89.0:
        return np.zeros(dem.shape, bool)
    s = _sun_vector(sun_az_deg, sun_el_deg)
    tan_el = np.tan(np.deg2rad(max(float(sun_el_deg), 0.25)))
    shadow = np.zeros(dem.shape, bool)
    for t in range(1, max_steps + 1):
        dy = int(round(float(s[1]) * t))
        dx = int(round(float(s[0]) * t))
        if dy == 0 and dx == 0:
            continue
        needed = dem + (np.hypot(dy, dx) * gsd_m) * tan_el
        shadow |= _shift(dem, dy, dx) > needed
    return shadow


def surface_normals(dem, gsd_m):
    """Per-pixel unit normals, (H,W,3), in the (x, y_row, z) frame."""
    gy, gx = np.gradient(dem.astype(np.float32), gsd_m)
    n = np.stack([-gx, -gy, np.ones_like(dem, np.float32)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def slope_aspect(dem, gsd_m):
    """Slope and aspect in degrees. Mirrors what Member A's GEO-03 will produce."""
    gy, gx = np.gradient(dem.astype(np.float32), gsd_m)
    slope = np.rad2deg(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    aspect = (np.rad2deg(np.arctan2(-gy, gx)) % 360.0).astype(np.float32)
    return slope, aspect


def render(dem, gsd_m, sun_az_deg, sun_el_deg, albedo=None,
           noise=0.006, seed=0, shadows=True):
    """Render the DEM under a given Sun.

    Returns (image 0..1 float32, incidence_deg, shadow_mask).

    Reflectance is Lunar-Lambert, the standard empirical model for the Moon and
    the one USGS ISIS3 fits in its photometric-correction tools:
        R = A * [ 2L*mu0 / (mu0 + mu) + (1 - L)*mu0 ]
    with mu0 = cos(incidence) on the local slope, mu = cos(emission) nadir.
    """
    rng = np.random.default_rng(seed)
    n = surface_normals(dem, gsd_m)
    s = _sun_vector(sun_az_deg, sun_el_deg)

    cos_i = n @ s
    mu0 = np.clip(cos_i, 0.0, 1.0)                 # cos incidence on the slope
    mu = np.clip(n[..., 2], 1e-3, 1.0)             # cos emission, nadir view
    if albedo is None:
        albedo = np.full(dem.shape, 0.12, np.float32)   # lunar mare is dark

    denom = np.maximum(mu0 + mu, 1e-6)
    refl = albedo * (2.0 * LUNAR_LAMBERT_L * mu0 / denom + (1.0 - LUNAR_LAMBERT_L) * mu0)

    shadow = (cast_shadow_mask(dem, gsd_m, sun_az_deg, sun_el_deg)
              if shadows else np.zeros(dem.shape, bool))
    refl = np.where(shadow, refl * AMBIENT, refl)

    img = refl / max(float(refl.max()), 1e-6)
    if noise > 0:
        img = img + rng.normal(0.0, noise, img.shape)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    incidence_deg = np.rad2deg(np.arccos(np.clip(cos_i, -1.0, 1.0))).astype(np.float32)
    return img, incidence_deg, shadow


# ---------------------------------------------------------------------------
# Cross-modal response
# ---------------------------------------------------------------------------
def cross_modal_remap(img, strength=1.0, invert=True, seed=0):
    """Simulate a different sensor's radiometric response.

    A panchromatic camera and a hyperspectral band do not respond to the same
    surface the same way, and the relationship is nonlinear and sometimes
    contrast-reversing. This is the condition RIFT / MIND-family descriptors
    exist for, so our cross-modal tests need it to be genuinely nonlinear
    rather than a linear brightness change any matcher survives.
    """
    rng = np.random.default_rng(seed)
    x = np.clip(img.astype(np.float32), 0.0, 1.0)
    gamma = 1.0 + 1.6 * strength
    y = x ** gamma
    # Add a smooth non-monotonic term so it is not a simple monotone curve.
    y = y + 0.28 * strength * np.sin(np.pi * x) ** 2
    if invert:
        y = 1.0 - y
    y = y + rng.normal(0.0, 0.004, y.shape)
    y = y - y.min()
    return np.clip(y / max(float(y.max()), 1e-6), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------
def homography(scale=1.0, rot_deg=0.0, shift=(0.0, 0.0), centre=(0.0, 0.0)):
    """Build the ground-truth source -> reference homography.

    Maps a source pixel (x, y) to the reference pixel showing the same ground.
    `scale` is reference pixels per source pixel: scale=2 means one source pixel
    covers two reference pixels, i.e. the source is the coarser image.
    """
    th = np.deg2rad(rot_deg)
    c, s = np.cos(th) * scale, np.sin(th) * scale
    cx, cy = centre
    a = np.array([[c, -s, 0.0],
                  [s, c, 0.0],
                  [0.0, 0.0, 1.0]], np.float64)
    # Rotate/scale about the given centre, then translate.
    pre = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    post = np.array([[1, 0, cx + shift[0]], [0, 1, cy + shift[1]], [0, 0, 1]], np.float64)
    return post @ a @ pre


def _meta(product_id, instrument, mission, gsd_m, shape, sun_az, sun_el):
    return SceneMeta(
        product_id=product_id,
        instrument=instrument,
        mission=mission,
        gsd_m=float(gsd_m),
        n_bands=1,
        wavelength_nm=(500.0, 800.0),
        array_shape=(int(shape[0]), int(shape[1])),
        dtype="float32",
        corner_latlon=[],
        sub_solar_azimuth_deg=float(sun_az),
        solar_incidence_deg=float(90.0 - sun_el),
        emission_deg=0.0,
        phase_deg=None,
        acquisition_utc=None,
        label_path=Path("<synthetic>"),
        raster_path=Path("<synthetic>"),
        # Nothing here came from a real label, and we say so (anti-stub honesty).
        label_fields_verified={},
    )


def make_pair(out_shape=(512, 512), scale=1.0, rot_deg=0.0, shift=(0.0, 0.0),
              sun_ref=(135.0, 45.0), sun_src=(135.0, 45.0), cross_modal=False,
              ref_gsd_m=0.5, ref_instrument="NAC", src_instrument="OHRC",
              n_craters=70, seed=0, noise=0.006, shadows=True):
    """Build a (source, reference) ImagePlane pair with an exact known transform.

    Returns (src_plane, ref_plane, H_true) where H_true maps source pixel
    coordinates to reference pixel coordinates, exactly, by construction.

    Both views are rendered from the SAME terrain, so a differing sun_src moves
    the shadows physically rather than just changing brightness.
    """
    h, w = out_shape
    # Render on a larger canvas so the warped source never samples empty space.
    margin = int(max(h, w) * (0.6 + 0.5 * max(scale, 1.0)))
    ch, cw = h + 2 * margin, w + 2 * margin

    dem = lunar_dem((ch, cw), gsd_m=ref_gsd_m, n_craters=int(n_craters * (ch * cw) / (h * w)),
                    seed=seed)
    img_ref_full, inc_ref_full, shadow_ref_full = render(
        dem, ref_gsd_m, sun_ref[0], sun_ref[1], noise=noise, seed=seed, shadows=shadows)
    same_sun = (abs(sun_src[0] - sun_ref[0]) < 1e-9 and abs(sun_src[1] - sun_ref[1]) < 1e-9)
    if same_sun:
        img_src_full, inc_src_full, shadow_src_full = img_ref_full, inc_ref_full, shadow_ref_full
    else:
        img_src_full, inc_src_full, shadow_src_full = render(
            dem, ref_gsd_m, sun_src[0], sun_src[1], noise=noise, seed=seed + 1, shadows=shadows)

    # Reference is a plain centre crop of the canvas.
    oy, ox = margin, margin
    ref_img = img_ref_full[oy:oy + h, ox:ox + w].copy()
    ref_inc = inc_ref_full[oy:oy + h, ox:ox + w].copy()
    ref_shadow = shadow_ref_full[oy:oy + h, ox:ox + w].copy()
    ref_dem = dem[oy:oy + h, ox:ox + w].copy()

    # H maps source pixels -> reference pixels; canvas coords add the crop offset.
    H = homography(scale=scale, rot_deg=rot_deg, shift=shift, centre=(w / 2.0, h / 2.0))
    offset = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float64)
    H_canvas = offset @ H       # source pixel -> canvas pixel

    src_full = img_src_full
    if scale > 1.0:
        # A coarser sensor integrates over a larger footprint: blur before
        # sampling, otherwise we would be simulating aliasing, not resolution.
        sigma = 0.5 * float(scale)
        k = int(2 * round(3 * sigma) + 1)
        src_full = cv2.GaussianBlur(src_full, (k, k), sigma)

    def warp(a, interp):
        return cv2.warpPerspective(a, H_canvas, (w, h),
                                   flags=interp | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REFLECT_101)

    src_img = warp(src_full, cv2.INTER_CUBIC)
    src_inc = warp(inc_src_full, cv2.INTER_LINEAR)
    src_dem = warp(dem, cv2.INTER_LINEAR)
    src_shadow = warp(shadow_src_full.astype(np.float32), cv2.INTER_NEAREST) > 0.5

    if cross_modal:
        src_img = cross_modal_remap(src_img, seed=seed + 7)

    src_gsd = ref_gsd_m * float(scale)
    src_slope, src_aspect = slope_aspect(src_dem, src_gsd)
    ref_slope, ref_aspect = slope_aspect(ref_dem, ref_gsd_m)

    src_plane = ImagePlane(
        array=np.clip(src_img, 0, 1).astype(np.float32),
        valid_mask=np.ones((h, w), bool),
        shadow_mask=src_shadow,
        gsd_m=src_gsd,
        meta=_meta("SYNTH_SRC", src_instrument, "SYNTH", src_gsd, (h, w),
                   sun_src[0], sun_src[1]),
        geo=GeometryLayers(incidence_deg=src_inc, dem_elev_m=src_dem,
                           slope_deg=src_slope, aspect_deg=src_aspect,
                           source="derived"),
        preprocess_chain=["synthetic"],
    )
    ref_plane = ImagePlane(
        array=np.clip(ref_img, 0, 1).astype(np.float32),
        valid_mask=np.ones((h, w), bool),
        shadow_mask=ref_shadow,
        gsd_m=float(ref_gsd_m),
        meta=_meta("SYNTH_REF", ref_instrument, "SYNTH", ref_gsd_m, (h, w),
                   sun_ref[0], sun_ref[1]),
        geo=GeometryLayers(incidence_deg=ref_inc, dem_elev_m=ref_dem,
                           slope_deg=ref_slope, aspect_deg=ref_aspect,
                           source="derived"),
        preprocess_chain=["synthetic"],
    )
    return src_plane, ref_plane, H


def transform_points(H, pts):
    """Apply a homography to (N,2) points."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    ones = np.ones((len(pts), 1))
    out = (H @ np.hstack([pts, ones]).T).T
    return out[:, :2] / out[:, 2:3]
