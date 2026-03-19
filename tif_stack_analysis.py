#!/usr/bin/env python3
"""Process TIFF fluorescence stacks with flexible, interactive preprocessing.

This script is intended for analyzing microscopy fluorescence image
sequences stored as multi-page TIFF stacks.  It guides the user through
several optional, interactive steps before computing dF/F₀ and exporting
the result as a float stack.

Features
--------
* **Background subtraction (optional)** – draw a rectangle on the mean
  image (across the whole stack) to specify a background ROI; its average
  intensity is subtracted from each frame.  Use ``--no-bg`` to bypass.
* **Baseline calculation** – compute F₀ as the per-pixel mean over a baseline
  frame range (default: first 20 frames), and compute per-pixel stddev.
* **Thresholding** – zero out per-pixel deltaF values that are smaller than
  a multiple of the baseline stddev (``--thr-mult``), matching the MATLAB
  approach.
* **Spatial smoothing** – apply a median filter to each output frame
  (``--median``) to reduce noise and emulate MATLAB's ``medfilt2``.
* **Gaussian blur** – apply optional Gaussian smoothing (``--blur``).

Configuration
-------------
``--baseline START END``
    Frame indices (or times if ``--fps`` is provided) used to compute
    the baseline F₀ (inclusive end index).

``--fps``
    Frames per second used to translate time values in ``--baseline``.

``--thr-mult``
    Multiplier for the baseline per-pixel stddev; deltaF values below
    ``thr_mult * stddev`` are set to 0 (matched to the MATLAB behavior).

``--median``
    Spatial median filter size (odd integer).  Used to smooth each output
    frame and emulate MATLAB ``medfilt2``.

``--blur SIGMA``
    Sigma for Gaussian blur in pixels (default 1.0).

``--no-bg``
    Skip background ROI selection and subtraction entirely.

Output
------
The output is a **grayscale** 32‑bit floating point TIFF stack with the
same number of frames as the input, containing the computed dF/F₀ values.

Dependencies [Environment name: TIFFPROCESS]
------------
- numpy
- tifffile
- matplotlib
- scipy (for gaussian_filter)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Slider
from scipy.ndimage import gaussian_filter, median_filter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def select_roi(image: np.ndarray) -> tuple[int, int, int, int]:
    """Interactively draw a rectangular ROI on ``image``.

    The user draws a single box with the mouse.  The figure window also
    contains two sliders that control the display *contrast* (vmin/vmax).
    The figure must be closed to continue.  Returns ``(x, y, width, height)``
    in pixel coordinates.
    """

    coords: list[int] = []

    # set up figure with room for two sliders at the bottom
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(image, cmap="gray")
    ax.set_title("Draw rectangle; close window when finished")

    # sliders for contrast
    vmax = float(np.nanmax(image))
    vmin = float(np.nanmin(image))
    axlow = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    axhigh = fig.add_axes([0.25, 0.01, 0.65, 0.03])
    slider_low = Slider(axlow, "Low", vmin, vmax, valinit=vmin)
    slider_high = Slider(axhigh, "High", vmin, vmax, valinit=vmax)

    def update_contrast(val):
        im.set_clim(slider_low.val, slider_high.val)
        fig.canvas.draw_idle()

    slider_low.on_changed(update_contrast)
    slider_high.on_changed(update_contrast)

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))
        coords[:] = [xmin, ymin, xmax - xmin, ymax - ymin]

    rs = RectangleSelector(ax, onselect,
                           useblit=True,
                           button=[1],  # left mouse button only
                           minspanx=5, minspany=5, spancoords="pixels",
                           interactive=True)
    plt.show()

    if not coords:
        raise RuntimeError("ROI selection cancelled or no box drawn")
    return tuple(coords)  # type: ignore[return-value]


def parse_baseline(baseline_args, fps: float | None) -> tuple[int, int]:
    """Convert command-line baseline specification to frame indices.

    The baseline range is inclusive (matching the MATLAB script behavior).
    """
    start, end = baseline_args
    if fps is not None:
        start = int(start * fps)
        end = int(end * fps)
    return int(start), int(end)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute background-subtracted dF/F0 from a TIFF stack")
    parser.add_argument("input", help="path to input TIFF stack")
    parser.add_argument("-o", "--output", default=None,
                        help="output TIFF path (default: same directory as input, named dFF0_<input>.tif)")
    parser.add_argument("--baseline", nargs=2, type=float, default=[0, 19],
                        help="baseline start and end (frames, inclusive); default is 0 19")
    parser.add_argument("--fps", type=float,
                        help="frames per second (used with --baseline)")
    parser.add_argument("--thr-mult", type=int, choices=[0, 1, 2], default=1,
                        help="threshold multiplier for baseline stddev; choose 0/1/2 (default 1)")
    parser.add_argument("--blur", type=float, default=1.0,
                        help="gaussian blur sigma (0 to disable, default 1)")
    parser.add_argument("--median", type=int, default=2,
                        help="spatial median filter size (odd integer, 0 to disable, default 2)")
    parser.add_argument("--bg-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="background ROI (x y w h); if omitted, ROI is selected interactively on the mean image")
    parser.add_argument("--no-bg", action="store_true",
                        help="skip background subtraction")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        args.output = str(input_path.parent / f"dFF0_{input_path.stem}.tif")

    # read stack
    print(f"Reading {args.input}...")
    stack = tifffile.imread(args.input)
    if stack.ndim != 3:
        raise ValueError("expected a 3D stack (frames, height, width)")

    # Background subtraction (ROI-based) is optional.
    if args.no_bg:
        print("skipping background subtraction")
        stack_bg = stack.astype(np.float32)
    else:
        # For interactive ROI selection use the mean image across all frames.
        if args.bg_roi is None:
            mean_image = stack.mean(axis=0)
            x0, y0, w, h = select_roi(mean_image)
        else:
            x0, y0, w, h = args.bg_roi

        print(f"ROI = x={x0}, y={y0}, w={w}, h={h}")

        # validate ROI is inside image bounds
        height, width = stack.shape[1:]
        if not (0 <= x0 < width and 0 <= y0 < height and
                w > 0 and h > 0 and x0 + w <= width and y0 + h <= height):
            raise ValueError("background ROI must lie within the image bounds")

        # calculate per-frame background mean
        roi = stack[:, y0 : y0 + h, x0 : x0 + w]
        bg_mean = roi.mean(axis=(1, 2))
        stack_bg = stack.astype(np.float32) - bg_mean[:, None, None]

    # optionally blur; sigma<=0 disables
    if args.blur and args.blur > 0:
        stack_bg = gaussian_filter(stack_bg,
                                   sigma=(0, args.blur, args.blur))

    # after background subtraction (and blur), let user pick a lower
    # threshold interactively.  Values below threshold will be set to NaN so
    # they are ignored in later calculations.
    def choose_threshold(image: np.ndarray) -> float:
        """Show ``image`` with sliders to pick threshold and contrast.

        Three sliders appear: ``Low``/``High`` control display range while
        ``Thresh`` selects the lower cutoff that will be applied to the data.
        """
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(image, cmap="gray")
        ax.set_title("Adjust contrast and threshold; close when satisfied")

        vmin = float(np.nanmin(image))
        vmax = float(np.nanmax(image))
        # axes for sliders
        axlow = fig.add_axes([0.25, 0.03, 0.65, 0.03])
        axhigh = fig.add_axes([0.25, 0.00, 0.65, 0.03])
        axth = fig.add_axes([0.25, 0.06, 0.65, 0.03])
        slider_low = Slider(axlow, "Low", vmin, vmax, valinit=vmin)
        slider_high = Slider(axhigh, "High", vmin, vmax, valinit=vmax)
        slider_thr = Slider(axth, "Thresh", vmin, vmax, valinit=vmin)

        def update_contrast(val):
            lo, hi = slider_low.val, slider_high.val
            im.set_clim(lo, hi)
            fig.canvas.draw_idle()

        def update_threshold(val):
            thr = slider_thr.val
            im.set_clim(thr, slider_high.val)
            fig.canvas.draw_idle()

        slider_low.on_changed(update_contrast)
        slider_high.on_changed(update_contrast)
        slider_thr.on_changed(update_threshold)

        plt.show()
        return slider_thr.val

    threshold = choose_threshold(stack_bg[0])
    print(f"using threshold {threshold:.3f}")
    # apply to whole stack
    stack_bg[stack_bg < threshold] = np.nan

    # determine baseline frames (inclusive end index)
    start, end = parse_baseline(args.baseline, args.fps)
    if not (0 <= start <= end < stack.shape[0]):
        raise ValueError("baseline indices must satisfy 0 <= start <= end < nframes")

    print(f"using baseline frames {start}..{end}")

    # compute baseline mean and stddev (MATLAB-style)
    baseline_frames = stack_bg[start : end + 1].astype(np.float64)
    n_baseline = baseline_frames.shape[0]

    sumimage = baseline_frames.sum(axis=0)
    sum2image = (baseline_frames**2).sum(axis=0)
    F0 = sumimage / n_baseline
    variance = sum2image / n_baseline - F0**2
    stddev = np.sqrt(np.maximum(variance, 0.0))

    # any non‑positive baseline values are invalid – mark as NaN
    F0[F0 <= 0] = np.nan

    # calculate deltaF and threshold it like MATLAB does
    deltaF = stack_bg.astype(np.float64) - F0
    deltaF[deltaF <= args.thr_mult * stddev] = 0.0

    # calculate dF/F0 per frame
    dff = deltaF / F0

    # optionally smooth result spatially (make output appear smoother)
    if args.median and args.median > 1:
        # median_filter uses (frames, y, x) size tuple; keep frames unchanged
        dff = median_filter(dff, size=(1, args.median, args.median))

    # output as single‑channel 32‑bit float TIFF (grayscale)
    print(f"writing dF/F0 stack to {args.output}...")
    tifffile.imwrite(args.output, dff.astype(np.float32))
    print("done")


if __name__ == "__main__":
    main()
