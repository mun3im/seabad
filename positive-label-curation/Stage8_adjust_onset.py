#!/usr/bin/env python3
"""
Stage8_adjust_onset.py

Interactive onset adjuster for Stage 7 QA review.
Opens a source FLAC, shows the full spectrogram + waveform, and lets you
drag a 3-second window to the correct position. Saves the result directly
to qa_corrections.csv for Stage 9 to apply.

Usage:
    # File picker — loops until cancelled
    python Stage8_adjust_onset.py

    # Jump straight to a specific clip
    python Stage8_adjust_onset.py xc422286_A_3000.wav
"""

import argparse
import csv
import json
import os
import re
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation

import config

# ── Config from config.py ─────────────────────────────────────────────────────
FLAC_OUTPUT_DIR = config.FLAC_OUTPUT_DIR
POSITIVE_FINAL_DIR = config.POSITIVE_FINAL_DIR
QA_CSV      = config.STAGE8_QA_CSV
SEGMENT_SEC = 3.0
SR          = config.TARGET_SAMPLE_RATE
N_FFT       = 512
HOP_LENGTH  = 128
FMAX        = 8_000
ISSUE_TYPES = ["wrong_onset", "noise_dominated", "no_bird"]
_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           config.STAGE8_STATE_FILE)
# ─────────────────────────────────────────────────────────────────────────────

_FNAME_RE = re.compile(r"^xc(\d+)_([A-Ea-e])_(\d+)\.wav$")


def _parse(fname: str):
    m = _FNAME_RE.match(Path(fname).name)
    if not m:
        raise ValueError(f"Cannot parse clip filename: {fname}")
    return m.group(1), m.group(2).upper(), int(m.group(3))


def _find_flac(xc_id: str, quality: str) -> Path:
    matches = list(Path(FLAC_OUTPUT_DIR).rglob(f"xc{xc_id}_{quality}.flac"))
    if not matches:
        raise FileNotFoundError(
            f"xc{xc_id}_{quality}.flac not found under {FLAC_OUTPUT_DIR}"
        )
    return matches[0]


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _load_qa() -> dict:
    rows = {}
    if os.path.exists(QA_CSV):
        with open(QA_CSV, newline="") as f:
            for row in csv.DictReader(f):
                rows[row["clip_filename"]] = row
    return rows


def _save_qa(rows: dict) -> None:
    with open(QA_CSV, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["clip_filename", "issue_type", "corrected_onset_ms"],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows.values())


# ── State persistence (remembers last browse directory) ───────────────────────

def _load_state() -> dict:
    try:
        with open(_STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def _save_state(d: dict) -> None:
    try:
        with open(_STATE_FILE, "w") as f:
            json.dump(d, f)
    except OSError:
        pass


# ── Interactive viewer ────────────────────────────────────────────────────────

def review_clip(clip_filename: str, qa_rows: dict) -> bool:
    """
    Show the interactive viewer for one clip.
    Returns True if the user saved, False if skipped/closed.
    """
    try:
        xc_id, quality, orig_onset_ms = _parse(clip_filename)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return False

    try:
        flac_path = _find_flac(xc_id, quality)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    print(f"\nLoading  {flac_path.name}  (current onset: {orig_onset_ms} ms)")
    y, sr = librosa.load(str(flac_path), sr=SR, mono=True)
    duration = len(y) / sr

    if duration < SEGMENT_SEC:
        print(f"  [SKIP] recording too short ({duration:.2f} s)")
        return False

    # Starting position — clamp to valid range
    start_s = max(0.0, min(orig_onset_ms / 1000.0, duration - SEGMENT_SEC))

    existing     = qa_rows.get(clip_filename, {})
    initial_issue = existing.get("issue_type", "wrong_onset")
    if initial_issue not in ISSUE_TYPES:
        initial_issue = "wrong_onset"

    state = {"start_s": start_s, "issue": initial_issue, "saved": False}

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_spec, ax_wave) = plt.subplots(
        2, 1, figsize=(20, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
    )
    fig.canvas.manager.set_window_title(
        f"Stage 8 QA — {clip_filename}  [{flac_path.name}]"
    )
    plt.subplots_adjust(bottom=0.30)

    # Spectrogram
    S_db = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=80, fmax=FMAX
        ),
        ref=np.max,
    )
    librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP_LENGTH,
        y_axis="mel", x_axis="time", ax=ax_spec, cmap="magma",
    )
    ax_spec.set_title("Spectrogram — drag the blue window to the correct onset")
    ax_spec.set_xlabel("")
    ax_spec.set_xlim(0, duration)

    # Mark original onset
    ax_spec.axvline(orig_onset_ms / 1000.0, color="white", linewidth=1,
                    linestyle=":", alpha=0.6, label="original")

    # Waveform
    times = np.linspace(0, duration, len(y))
    ax_wave.plot(times, y, color="black", linewidth=0.5)
    ax_wave.axvline(orig_onset_ms / 1000.0, color="grey", linewidth=1,
                    linestyle=":", alpha=0.6)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_xlim(0, duration)

    # Draggable 3-second selection window - store in list so we can replace
    spans = {
        "spec": ax_spec.axvspan(state["start_s"], state["start_s"] + SEGMENT_SEC,
                                 alpha=0.30, color="deepskyblue", zorder=3),
        "wave": ax_wave.axvspan(state["start_s"], state["start_s"] + SEGMENT_SEC,
                                 alpha=0.30, color="deepskyblue", zorder=3)
    }

    def _refresh_spans():
        s = state["start_s"]
        # Remove old spans and create new ones
        spans["spec"].remove()
        spans["wave"].remove()
        spans["spec"] = ax_spec.axvspan(s, s + SEGMENT_SEC,
                                         alpha=0.30, color="deepskyblue", zorder=3)
        spans["wave"] = ax_wave.axvspan(s, s + SEGMENT_SEC,
                                         alpha=0.30, color="deepskyblue", zorder=3)
        fig.canvas.draw_idle()

    # ── Widgets ───────────────────────────────────────────────────────────────
    # Issue-type radio buttons
    ax_radio = plt.axes([0.02, 0.04, 0.16, 0.18])
    ax_radio.set_title("Issue type", fontsize=8)
    radio = widgets.RadioButtons(
        ax_radio, ISSUE_TYPES, active=ISSUE_TYPES.index(initial_issue)
    )

    # Live onset readout
    ax_info = plt.axes([0.25, 0.13, 0.28, 0.06])
    ax_info.axis("off")
    info_text = ax_info.text(
        0.0, 0.5, f"onset: {int(round(state['start_s'] * 1000))} ms",
        fontsize=11, va="center",
    )

    # Manual onset text box
    ax_tb = plt.axes([0.25, 0.06, 0.20, 0.05])
    tb = widgets.TextBox(ax_tb, "set onset (ms) ", initial=str(orig_onset_ms))

    def _apply_textbox(text):
        try:
            s = max(0.0, min(int(text) / 1000.0, duration - SEGMENT_SEC))
            state["start_s"] = s
            info_text.set_text(f"onset: {int(round(s * 1000))} ms")
            _refresh_spans()
        except ValueError:
            pass

    tb.on_submit(_apply_textbox)

    # Play / Stop / Save / Skip
    ax_play = plt.axes([0.58, 0.08, 0.08, 0.05])
    ax_save = plt.axes([0.72, 0.08, 0.10, 0.05])
    ax_skip = plt.axes([0.84, 0.08, 0.08, 0.05])
    btn_play = widgets.Button(ax_play, "Play")
    btn_save = widgets.Button(ax_save, "Save")
    btn_skip = widgets.Button(ax_skip, "Skip")

    # ── Playback ──────────────────────────────────────────────────────────────
    prog_spec = ax_spec.axvline(0, color="red", linewidth=1.5, visible=False)
    prog_wave = ax_wave.axvline(0, color="red", linewidth=1.5, visible=False)
    pb = {"on": False, "t0": None, "anim": None, "start_offset": 0.0}

    def _stop_pb():
        sd.stop()
        pb["on"] = False
        btn_play.label.set_text("Play")
        prog_spec.set_visible(False)
        prog_wave.set_visible(False)
        if pb["anim"]:
            pb["anim"].event_source.stop()
            pb["anim"] = None
        fig.canvas.draw_idle()

    def _tick(frame):
        if not pb["on"]:
            return []
        elapsed = time.time() - pb["t0"]
        if elapsed >= SEGMENT_SEC:
            _stop_pb()
            return []
        current_time = pb["start_offset"] + elapsed
        for ln in (prog_spec, prog_wave):
            ln.set_xdata([current_time, current_time])
        return [prog_spec, prog_wave]

    def toggle_play(event):
        if pb["on"]:
            _stop_pb()
        else:
            # Extract and play only the 3-second segment
            start_s = state["start_s"]
            start_sample = int(start_s * sr)
            end_sample = int((start_s + SEGMENT_SEC) * sr)
            segment = y[start_sample:end_sample]

            sd.play(segment, sr)
            pb.update(on=True, t0=time.time(), start_offset=start_s)
            btn_play.label.set_text("Stop")
            for ln in (prog_spec, prog_wave):
                ln.set_visible(True)
                ln.set_xdata([start_s, start_s])
            pb["anim"] = FuncAnimation(
                fig, _tick, interval=50, blit=True, cache_frame_data=False
            )
            fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)

    # ── Drag interaction ──────────────────────────────────────────────────────
    drag = {"on": False, "x0": None, "s0": None}

    def on_press(event):
        if event.inaxes not in (ax_spec, ax_wave) or event.xdata is None:
            return
        s = state["start_s"]
        if s <= event.xdata <= s + SEGMENT_SEC:
            drag.update(on=True, x0=event.xdata, s0=s)

    def on_motion(event):
        if not drag["on"] or event.xdata is None:
            return
        new_s = max(0.0, min(drag["s0"] + event.xdata - drag["x0"],
                             duration - SEGMENT_SEC))
        state["start_s"] = new_s
        ms = int(round(new_s * 1000))
        info_text.set_text(f"onset: {ms} ms")
        tb.set_val(str(ms))
        _refresh_spans()

    def on_release(event):
        drag["on"] = False

    fig.canvas.mpl_connect("button_press_event",   on_press)
    fig.canvas.mpl_connect("motion_notify_event",  on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    # ── Radio callback ────────────────────────────────────────────────────────
    radio.on_clicked(lambda label: state.update(issue=label))

    # ── Save / Skip ───────────────────────────────────────────────────────────
    def do_save(event):
        _stop_pb()
        ms    = int(round(state["start_s"] * 1000))
        issue = state["issue"]
        qa_rows[clip_filename] = {
            "clip_filename":      clip_filename,
            "issue_type":         issue,
            "corrected_onset_ms": "" if issue == "no_bird" else str(ms),
        }
        _save_qa(qa_rows)
        print(f"  Saved  {clip_filename}  issue={issue}  onset={ms} ms")
        state["saved"] = True
        plt.close(fig)

    def do_skip(event):
        _stop_pb()
        print(f"  Skipped  {clip_filename}")
        plt.close(fig)

    btn_save.on_clicked(do_save)
    btn_skip.on_clicked(do_skip)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)
    plt.show()
    return state["saved"]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive onset adjuster — writes to qa_corrections.csv"
    )
    parser.add_argument(
        "clip_filename", nargs="?",
        help="Clip to review (e.g. xc422286_A_3000.wav). "
             "Omit to open a file picker.",
    )
    args = parser.parse_args()

    # Ensure metadata directory exists
    os.makedirs(config.METADATA_DIR, exist_ok=True)

    qa_rows = _load_qa()
    app_state = _load_state()

    if args.clip_filename:
        review_clip(args.clip_filename, qa_rows)
        return

    # File-picker loop — keeps opening clips until the user cancels
    initial_dir = app_state.get("last_dir", str(POSITIVE_FINAL_DIR))
    if not os.path.isdir(initial_dir):
        initial_dir = str(POSITIVE_FINAL_DIR)

    while True:
        root = tk.Tk()
        root.withdraw()
        chosen = filedialog.askopenfilename(
            title="Select a positive clip to review  (Cancel to exit)",
            initialdir=initial_dir,
            filetypes=[("WAV clips", "*.wav"), ("All files", "*.*")],
        )
        root.destroy()

        if not chosen:
            print("Done.")
            break

        initial_dir = os.path.dirname(os.path.abspath(chosen))
        _save_state({"last_dir": initial_dir})

        review_clip(Path(chosen).name, qa_rows)
        qa_rows = _load_qa()   # reload in case of concurrent edits


if __name__ == "__main__":
    main()
