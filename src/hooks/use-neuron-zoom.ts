/**
 * useNeuronZoom — Focus-mode zoom controller
 *
 * Manages zoom state for the 3D neuron canvas without hijacking
 * page scroll. Zoom is only active when "Focus Mode" is explicitly
 * toggled on, which also locks page scroll.
 *
 * @module hooks/use-neuron-zoom
 */

"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { clamp } from "@/lib/utils";

interface UseNeuronZoomOptions {
  /** Zoom value (0–1) at which the warp-speed animation triggers */
  warpThreshold?: number;
}

interface UseNeuronZoomReturn {
  /** Current zoom level, 0–1 */
  zoom: number;
  /** Whether focus mode (scroll-zoom) is active */
  focusMode: boolean;
  /** Whether the warp animation is currently playing */
  isWarping: boolean;
  /** Whether the user has completed the warp & reached the nucleus */
  hasReachedNucleus: boolean;
  /** Toggle focus mode on/off */
  toggleFocusMode: () => void;
  /** Exit focus mode and reset zoom */
  exitFocusMode: () => void;
  /** The wheel handler to attach to the canvas */
  handleWheel: (e: WheelEvent) => void;
  /** Reset warp state (e.g. after closing fact carousel) */
  resetWarp: () => void;
}

export function useNeuronZoom(
  opts: UseNeuronZoomOptions = {}
): UseNeuronZoomReturn {
  const { warpThreshold = 0.85 } = opts;

  const [zoom, setZoom] = useState(0);
  const [focusMode, setFocusMode] = useState(false);
  const [isWarping, setIsWarping] = useState(false);
  const [hasReachedNucleus, setHasReachedNucleus] = useState(false);

  const warpTriggered = useRef(false);

  /* ─── Lock / Unlock Page Scroll ──────────────────────────── */

  useEffect(() => {
    if (focusMode) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [focusMode]);

  /* ─── Wheel Handler (only works when focusMode is on) ───── */

  const handleWheel = useCallback(
    (e: WheelEvent) => {
      if (!focusMode || isWarping) return;

      e.preventDefault();
      const delta = e.deltaY * -0.003;
      setZoom((prev) => {
        const next = clamp(prev + delta, 0, 1);

        if (next >= warpThreshold && !warpTriggered.current) {
          warpTriggered.current = true;
          setIsWarping(true);
          // After warp animation completes, mark nucleus reached
          setTimeout(() => {
            setIsWarping(false);
            setHasReachedNucleus(true);
          }, 1800);
        }

        return next;
      });
    },
    [focusMode, isWarping, warpThreshold]
  );

  /* ─── Toggle ────────────────────────────────────────────── */

  const toggleFocusMode = useCallback(() => {
    setFocusMode((prev) => {
      if (prev) {
        // Exiting focus mode — reset zoom
        setZoom(0);
        warpTriggered.current = false;
        setIsWarping(false);
        setHasReachedNucleus(false);
      }
      return !prev;
    });
  }, []);

  const exitFocusMode = useCallback(() => {
    setFocusMode(false);
    setZoom(0);
    warpTriggered.current = false;
    setIsWarping(false);
    setHasReachedNucleus(false);
  }, []);

  const resetWarp = useCallback(() => {
    setHasReachedNucleus(false);
    setIsWarping(false);
    warpTriggered.current = false;
    setZoom(0);
    setFocusMode(false);
  }, []);

  return {
    zoom,
    focusMode,
    isWarping,
    hasReachedNucleus,
    toggleFocusMode,
    exitFocusMode,
    handleWheel,
    resetWarp,
  };
}
