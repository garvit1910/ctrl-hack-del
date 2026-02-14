/**
 * NeuroDetect — Living Neuron 3D Canvas Engine
 *
 * Recursive 3D projected neuron visualization rendered on HTML5 Canvas.
 * Features a Focus Mode toggle for intentional zoom interaction (no
 * scroll-hijacking), a warp-speed entry animation when approaching
 * the nucleus, and an animated fact carousel on arrival.
 *
 * @module components/canvas/HeroNeuron
 */

"use client";

import React, {
  useRef,
  useEffect,
  useMemo,
} from "react";
import { AnimatePresence } from "framer-motion";
import {
  project3D,
  rotateX,
  rotateY,
  lerp,
  clamp,
  type Vec3,
} from "@/lib/utils";
import { useNeuronZoom } from "@/hooks/use-neuron-zoom";
import { FactCarousel } from "@/components/ui/fact-carousel";
import { Crosshair, X } from "lucide-react";
import { Button } from "@/components/ui/button";

/* ─── Branch Node Type ─────────────────────────────────────────────────── */

interface BranchNode {
  start: Vec3;
  end: Vec3;
  depth: number;
  children: BranchNode[];
  pulsePhase: number;
}

/* ─── Neuron Tree Generator ────────────────────────────────────────────── */

function buildNeuronTree(
  origin: Vec3,
  dir: Vec3,
  length: number,
  depth: number,
  maxDepth: number
): BranchNode {
  const end: Vec3 = {
    x: origin.x + dir.x * length,
    y: origin.y + dir.y * length,
    z: origin.z + dir.z * length,
  };

  const children: BranchNode[] = [];

  if (depth < maxDepth) {
    const numBranches = depth < 2 ? 3 : 2;
    for (let i = 0; i < numBranches; i++) {
      const spread = 0.45 + depth * 0.05;
      const offset = spread * (i - (numBranches - 1) / 2);
      const jitter = (Math.random() - 0.5) * 0.25;

      let newDir = rotateY(dir, offset + jitter);
      newDir = rotateX(newDir, offset * 0.6 + jitter * 0.5);

      const mag = Math.sqrt(
        newDir.x ** 2 + newDir.y ** 2 + newDir.z ** 2
      );
      const normalized: Vec3 = {
        x: newDir.x / mag,
        y: newDir.y / mag,
        z: newDir.z / mag,
      };

      children.push(
        buildNeuronTree(
          end,
          normalized,
          length * 0.7,
          depth + 1,
          maxDepth
        )
      );
    }
  }

  return {
    start: origin,
    end,
    depth,
    children,
    pulsePhase: Math.random() * Math.PI * 2,
  };
}

/* ─── Synapse Particle ─────────────────────────────────────────────────── */

interface Particle {
  branch: BranchNode;
  t: number;
  speed: number;
  baseSpeed: number;
  size: number;
}

/* ─── Main Component ──────────────────────────────────────────────────── */

export default function HeroNeuron() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const timeRef = useRef<number>(0);
  const warpProgressRef = useRef<number>(0);

  const {
    zoom,
    focusMode,
    isWarping,
    hasReachedNucleus,
    toggleFocusMode,
    handleWheel,
    resetWarp,
  } = useNeuronZoom({ warpThreshold: 0.85 });

  // Build neuron tree once using multiple root directions
  const trees = useMemo(() => {
    const directions: Vec3[] = [
      { x: 0, y: -1, z: 0 },
      { x: 0.7, y: -0.7, z: 0 },
      { x: -0.7, y: -0.7, z: 0 },
      { x: 0, y: -0.7, z: 0.7 },
      { x: 0, y: -0.7, z: -0.7 },
      { x: 0, y: 1, z: 0 },
      { x: 0.5, y: 0.85, z: 0 },
      { x: -0.5, y: 0.85, z: 0 },
    ];

    return directions.map((d) =>
      buildNeuronTree({ x: 0, y: 0, z: 0 }, d, 120, 0, 7)
    );
  }, []);

  // Generate particles along branches
  const particles = useMemo(() => {
    const parts: Particle[] = [];
    function collect(node: BranchNode) {
      if (Math.random() < 0.3) {
        const spd = 0.002 + Math.random() * 0.004;
        parts.push({
          branch: node,
          t: Math.random(),
          speed: spd,
          baseSpeed: spd,
          size: 1 + Math.random() * 2,
        });
      }
      node.children.forEach(collect);
    }
    trees.forEach(collect);
    return parts;
  }, [trees]);

  /* ─── Canvas Setup & Animation Loop ────────────────────────── */

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Only add wheel listener when in focus mode
    if (focusMode) {
      canvas.addEventListener("wheel", handleWheel, { passive: false });
    }

    const ctx = canvas.getContext("2d")!;
    let width = 0;
    let height = 0;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      width = canvas!.clientWidth;
      height = canvas!.clientHeight;
      canvas!.width = width * dpr;
      canvas!.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    resize();
    window.addEventListener("resize", resize);

    /* ─── Render Branch Recursively ────────────────────────── */

    function renderBranch(
      node: BranchNode,
      time: number,
      focalLength: number,
      cx: number,
      cy: number,
      rotAngleY: number,
      rotAngleX: number,
      globalAlpha: number,
      cameraZ: number,
      warpFactor: number
    ) {
      // Rotate points
      let s = rotateY(node.start, rotAngleY);
      s = rotateX(s, rotAngleX);
      let e = rotateY(node.end, rotAngleY);
      e = rotateX(e, rotAngleX);

      // Move scene toward camera (fly into the neuron)
      s = { ...s, z: s.z + cameraZ };
      e = { ...e, z: e.z + cameraZ };

      const p1 = project3D(s, focalLength, cx, cy);
      const p2 = project3D(e, focalLength, cx, cy);

      // Pulse brightness
      const pulse =
        0.6 + 0.4 * Math.sin(time * 2 + node.pulsePhase);

      // Depth-based color: cyan core → blue tips
      const depthRatio = node.depth / 7;

      // During warp, branches get a speed-line stretch effect
      const warpStretch = 1 + warpFactor * 3 * (1 - depthRatio);

      const r = Math.round(lerp(0, 80, depthRatio));
      const g = Math.round(lerp(220, 140, depthRatio));
      const b = Math.round(lerp(255, 255, depthRatio));

      // Warp brightens everything
      const warpBright = 1 + warpFactor * 2;
      const lineWidth = Math.max(0.5, (1 - depthRatio) * 3 * p1.scale * warpStretch);
      const alpha = clamp(
        globalAlpha * pulse * (1 - depthRatio * 0.4) * p1.scale * warpBright,
        0,
        1
      );

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);

      // During warp, extend lines outward from center for speed-line effect
      if (warpFactor > 0.01) {
        const dx = p2.x - cx;
        const dy = p2.y - cy;
        ctx.lineTo(
          p2.x + dx * warpFactor * 2,
          p2.y + dy * warpFactor * 2
        );
      } else {
        ctx.lineTo(p2.x, p2.y);
      }
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = "round";
      ctx.stroke();

      // Render children
      node.children.forEach((child) =>
        renderBranch(
          child,
          time,
          focalLength,
          cx,
          cy,
          rotAngleY,
          rotAngleX,
          globalAlpha,
          cameraZ,
          warpFactor
        )
      );
    }

    /* ─── Render Nucleus ───────────────────────────────────── */

    function renderNucleus(
      time: number,
      focalLength: number,
      cx: number,
      cy: number,
      rotAngleY: number,
      rotAngleX: number,
      zoomLevel: number,
      cameraZ: number,
      warpFactor: number
    ) {
      let nucleusPos: Vec3 = { x: 0, y: 0, z: 0 };
      nucleusPos = rotateY(nucleusPos, rotAngleY);
      nucleusPos = rotateX(nucleusPos, rotAngleX);
      nucleusPos = { ...nucleusPos, z: nucleusPos.z + cameraZ };

      const projected = project3D(nucleusPos, focalLength, cx, cy);
      const baseRadius = 12 + zoomLevel * 30;
      const pulse = 1 + 0.15 * Math.sin(time * 3);
      // During warp, nucleus expands dramatically
      const warpExpand = 1 + warpFactor * 8;
      const radius = baseRadius * projected.scale * pulse * warpExpand;

      // Outer glow — amplified during warp
      const glowAlpha = (0.3 + warpFactor * 0.5) * pulse;
      const glowGrad = ctx.createRadialGradient(
        projected.x,
        projected.y,
        radius * 0.3,
        projected.x,
        projected.y,
        radius * (2.5 + warpFactor * 4)
      );
      glowGrad.addColorStop(0, `rgba(0, 220, 255, ${clamp(glowAlpha, 0, 1)})`);
      glowGrad.addColorStop(0.5, `rgba(0, 180, 255, ${clamp(glowAlpha * 0.4, 0, 1)})`);
      glowGrad.addColorStop(1, "rgba(0, 0, 0, 0)");

      ctx.beginPath();
      ctx.arc(
        projected.x,
        projected.y,
        radius * (2.5 + warpFactor * 4),
        0,
        Math.PI * 2
      );
      ctx.fillStyle = glowGrad;
      ctx.fill();

      // Core
      const coreGrad = ctx.createRadialGradient(
        projected.x,
        projected.y,
        0,
        projected.x,
        projected.y,
        radius
      );
      coreGrad.addColorStop(0, "rgba(200, 255, 255, 0.95)");
      coreGrad.addColorStop(0.4, "rgba(0, 220, 255, 0.8)");
      coreGrad.addColorStop(1, "rgba(0, 100, 200, 0.2)");

      ctx.beginPath();
      ctx.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = coreGrad;
      ctx.fill();
    }

    /* ─── Render Particles ─────────────────────────────────── */

    function renderParticles(
      time: number,
      focalLength: number,
      cx: number,
      cy: number,
      rotAngleY: number,
      rotAngleX: number,
      cameraZ: number,
      warpFactor: number
    ) {
      particles.forEach((p) => {
        // During warp, particles accelerate dramatically
        const speedMult = 1 + warpFactor * 15;
        p.t = (p.t + p.baseSpeed * speedMult) % 1;

        const pos: Vec3 = {
          x: lerp(p.branch.start.x, p.branch.end.x, p.t),
          y: lerp(p.branch.start.y, p.branch.end.y, p.t),
          z: lerp(p.branch.start.z, p.branch.end.z, p.t),
        };

        let rPos = rotateY(pos, rotAngleY);
        rPos = rotateX(rPos, rotAngleX);
        rPos = { ...rPos, z: rPos.z + cameraZ };

        const proj = project3D(rPos, focalLength, cx, cy);
        const alpha = 0.4 + 0.6 * Math.sin(time * 5 + p.branch.pulsePhase);

        // During warp, particles get trail effect
        const trailLen = warpFactor * 6;
        if (trailLen > 0.1) {
          const dx = proj.x - cx;
          const dy = proj.y - cy;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const nx = dx / dist;
          const ny = dy / dist;

          ctx.beginPath();
          ctx.moveTo(proj.x, proj.y);
          ctx.lineTo(proj.x + nx * trailLen * 8, proj.y + ny * trailLen * 8);
          ctx.strokeStyle = `rgba(0, 255, 220, ${clamp(alpha * proj.scale * 0.5, 0, 0.6)})`;
          ctx.lineWidth = p.size * proj.scale * 0.5;
          ctx.stroke();
        }

        ctx.beginPath();
        ctx.arc(
          proj.x,
          proj.y,
          p.size * proj.scale * (1 + warpFactor),
          0,
          Math.PI * 2
        );
        ctx.fillStyle = `rgba(0, 255, 220, ${clamp(
          alpha * proj.scale * (1 + warpFactor * 2),
          0,
          1
        )})`;
        ctx.fill();
      });
    }

    /* ─── Warp-Speed Star Lines ────────────────────────────── */

    function renderWarpLines(
      cx: number,
      cy: number,
      warpFactor: number,
      time: number
    ) {
      if (warpFactor < 0.01) return;

      const numLines = Math.floor(warpFactor * 80);
      ctx.save();
      for (let i = 0; i < numLines; i++) {
        const angle = (i / numLines) * Math.PI * 2 + time * 0.5;
        const innerR = 30 + Math.sin(time * 3 + i) * 20;
        const outerR = innerR + warpFactor * 400 * (0.5 + Math.random() * 0.5);

        const x1 = cx + Math.cos(angle) * innerR;
        const y1 = cy + Math.sin(angle) * innerR;
        const x2 = cx + Math.cos(angle) * outerR;
        const y2 = cy + Math.sin(angle) * outerR;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = `rgba(0, 220, 255, ${clamp(warpFactor * 0.3, 0, 0.4)})`;
        ctx.lineWidth = 0.5 + warpFactor;
        ctx.stroke();
      }
      ctx.restore();
    }

    /* ─── Animation Frame ──────────────────────────────────── */

    let currentZoom = 0;

    function animate() {
      const time = (timeRef.current += 0.016);

      // Smooth zoom interpolation
      currentZoom = lerp(currentZoom, zoom, 0.08);

      // Warp progress: ramps up during isWarping, decays after
      if (isWarping) {
        warpProgressRef.current = lerp(warpProgressRef.current, 1, 0.04);
      } else {
        warpProgressRef.current = lerp(warpProgressRef.current, 0, 0.06);
      }
      const warpFactor = warpProgressRef.current;

      ctx.clearRect(0, 0, width, height);

      // Background — shifts to deep blue during warp
      const bgGrad = ctx.createRadialGradient(
        width / 2,
        height / 2,
        0,
        width / 2,
        height / 2,
        width * 0.7
      );

      const bgCenterR = Math.round(lerp(5, 0, warpFactor));
      const bgCenterG = Math.round(lerp(15, 10, warpFactor));
      const bgCenterB = Math.round(lerp(25, 40, warpFactor));
      bgGrad.addColorStop(0, `rgba(${bgCenterR}, ${bgCenterG}, ${bgCenterB}, 1)`);
      bgGrad.addColorStop(1, "rgba(0, 0, 0, 1)");
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const focalLength = 400;

      // Camera flies forward — during warp, accelerates dramatically
      const warpCameraBoost = warpFactor * 600;
      const cameraZ = currentZoom * 350 + warpCameraBoost;

      // Slow rotation as user zooms in
      const rotSpeed = lerp(0.15, 0.03, currentZoom);
      const rotY = time * rotSpeed;
      const rotXAngle = Math.sin(time * 0.1) * lerp(0.15, 0.03, currentZoom);

      // Render warp-speed lines behind everything
      renderWarpLines(cx, cy, warpFactor, time);

      // Render neuron branches
      trees.forEach((tree) => {
        renderBranch(tree, time, focalLength, cx, cy, rotY, rotXAngle, 0.9, cameraZ, warpFactor);
      });

      // Render nucleus
      renderNucleus(time, focalLength, cx, cy, rotY, rotXAngle, currentZoom, cameraZ, warpFactor);

      // Render synapse particles
      renderParticles(time, focalLength, cx, cy, rotY, rotXAngle, cameraZ, warpFactor);

      // Zoom indicator (only in focus mode)
      if (focusMode && currentZoom > 0.01) {
        const indicatorText = `DEPTH: ${Math.round(currentZoom * 100)}%`;
        ctx.font = "12px monospace";
        ctx.fillStyle = `rgba(0, 220, 255, ${0.4 + currentZoom * 0.4})`;
        ctx.textAlign = "right";
        ctx.fillText(indicatorText, width - 20, height - 20);

        if (currentZoom < 0.8 && !isWarping) {
          ctx.font = "11px monospace";
          ctx.fillStyle = "rgba(0, 220, 255, 0.3)";
          ctx.textAlign = "center";
          ctx.fillText(
            "Scroll to dive into the nucleus...",
            cx,
            height - 20
          );
        }
      }

      // White flash overlay during peak warp
      if (warpFactor > 0.7) {
        const flashAlpha = clamp((warpFactor - 0.7) / 0.3 * 0.6, 0, 0.6);
        ctx.fillStyle = `rgba(200, 255, 255, ${flashAlpha})`;
        ctx.fillRect(0, 0, width, height);
      }

      animFrameRef.current = requestAnimationFrame(animate);
    }

    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      window.removeEventListener("resize", resize);
      if (focusMode) {
        canvas.removeEventListener("wheel", handleWheel);
      }
    };
  }, [trees, particles, zoom, handleWheel, focusMode, isWarping]);

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className={`w-full h-full ${
          focusMode
            ? "cursor-grab active:cursor-grabbing"
            : "cursor-default"
        }`}
        style={{ touchAction: "none" }}
      />

      {/* Focus Mode Toggle Button */}
      <div className="absolute top-4 right-4 z-20">
        <Button
          variant={focusMode ? "default" : "outline"}
          size="sm"
          onClick={toggleFocusMode}
          className={`gap-2 font-mono text-xs transition-all duration-300 ${
            focusMode
              ? "bg-cyan-500/20 border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/30 shadow-[0_0_20px_rgba(0,220,255,0.15)]"
              : "border-zinc-700 text-zinc-400 hover:text-cyan-400 hover:border-cyan-500/30"
          }`}
        >
          {focusMode ? (
            <>
              <X className="w-3.5 h-3.5" />
              Exit Focus
            </>
          ) : (
            <>
              <Crosshair className="w-3.5 h-3.5" />
              Explore Neuron
            </>
          )}
        </Button>
      </div>

      {/* Focus mode hint overlay */}
      {focusMode && zoom < 0.05 && !isWarping && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 animate-pulse z-20">
          <span className="text-xs text-cyan-400/60 font-mono tracking-widest uppercase">
            Scroll to explore
          </span>
          <div className="w-5 h-8 border-2 border-cyan-500/30 rounded-full flex justify-center pt-1">
            <div className="w-1 h-2 bg-cyan-400/60 rounded-full animate-bounce" />
          </div>
        </div>
      )}

      {/* Focus mode active indicator */}
      {focusMode && !isWarping && !hasReachedNucleus && (
        <div className="absolute top-4 left-4 z-20">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-cyan-500/20 bg-black/60 backdrop-blur-sm text-cyan-400 text-xs font-mono">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            Focus Mode Active
          </div>
        </div>
      )}

      {/* Warp in-progress overlay */}
      {isWarping && (
        <div className="absolute inset-0 z-20 flex items-center justify-center pointer-events-none">
          <div className="text-center animate-pulse">
            <p className="text-cyan-300 font-mono text-sm tracking-[0.3em] uppercase">
              Entering Nucleus
            </p>
          </div>
        </div>
      )}

      {/* Fact carousel (shown after warp completes) */}
      <AnimatePresence>
        {hasReachedNucleus && (
          <FactCarousel onClose={resetWarp} />
        )}
      </AnimatePresence>
    </div>
  );
}
