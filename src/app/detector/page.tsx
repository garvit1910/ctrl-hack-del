/**
 * NeuroDetect — Detector Dashboard
 *
 * The /detector route. Provides an Archimedes spiral drawing canvas
 * with real-time tremor analysis, clinical metrics display, and
 * the NeuroDetect dark aesthetic.
 *
 * @module app/detector/page
 */

"use client";

import React, { useRef, useState, useCallback } from "react";
import {
  Brain,
  Activity,
  RotateCcw,
  TrendingUp,
  Timer,
  Target,
  AlertTriangle,
  CheckCircle2,
  Info,
  Upload,
  Pencil,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import SpiralCanvas, {
  type SpiralAnalysis,
  type SpiralCanvasRef,
} from "@/components/canvas/SpiralCanvas";
import SpiralUpload from "@/components/canvas/SpiralUpload";

/* ─── Risk Classification ─────────────────────────────────────────────── */

function classifyRisk(score: number): {
  level: "low" | "moderate" | "high";
  label: string;
  color: string;
  bgColor: string;
  description: string;
} {
  if (score < 30) {
    return {
      level: "low",
      label: "Low Risk",
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10 border-emerald-500/20",
      description:
        "Drawing patterns appear within normal range. Consistent speed and minimal radial deviation detected.",
    };
  }
  if (score < 65) {
    return {
      level: "moderate",
      label: "Moderate Indicators",
      color: "text-amber-400",
      bgColor: "bg-amber-500/10 border-amber-500/20",
      description:
        "Some irregularity detected in velocity or trajectory. This could be environmental. Consider retesting in a controlled setting.",
    };
  }
  return {
    level: "high",
    label: "Elevated Indicators",
    color: "text-red-400",
    bgColor: "bg-red-500/10 border-red-500/20",
    description:
      "Significant drawing irregularity detected. Please note: this is a screening tool, not a diagnosis. Consult a neurologist for clinical evaluation.",
  };
}

/* ─── Metric Card ──────────────────────────────────────────────────────── */

function MetricCard({
  icon: Icon,
  label,
  value,
  unit,
  subtext,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  unit?: string;
  subtext?: string;
}) {
  return (
    <div className="flex items-start gap-3 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800/50">
      <div className="mt-0.5 p-2 rounded-md bg-cyan-500/10">
        <Icon className="w-4 h-4 text-cyan-400" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider">
          {label}
        </p>
        <p className="text-lg font-semibold text-white mt-0.5">
          {value}
          {unit && (
            <span className="text-xs text-zinc-500 ml-1">{unit}</span>
          )}
        </p>
        {subtext && (
          <p className="text-xs text-zinc-600 mt-0.5">{subtext}</p>
        )}
      </div>
    </div>
  );
}

/* ─── Main Page ────────────────────────────────────────────────────────── */

export default function DetectorPage() {
  const canvasRef = useRef<SpiralCanvasRef>(null);
  const [analysis, setAnalysis] = useState<SpiralAnalysis | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [showInfoDialog, setShowInfoDialog] = useState(false);
  const [activeTab, setActiveTab] = useState<"draw" | "upload">("draw");
  const [isUploading, setIsUploading] = useState(false);

  const handleAnalysisUpdate = useCallback((a: SpiralAnalysis) => {
    setAnalysis(a);
  }, []);

  const handleDrawingEnd = useCallback(() => {
    setIsDrawing(false);
  }, []);

  const handleReset = useCallback(() => {
    canvasRef.current?.clear();
    setAnalysis(null);
  }, []);

  const risk = analysis ? classifyRisk(analysis.tremorScore) : null;

  return (
    <div className="min-h-screen bg-black text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-900 bg-black/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cyan-500/10">
              <Brain className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">
                NeuroDetect
              </h1>
              <p className="text-xs text-zinc-500">
                Spiral Tremor Analysis
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowInfoDialog(true)}
            >
              <Info className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm" asChild>
              <a href="/">Home</a>
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Instructions */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold tracking-tight mb-2">
            Archimedes Spiral Test
          </h2>
          <p className="text-zinc-400 max-w-2xl">
            Draw a spiral on screen or upload an existing image. Your
            drawing will be analyzed for motor function indicators — either
            in real-time or via our ML pipeline.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Canvas Column */}
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {/* Tab Switcher */}
                    <div className="flex items-center rounded-lg border border-zinc-800 p-0.5 bg-zinc-900/50">
                      <button
                        onClick={() => setActiveTab("draw")}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                          activeTab === "draw"
                            ? "bg-cyan-500/15 text-cyan-400"
                            : "text-zinc-500 hover:text-zinc-300"
                        }`}
                      >
                        <Pencil className="w-3.5 h-3.5" />
                        Draw
                      </button>
                      <button
                        onClick={() => setActiveTab("upload")}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                          activeTab === "upload"
                            ? "bg-cyan-500/15 text-cyan-400"
                            : "text-zinc-500 hover:text-zinc-300"
                        }`}
                      >
                        <Upload className="w-3.5 h-3.5" />
                        Upload
                      </button>
                    </div>
                    {isDrawing && activeTab === "draw" && (
                      <span className="inline-flex items-center gap-1 text-xs bg-cyan-500/10 text-cyan-400 px-2 py-0.5 rounded-full">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                        Recording
                      </span>
                    )}
                    {isUploading && activeTab === "upload" && (
                      <span className="inline-flex items-center gap-1 text-xs bg-cyan-500/10 text-cyan-400 px-2 py-0.5 rounded-full">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                        Processing
                      </span>
                    )}
                  </div>
                  {activeTab === "draw" && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleReset}
                      className="gap-1.5"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      Reset
                    </Button>
                  )}
                </div>
                <CardDescription>
                  {activeTab === "draw"
                    ? "Click and drag to draw. Stay as close to the guide spiral as possible."
                    : "Upload a photo of a hand-drawn spiral for ML-powered analysis."}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {activeTab === "draw" ? (
                  <div className="flex justify-center">
                    <SpiralCanvas
                      ref={canvasRef}
                      width={500}
                      height={500}
                      showGuide={true}
                      onAnalysisUpdate={handleAnalysisUpdate}
                      onDrawingStart={() => setIsDrawing(true)}
                      onDrawingEnd={handleDrawingEnd}
                    />
                  </div>
                ) : (
                  <SpiralUpload
                    onUploadStart={() => setIsUploading(true)}
                    onUploadComplete={() => {
                      setIsUploading(false);
                    }}
                  />
                )}
              </CardContent>
            </Card>
          </div>

          {/* Analysis Column */}
          <div className="space-y-4">
            {/* Risk Assessment */}
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  <CardTitle className="text-base">Risk Assessment</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {analysis && risk ? (
                  <>
                    <div
                      className={`p-4 rounded-lg border ${risk.bgColor}`}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        {risk.level === "low" ? (
                          <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                        ) : risk.level === "moderate" ? (
                          <AlertTriangle className="w-5 h-5 text-amber-400" />
                        ) : (
                          <AlertTriangle className="w-5 h-5 text-red-400" />
                        )}
                        <span
                          className={`font-semibold text-sm ${risk.color}`}
                        >
                          {risk.label}
                        </span>
                      </div>
                      <p className="text-xs text-zinc-400 leading-relaxed">
                        {risk.description}
                      </p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-zinc-400">Tremor Score</span>
                        <span className={`font-mono font-bold ${risk.color}`}>
                          {analysis.tremorScore}/100
                        </span>
                      </div>
                      <Progress value={analysis.tremorScore} />
                    </div>
                  </>
                ) : (
                  <div className="text-center py-8 text-zinc-600">
                    <Activity className="w-8 h-8 mx-auto mb-2 opacity-30" />
                    <p className="text-sm">
                      Draw a spiral to begin analysis
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Detailed Metrics */}
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-cyan-400" />
                  <CardTitle className="text-base">Metrics</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="space-y-2">
                {analysis ? (
                  <>
                    <MetricCard
                      icon={Target}
                      label="Radial Deviation"
                      value={analysis.meanDeviation}
                      unit="px"
                      subtext={`σ = ${analysis.stdDeviation}`}
                    />
                    <MetricCard
                      icon={TrendingUp}
                      label="Drawing Speed"
                      value={analysis.meanSpeed}
                      unit="px/ms"
                      subtext={`variance = ${analysis.speedVariance}`}
                    />
                    <MetricCard
                      icon={Timer}
                      label="Duration"
                      value={(analysis.drawingTime / 1000).toFixed(1)}
                      unit="sec"
                      subtext={`${analysis.pointCount} sample points`}
                    />
                  </>
                ) : (
                  <div className="text-center py-6 text-zinc-600">
                    <p className="text-xs">No data yet</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Disclaimer */}
            <div className="text-xs text-zinc-600 px-2 leading-relaxed">
              <strong className="text-zinc-500">Disclaimer:</strong> This
              tool is for educational and screening purposes only. It is
              not a medical diagnostic device. Always consult a qualified
              healthcare professional for clinical evaluation.
            </div>
          </div>
        </div>
      </main>

      {/* Info Dialog */}
      <Dialog open={showInfoDialog} onOpenChange={setShowInfoDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>About the Spiral Test</DialogTitle>
            <DialogDescription className="space-y-3 pt-2">
              <p>
                The Archimedes spiral drawing test is a validated clinical
                tool used by neurologists to assess motor function. It
                measures:
              </p>
              <ul className="list-disc pl-5 space-y-1 text-zinc-300">
                <li>
                  <strong>Radial deviation</strong> — how far drawn points
                  stray from the ideal spiral
                </li>
                <li>
                  <strong>Speed consistency</strong> — uniform drawing speed
                  indicates healthy motor control
                </li>
                <li>
                  <strong>Tremor indicators</strong> — high-frequency
                  oscillations in the stroke path
                </li>
              </ul>
              <p className="text-zinc-500 text-xs">
                Reference: Pullman SL. Spiral analysis: a new technique for
                measuring tremor with a digitizing tablet. Mov Disord. 1998.
              </p>
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </div>
  );
}
