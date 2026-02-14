/**
 * ScrollReveal — Scroll-triggered entrance wrapper
 *
 * Uses Framer Motion + IntersectionObserver (via whileInView)
 * to fade-in and scale-up children when they enter the viewport,
 * and fade-out when they leave.
 *
 * @module components/ui/scroll-reveal
 */

"use client";

import React from "react";
import { motion, type Variants } from "framer-motion";

interface ScrollRevealProps {
  children: React.ReactNode;
  className?: string;
  /** Delay in seconds before the entrance animation starts */
  delay?: number;
  /** Viewport margin for early trigger (default: "-60px") */
  margin?: string;
}

const revealVariants: Variants = {
  hidden: { opacity: 0, y: 32, scale: 0.95 },
  visible: { opacity: 1, y: 0, scale: 1 },
};

export default function ScrollReveal({
  children,
  className,
  delay = 0,
  margin = "-60px",
}: ScrollRevealProps) {
  return (
    <motion.div
      variants={revealVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: false, margin }}
      transition={{
        type: "spring",
        stiffness: 80,
        damping: 20,
        delay,
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
