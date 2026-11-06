/**
 * Game logic and graphics engine.
 *
 * Two independent engines live in this file:
 *
 * 1. Hand motion reading — HandMotionEngine matches raw detections to stable hand
 *    identities and keeps one independent HandMotionReader per hand. Each reader
 *    owns its own smoothing, velocities and gesture state, so a hand can be read
 *    without looking at the other.
 *
 *    Coordinate spaces:
 *    - imagePosition: normalized [0..1] on the camera frame (what MediaPipe returns).
 *    - position:      "view" space, i.e. imagePosition with x flipped when the preview
 *                     is mirrored. Use this for anything the player sees.
 *    - velocity:      view-space units per second, so its direction matches the screen.
 *
 * 2. Graphics engine (p5.js) — GraphicsEngine owns the fullscreen p5 canvas for the
 *    interactive "La noche estrellada" experience. It paints the original artwork as
 *    the base layer and stacks animated light layers (sky swirls, star and moon glow,
 *    luminous particles) on top. It receives interpreted hand frames through
 *    updateHands() so a later phase can drive the scene with hand coordinates.
 */

import { FINGERTIPS, LANDMARK_NAMES } from "./handpose.js";

const PALM_INDICES = [0, 5, 9, 13, 17];
const MIN_HAND_SCALE = 0.02;
const MAX_DELTA_SECONDS = 0.2;

/** Non-thumb fingers, used to tell an open hand from a fist without camera-range calibration. */
const FINGERS = [
  { name: "index", mcp: 5, tip: 8 },
  { name: "middle", mcp: 9, tip: 12 },
  { name: "ring", mcp: 13, tip: 16 },
  { name: "pinky", mcp: 17, tip: 20 },
];

const distance2D = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

const average = (points) => {
  const sum = points.reduce(
    (acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y, z: acc.z + point.z }),
    { x: 0, y: 0, z: 0 },
  );
  return { x: sum.x / points.length, y: sum.y / points.length, z: sum.z / points.length };
};

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

export const DEFAULT_READER_OPTIONS = {
  mirror: true,
  smoothing: 0.55,
  pinchEnter: 0.6,
  pinchExit: 0.9,
  /** tip/mcp distance ratio above which a finger counts as extended. */
  extendedRatio: 1.3,
};

export class HandMotionReader {
  constructor({ id, label, ...overrides } = {}) {
    this.id = id ?? label ?? "hand";
    this.label = label ?? "Unknown";
    this.options = { ...DEFAULT_READER_OPTIONS, ...overrides };
    this.lastSeen = 0;
    this.lastPosition = { x: 0.5, y: 0.5 };

    this._smoothed = null;
    this._previous = null;
    this._previousTime = 0;
    this._pinch = false;
    this._fist = false;
    this._open = false;
    this._gesture = "none";
    this._events = [];
  }

  /**
   * Feeds one detection for this hand.
   * @returns {object|null} sample with joints, velocities, gestures and events.
   */
  update({ landmarks, worldLandmarks = [], score = 0, timestamp }) {
    if (!Array.isArray(landmarks) || landmarks.length < 21) return null;

    const { mirror, smoothing, pinchEnter, pinchExit, extendedRatio } = this.options;

    let deltaSeconds = this._previousTime ? (timestamp - this._previousTime) / 1000 : 0;
    const reacquired = deltaSeconds > MAX_DELTA_SECONDS || deltaSeconds <= 0;
    if (reacquired) deltaSeconds = 0;
    this._previousTime = timestamp;

    const alpha = clamp(1 - smoothing, 0.05, 1);
    this._smoothed =
      !this._smoothed || reacquired
        ? landmarks.map((point) => ({ x: point.x, y: point.y, z: point.z ?? 0 }))
        : landmarks.map((point, index) => {
            const previous = this._smoothed[index];
            return {
              x: previous.x + (point.x - previous.x) * alpha,
              y: previous.y + (point.y - previous.y) * alpha,
              z: previous.z + ((point.z ?? 0) - previous.z) * alpha,
            };
          });

    const points = this._smoothed;
    const toView = (point) => ({ x: mirror ? 1 - point.x : point.x, y: point.y, z: point.z });

    const previous = this._previous;
    const velocities = points.map((point, index) => {
      if (!previous || deltaSeconds === 0) return { x: 0, y: 0, z: 0 };
      const current = toView(point);
      const before = toView(previous[index]);
      return {
        x: (current.x - before.x) / deltaSeconds,
        y: (current.y - before.y) / deltaSeconds,
        z: (current.z - before.z) / deltaSeconds,
      };
    });

    const handScale = Math.max(distance2D(points[0], points[9]), MIN_HAND_SCALE);
    const palmPoints = PALM_INDICES.map((index) => points[index]);
    const palmView = average(palmPoints.map(toView));
    const palmVelocity = average(PALM_INDICES.map((index) => velocities[index]));

    const spread =
      FINGERTIPS.reduce((total, index) => total + distance2D(points[index], points[0]), 0) /
      FINGERTIPS.length /
      handScale;

    // A finger is extended when its tip is clearly farther from the wrist than its
    // knuckle. This is scale-invariant, unlike an absolute distance threshold.
    const extensionRatios = FINGERS.map(
      (finger) =>
        distance2D(points[finger.tip], points[0]) /
        Math.max(distance2D(points[finger.mcp], points[0]), 1e-4),
    );
    const extendedFingers = extensionRatios.filter((ratio) => ratio > extendedRatio).length;

    // A closed fist also brings the thumb next to the index, so a pinch only counts
    // while the index finger is still extended.
    const indexExtended = extensionRatios[0] > extendedRatio * 0.85;
    const pinchRatio = distance2D(points[4], points[8]) / handScale;
    const pinchNow =
      indexExtended && (this._pinch ? pinchRatio < pinchExit : pinchRatio < pinchEnter);

    const fistNow = this._fist ? extendedFingers <= 1 : extendedFingers === 0;
    const openNow = this._open ? extendedFingers >= 3 : extendedFingers >= 4;

    let gesture = "none";
    if (pinchNow) gesture = "pinch";
    else if (openNow && !fistNow) gesture = "open";
    else if (fistNow && !openNow) gesture = "fist";

    if (pinchNow !== this._pinch) {
      this._events.push({
        type: pinchNow ? "pinch:start" : "pinch:end",
        handId: this.id,
        at: timestamp,
      });
    }
    if (gesture !== this._gesture) {
      this._events.push({
        type: "gesture:change",
        handId: this.id,
        from: this._gesture,
        to: gesture,
        at: timestamp,
      });
    }
    this._pinch = pinchNow;
    this._fist = fistNow;
    this._open = openNow;
    this._gesture = gesture;

    this._previous = points.map((point) => ({ x: point.x, y: point.y, z: point.z }));
    this.lastSeen = timestamp;
    this.lastPosition = { x: points[0].x, y: points[0].y };

    return {
      id: this.id,
      label: this.label,
      score,
      timestamp,
      gesture,
      pinch: { active: pinchNow, ratio: pinchRatio },
      spread,
      fingers: { extended: extendedFingers, ratios: extensionRatios },
      palm: {
        position: palmView,
        velocity: palmVelocity,
        speed: Math.hypot(palmVelocity.x, palmVelocity.y),
      },
      landmarks: points,
      worldLandmarks,
      joints: points.map((point, index) => {
        const view = toView(point);
        const velocity = velocities[index];
        return {
          index,
          name: LANDMARK_NAMES[index],
          position: view,
          imagePosition: { x: point.x, y: point.y, z: point.z },
          velocity,
          speed: Math.hypot(velocity.x, velocity.y),
        };
      }),
      events: this._events.splice(0),
    };
  }
}

export const DEFAULT_ENGINE_OPTIONS = {
  ...DEFAULT_READER_OPTIONS,
  lostTimeoutMs: 250,
  matchGate: 0.45,
  labelPenalty: 0.18,
};

export class HandMotionEngine {
  constructor(options = {}) {
    this.options = { ...DEFAULT_ENGINE_OPTIONS, ...options };
    this.readers = new Map();
  }

  /** Returns the live reader for a hand id, e.g. "Left". */
  get(id) {
    return this.readers.get(id) ?? null;
  }

  /**
   * Assigns stable identities to the raw detections and reads each hand on its own.
   *
   * @param {Array<{label: string, score: number, landmarks: Array, worldLandmarks: Array}>} rawHands
   * @param {number} [timestamp]
   * @returns {{hands: Array, events: Array, byId: Record<string, object>}}
   */
  update(rawHands = [], timestamp = performance.now()) {
    const { lostTimeoutMs, matchGate, labelPenalty } = this.options;
    const detected = rawHands.filter((hand) => hand?.landmarks?.length >= 21);
    const events = [];

    // Match detections to known hands: nearest wrist wins, with a bonus for the
    // same handedness so identity does not flip when the model hesitates.
    const candidates = [];
    for (const [id, reader] of this.readers) {
      detected.forEach((hand, index) => {
        const wrist = hand.landmarks[0];
        if (!wrist) return;
        const distance = distance2D(wrist, reader.lastPosition);
        candidates.push({
          id,
          index,
          distance,
          cost: distance + (hand.label === reader.label ? 0 : labelPenalty),
        });
      });
    }
    candidates.sort((a, b) => a.cost - b.cost);

    const assignment = new Map();
    const usedIndexes = new Set();
    for (const candidate of candidates) {
      if (usedIndexes.has(candidate.index) || assignment.has(candidate.id)) continue;
      if (candidate.distance > matchGate) continue;
      assignment.set(candidate.id, candidate.index);
      usedIndexes.add(candidate.index);
    }

    detected.forEach((hand, index) => {
      if (usedIndexes.has(index)) return;

      const label = hand.label || "Unknown";
      let id = label;
      let suffix = 2;
      while (assignment.has(id)) id = `${label}-${suffix++}`;

      // Reclaim the id of a hand that was visible before but went unmatched.
      this.readers.delete(id);

      const reader = new HandMotionReader({ id, label, ...this._readerOptions() });
      reader.lastSeen = timestamp;
      this.readers.set(id, reader);
      assignment.set(id, index);
      usedIndexes.add(index);
      events.push({ type: "hand:found", handId: id, label, at: timestamp });
    });

    const hands = [];
    for (const [id, index] of assignment) {
      const reader = this.readers.get(id);
      if (!reader) continue;
      const hand = detected[index];
      const sample = reader.update({
        landmarks: hand.landmarks,
        worldLandmarks: hand.worldLandmarks ?? [],
        score: hand.score ?? 0,
        timestamp,
      });
      if (!sample) continue;
      hands.push(sample);
      events.push(...sample.events);
    }

    // A hand is only released when it is absent from this frame for longer than
    // the timeout, so a single dropped frame never changes its identity.
    for (const [id, reader] of this.readers) {
      if (assignment.has(id)) continue;
      if (timestamp - reader.lastSeen > lostTimeoutMs) {
        this.readers.delete(id);
        events.push({ type: "hand:lost", handId: id, at: timestamp });
      }
    }

    const byId = {};
    for (const hand of hands) byId[hand.id] = hand;

    return { hands, events, byId, liveIds: [...this.readers.keys()] };
  }

  _readerOptions() {
    const { mirror, smoothing, pinchEnter, pinchExit, extendedRatio } = this.options;
    return { mirror, smoothing, pinchEnter, pinchExit, extendedRatio };
  }
}

// ---------------------------------------------------------------------------
// p5.js graphics engine — "La noche estrellada"
// ---------------------------------------------------------------------------

const P5_VERSION = "2.3.4";
const P5_BUNDLE = `https://cdn.jsdelivr.net/npm/p5@${P5_VERSION}/lib/p5.min.js`;

export const DEFAULT_GRAPHICS_OPTIONS = {
  /** Reference proportions of the canvas (16:9). Pass null to fill the
   *  available space completely. */
  aspect: 16 / 9,
  /** Provisional deep-blue backdrop, shown where the painting does not cover. */
  background: "#0d1f45",
  /** How the painting fits the canvas without distortion: "contain" shows it
   *  whole (the default, so no important element is ever cropped away),
   *  "cover" fills the canvas cropping the overflow. */
  paintFit: "contain",
};

/** Path of the artwork, matching the project's asset layout (URL-encoded). */
const PAINTING_URL = "assets/Obras/noche%20estrellada/noche%20estrellada.jpg";

let p5ScriptPromise = null;

/**
 * Loads the p5 browser bundle from the CDN and resolves with the p5 class.
 *
 * p5 ships its browser bundle as a classic global script (IIFE), not as an ES
 * module, so it is injected as a <script> tag and read back from globalThis.
 * This mirrors how handpose.js pulls MediaPipe from a CDN at runtime, except
 * p5 needs a classic script instead of a dynamic import().
 */
function loadP5() {
  if (!p5ScriptPromise) {
    p5ScriptPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = P5_BUNDLE;
      script.async = true;
      script.onload = () => {
        const p5 = globalThis.p5;
        if (typeof p5 === "function") resolve(p5);
        else reject(new Error("p5 loaded but did not attach to globalThis"));
      };
      script.onerror = () => {
        p5ScriptPromise = null;
        reject(new Error(`Failed to load p5 from ${P5_BUNDLE}`));
      };
      document.head.appendChild(script);
    });
  }
  return p5ScriptPromise;
}

/**
 * Largest canvas that keeps `aspect` (width / height) inside the available box.
 * The result never overflows availableWidth x availableHeight.
 */
export function fitAspectSize(availableWidth, availableHeight, aspect = 16 / 9) {
  const width = Math.min(availableWidth, availableHeight * aspect);
  const height = width / aspect;
  return {
    width: Math.max(1, Math.floor(width)),
    height: Math.max(1, Math.floor(height)),
  };
}

/**
 * Rect that scales an image onto a canvas without distortion, centered.
 *
 * @param {"cover"|"contain"} mode cover fills the canvas (cropping overflow),
 *   contain shows the whole image (leaving margins).
 * @returns {{x: number, y: number, width: number, height: number}} canvas coords.
 */
export function fitImageRect(mode, imageWidth, imageHeight, canvasWidth, canvasHeight) {
  const scale =
    mode === "contain"
      ? Math.min(canvasWidth / imageWidth, canvasHeight / imageHeight)
      : Math.max(canvasWidth / imageWidth, canvasHeight / imageHeight);
  const width = imageWidth * scale;
  const height = imageHeight * scale;
  return {
    x: (canvasWidth - width) / 2,
    y: (canvasHeight - height) / 2,
    width,
    height,
  };
}

// ---------------------------------------------------------------------------
// "La noche estrellada" layer configuration.
//
// Positions are normalized [0..1] over the PAINTING (x left->right, y top->
// bottom), so the effects stay glued to the artwork whatever the canvas size
// or crop. When the cover crop hides a position, its glow simply falls off
// the canvas edge.
// ---------------------------------------------------------------------------

/** Crescent moon, top-right corner of the painting. */
const MOON = { x: 0.88, y: 0.12, size: 0.09 };

/** Bright stars, spread across the central band so they survive the crop. */
const STARS = [
  { x: 0.52, y: 0.28, size: 0.045, speed: 1.2, phase: 0.0 },
  { x: 0.60, y: 0.20, size: 0.030, speed: 1.8, phase: 1.4 },
  { x: 0.44, y: 0.24, size: 0.026, speed: 1.5, phase: 2.2 },
  { x: 0.66, y: 0.32, size: 0.022, speed: 2.1, phase: 3.1 },
  { x: 0.36, y: 0.34, size: 0.018, speed: 1.7, phase: 4.0 },
  { x: 0.58, y: 0.40, size: 0.016, speed: 2.4, phase: 5.0 },
  { x: 0.70, y: 0.26, size: 0.014, speed: 1.9, phase: 0.8 },
  { x: 0.30, y: 0.26, size: 0.014, speed: 2.6, phase: 2.7 },
];

/** Flowing sky brushstrokes: each entry is one polyline of normalized points. */
const SWIRLS = [
  {
    points: [
      { x: 0.28, y: 0.40 },
      { x: 0.38, y: 0.30 },
      { x: 0.50, y: 0.24 },
      { x: 0.62, y: 0.28 },
      { x: 0.70, y: 0.40 },
    ],
    lanes: 5,
    gap: 0.006,
    ripple: 0.012,
    speed: 0.9,
    alpha: 46,
    weight: 2.2,
    color: [140, 190, 255],
  },
  {
    points: [
      { x: 0.30, y: 0.34 },
      { x: 0.44, y: 0.28 },
      { x: 0.58, y: 0.32 },
      { x: 0.68, y: 0.44 },
    ],
    lanes: 4,
    gap: 0.008,
    ripple: 0.014,
    speed: 0.7,
    alpha: 40,
    weight: 2.6,
    color: [120, 170, 240],
  },
  {
    points: [
      { x: 0.36, y: 0.46 },
      { x: 0.46, y: 0.38 },
      { x: 0.58, y: 0.42 },
      { x: 0.66, y: 0.52 },
    ],
    lanes: 3,
    gap: 0.010,
    ripple: 0.010,
    speed: 1.1,
    alpha: 34,
    weight: 2.0,
    color: [150, 200, 255],
  },
];

/** Ambient luminous particles that drift upward over the whole canvas. */
const PARTICLE_COUNT = 70;

/** Particle tones: warm yellow, white, and sky blue. */
const PARTICLE_TONES = [
  [255, 232, 160],
  [245, 245, 255],
  [150, 190, 255],
];

/**
 * Graphics engine that owns the p5 canvas.
 *
 * Phase 1 set the stage: a canvas that fills its container (or keeps a 16:9
 * fit when an aspect is set) with a provisional dark-blue background.
 * Phase 2 paints the original artwork as the base layer and stacks animated
 * light layers (sky swirls, star and moon glow, particles) on top. Loading,
 * drawing and animation live in separate methods. The autonomous animations
 * can be paused and resumed without stopping the painting (pauseAnimations /
 * resumeAnimations / setAnimationsEnabled), and updateHands() remains the
 * hook for the future hand-driven phase.
 */
export class GraphicsEngine {
  #container = null;
  #p5 = null;
  #hands = [];
  #events = [];
  #byId = {};
  #painting = null;
  #animTime = 0;
  #animationsEnabled = true;
  #particles = null;

  constructor(container, options = {}) {
    const { p5, aspect, background, paintFit } = { ...DEFAULT_GRAPHICS_OPTIONS, ...options };
    if (!container || !(container instanceof HTMLElement)) {
      throw new Error("GraphicsEngine requires a container HTMLElement");
    }
    this.#container = container;
    this.aspect = aspect;
    this.background = background;
    this.paintFit = paintFit;

    const engine = this;
    this.#p5 = new p5(
      (sketch) => {
        sketch.setup = () => engine.#setup(sketch);
        sketch.draw = () => engine.#draw(sketch);
        sketch.windowResized = () => engine.#windowResized(sketch);
      },
      container,
    );
  }

  /** Creates the engine once p5 has finished loading from the CDN. */
  static async create(container, options = {}) {
    const p5 = await loadP5();
    return new GraphicsEngine(container, { ...options, p5 });
  }

  /** Live p5 instance, or null after destroy(). */
  get p5() {
    return this.#p5;
  }

  /** The p5 canvas element, or null before setup or after destroy(). */
  get canvas() {
    return this.#p5?.canvas ?? null;
  }

  /** True once the painting image has loaded successfully. */
  get paintingLoaded() {
    return this.#painting != null;
  }

  /**
   * Feeds one interpreted hand frame from HandMotionEngine.
   *
   * Stored for later phases only: nothing is drawn yet. Each hand's
   * `position`/`palm.position` is normalized view space [0..1], ready to be
   * projected onto the canvas when the hand-driven effects land.
   */
  updateHands({ hands = [], events = [], byId = {} } = {}) {
    this.#hands = hands;
    this.#events = events;
    this.#byId = byId;
  }

  /** True while the autonomous animations advance. */
  get animationsRunning() {
    return this.#animationsEnabled;
  }

  /** Freezes every autonomous animation; the painting keeps being drawn. */
  pauseAnimations() {
    this.#animationsEnabled = false;
  }

  /** Resumes the autonomous animations from where they were frozen. */
  resumeAnimations() {
    this.#animationsEnabled = true;
  }

  /** @param {boolean} enabled Whether the autonomous animations should run. */
  setAnimationsEnabled(enabled) {
    this.#animationsEnabled = Boolean(enabled);
  }

  /** Stops the sketch and releases the p5 instance. */
  destroy() {
    this.#p5?.remove();
    this.#p5 = null;
    this.#hands = [];
    this.#events = [];
    this.#byId = {};
    this.#painting = null;
    this.#particles = null;
    this.#animTime = 0;
    this.#animationsEnabled = true;
  }

  #availableSize() {
    const width = this.#container.clientWidth || window.innerWidth;
    const height = this.#container.clientHeight || window.innerHeight;
    return { width, height };
  }

  /** Canvas size for the current container: full space, or aspect-fit when set. */
  #canvasSize() {
    const available = this.#availableSize();
    if (this.aspect == null) {
      return { width: available.width, height: available.height };
    }
    return fitAspectSize(available.width, available.height, this.aspect);
  }

  // --- Loading -----------------------------------------------------------

  async #setup(p) {
    const size = this.#canvasSize();
    // p5 appends the canvas to the container passed to `new p5(sketch, node)`,
    // so the CSS centering of #stage applies without any extra re-parenting.
    p.createCanvas(size.width, size.height);
    await this.#loadPainting(p);
  }

  /** Loads the artwork once; the dark-blue backdrop stays until it exists. */
  async #loadPainting(p) {
    try {
      this.#painting = await p.loadImage(PAINTING_URL);
      if (!this.#painting || !this.#painting.width) {
        this.#painting = null;
        console.warn(`[graphics] painting at ${PAINTING_URL} loaded empty`);
      }
    } catch (error) {
      this.#painting = null;
      console.warn(
        `[graphics] painting not found at ${PAINTING_URL} — ` +
          "showing the dark-blue backdrop until the file is added",
        error,
      );
    }
  }

  // --- Animation (state updates, no drawing) -----------------------------

  /** Advances every time-based animation state by one frame. */
  #updateAnimations(deltaSeconds) {
    this.#animTime += deltaSeconds;
    this.#ensureParticles();
    this.#updateParticles(deltaSeconds);
  }

  #ensureParticles() {
    if (this.#particles) return;
    this.#particles = Array.from({ length: PARTICLE_COUNT }, () => ({
      x: Math.random(),
      y: Math.random(),
      rise: 0.008 + Math.random() * 0.025, // canvas fractions per second
      driftPhase: Math.random() * Math.PI * 2,
      swayPhase: Math.random() * Math.PI * 2,
      twinklePhase: Math.random() * Math.PI * 2,
      size: 1.5 + Math.random() * 2.5,
      tone: Math.floor(Math.random() * PARTICLE_TONES.length),
    }));
  }

  #updateParticles(deltaSeconds) {
    for (const particle of this.#particles) {
      particle.y -= particle.rise * deltaSeconds;
      // Two layered sines = soft organic wandering, never frantic.
      particle.x +=
        (Math.sin(this.#animTime * 0.5 + particle.driftPhase) * 0.010 +
          Math.sin(this.#animTime * 1.3 + particle.swayPhase) * 0.005) *
        deltaSeconds;
      if (particle.y < -0.02) {
        particle.y = 1.02;
        particle.x = Math.random();
      }
    }
  }

  // --- Drawing (one method per layer) -------------------------------------

  #draw(p) {
    const deltaSeconds = Math.min((p.deltaTime ?? 16.67) / 1000, 0.1);
    if (this.#animationsEnabled) {
      this.#updateAnimations(deltaSeconds);
    }

    // Underlayer, visible only where the painting does not cover the canvas.
    p.background(this.background);

    const rect = this.#painting ? this.#imageRect(p) : null;
    this.#drawPainting(p, rect);

    if (rect) {
      // All effects read the accumulated animation clock, so pausing freezes
      // them while the painting itself keeps being drawn.
      this.#drawSkySwirls(p, rect, this.#animTime);
      this.#drawStarGlow(p, rect, this.#animTime);
      this.#drawMoonGlow(p, rect, this.#animTime);
    }
    this.#drawParticles(p, this.#animTime);

    // Phase 3 — hands: project this.#hands onto the canvas and react to them.
  }

  /** Rect that maps the painting onto the canvas without distortion. */
  #imageRect(p) {
    return fitImageRect(
      this.paintFit,
      this.#painting.width,
      this.#painting.height,
      p.width,
      p.height,
    );
  }

  /** Painting-normalized [0..1] point -> canvas coordinates. */
  #imageToCanvas(nx, ny, rect) {
    return { x: rect.x + nx * rect.width, y: rect.y + ny * rect.height };
  }

  #drawPainting(p, rect) {
    if (!this.#painting || !rect) return;
    p.image(this.#painting, rect.x, rect.y, rect.width, rect.height);
  }

  /** Animated flowing brushstrokes over the sky, following the artwork's swirls. */
  #drawSkySwirls(p, rect, time) {
    p.push();
    p.blendMode(p.SCREEN);
    p.noFill();
    const segments = 28;
    for (const swirl of SWIRLS) {
      for (let lane = 0; lane < swirl.lanes; lane += 1) {
        const laneOffset = (lane - (swirl.lanes - 1) / 2) * swirl.gap;
        p.stroke(...swirl.color, swirl.alpha);
        p.strokeWeight(swirl.weight);
        p.beginShape();
        for (let i = 0; i <= segments; i += 1) {
          const t = i / segments;
          const point = this.#sampleSwirl(swirl.points, t);
          const tangent = this.#swirlTangent(swirl.points, t);
          const wobble =
            Math.sin(time * swirl.speed * Math.PI * 2 + t * 10 + lane * 1.3) * swirl.ripple;
          const offset = laneOffset + wobble;
          // Perpendicular to the tangent keeps the lane band flowing along the path.
          const nx = point.x - tangent.y * offset;
          const ny = point.y + tangent.x * offset;
          const c = this.#imageToCanvas(nx, ny, rect);
          // p5 2.x: curveVertex() was renamed to splineVertex().
          p.splineVertex(c.x, c.y);
        }
        p.endShape();
      }
    }
    p.pop();
  }

  /** Point along a polyline at t in [0..1]. */
  #sampleSwirl(points, t) {
    const total = points.length - 1;
    const position = Math.min(Math.max(t, 0), 1) * total;
    const index = Math.min(Math.floor(position), total - 1);
    const fraction = position - index;
    const a = points[index];
    const b = points[index + 1];
    return {
      x: a.x + (b.x - a.x) * fraction,
      y: a.y + (b.y - a.y) * fraction,
    };
  }

  /** Normalized tangent of the polyline at t (finite difference). */
  #swirlTangent(points, t) {
    const epsilon = 0.001;
    const a = this.#sampleSwirl(points, Math.max(0, t - epsilon));
    const b = this.#sampleSwirl(points, Math.min(1, t + epsilon));
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const length = Math.hypot(dx, dy) || 1;
    return { x: dx / length, y: dy / length };
  }

  /** Soft pulsing halos over the bright stars of the painting. */
  #drawStarGlow(p, rect, time) {
    const ctx = p.drawingContext;
    p.push();
    p.blendMode(p.SCREEN);
    p.noStroke();
    for (const star of STARS) {
      const c = this.#imageToCanvas(star.x, star.y, rect);
      const pulse = 0.5 + 0.5 * Math.sin(time * star.speed * Math.PI * 2 + star.phase);
      const radius = star.size * rect.width * (0.8 + 0.35 * pulse);
      const gradient = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, radius);
      gradient.addColorStop(0, "rgba(255, 236, 170, 0.85)");
      gradient.addColorStop(0.35, "rgba(255, 214, 120, 0.35)");
      gradient.addColorStop(1, "rgba(255, 214, 120, 0)");
      p.fill(gradient);
      p.circle(c.x, c.y, radius * 2);
      p.fill(255, 244, 200, 90 + 60 * pulse);
      p.circle(c.x, c.y, radius * 0.28);

      // Small golden motes slowly orbiting the star's halo.
      const motes = 4;
      for (let k = 0; k < motes; k += 1) {
        const angle = time * (0.25 + 0.1 * (star.phase % 1)) + (k * Math.PI * 2) / motes;
        const orbit = radius * (0.55 + 0.3 * Math.sin(time * 0.8 + k * 1.7));
        const moteX = c.x + Math.cos(angle) * orbit;
        const moteY = c.y + Math.sin(angle) * orbit * 0.6;
        const moteRadius = 1.2 + 1.8 * Math.abs(Math.sin(time * 1.1 + k + star.phase));
        p.fill(255, 216, 120, 60 + 110 * Math.abs(Math.sin(time * 1.1 + k + star.phase)));
        p.circle(moteX, moteY, moteRadius);
      }
    }
    p.pop();
  }

  /** Slow breathing halo around the crescent moon. */
  #drawMoonGlow(p, rect, time) {
    const c = this.#imageToCanvas(MOON.x, MOON.y, rect);
    const breathe = 0.5 + 0.5 * Math.sin(time * 0.6 * Math.PI * 2);
    const radius = MOON.size * rect.width * (1 + 0.15 * breathe);
    const ctx = p.drawingContext;
    p.push();
    p.blendMode(p.SCREEN);
    p.noStroke();
    const gradient = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, radius);
    gradient.addColorStop(0, "rgba(255, 243, 190, 0.55)");
    gradient.addColorStop(0.5, "rgba(255, 232, 150, 0.18)");
    gradient.addColorStop(1, "rgba(255, 232, 150, 0)");
    p.fill(gradient);
    p.circle(c.x, c.y, radius * 2);
    p.pop();
  }

  /** Luminous motes in yellow, white and blue tones, drifting upward. */
  #drawParticles(p, time) {
    p.push();
    p.blendMode(p.ADD);
    p.noStroke();
    for (const particle of this.#particles) {
      const twinkle = 0.4 + 0.6 * Math.abs(Math.sin(time * 1.6 + particle.twinklePhase));
      const tone = PARTICLE_TONES[particle.tone];
      const x = particle.x * p.width;
      const y = particle.y * p.height;
      const radius =
        particle.size * (0.8 + 0.5 * twinkle) * (Math.min(p.width, p.height) / 500);
      p.fill(tone[0], tone[1], tone[2], 60 + 90 * twinkle);
      p.circle(x, y, radius * 2);
    }
    p.pop();
  }

  #windowResized(p) {
    const size = this.#canvasSize();
    p.resizeCanvas(size.width, size.height);
  }
}
