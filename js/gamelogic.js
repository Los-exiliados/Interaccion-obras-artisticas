/**
 * Per-hand motion reading.
 *
 * HandMotionEngine matches raw detections to stable hand identities and keeps one
 * independent HandMotionReader per hand. Each reader owns its own smoothing,
 * velocities and gesture state, so a hand can be read without looking at the other.
 *
 * Coordinate spaces:
 * - imagePosition: normalized [0..1] on the camera frame (what MediaPipe returns).
 * - position:      "view" space, i.e. imagePosition with x flipped when the preview
 *                  is mirrored. Use this for anything the player sees.
 * - velocity:      view-space units per second, so its direction matches the screen.
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
