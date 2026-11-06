/**
 * Hand detection layer for MediaPipe Tasks Vision.
 *
 * Responsibility: load the runtime + model, run detection on the camera feed and
 * emit RAW two-hand detections. Hand identity, smoothing, velocities and gestures
 * live in gamelogic.js; drawing lives in ui.js.
 */

const VISION_VERSION = "1.2.2";
const VISION_BUNDLE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VISION_VERSION}/vision_bundle.mjs`;
const VISION_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VISION_VERSION}/wasm`;

export const DEFAULT_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

/** MediaPipe hand landmark names, index === landmark id (0..20). */
export const LANDMARK_NAMES = [
  "wrist",
  "thumb_cmc",
  "thumb_mcp",
  "thumb_ip",
  "thumb_tip",
  "index_finger_mcp",
  "index_finger_pip",
  "index_finger_dip",
  "index_finger_tip",
  "middle_finger_mcp",
  "middle_finger_pip",
  "middle_finger_dip",
  "middle_finger_tip",
  "ring_finger_mcp",
  "ring_finger_pip",
  "ring_finger_dip",
  "ring_finger_tip",
  "pinky_mcp",
  "pinky_pip",
  "pinky_dip",
  "pinky_tip",
];

/** Bones of the hand skeleton, as pairs of landmark indexes. */
export const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

/** Tip of each finger, used for pinch/spread measurements. */
export const FINGERTIPS = [4, 8, 12, 16, 20];

let visionModulePromise = null;

function loadVision() {
  if (!visionModulePromise) {
    visionModulePromise = import(VISION_BUNDLE).catch((error) => {
      visionModulePromise = null;
      throw error;
    });
  }
  return visionModulePromise;
}

/**
 * Creates a hand tracker bound to a <video> element.
 *
 * @param {object} options
 * @param {HTMLVideoElement} options.video Camera feed already playing.
 * @param {string} [options.modelUrl]
 * @param {number} [options.numHands=2] Maximum hands to detect; raise it to 4 for two people.
 * @param {(status: {phase: string, delegate?: string, reason?: string}) => void} [options.onStatus]
 * @returns {Promise<HandTracker>}
 */
export async function createHandTracker(options = {}) {
  const {
    video,
    modelUrl = DEFAULT_MODEL_URL,
    numHands = 2,
    minHandDetectionConfidence = 0.5,
    minHandPresenceConfidence = 0.5,
    minTrackingConfidence = 0.5,
    onStatus = () => {},
  } = options;

  if (!(video instanceof HTMLVideoElement)) {
    throw new Error("createHandTracker requires a <video> element");
  }

  onStatus({ phase: "runtime" });
  const vision = await loadVision();
  const wasmFileset = await vision.FilesetResolver.forVisionTasks(VISION_WASM);

  const buildOptions = (delegate) => ({
    baseOptions: { modelAssetPath: modelUrl, delegate },
    runningMode: "VIDEO",
    numHands,
    minHandDetectionConfidence,
    minHandPresenceConfidence,
    minTrackingConfidence,
  });

  let landmarker;
  try {
    onStatus({ phase: "model", delegate: "GPU" });
    landmarker = await vision.HandLandmarker.createFromOptions(
      wasmFileset,
      buildOptions("GPU"),
    );
  } catch (gpuError) {
    onStatus({ phase: "model", delegate: "CPU", reason: gpuError?.message });
    landmarker = await vision.HandLandmarker.createFromOptions(
      wasmFileset,
      buildOptions("CPU"),
    );
  }

  return new HandTracker(video, landmarker, { numHands });
}

/**
 * Runs HandLandmarker once per camera frame and forwards raw results.
 */
export class HandTracker {
  #video;
  #landmarker;
  #running = false;
  #handle = null;
  #lastVideoTime = -1;
  #frameTimes = [];
  #onFrame = null;

  constructor(video, landmarker, { numHands = 2 } = {}) {
    this.#video = video;
    this.#landmarker = landmarker;
    this.numHands = numHands;
    this.fps = 0;
  }

  get running() {
    return this.#running;
  }

  /**
   * @param {(frame: {timestamp: number, hands: RawHand[], fps: number}) => void} onFrame
   */
  start(onFrame) {
    if (this.#running) return;
    this.#running = true;
    this.#onFrame = onFrame;
    this.#lastVideoTime = -1;
    this.#frameTimes = [];
    this.#schedule();
  }

  stop() {
    if (!this.#running) return;
    this.#running = false;
    this.#onFrame = null;
    this.#lastVideoTime = -1;

    if (this.#handle !== null) {
      if (typeof this.#video.cancelVideoFrameCallback === "function") {
        this.#video.cancelVideoFrameCallback(this.#handle);
      } else {
        cancelAnimationFrame(this.#handle);
      }
      this.#handle = null;
    }
  }

  /** Releases the WASM/GPU resources held by the task. */
  close() {
    this.stop();
    this.#landmarker?.close?.();
    this.#landmarker = null;
  }

  /** Runs detection on one frame and returns raw hands (no identity, no smoothing). */
  detect(timestamp = performance.now()) {
    const video = this.#video;
    if (!this.#landmarker || video.readyState < 2) return [];

    const result = this.#landmarker.detectForVideo(video, timestamp);
    const landmarksPerHand = result.landmarks ?? [];
    const worldPerHand = result.worldLandmarks ?? [];
    const handednessPerHand = result.handedness ?? result.handednesses ?? [];

    return landmarksPerHand.map((landmarks, index) => {
      const category = handednessPerHand[index]?.[0];
      return {
        label: category?.categoryName ?? category?.displayName ?? "Unknown",
        score: category?.score ?? 0,
        landmarks,
        worldLandmarks: worldPerHand[index] ?? [],
      };
    });
  }

  #schedule() {
    if (!this.#running) return;

    const onVideoFrame = () => {
      this.#handle = null;
      if (!this.#running) return;
      try {
        this.#tick();
      } finally {
        this.#schedule();
      }
    };

    if (typeof this.#video.requestVideoFrameCallback === "function") {
      this.#handle = this.#video.requestVideoFrameCallback(onVideoFrame);
    } else {
      this.#handle = requestAnimationFrame(onVideoFrame);
    }
  }

  #tick() {
    const video = this.#video;
    if (
      !video ||
      video.readyState < 2 ||
      video.currentTime === this.#lastVideoTime
    )
      return;
    this.#lastVideoTime = video.currentTime;

    const timestamp = performance.now();
    let hands;
    try {
      hands = this.detect(timestamp);
    } catch (error) {
      console.error("[handpose] detection failed", error);
      return;
    }

    this.#updateFps(timestamp);
    this.#onFrame?.({ timestamp, hands, fps: this.fps });
  }

  #updateFps(timestamp) {
    this.#frameTimes.push(timestamp);
    while (this.#frameTimes.length > 31) this.#frameTimes.shift();
    if (this.#frameTimes.length > 1) {
      const span = this.#frameTimes.at(-1) - this.#frameTimes[0];
      this.fps = span > 0 ? ((this.#frameTimes.length - 1) * 1000) / span : 0;
    }
  }
}
