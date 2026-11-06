/**
 * Rendering layer: skeleton overlay on the camera feed and one panel per hand.
 */

import { HAND_CONNECTIONS } from "./handpose.js";

const PALETTE = ["#7ee787", "#79c0ff", "#d2a8ff", "#ffa657"];
const SPEED_FOR_FULL_HEAT = 1.4; // view-space units per second
const TWO_PI = Math.PI * 2;

/** Stable color per hand identity, so up to four hands never look the same. */
export function handColor(id) {
  const key = String(id ?? "");
  const match = /^(Left|Right)(?:-(\d+))?$/.exec(key);
  if (match) {
    const base = match[1] === "Left" ? 0 : 1;
    const extra = match[2] ? Number(match[2]) - 1 : 0;
    return PALETTE[(base + extra * 2) % PALETTE.length];
  }

  let hash = 0;
  for (let index = 0; index < key.length; index += 1) {
    hash = (hash * 31 + key.charCodeAt(index)) % 9973;
  }
  return PALETTE[hash % PALETTE.length];
}

export class HandOverlay {
  constructor({ canvas, video, mirror = true }) {
    this.canvas = canvas;
    this.video = video;
    this.mirror = mirror;
    this.ctx = canvas.getContext("2d");
    this.dpr = 1;
    this.viewport = { scale: 1, dx: 0, dy: 0, videoWidth: 1280, videoHeight: 720 };
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.dpr = dpr;
    this.#refreshViewport();
  }

  clear() {
    if (!this.ctx) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  draw(hands = [], { fps = 0 } = {}) {
    const ctx = this.ctx;
    if (!ctx) return;

    this.#refreshViewport();
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    for (const hand of hands) this.#drawHand(hand);
    this.#drawFps(fps);
  }

  mapPoint(point) {
    const { scale, dx, dy, videoWidth, videoHeight } = this.viewport;
    return { x: dx + point.x * videoWidth * scale, y: dy + point.y * videoHeight * scale };
  }

  #refreshViewport() {
    const videoWidth = this.video.videoWidth || 1280;
    const videoHeight = this.video.videoHeight || 720;
    const scale = Math.max(this.canvas.width / videoWidth, this.canvas.height / videoHeight);
    this.viewport = {
      scale,
      videoWidth,
      videoHeight,
      dx: (this.canvas.width - videoWidth * scale) / 2,
      dy: (this.canvas.height - videoHeight * scale) / 2,
    };
  }

  #drawHand(hand) {
    const ctx = this.ctx;
    const dpr = this.dpr;
    const accent = handColor(hand.id);
    const points = hand.joints.map((joint) => this.mapPoint(joint.position));

    ctx.lineWidth = Math.max(1, 2 * dpr);
    ctx.strokeStyle = `${accent}99`;
    ctx.beginPath();
    for (const [from, to] of HAND_CONNECTIONS) {
      ctx.moveTo(points[from].x, points[from].y);
      ctx.lineTo(points[to].x, points[to].y);
    }
    ctx.stroke();

    if (hand.pinch.active) {
      ctx.strokeStyle = "#f2cc60";
      ctx.lineWidth = 3 * dpr;
      ctx.beginPath();
      ctx.moveTo(points[4].x, points[4].y);
      ctx.lineTo(points[8].x, points[8].y);
      ctx.stroke();
    }

    for (const joint of hand.joints) {
      const { x, y } = points[joint.index];
      const heat = Math.min(joint.speed / SPEED_FOR_FULL_HEAT, 1);
      const radius = (joint.index === 4 || joint.index === 8 ? 6 : 4) * dpr;

      ctx.beginPath();
      ctx.fillStyle = `hsl(${200 - 200 * heat} 90% ${58 + 8 * heat}%)`;
      ctx.arc(x, y, radius, 0, TWO_PI);
      ctx.fill();
      ctx.lineWidth = dpr;
      ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
      ctx.stroke();
    }

    const palm = this.mapPoint(hand.palm.position);
    ctx.beginPath();
    ctx.arc(palm.x, palm.y, 9 * dpr, 0, TWO_PI);
    ctx.lineWidth = 2 * dpr;
    ctx.strokeStyle = accent;
    ctx.stroke();

    this.#drawChip(points[0], `${hand.id} · ${hand.gesture} · ${Math.round(hand.score * 100)}%`, accent);
  }

  #drawChip(anchor, text, accent) {
    const ctx = this.ctx;
    const dpr = this.dpr;
    ctx.font = `600 ${12 * dpr}px "Segoe UI", system-ui, sans-serif`;
    const width = ctx.measureText(text).width + 16 * dpr;
    const height = 22 * dpr;
    const x = Math.min(Math.max(anchor.x - width / 2, 4 * dpr), this.canvas.width - width - 4 * dpr);
    const y = Math.min(Math.max(anchor.y + 22 * dpr, 4 * dpr), this.canvas.height - height - 4 * dpr);

    ctx.beginPath();
    if (typeof ctx.roundRect === "function") ctx.roundRect(x, y, width, height, 6 * dpr);
    else ctx.rect(x, y, width, height);
    ctx.fillStyle = "rgba(9, 12, 18, 0.82)";
    ctx.fill();

    ctx.fillStyle = accent;
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + 8 * dpr, y + height / 2);
  }

  #drawFps(fps) {
    if (!fps) return;
    const ctx = this.ctx;
    const dpr = this.dpr;
    ctx.font = `600 ${12 * dpr}px "Segoe UI", system-ui, sans-serif`;
    ctx.fillStyle = "rgba(230, 237, 243, 0.85)";
    ctx.textBaseline = "top";
    ctx.fillText(`${fps.toFixed(0)} fps`, 10 * dpr, 10 * dpr);
  }
}

function cssEscape(value) {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") return CSS.escape(value);
  return String(value).replace(/["\\]/g, "\\$&");
}

function createCard(hand) {
  const card = document.createElement("article");
  card.className = "hand-card";
  card.dataset.hand = hand.id;

  const header = document.createElement("header");
  header.className = "hand-card__header";

  const dot = document.createElement("span");
  dot.className = "hand-card__dot";
  dot.style.background = handColor(hand.id);

  const title = document.createElement("h2");
  title.className = "hand-card__title";
  title.textContent = hand.id;

  const score = document.createElement("span");
  score.className = "hand-card__score";
  score.dataset.field = "score";

  header.append(dot, title, score);

  const list = document.createElement("dl");
  list.className = "hand-card__stats";
  const rows = [
    ["Gesture", "gesture"],
    ["Pinch", "pinch"],
    ["Fingers", "fingers"],
    ["Palm speed", "speed"],
    ["Active joint", "activeJoint"],
  ];
  for (const [label, field] of rows) {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = label;
    const value = document.createElement("dd");
    value.dataset.field = field;
    row.append(term, value);
    list.append(row);
  }

  const meter = document.createElement("div");
  meter.className = "hand-card__meter";
  const bar = document.createElement("span");
  bar.dataset.field = "meter";
  meter.append(bar);

  card.append(header, list, meter);
  return card;
}

function updateCard(card, hand) {
  card.dataset.gesture = hand.gesture;
  const set = (field, value) => {
    const node = card.querySelector(`[data-field="${field}"]`);
    if (node) node.textContent = value;
  };

  set("score", `${Math.round(hand.score * 100)}%`);
  set("gesture", hand.gesture);
  set("pinch", hand.pinch.ratio.toFixed(2));
  set("fingers", `${hand.fingers.extended}/${hand.fingers.ratios.length}`);
  set("speed", hand.palm.speed.toFixed(2));

  let active = hand.joints[0];
  for (const joint of hand.joints) {
    if (joint.speed > active.speed) active = joint;
  }
  set("activeJoint", active.name);

  const meter = card.querySelector('[data-field="meter"]');
  if (meter) {
    const ratio = Math.min(hand.palm.speed / SPEED_FOR_FULL_HEAT, 1);
    meter.style.width = `${(ratio * 100).toFixed(1)}%`;
  }
}

/**
 * Creates/updates/removes one card per hand currently detected.
 *
 * @param {HTMLElement} container
 * @param {Array} hands samples returned by HandMotionEngine.update()
 * @param {string[]|null} [liveIds] ids still tracked; keeps a card alive through a dropped frame
 */
export function renderHandPanels(container, hands = [], liveIds = null) {
  if (!container) return;
  const alive = new Set();
  const keep = liveIds ? new Set(liveIds) : alive;

  for (const hand of hands) {
    alive.add(hand.id);
    let card = container.querySelector(`[data-hand="${cssEscape(hand.id)}"]`);
    if (!card) {
      card = createCard(hand);
      container.append(card);
    }
    updateCard(card, hand);
  }

  for (const card of container.querySelectorAll("[data-hand]")) {
    if (!keep.has(card.dataset.hand)) card.remove();
  }

  container.classList.toggle("is-empty", keep.size === 0);
}
