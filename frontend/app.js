/**
 * PRobe Frontend — WebSocket client & UI controller
 *
 * Connects to the backend WebSocket at /ws, drives a full episode
 * lifecycle: reset → step* → terminal, and renders all state changes
 * (code viewer, reward bars, history feed, episode-end modal) in real time.
 *
 * Architecture
 * ------------
 *   WsClient         — thin wrapper around native WebSocket with reconnect
 *   RewardDashboard  — renders ring, component bars, issues progress
 *   CodeViewer       — renders syntax-highlighted code with line decorations
 *   HistoryFeed      — append-only action history list
 *   ProbeController  — orchestrates all of the above; owns episode state
 */

"use strict";

// ═══════════════════════════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════════════════════════

const CONFIG = {
  // WebSocket URL — auto-detects host so the page works on any deployment
  wsUrl: `ws://${window.location.hostname}:8000/ws`,
  reconnectDelayMs: 2000,
  ringCircumference: 314,  // 2π × r=50
};


// ═══════════════════════════════════════════════════════════════════
// WsClient — WebSocket with auto-reconnect
// ═══════════════════════════════════════════════════════════════════

class WsClient {
  /**
   * @param {string} url            WebSocket endpoint
   * @param {function} onMessage    Called with parsed JSON message objects
   * @param {function} onStatusChange Called with ('connected'|'disconnected')
   */
  constructor(url, onMessage, onStatusChange) {
    this._url            = url;
    this._onMessage      = onMessage;
    this._onStatusChange = onStatusChange;
    this._socket         = null;
    this._connected      = false;
  }

  connect() {
    if (this._socket) this._socket.close();

    this._socket = new WebSocket(this._url);

    this._socket.onopen = () => {
      this._connected = true;
      this._onStatusChange("connected");
    };

    this._socket.onclose = () => {
      this._connected = false;
      this._onStatusChange("disconnected");
    };

    this._socket.onerror = (err) => {
      console.error("[WsClient] error:", err);
      this._connected = false;
      this._onStatusChange("disconnected");
    };

    this._socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this._onMessage(msg);
      } catch (e) {
        console.warn("[WsClient] unparseable message:", event.data);
      }
    };
  }

  send(payload) {
    if (!this._connected) {
      console.warn("[WsClient] send called while disconnected");
      return;
    }
    this._socket.send(JSON.stringify(payload));
  }

  get isConnected() { return this._connected; }
}


// ═══════════════════════════════════════════════════════════════════
// CodeViewer — renders code with per-line decorations
// ═══════════════════════════════════════════════════════════════════

class CodeViewer {
  constructor(preEl) {
    this._pre = preEl;
    this._lines = [];
    // Track which lines have active highlights so we can clear them
    this._decoratedLines = new Set();
  }

  /**
   * Render source code as numbered, individually addressable lines.
   * Clears any previous decorations.
   */
  render(sourceCode) {
    this._lines = sourceCode.split("\n");
    this._decoratedLines.clear();
    this._pre.innerHTML = this._lines.map((text, idx) => {
      const lineNum = idx + 1;
      return `<span class="code-line" id="cl-${lineNum}">`
           + `<span class="code-line-num">${lineNum}</span>`
           + escapeHtml(text)
           + `</span>`;
    }).join("\n");
  }

  /**
   * Apply a CSS class to a specific line.
   * @param {number} lineNumber   1-based
   * @param {string} cssClass     e.g. 'hl-comment'
   */
  decorateLine(lineNumber, cssClass) {
    const el = document.getElementById(`cl-${lineNumber}`);
    if (!el) return;
    // Remove any previous highlight class on this line before adding the new one
    el.classList.remove("hl-comment", "hl-issue", "hl-scanner", "hl-context");
    el.classList.add(cssClass);
    this._decoratedLines.add(lineNumber);
  }

  /** Scroll the given 1-based line number into view. */
  scrollToLine(lineNumber) {
    const el = document.getElementById(`cl-${lineNumber}`);
    if (el) el.scrollIntoView({ block: "center", behavior: "smooth" });
  }

  clearDecorations() {
    for (const lineNum of this._decoratedLines) {
      const el = document.getElementById(`cl-${lineNum}`);
      if (el) el.classList.remove("hl-comment", "hl-issue", "hl-scanner", "hl-context");
    }
    this._decoratedLines.clear();
  }
}


// ═══════════════════════════════════════════════════════════════════
// RewardDashboard — ring + bars + issues progress
// ═══════════════════════════════════════════════════════════════════

class RewardDashboard {
  constructor() {
    this._ringTrack     = document.getElementById("ring-track");
    this._ringValue     = document.getElementById("ring-value");
    this._issuesFill    = document.getElementById("issues-bar-fill");
    this._issuesLabel   = document.getElementById("issues-found-label");

    // Component bar element pairs { fill, val }
    this._bars = {
      issue_credit:          this._barPair("issue_credit"),
      classification_credit: this._barPair("classification_credit"),
      false_positive_penalty:this._barPair("false_positive_penalty"),
      coverage_bonus:        this._barPair("coverage_bonus"),
      decision_score:        this._barPair("decision_score"),
      efficiency_bonus:      this._barPair("efficiency_bonus"),
    };
  }

  _barPair(key) {
    return {
      fill: document.getElementById(`bar-${key}`),
      val:  document.getElementById(`val-${key}`),
    };
  }

  /**
   * Update the cumulative reward ring.
   * Clamps input to [-1, 1] and maps to ring arc.
   */
  updateRing(cumulativeReward) {
    const clamped   = Math.max(-1, Math.min(1, cumulativeReward));
    // Map [-1, 1] → [0, circumference]: negative reward still shows a partial arc
    const fraction  = (clamped + 1) / 2;
    const offset    = CONFIG.ringCircumference * (1 - fraction);

    this._ringTrack.style.strokeDashoffset = offset;
    // Colour: green above 0, red below
    this._ringTrack.style.stroke = clamped >= 0 ? "var(--green)" : "var(--red)";
    this._ringValue.textContent  = clamped.toFixed(2);
    this._ringValue.style.color  = clamped >= 0 ? "var(--green)" : "var(--red)";
  }

  /**
   * Render per-component score bars from a components dict.
   * The bar width maps the absolute value to a 0-100% scale capped at 0.40.
   */
  updateBars(components) {
    const MAX_BAR_VALUE = 0.40;

    for (const [key, pair] of Object.entries(this._bars)) {
      const rawValue = components[key] ?? 0;
      const absWidth = Math.min(Math.abs(rawValue) / MAX_BAR_VALUE * 100, 100);

      pair.fill.style.width   = `${absWidth}%`;
      pair.val.textContent    = rawValue.toFixed(2);

      // Positive/negative/neutral colouring
      pair.fill.classList.remove("positive", "negative", "neutral");
      if (rawValue > 0)       pair.fill.classList.add("positive");
      else if (rawValue < 0)  pair.fill.classList.add("negative");
      else                    pair.fill.classList.add("neutral");
    }
  }

  /** Update the issues-found progress bar. */
  updateIssues(found, total) {
    const pct = total > 0 ? (found / total) * 100 : 0;
    this._issuesFill.style.width  = `${pct}%`;
    this._issuesLabel.textContent = `${found} / ${total}`;
  }

  reset() {
    this.updateRing(0);
    this.updateBars({});
    this.updateIssues(0, 0);
  }
}


// ═══════════════════════════════════════════════════════════════════
// HistoryFeed — append-only episode action log
// ═══════════════════════════════════════════════════════════════════

class HistoryFeed {
  constructor(containerEl) {
    this._container = containerEl;
    this._count = 0;
  }

  clear() {
    this._container.innerHTML = '<div class="history-empty">No actions yet.</div>';
    this._count = 0;
  }

  /**
   * Append one step to the feed.
   * @param {string} actionType   Human-readable action label
   * @param {object} reward       RewardType object from server
   */
  append(actionType, reward) {
    if (this._count === 0) {
      this._container.innerHTML = "";
    }
    this._count++;

    const total    = reward.total ?? 0;
    const polarity = total > 0.001 ? "positive" : total < -0.001 ? "negative" : "neutral";
    const rewardClass = total >= 0 ? "pos" : "neg";
    const sign        = total >= 0 ? "+" : "";

    const item = document.createElement("div");
    item.className = `history-item ${polarity}`;
    item.innerHTML = `
      <div>
        <span class="h-action">${escapeHtml(actionType)}</span>
        &nbsp;→&nbsp;
        <span class="h-reward ${rewardClass}">${sign}${total.toFixed(3)}</span>
      </div>
      <div class="h-explain">${escapeHtml(reward.explanation ?? "")}</div>
    `;
    this._container.prepend(item);   // newest at top
  }
}


// ═══════════════════════════════════════════════════════════════════
// ProbeController — owns all state, wires UI ↔ WsClient
// ═══════════════════════════════════════════════════════════════════

class ProbeController {
  constructor() {
    // Sub-components
    this._ws        = null;
    this._viewer    = new CodeViewer(document.getElementById("code-block"));
    this._dashboard = new RewardDashboard();
    this._feed      = new HistoryFeed(document.getElementById("history-feed"));

    // Episode state
    this._episodeActive   = false;
    this._cumulativeReward = 0;
    this._stepCount       = 0;
    this._maxSteps        = 0;
    this._totalIssues     = 0;
    this._foundCount      = 0;
    this._lastObs         = null;

    this._bindStaticButtons();
  }

  // ── Initialisation ──────────────────────────────────────────────

  _bindStaticButtons() {
    document.getElementById("btn-connect").addEventListener("click", () => this._connect());
    document.getElementById("btn-reset").addEventListener("click",   () => this._sendReset());
    document.getElementById("btn-comment").addEventListener("click", () => this._sendComment());
    document.getElementById("btn-get-context").addEventListener("click", () => this._sendGetContext());
    document.getElementById("btn-run-scanner").addEventListener("click", () => this._sendAction("run_scanner"));
    document.getElementById("btn-request-changes").addEventListener("click", () => this._sendAction("request_changes"));
    document.getElementById("btn-approve").addEventListener("click", () => this._sendAction("approve"));
    document.getElementById("btn-submit").addEventListener("click",  () => this._sendAction("submit_review"));
    document.getElementById("btn-escalate").addEventListener("click",() => this._sendAction("escalate_to_security_review"));
    document.getElementById("modal-close").addEventListener("click", () => {
      document.getElementById("modal-overlay").style.display = "none";
      this._sendReset();
    });
  }

  // ── WebSocket lifecycle ──────────────────────────────────────────

  _connect() {
    this._ws = new WsClient(
      CONFIG.wsUrl,
      (msg) => this._handleMessage(msg),
      (status) => this._handleConnectionStatus(status),
    );
    this._ws.connect();
  }

  _handleConnectionStatus(status) {
    const badge  = document.getElementById("conn-badge");
    const btnReset  = document.getElementById("btn-reset");
    const btnConnect = document.getElementById("btn-connect");

    if (status === "connected") {
      badge.textContent = "🟢 Connected";
      badge.className   = "badge connected";
      btnConnect.textContent = "Reconnect";
      btnReset.disabled = false;
      // Auto-start first episode on successful connect
      this._sendReset();
    } else {
      badge.textContent = "⚫ Disconnected";
      badge.className   = "badge disconnected";
      this._setActionButtonsEnabled(false);
    }
  }

  // ── Message dispatch ─────────────────────────────────────────────

  _handleMessage(msg) {
    switch (msg.type) {
      case "reset": this._applyObservation(msg.observation, null, false); break;
      case "step":  this._applyStep(msg);  break;
      case "error": this._showError(msg.detail); break;
      default: console.warn("[ProbeController] unknown message type:", msg.type);
    }
  }

  // ── Episode state application ────────────────────────────────────

  /**
   * Apply a fresh observation (after reset or step).
   * Updates every UI component from the single observation object.
   */
  _applyObservation(obs, reward, done) {
    this._lastObs     = obs;
    this._stepCount   = obs.step_count;
    this._maxSteps    = obs.max_steps;
    this._totalIssues = obs.total_issues;
    this._foundCount  = obs.issues_found_count;

    // ── Task metadata ──
    document.getElementById("task-label").textContent =
      `Task ${obs.task_id} — ${obs.file_name}`;
    document.getElementById("task-desc").textContent  = obs.task_description;
    document.getElementById("steps-counter").textContent =
      `Step ${obs.step_count} / ${obs.max_steps}`;

    const diffBadge = document.getElementById("difficulty-badge");
    diffBadge.textContent = obs.task_difficulty;
    diffBadge.className   = `difficulty-badge ${obs.task_difficulty.replace(/\s+/g, "-")}`;

    // ── Adversarial hint ──
    const advEl = document.getElementById("adv-hint");
    if (obs.adversarial_hint) {
      advEl.textContent    = `⚠️ ${obs.adversarial_hint}`;
      advEl.style.display  = "block";
    } else {
      advEl.style.display  = "none";
    }

    // ── Code viewer ── (only re-render if code changed, i.e. on reset)
    if (!reward) {
      this._viewer.render(obs.code_snippet);
      this._viewer.clearDecorations();
    }

    // ── Highlight lines mentioned in review history ──
    this._decorateHistoryLines(obs.review_history);

    // ── Context hints ──
    this._renderHints(obs.context_hints);

    // ── Dashboard ──
    this._cumulativeReward = obs.metadata?.cumulative_reward ?? 0;
    this._dashboard.updateRing(this._cumulativeReward);
    this._dashboard.updateIssues(this._foundCount, this._totalIssues);

    if (reward) {
      this._dashboard.updateBars(reward.components ?? {});
      this._feed.append(this._lastActionLabel, reward);
    }

    // ── Terminal handling ──
    if (done) {
      this._episodeActive = false;
      this._setActionButtonsEnabled(false);
      this._showEpisodeEndModal(obs, reward);
    } else {
      this._episodeActive = true;
      this._setActionButtonsEnabled(true);
    }
  }

  _applyStep(msg) {
    this._applyObservation(msg.observation, msg.reward, msg.done);
  }

  // ── Line decorations ─────────────────────────────────────────────

  /**
   * Walk review_history and apply colour-coded line highlights.
   * Later entries overwrite earlier ones on the same line, so the most
   * recent action's highlight takes priority.
   */
  _decorateHistoryLines(history) {
    this._viewer.clearDecorations();
    for (const entry of history) {
      if (!entry.line) continue;
      let cssClass = "hl-comment";
      if (entry.type === "scanner_result") continue;      // no single line
      if (entry.type === "context_probe")  cssClass = "hl-context";
      if (entry.type === "comment")        cssClass = "hl-comment";
      this._viewer.decorateLine(entry.line, cssClass);
    }
  }

  // ── Hints ────────────────────────────────────────────────────────

  _renderHints(hints) {
    const container = document.getElementById("hints-container");
    const list      = document.getElementById("hints-list");

    if (!hints || hints.length === 0) {
      container.style.display = "none";
      return;
    }
    container.style.display = "block";
    list.innerHTML = hints.map(h =>
      `<div class="hint-item">${escapeHtml(h)}</div>`
    ).join("");
  }

  // ── Action senders ───────────────────────────────────────────────

  _sendReset() {
    if (!this._ws?.isConnected) return;
    this._episodeActive = false;
    this._setActionButtonsEnabled(false);
    this._dashboard.reset();
    this._feed.clear();
    this._viewer._pre.innerHTML = '<span class="placeholder-text">Loading…</span>';
    document.getElementById("hints-container").style.display = "none";
    document.getElementById("adv-hint").style.display = "none";
    this._ws.send({ command: "reset" });
  }

  _sendComment() {
    const line           = parseInt(document.getElementById("inp-line").value, 10) || null;
    const comment        = document.getElementById("inp-comment").value.trim();
    const severity       = document.getElementById("inp-severity").value     || null;
    const category       = document.getElementById("inp-category").value     || null;
    const classification = document.getElementById("inp-classification").value || null;

    if (!comment) {
      alert("Please enter a comment before submitting.");
      return;
    }
    this._lastActionLabel = `ADD_COMMENT (L${line ?? "?"})`;
    this._sendAction("add_comment", {
      line_number: line,
      comment,
      severity,
      category,
      classification,
    });
    // Clear comment fields after send
    document.getElementById("inp-comment").value = "";
  }

  _sendGetContext() {
    const line = parseInt(document.getElementById("inp-probe-line").value, 10) || null;
    if (!line) { alert("Enter a line number to probe."); return; }
    this._lastActionLabel = `GET_CONTEXT (L${line})`;
    this._sendAction("get_context", { line_number: line });
  }

  /**
   * Send a step action to the server.
   * @param {string} actionType   snake_case action type string
   * @param {object} extra        Additional fields (line_number, comment, …)
   */
  _sendAction(actionType, extra = {}) {
    if (!this._ws?.isConnected || !this._episodeActive) return;
    this._lastActionLabel = actionType.toUpperCase().replace(/_/g, " ");
    this._ws.send({
      command: "step",
      action: { action_type: actionType, ...extra },
    });
  }

  // ── UI helpers ───────────────────────────────────────────────────

  _setActionButtonsEnabled(enabled) {
    const ids = [
      "btn-comment", "btn-get-context", "btn-run-scanner",
      "btn-request-changes", "btn-approve", "btn-submit", "btn-escalate",
    ];
    for (const id of ids) {
      document.getElementById(id).disabled = !enabled;
    }
  }

  _showEpisodeEndModal(obs, reward) {
    const totalReward = this._cumulativeReward;
    const passed      = reward?.passed ?? false;

    document.getElementById("modal-overlay").style.display = "flex";
    document.getElementById("modal-icon").textContent =
      totalReward >= 0.5 ? "🏆" : totalReward >= 0 ? "🏁" : "💔";
    document.getElementById("modal-title").textContent =
      passed ? "Episode Passed!" : "Episode Complete";
    document.getElementById("modal-body").textContent =
      reward?.explanation ?? "Episode ended.";

    // Render a small stats grid inside the modal
    const decision  = obs.metadata?.review_decision ?? "—";
    const esc       = obs.metadata?.escalation_required ? "Yes" : "No";
    document.getElementById("modal-stats").innerHTML = `
      <span class="stat-label">Cumulative reward</span>
      <span class="stat-value">${totalReward.toFixed(3)}</span>
      <span class="stat-label">Issues found</span>
      <span class="stat-value">${obs.issues_found_count} / ${obs.total_issues}</span>
      <span class="stat-label">Steps used</span>
      <span class="stat-value">${obs.step_count} / ${obs.max_steps}</span>
      <span class="stat-label">Decision</span>
      <span class="stat-value">${decision}</span>
      <span class="stat-label">Escalation required</span>
      <span class="stat-value">${esc}</span>
    `;
  }

  _showError(detail) {
    console.error("[ProbeController] server error:", detail);
    // Non-intrusive: just log and append to feed as a red entry
    this._feed.append("ERROR", {
      total: 0,
      explanation: detail ?? "Unknown server error",
    });
  }
}


// ═══════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════

/** Escape HTML special chars to prevent XSS when inserting code/text. */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}


// ═══════════════════════════════════════════════════════════════════
// Bootstrap
// ═══════════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
  window._probe = new ProbeController();
});
