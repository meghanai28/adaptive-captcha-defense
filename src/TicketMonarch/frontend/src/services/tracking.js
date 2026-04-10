import { v4 as uuidv4 } from 'uuid'

const API_BASE_URL = '/api'

// Target sampling rate ~50–100 Hz
const MOUSE_SAMPLE_INTERVAL_MS = 15 // ~66 Hz
const FLUSH_INTERVAL_MS = 5000 // 5 seconds

let sessionId = null
let currentPage = null
let trackingEnabled = true // false on /dev page

let mouseBuffer = []
let clickBuffer = []
let keystrokeBuffer = []
let scrollBuffer = []

let lastMouseEvent = null
let mouseMovedSinceSample = false
let lastFlushTime = Date.now()
let lastClickTimestamp = null
let lastKeyTimestampByField = {}
let lastScrollTimestamp = null

let mouseIntervalId = null
let flushIntervalId = null
let isInitialized = false
let flushInFlight = null

// Non-sensitive special keys worth logging for behavioral analysis.
// Letters, digits, and modifiers (Shift, Ctrl, Alt, Meta) are excluded.
const LOGGABLE_KEYS = new Set([
  'Backspace', 'Delete', 'Tab', 'Enter', 'Escape',
  'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
  'Home', 'End', 'PageUp', 'PageDown',
  'Insert', 'CapsLock', 'NumLock', 'ScrollLock',
  'ContextMenu', 'PrintScreen', 'Pause',
  'F1', 'F2', 'F3', 'F4', 'F5', 'F6',
  'F7', 'F8', 'F9', 'F10', 'F11', 'F12'
])

export function getSessionId() {
  if (sessionId) return sessionId

  // Try to reuse from storage if available
  const stored = window.sessionStorage.getItem('tm_session_id')
  if (stored) {
    sessionId = stored
    return sessionId
  }

  sessionId = uuidv4()
  window.sessionStorage.setItem('tm_session_id', sessionId)
  // Also store in localStorage so the /dev dashboard in another tab can find it
  window.localStorage.setItem('tm_active_session_id', sessionId)
  return sessionId
}

export function setTrackingPage(pageName) {
  currentPage = pageName
  // null means tracking is disabled (e.g. on /dev dashboard)
  trackingEnabled = pageName !== null
}

function handleRawMouseMove(event) {
  if (!trackingEnabled) return
  lastMouseEvent = {
    x: event.clientX,
    y: event.clientY
  }
  mouseMovedSinceSample = true
}

function sampleMousePosition() {
  if (!lastMouseEvent || !trackingEnabled || !mouseMovedSinceSample) return

  const timestamp = performance.now()

  mouseBuffer.push({
    x: lastMouseEvent.x,
    y: lastMouseEvent.y,
    t: timestamp
  })
  mouseMovedSinceSample = false
}

function handleClick(event) {
  if (!trackingEnabled) return

  const now = performance.now()

  const timeSinceLastClick = lastClickTimestamp != null ? now - lastClickTimestamp : null
  lastClickTimestamp = now

  const buttonMap = {
    0: 'left',
    1: 'middle',
    2: 'right'
  }

  const targetInfo = event.target
    ? {
        tag: event.target.tagName,
        id: event.target.id || null,
        classes: event.target.className || null,
        name: event.target.name || null,
        type: event.target.type || null,
        text: (event.target.innerText || '').slice(0, 64)
      }
    : null

  clickBuffer.push({
    t: now,
    x: event.clientX,
    y: event.clientY,
    button: buttonMap[event.button] || 'unknown',
    target: targetInfo,
    dt_since_last: timeSinceLastClick
  })
}

function handleWheel(event) {
  if (!trackingEnabled) return

  const now = performance.now()
  const timeSinceLastScroll = lastScrollTimestamp != null ? now - lastScrollTimestamp : null
  lastScrollTimestamp = now

  scrollBuffer.push({
    t: now,
    scrollX: window.scrollX,
    scrollY: window.scrollY,
    dy: event.deltaY,
    dt_since_last: timeSinceLastScroll
  })
}

async function flushBuffers(force = false) {
  if (flushInFlight) {
    await flushInFlight
    // If this is a forced flush, we need to actually run again after the
    // in-flight one finishes, since it may have been a no-op interval tick.
    if (!force) return
  }

  flushInFlight = (async () => {
    try {
      const now = Date.now()
      const elapsed = now - lastFlushTime

      if (!force && elapsed < FLUSH_INTERVAL_MS) {
        return
      }

      lastFlushTime = now

      if ((!mouseBuffer.length && !clickBuffer.length && !keystrokeBuffer.length && !scrollBuffer.length) || !sessionId) {
        return
      }

      const payloadBase = {
        session_id: getSessionId(),
        page: currentPage
      }

      const mousePayload = mouseBuffer.length
        ? {
            ...payloadBase,
            samples: mouseBuffer
          }
        : null

      const clickPayload = clickBuffer.length
        ? {
            ...payloadBase,
            clicks: clickBuffer
          }
        : null

      const keystrokePayload = keystrokeBuffer.length
        ? {
            ...payloadBase,
            keystrokes: keystrokeBuffer
          }
        : null

      const scrollPayload = scrollBuffer.length
        ? {
            ...payloadBase,
            scrolls: scrollBuffer
          }
        : null

      const mouseBatch = mouseBuffer
      const clickBatch = clickBuffer
      const keystrokeBatch = keystrokeBuffer
      const scrollBatch = scrollBuffer

      mouseBuffer = []
      clickBuffer = []
      keystrokeBuffer = []
      scrollBuffer = []

      try {
        const requests = []

        const postJson = async (url, payload) => {
          const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          })

          if (!response.ok) {
            throw new Error(`Request failed: ${response.status}`)
          }
        }

        if (mousePayload) {
          requests.push(
            postJson(`${API_BASE_URL}/tracking/mouse`, mousePayload)
          )
        }

        if (clickPayload) {
          requests.push(
            postJson(`${API_BASE_URL}/tracking/clicks`, clickPayload)
          )
        }

        if (keystrokePayload) {
          requests.push(
            postJson(`${API_BASE_URL}/tracking/keystrokes`, keystrokePayload)
          )
        }

        if (scrollPayload) {
          requests.push(
            postJson(`${API_BASE_URL}/tracking/scroll`, scrollPayload)
          )
        }

        if (requests.length) {
          await Promise.all(requests)
        }
      } catch {
        // Restore unsent telemetry at the front so old events keep their order.
        mouseBuffer = mouseBatch.concat(mouseBuffer)
        clickBuffer = clickBatch.concat(clickBuffer)
        keystrokeBuffer = keystrokeBatch.concat(keystrokeBuffer)
        scrollBuffer = scrollBatch.concat(scrollBuffer)
      }
    } finally {
      flushInFlight = null
    }
  })()

  await flushInFlight
}

export function initTracking() {
  if (isInitialized || typeof window === 'undefined') return
  // Never initialize tracking on the dev dashboard — it would create
  // a new session ID and overwrite localStorage, hiding the real session.
  if (window.location.pathname.startsWith('/dev')) return
  isInitialized = true

  getSessionId()

  window.addEventListener('mousemove', handleRawMouseMove, { passive: true })
  window.addEventListener('click', handleClick, { passive: true })
  window.addEventListener('wheel', handleWheel, { passive: true })

  // Keystroke tracking — captures form field typing on any page
  window.addEventListener(
    'keydown',
    (event) => {
      if (!trackingEnabled) return
      const target = event.target
      if (!target) return
      const tag = target.tagName
      const isFormField =
        tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
      if (!isFormField) return

      const now = performance.now()
      const fieldId = target.name || target.id || 'unknown'
      const last = lastKeyTimestampByField[fieldId]
      const dt = last != null ? now - last : null
      lastKeyTimestampByField[fieldId] = now

      // Log non-sensitive special key names; letters/digits/modifiers stay null
      const key = LOGGABLE_KEYS.has(event.key) ? event.key : null
      const keystroke = {
        field: fieldId,
        type: 'down',
        t: now,
        dt_since_last: dt
      }

      if (key) keystroke.key = key
      keystrokeBuffer.push(keystroke)
    },
    false
  )

  window.addEventListener(
    'keyup',
    (event) => {
      if (!trackingEnabled) return
      const target = event.target
      if (!target) return
      const tag = target.tagName
      const isFormField =
        tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
      if (!isFormField) return

      const now = performance.now()
      const fieldId = target.name || target.id || 'unknown'
      const key = LOGGABLE_KEYS.has(event.key) ? event.key : null
      const keystroke = {
        field: fieldId,
        type: 'up',
        t: now
      }

      if (key) keystroke.key = key
      keystrokeBuffer.push(keystroke)
    },
    false
  )

  mouseIntervalId = window.setInterval(sampleMousePosition, MOUSE_SAMPLE_INTERVAL_MS)

  // Periodic flush
  flushIntervalId = window.setInterval(flushBuffers, 1000)
}

/**
 * Force-flush all buffered telemetry immediately.
 * Call before agent evaluation to ensure DB has latest data.
 */
export async function forceFlush() {
  await flushBuffers(true)
}

/**
 * Reset the session — generates a new session ID and clears all buffers.
 * Call after a successful purchase so the next flow starts fresh.
 */
export function resetSession() {
  // Clear buffers
  mouseBuffer = []
  clickBuffer = []
  keystrokeBuffer = []
  scrollBuffer = []

  // Reset timing state — intervals keep running (they're guarded by
  // isInitialized and will pick up the new session ID automatically)
  lastMouseEvent = null
  mouseMovedSinceSample = false
  lastClickTimestamp = null
  lastKeyTimestampByField = {}
  lastScrollTimestamp = null
  lastFlushTime = Date.now()

  // Generate new session ID
  sessionId = uuidv4()
  window.sessionStorage.setItem('tm_session_id', sessionId)
  window.sessionStorage.removeItem('tm_is_bot')
  window.localStorage.setItem('tm_active_session_id', sessionId)

  console.log(`[Tracking] Session reset — new ID: ${sessionId}`)
}

