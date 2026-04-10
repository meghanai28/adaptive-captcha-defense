# TicketMonarch Frontend

React + Vite frontend for the TicketMonarch demo app.

## Scripts

In the project directory, you can run:

### `npm run dev`

Starts the Vite dev server on [http://localhost:3000](http://localhost:3000).

The frontend proxies `/api/*` requests to the Flask backend.

### `npm run build`

Builds the production bundle.

### `npm run preview`

Previews the production build locally.

## App Behavior

- Tracks mouse, click, keystroke, and scroll telemetry during the booking flow
- Only records mouse samples after real movement, so stationary cursors do not inflate sessions
- Re-queues failed telemetry flushes instead of silently dropping them
- Calls `/api/agent/rolling` during checkout for rolling risk and honeypot deployment
- Calls `/api/agent/evaluate` on purchase, and uses the returned policy action directly (`allow`, `block`, or puzzle)

## Notes

- Run the Flask backend on `http://localhost:5000`
- The main application README lives at `src/TicketMonarch/README.md`
