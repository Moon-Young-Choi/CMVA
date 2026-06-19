(function () {
  const refreshablePaths = new Set([
    "/",
    "/markets",
    "/volatility",
    "/trend",
    "/correlation-pca",
    "/shock-regime",
    "/models",
    "/validation",
  ]);
  let lastClosedTime = getLatestClosedTime(window.CMVA_SNAPSHOT);
  let reconnectTimer = null;
  let reloadTimer = null;

  function getLatestClosedTime(snapshot) {
    return (
      snapshot &&
      snapshot.summary &&
      (snapshot.summary.latest_closed_time || snapshot.data_accumulation.latest_open_time)
    ) || null;
  }

  function setText(id, value) {
    const element = document.getElementById(id);
    if (!element) return;
    element.textContent = value || "-";
  }

  function updateStatus(snapshot) {
    setText("live-mode", snapshot.mode);
    setText("live-ws", snapshot.summary && snapshot.summary.websocket_status);
    setText("live-bootstrap", snapshot.bootstrap_progress && snapshot.bootstrap_progress.phase);
  }

  function scheduleReload() {
    if (!refreshablePaths.has(window.location.pathname) || reloadTimer) return;
    reloadTimer = window.setTimeout(() => {
      window.location.reload();
    }, 300);
  }

  function handleSnapshot(snapshot) {
    window.CMVA_SNAPSHOT = snapshot;
    updateStatus(snapshot);
    if (window.renderCmvaCharts) {
      window.renderCmvaCharts(snapshot);
    }
    const latestClosedTime = getLatestClosedTime(snapshot);
    if (latestClosedTime && latestClosedTime !== lastClosedTime) {
      lastClosedTime = latestClosedTime;
      scheduleReload();
    }
  }

  function connect() {
    if (!("WebSocket" in window)) return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws/snapshot`);
    socket.onmessage = (event) => {
      try {
        handleSnapshot(JSON.parse(event.data));
      } catch (_error) {
        setText("live-ws", "snapshot parse error");
      }
    };
    socket.onclose = () => {
      window.clearTimeout(reconnectTimer);
      reconnectTimer = window.setTimeout(connect, 2500);
    };
    socket.onerror = () => {
      socket.close();
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", connect, { once: true });
  } else {
    connect();
  }
})();
