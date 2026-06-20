(function () {
  const refreshablePaths = new Set([
    "/",
    "/setup",
    "/overview",
    "/dashboard",
    "/data-quality",
    "/model-lab",
    "/current-market-state",
    "/trend-seasonality",
    "/diagnostics",
    "/rolling-evaluation",
    "/model-comparison",
    "/markets",
    "/volatility",
    "/trend",
    "/correlation-pca",
    "/shock-regime",
    "/models",
    "/validation",
  ]);
  let lastClosedTime = getLatestClosedTime(window.CMVA_SNAPSHOT);
  let lastModelJobKey = getModelJobKey(window.CMVA_SNAPSHOT);
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

  function updateModelLabProgress(snapshot) {
    const status = snapshot && snapshot.model_lab && snapshot.model_lab.job_status;
    if (!status) return;
    setText("model-lab-status", status.status);
    setText("model-lab-stage", status.active_stage);
    setText("model-lab-target", status.active_target);
    setText("model-lab-candidate", status.active_candidate);
    setText("model-lab-completed-fits", status.completed_fits || 0);
    setText("model-lab-total-fits", status.total_fits || 0);
    const bar = document.getElementById("model-lab-progress-bar");
    if (!bar) return;
    const progress = Number(status.progress_pct);
    const percent = Number.isFinite(progress) ? Math.max(0, Math.min(100, progress * 100)) : 0;
    bar.style.width = `${percent.toFixed(1)}%`;
  }

  function getModelJobKey(snapshot) {
    const status = snapshot && snapshot.model_lab && snapshot.model_lab.job_status;
    if (!status) return "";
    return `${status.status || ""}:${status.updated_at || ""}`;
  }

  function applyTableFilter(form) {
    const tableSelector = form.dataset.tableFilter;
    const table = tableSelector && document.querySelector(tableSelector);
    if (!table || !table.tBodies.length) return;
    const family = (form.querySelector("[name='family']") || {}).value || "";
    const target = (form.querySelector("[name='target']") || {}).value || "";
    const symbol = (form.querySelector("[name='symbol']") || {}).value || "";
    const topValue = Number((form.querySelector("[name='top_n']") || {}).value);
    Array.from(table.tBodies[0].rows).forEach((row) => {
      const rank = Number(row.dataset.rank);
      const visible =
        (!family || row.dataset.family === family) &&
        (!target || row.dataset.target === target) &&
        (!symbol || row.dataset.symbol === symbol) &&
        (!Number.isFinite(topValue) || topValue <= 0 || !Number.isFinite(rank) || rank <= topValue);
      row.hidden = !visible;
    });
  }

  function applyMetricFilter(form) {
    const metric = (form.querySelector("[name='metric']") || {}).value || "";
    if (!metric) return;
    document.querySelectorAll("[data-metric]").forEach((element) => {
      element.hidden = element.dataset.metric !== metric;
      if (!element.hidden && window.echarts) {
        const chart = window.echarts.getInstanceByDom(element);
        if (chart) chart.resize();
      }
    });
  }

  function applyFilters() {
    document.querySelectorAll(".js-table-filter").forEach(applyTableFilter);
    document.querySelectorAll(".js-metric-filter").forEach(applyMetricFilter);
  }

  function flashVisibleRows() {
    Array.from(document.querySelectorAll("tbody tr:not([hidden])"))
      .slice(0, 16)
      .forEach((row) => {
        row.classList.remove("flash-update");
        void row.offsetWidth;
        row.classList.add("flash-update");
      });
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
    updateModelLabProgress(snapshot);
    if (window.renderCmvaCharts) {
      window.renderCmvaCharts(snapshot);
    }
    applyFilters();
    const latestClosedTime = getLatestClosedTime(snapshot);
    const modelJobKey = getModelJobKey(snapshot);
    if (latestClosedTime && latestClosedTime !== lastClosedTime) {
      lastClosedTime = latestClosedTime;
      flashVisibleRows();
      scheduleReload();
    }
    if (modelJobKey && modelJobKey !== lastModelJobKey) {
      lastModelJobKey = modelJobKey;
      flashVisibleRows();
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

  function initialize() {
    applyFilters();
    document.addEventListener("change", applyFilters);
    document.addEventListener("input", (event) => {
      if (event.target && event.target.closest && event.target.closest(".js-table-filter, .js-metric-filter")) {
        applyFilters();
      }
    });
    connect();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
