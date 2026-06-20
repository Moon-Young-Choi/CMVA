(function () {
  const root = document.getElementById("simulation-app") || document.getElementById("simulation-results");
  const initialSnapshot = window.CMVA_SIMULATION_SNAPSHOT || null;
  const charts = new WeakMap();
  let reconnectTimer = null;

  function text(id, value) {
    const element = document.getElementById(id);
    if (!element) return;
    element.textContent = value === null || value === undefined || value === "" ? "-" : String(value);
  }

  function pct(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "0.0%";
    return `${Math.max(0, Math.min(100, number * 100)).toFixed(1)}%`;
  }

  function patch(snapshot) {
    if (!snapshot) return;
    patchProgress(snapshot.progress || {});
    patchScoreCards(snapshot.latest_score || {});
    patchScoreRows(snapshot.scores || []);
    patchWarnings(snapshot.warnings || []);
    renderSimulationCharts(snapshot.metric_series || {});
  }

  function patchProgress(progress) {
    text("sim-status", progress.status_label || progress.status);
    text("sim-progress-text", pct(progress.progress_pct));
    text("sim-current-origin", progress.current_origin_time);
    text("sim-active-symbol", progress.active_symbol);
    text("sim-active-model", progress.active_model);
    text("sim-completed-fits", progress.completed_fits || 0);
    text("sim-total-fits", progress.total_fits || 0);
    text("sim-updated-at", progress.updated_at);
    const bar = document.getElementById("simulation-progress-bar");
    if (bar) bar.style.width = pct(progress.progress_pct);
  }

  function patchScoreCards(score) {
    const values = [
      score.volatility_score_0_100,
      score.expectation_score_0_100,
      score.trend_score_minus100_100,
      score.seasonality_score_0_100,
      score.volatility_confidence_0_1,
    ];
    document.querySelectorAll("#simulation-app .metric-value").forEach((element, index) => {
      const value = values[index];
      element.textContent = value === null || value === undefined ? "-" : String(value);
    });
  }

  function patchScoreRows(rows) {
    const body = document.getElementById("simulation-score-rows");
    if (!body) return;
    const latest = rows.slice(-30).reverse();
    body.innerHTML =
      latest
        .map(
          (row) => `<tr>
            <td>${escapeHtml(row.origin_time)}</td>
            <td>${escapeHtml(row.symbol)}</td>
            <td>${escapeHtml(row.selected_model_id)}</td>
            <td>${escapeHtml(row.volatility_score_0_100)}</td>
            <td>${escapeHtml(row.expectation_score_0_100)}</td>
            <td>${escapeHtml(row.trend_score_minus100_100)}</td>
            <td>${escapeHtml(row.seasonality_score_0_100)}</td>
            <td>${escapeHtml(row.volatility_confidence_0_1)}</td>
            <td>${escapeHtml((row.warnings || []).join("; "))}</td>
          </tr>`
        )
        .join("") || `<tr><td colspan="9">아직 표시할 결과가 없습니다.</td></tr>`;
  }

  function patchWarnings(warnings) {
    const list = document.getElementById("simulation-warning-list");
    if (!list) return;
    list.innerHTML =
      warnings.length > 0
        ? warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")
        : "<li>표시할 경고가 없습니다.</li>";
  }

  function renderSimulationCharts(metricSeries) {
    if (!window.echarts) return;
    document.querySelectorAll(".simulation-chart").forEach((element) => {
      const key = element.dataset.simSeries;
      drawChart(element, key, metricSeries[key]);
    });
  }

  function drawChart(element, key, rawRows) {
    let chart = charts.get(element);
    if (!chart) {
      chart = window.echarts.init(element, null, { renderer: "canvas" });
      charts.set(element, chart);
      window.addEventListener("resize", () => chart.resize());
    }
    const title = element.getAttribute("aria-label") || koreanTitle(key);
    const option = buildOption(title, key, rawRows);
    chart.setOption(option, true);
  }

  function buildOption(title, key, rawRows) {
    const base = {
      animation: false,
      backgroundColor: "#eef2f6",
      title: { text: title, left: 10, top: 8, textStyle: { color: "#526071", fontSize: 12, fontWeight: 500 } },
      grid: { left: 46, right: 18, top: 44, bottom: 32 },
      tooltip: { trigger: "axis", confine: true },
      legend: { top: 8, right: 8, textStyle: { color: "#526071" } },
      xAxis: { type: "category", axisLabel: { color: "#667085", hideOverlap: true }, axisTick: { show: false } },
      yAxis: { type: "value", scale: true, axisLabel: { color: "#667085" }, splitLine: { lineStyle: { color: "#d8dee6" } } },
      series: [],
    };
    if (key && key.endsWith("_by_model")) {
      const rowsByModel = rawRows || {};
      const times = collectTimes(Object.values(rowsByModel).flat());
      base.xAxis.data = times;
      base.series = Object.entries(rowsByModel).map(([model, rows]) => ({
        name: model,
        type: "line",
        showSymbol: false,
        data: times.map((time) => valueAt(rows, time, "value")),
      }));
      return base;
    }
    if (key === "rmse_mae_by_model") {
      const rowsByModel = rawRows || {};
      const flattened = Object.values(rowsByModel).flat();
      const times = collectTimes(flattened);
      base.xAxis.data = times;
      base.series = Object.entries(rowsByModel).flatMap(([model, rows]) => [
        { name: `${model} RMSE`, type: "line", showSymbol: false, data: times.map((time) => valueAt(rows, time, "rmse")) },
        { name: `${model} MAE`, type: "line", showSymbol: false, data: times.map((time) => valueAt(rows, time, "mae")) },
      ]);
      return base;
    }
    if (key === "forecast_vs_realized_volatility") {
      const rows = rawRows || [];
      const times = rows.map((row) => row.time);
      base.xAxis.data = times;
      base.series = [
        { name: "예측 변동성", type: "line", showSymbol: false, data: rows.map((row) => numeric(row.forecast)) },
        { name: "실현 변동성", type: "line", showSymbol: false, data: rows.map((row) => numeric(row.realized)) },
      ];
      return base;
    }
    if (key === "selected_model") {
      const rows = rawRows || [];
      const labels = Array.from(new Set(rows.map((row) => row.value).filter(Boolean)));
      base.xAxis.data = rows.map((row) => row.time);
      base.yAxis.axisLabel.formatter = (value) => labels[Number(value) - 1] || "";
      base.series = [
        {
          name: "선택 모델",
          type: "line",
          step: "middle",
          showSymbol: false,
          data: rows.map((row) => Math.max(1, labels.indexOf(row.value) + 1)),
        },
      ];
      return base;
    }
    const rows = rawRows || [];
    base.xAxis.data = rows.map((row) => row.time);
    base.series = [{ name: title, type: "line", showSymbol: false, data: rows.map((row) => numeric(row.value)) }];
    return base;
  }

  function connect() {
    if (!root || !("WebSocket" in window)) return;
    const runId = root.dataset.runId;
    if (!runId || root.id === "simulation-results") return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws/simulations/${runId}`);
    socket.onmessage = (event) => {
      try {
        patch(JSON.parse(event.data));
      } catch (_error) {
        text("sim-status", "스냅샷 해석 오류");
      }
    };
    socket.onclose = () => {
      window.clearTimeout(reconnectTimer);
      reconnectTimer = window.setTimeout(connect, 2000);
    };
    socket.onerror = () => socket.close();
  }

  function collectTimes(rows) {
    return Array.from(new Set((rows || []).map((row) => row.time).filter(Boolean))).sort();
  }

  function valueAt(rows, time, key) {
    const row = (rows || []).find((item) => item.time === time);
    return row ? numeric(row[key]) : null;
  }

  function numeric(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function koreanTitle(key) {
    return (
      {
        volatility_score: "변동성 점수",
        expectation_score: "기대값 변화 점수",
        trend_score: "추세 점수",
        seasonality_score: "계절성 점수",
        confidence: "신뢰도",
      }[key] || key
    );
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function initialize() {
    patch(initialSnapshot);
    connect();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
