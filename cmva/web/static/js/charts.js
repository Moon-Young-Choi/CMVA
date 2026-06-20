(function () {
  const chartInstances = new WeakMap();

  function renderCmvaCharts(snapshot) {
    const series = (snapshot && snapshot.series) || {};
    if (window.echarts) {
      upgradeCanvasCharts();
      document.querySelectorAll(".echart.line-chart").forEach((element) => {
        const name = element.dataset.series;
        drawEChart(element, series[name] || [], name || "series");
      });
    } else {
      document.querySelectorAll("canvas.line-chart").forEach((canvas) => {
        const name = canvas.dataset.series;
        drawLine(canvas, series[name] || [], name || "series");
      });
    }
  }

  function upgradeCanvasCharts() {
    document.querySelectorAll("canvas.line-chart").forEach((canvas) => {
      const div = document.createElement("div");
      div.className = canvas.className.replace("line-chart", "line-chart echart");
      div.dataset.series = canvas.dataset.series || "";
      div.dataset.metric = canvas.dataset.metric || "";
      div.setAttribute("role", "img");
      div.setAttribute("aria-label", canvas.getAttribute("aria-label") || `${div.dataset.series} chart`);
      canvas.replaceWith(div);
    });
  }

  function drawEChart(element, rows, label) {
    const values = rows
      .map((row) => ({
        time: row.time,
        value: Number(row.value),
        label: row.label,
      }))
      .filter((row) => Number.isFinite(row.value));
    let chart = chartInstances.get(element);
    if (!chart) {
      chart = window.echarts.init(element, null, { renderer: "canvas" });
      chartInstances.set(element, chart);
      window.addEventListener("resize", () => chart.resize());
    }
    const title = label.replaceAll("_", " ");
    chart.setOption(
      {
        animation: false,
        backgroundColor: "#eef2f6",
        title: {
          text: title,
          left: 10,
          top: 8,
          textStyle: {
            color: "#526071",
            fontSize: 12,
            fontWeight: 500,
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          },
        },
        grid: { left: 42, right: 16, top: 42, bottom: 30 },
        tooltip: {
          trigger: "axis",
          confine: true,
          valueFormatter: (value) => Number(value).toExponential(3),
        },
        xAxis: {
          type: "category",
          data: values.map((row) => row.time),
          axisLabel: { color: "#667085", hideOverlap: true },
          axisLine: { lineStyle: { color: "#c4ccd6" } },
          axisTick: { show: false },
        },
        yAxis: {
          type: "value",
          scale: true,
          axisLabel: { color: "#667085" },
          splitLine: { lineStyle: { color: "#d8dee6" } },
        },
        series: [
          {
            name: title,
            type: "line",
            data: values.map((row) => row.value),
            showSymbol: false,
            smooth: false,
            lineStyle: { color: "#006d77", width: 2 },
            areaStyle: { color: "rgba(0, 109, 119, 0.08)" },
          },
        ],
      },
      true
    );
  }

  function drawLine(canvas, rows, label) {
    const ratio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * ratio));
    canvas.height = Math.max(1, Math.floor(rect.height * ratio));
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#eef2f6";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#526071";
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText(label.replaceAll("_", " "), 12, 20);
    const values = rows.map((row) => Number(row.value)).filter((value) => Number.isFinite(value));
    if (values.length < 2) {
      ctx.fillText("waiting for data", 12, 44);
      return;
    }
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(max - min, 1e-12);
    const pad = 28;
    ctx.strokeStyle = "#c4ccd6";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, height - pad);
    ctx.lineTo(width - 10, height - pad);
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, height - pad);
    ctx.stroke();
    ctx.strokeStyle = "#006d77";
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((value, idx) => {
      const x = pad + (idx * (width - pad - 18)) / (values.length - 1);
      const y = height - pad - ((value - min) / span) * (height - pad * 2);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    const latestRow = rows
      .slice()
      .reverse()
      .find((row) => Number.isFinite(Number(row.value)));
    const latestText = latestRow && latestRow.label ? latestRow.label : values[values.length - 1].toExponential(3);
    ctx.fillStyle = "#667085";
    ctx.fillText(`latest ${latestText}`, 12, height - 8);
  }

  window.renderCmvaCharts = renderCmvaCharts;
})();
