(function () {
  function renderCmvaCharts(snapshot) {
    const series = (snapshot && snapshot.series) || {};
    document.querySelectorAll("canvas.line-chart").forEach((canvas) => {
      const name = canvas.dataset.series;
      drawLine(canvas, series[name] || [], name || "series");
    });
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
