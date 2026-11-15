/**
 * Dual-Layer Canvas Viewport Engine
 */

class CanvasEngine {
  constructor(bottomCanvas, topCanvas, onRenderTop) {
    this.bottomCanvas = bottomCanvas;
    this.topCanvas = topCanvas;
    this.ctxBottom = bottomCanvas.getContext('2d');
    this.ctxTop = topCanvas.getContext('2d');
    this.onRenderTop = onRenderTop;
    this.onZoomChange = null;

    // 视口变换参数
    this.scale = 1.0;
    this.panX = 0;
    this.panY = 0;

    // 底图对象
    this.bgImage = null;
    this.imageSize = [800, 600];

    // 图层滤镜参数
    this.imageOpacity = 0.85;
    this.molOpacity = 1.0;
    this.invertImage = false;
    this.showGrid = false;
    this.showRefBrackets = false;
    this.showMolLayer = true;  // 是否显示 MolScribe 预标注的分子图结构

    this.dpr = window.devicePixelRatio || 1;
    this.initResizeObserver();
    this.initWheelZoom();
  }

  initResizeObserver() {
    const resize = () => {
      const rect = this.topCanvas.parentElement.getBoundingClientRect();
      const w = Math.floor(rect.width);
      const h = Math.floor(rect.height);

      this.bottomCanvas.width = w * this.dpr;
      this.bottomCanvas.height = h * this.dpr;
      this.topCanvas.width = w * this.dpr;
      this.topCanvas.height = h * this.dpr;

      this.bottomCanvas.style.width = `${w}px`;
      this.bottomCanvas.style.height = `${h}px`;
      this.topCanvas.style.width = `${w}px`;
      this.topCanvas.style.height = `${h}px`;

      this.ctxBottom.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      this.ctxTop.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

      this.render();
    };

    window.addEventListener('resize', resize);
    setTimeout(resize, 50);
  }

  initWheelZoom() {
    this.topCanvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = this.topCanvas.getBoundingClientRect();
      const screenX = e.clientX - rect.left;
      const screenY = e.clientY - rect.top;
      
      // 向上滚动放大，向下滚动缩小
      const factor = e.deltaY < 0 ? 1.15 : (1 / 1.15);
      this.zoomAt(screenX, screenY, factor);

      if (typeof this.onZoomChange === 'function') {
        this.onZoomChange(this.scale);
      }
    }, { passive: false });
  }

  setImage(imageElement, size) {
    this.bgImage = imageElement;
    if (size) this.imageSize = size;
    this.fitToScreen();
  }

  fitToScreen() {
    const rect = this.topCanvas.parentElement.getBoundingClientRect();
    const padding = 40;
    const availW = rect.width - padding * 2;
    const availH = rect.height - padding * 2;
    const [imgW, imgH] = this.imageSize;

    if (imgW <= 0 || imgH <= 0) return;

    const scaleX = availW / imgW;
    const scaleY = availH / imgH;
    this.scale = Math.min(scaleX, scaleY, 2.5); // 适中放大
    this.scale = Math.max(this.scale, 0.2);

    this.panX = (rect.width - imgW * this.scale) / 2;
    this.panY = (rect.height - imgH * this.scale) / 2;
    this.render();

    if (typeof this.onZoomChange === 'function') {
      this.onZoomChange(this.scale);
    }
  }

  resetZoom() {
    const rect = this.topCanvas.parentElement.getBoundingClientRect();
    const [imgW, imgH] = this.imageSize;
    this.scale = 1.0;
    this.panX = (rect.width - imgW) / 2;
    this.panY = (rect.height - imgH) / 2;
    this.render();

    if (typeof this.onZoomChange === 'function') {
      this.onZoomChange(this.scale);
    }
  }

  pan(dx, dy) {
    this.panX += dx;
    this.panY += dy;
    this.render();
  }

  zoomAt(screenX, screenY, factor) {
    const prevScale = this.scale;
    const newScale = Math.min(Math.max(prevScale * factor, 0.1), 12.0);
    if (prevScale === newScale) return;

    // 保持鼠标所在世界坐标不动
    const world = this.screenToWorld(screenX, screenY);
    this.scale = newScale;
    this.panX = screenX - world.x * newScale;
    this.panY = screenY - world.y * newScale;
    this.render();

    if (typeof this.onZoomChange === 'function') {
      this.onZoomChange(this.scale);
    }
  }

  screenToWorld(sx, sy) {
    return {
      x: (sx - this.panX) / this.scale,
      y: (sy - this.panY) / this.scale,
    };
  }

  worldToScreen(wx, wy) {
    return {
      x: wx * this.scale + this.panX,
      y: wy * this.scale + this.panY,
    };
  }

  render() {
    this.renderBottom();
    this.renderTop();
  }

  renderBottom() {
    const ctx = this.ctxBottom;
    const w = this.bottomCanvas.width / this.dpr;
    const h = this.bottomCanvas.height / this.dpr;

    ctx.clearRect(0, 0, w, h);

    const isDark = ChemUtils.isDarkTheme();

    // 绘制背景对齐网格
    if (this.showGrid) {
      this.drawGrid(ctx, w, h, isDark);
    }

    if (!this.bgImage) {
      ctx.fillStyle = isDark ? '#94a3b8' : '#64748b';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('正在加载原图...', w / 2, h / 2);
      return;
    }

    ctx.save();
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.scale, this.scale);

    ctx.globalAlpha = this.imageOpacity;
    if (this.invertImage) {
      ctx.filter = 'invert(1) hue-rotate(180deg)';
    } else {
      ctx.filter = 'none';
    }

    ctx.drawImage(this.bgImage, 0, 0, this.imageSize[0], this.imageSize[1]);
    
    // 绘制原图边界虚线
    ctx.strokeStyle = isDark ? '#475569' : '#94a3b8';
    ctx.lineWidth = 1 / this.scale;
    ctx.setLineDash([4 / this.scale, 4 / this.scale]);
    ctx.strokeRect(0, 0, this.imageSize[0], this.imageSize[1]);

    ctx.restore();
  }

  renderTop() {
    const ctx = this.ctxTop;
    const w = this.topCanvas.width / this.dpr;
    const h = this.topCanvas.height / this.dpr;

    ctx.clearRect(0, 0, w, h);

    // 关闭图结构显示时只保留底图 (原图)
    if (!this.showMolLayer) return;

    ctx.save();
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.scale, this.scale);
    ctx.globalAlpha = this.molOpacity;

    if (typeof this.onRenderTop === 'function') {
      this.onRenderTop(ctx, this.scale);
    }

    ctx.restore();
  }

  drawGrid(ctx, w, h, isDark) {
    ctx.save();
    ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';
    ctx.lineWidth = 1;
    const gridSize = 30 * this.scale;
    const startX = this.panX % gridSize;
    const startY = this.panY % gridSize;

    ctx.beginPath();
    for (let x = startX; x < w; x += gridSize) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = startY; y < h; y += gridSize) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();
    ctx.restore();
  }
}
