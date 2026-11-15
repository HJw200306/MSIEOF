/**
 * OBB (Oriented Bounding Box) Visual Annotator Engine
 * 高性能旋转矩形标注与交互编辑器
 */

class OBBEditor {
  constructor(options = {}) {
    this.bottomCanvas = options.bottomCanvas;
    this.topCanvas = options.topCanvas;
    this.ctxBottom = this.bottomCanvas.getContext('2d');
    this.ctxTop = this.topCanvas.getContext('2d');

    this.onSelectionChange = options.onSelectionChange || null;
    this.onBoxesChange = options.onBoxesChange || null;
    this.onStatusChange = options.onStatusChange || null;
    this.onModeChange = options.onModeChange || null;
    this.onCoordsChange = options.onCoordsChange || null;

    // 类别与色彩映射
    this.classNames = {
      0: "text",
      1: "l_bracket",
      2: "r_bracket",
      3: "bracket_sub",
      4: "wave_ap",
      5: "dash_ap"
    };

    this.classColors = {
      0: "#eab308", // text: gold / yellow
      1: "#00e5ff", // l_bracket: cyan
      2: "#2979ff", // r_bracket: blue
      3: "#ff9100", // bracket_sub: amber
      4: "#ec4899", // wave_ap: pink
      5: "#10b981"  // dash_ap: emerald
    };

    // 数据状态
    this.boxes = [];
    this.selectedBoxId = null;
    this.hoveredBoxId = null;
    this.activeHandle = null; // 'tl', 'tr', 'br', 'bl', 'rotate', 'body'
    
    // 工具模式: 'select' | 'draw'
    this.currentMode = 'select';
    this.currentClassId = 1; // 默认左括号

    // 视口变换参数 (同分子图视口)
    this.scale = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.image = null;
    this.imageSize = [800, 600];

    // 图层滤镜
    this.imageOpacity = 0.9;
    this.invertImage = false;
    this.showLabels = true;
    this.showOrientation = true;

    // 交互过程状态
    this.isPanning = false;
    this.panStartX = 0;
    this.panStartY = 0;
    this.dragStartScreen = [0, 0];
    this.dragStartWorld = [0, 0];
    this.dragInitialBox = null;
    this.drawStep = 0;
    this.drawStartPoint = null;
    this.drawEndPoint = null;

    // 撤销 / 重做历史记录
    this.history = [];
    this.historyIndex = -1;
    this.maxHistory = 50;

    this.dpr = window.devicePixelRatio || 1;
    this.initResizeObserver();
    this.initEvents();
  }

  handleResize() {
    const parent = this.topCanvas.parentElement;
    if (!parent) return false;
    const rect = parent.getBoundingClientRect();
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);
    if (w <= 0 || h <= 0) return false;

    const targetW = w * this.dpr;
    const targetH = h * this.dpr;

    if (this.bottomCanvas.width !== targetW || this.bottomCanvas.height !== targetH) {
      this.bottomCanvas.width = targetW;
      this.bottomCanvas.height = targetH;
      this.topCanvas.width = targetW;
      this.topCanvas.height = targetH;

      this.bottomCanvas.style.width = `${w}px`;
      this.bottomCanvas.style.height = `${h}px`;
      this.topCanvas.style.width = `${w}px`;
      this.topCanvas.style.height = `${h}px`;

      this.ctxBottom.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      this.ctxTop.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    }
    return true;
  }

  initResizeObserver() {
    const onResize = () => {
      if (this.handleResize()) {
        this.render();
      }
    };

    if (window.ResizeObserver && this.topCanvas.parentElement) {
      this.resizeObserver = new ResizeObserver(() => onResize());
      this.resizeObserver.observe(this.topCanvas.parentElement);
    }
    window.addEventListener('resize', onResize);
    setTimeout(onResize, 50);
  }

  initEvents() {
    this.topCanvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
    window.addEventListener('mousemove', (e) => this.onMouseMove(e));
    window.addEventListener('mouseup', (e) => this.onMouseUp(e));
    this.topCanvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
  }

  // =========================================================================
  // 视口与坐标变换
  // =========================================================================

  screenToWorld(sx, sy) {
    return [
      (sx - this.panX) / this.scale,
      (sy - this.panY) / this.scale
    ];
  }

  worldToScreen(wx, wy) {
    return [
      wx * this.scale + this.panX,
      wy * this.scale + this.panY
    ];
  }

  onWheel(e) {
    e.preventDefault();
    const rect = this.topCanvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.15 : (1 / 1.15);

    const newScale = Math.min(Math.max(this.scale * factor, 0.1), 15.0);
    this.panX = sx - (sx - this.panX) * (newScale / this.scale);
    this.panY = sy - (sy - this.panY) * (newScale / this.scale);
    this.scale = newScale;

    this.render();
    if (this.onStatusChange) this.onStatusChange();
  }

  fitToScreen() {
    this.handleResize();
    const parent = this.topCanvas.parentElement;
    if (!parent) return;
    const rect = parent.getBoundingClientRect();
    const pad = 40;
    const availW = Math.max(rect.width - pad * 2, 50);
    const availH = Math.max(rect.height - pad * 2, 50);
    const [imgW, imgH] = this.imageSize;
    if (imgW <= 0 || imgH <= 0) return;

    const scaleX = availW / imgW;
    const scaleY = availH / imgH;
    this.scale = Math.min(scaleX, scaleY, 2.5);
    this.scale = Math.max(this.scale, 0.1);

    this.panX = (rect.width - imgW * this.scale) / 2;
    this.panY = (rect.height - imgH * this.scale) / 2;
    this.render();
    if (this.onStatusChange) this.onStatusChange();
  }

  // =========================================================================
  // 数据载入与管理
  // =========================================================================

  loadData(imageElement, size, boxes = []) {
    this.image = imageElement;
    if (size && size.length === 2 && size[0] > 0 && size[1] > 0) {
      this.imageSize = size;
    } else if (imageElement) {
      this.imageSize = [imageElement.naturalWidth || imageElement.width, imageElement.naturalHeight || imageElement.height];
    }

    this.boxes = (boxes || []).map((b, idx) => {
      const id = b.id || `box_${Date.now()}_${idx}`;
      const cls_id = b.class_id !== undefined ? parseInt(b.class_id) : 1;
      const rawCorners = b.corners.map(pt => [parseFloat(pt[0]), parseFloat(pt[1])]);
      const canonicalCorners = this.canonicalizeCorners(rawCorners);
      return {
        id: id,
        class_id: cls_id,
        class_name: b.class_name || this.classNames[cls_id] || `cls_${cls_id}`,
        corners: canonicalCorners,
        conf: b.conf !== undefined ? parseFloat(b.conf) : 1.0,
      };
    });

    this.selectedBoxId = null;
    this.hoveredBoxId = null;
    this.activeHandle = null;
    this.history = [];
    this.historyIndex = -1;
    this.recordHistory();

    this.fitToScreen();
    if (this.onBoxesChange) this.onBoxesChange(this.boxes);
  }

  getBoxes() {
    return this.boxes;
  }

  setMode(mode) {
    this.currentMode = mode;
    this.activeHandle = null;
    this.drawStep = 0;
    this.drawStartPoint = null;
    this.topCanvas.style.cursor = mode === 'draw' ? 'crosshair' : 'default';
    this.renderTop();
  }

  setClass(cls_id) {
    this.currentClassId = parseInt(cls_id);
    if (this.selectedBoxId) {
      const box = this.boxes.find(b => b.id === this.selectedBoxId);
      if (box && box.class_id !== this.currentClassId) {
        box.class_id = this.currentClassId;
        box.class_name = this.classNames[this.currentClassId] || `cls_${this.currentClassId}`;
        this.recordHistory();
        this.renderTop();
        if (this.onBoxesChange) this.onBoxesChange(this.boxes);
      }
    }
  }

  deleteSelected() {
    if (!this.selectedBoxId) return;
    this.boxes = this.boxes.filter(b => b.id !== this.selectedBoxId);
    this.selectedBoxId = null;
    this.activeHandle = null;
    this.recordHistory();
    this.renderTop();
    if (this.onBoxesChange) this.onBoxesChange(this.boxes);
    if (this.onSelectionChange) this.onSelectionChange(null);
  }

  clearAll() {
    if (this.boxes.length === 0) return;
    this.boxes = [];
    this.selectedBoxId = null;
    this.activeHandle = null;
    this.recordHistory();
    this.renderTop();
    if (this.onBoxesChange) this.onBoxesChange(this.boxes);
    if (this.onSelectionChange) this.onSelectionChange(null);
  }

  selectBox(boxId) {
    this.selectedBoxId = boxId;
    this.renderTop();
    const box = this.boxes.find(b => b.id === boxId);
    if (this.onSelectionChange) this.onSelectionChange(box || null);
  }

  // =========================================================================
  // 历史记录 (Undo / Redo)
  // =========================================================================

  recordHistory() {
    const snapshot = JSON.stringify(this.boxes);
    if (this.historyIndex >= 0 && this.history[this.historyIndex] === snapshot) {
      return;
    }
    this.history = this.history.slice(0, this.historyIndex + 1);
    this.history.push(snapshot);
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    } else {
      this.historyIndex++;
    }
  }

  undo() {
    if (this.historyIndex > 0) {
      this.historyIndex--;
      this.boxes = JSON.parse(this.history[this.historyIndex]);
      if (!this.boxes.find(b => b.id === this.selectedBoxId)) {
        this.selectedBoxId = null;
      }
      this.renderTop();
      if (this.onBoxesChange) this.onBoxesChange(this.boxes);
      if (this.onSelectionChange) {
        const box = this.boxes.find(b => b.id === this.selectedBoxId);
        this.onSelectionChange(box || null);
      }
    }
  }

  redo() {
    if (this.historyIndex < this.history.length - 1) {
      this.historyIndex++;
      this.boxes = JSON.parse(this.history[this.historyIndex]);
      this.renderTop();
      if (this.onBoxesChange) this.onBoxesChange(this.boxes);
      if (this.onSelectionChange) {
        const box = this.boxes.find(b => b.id === this.selectedBoxId);
        this.onSelectionChange(box || null);
      }
    }
  }

  // =========================================================================
  // OBB 几何数学计算 (严格标准化坐标系)
  // =========================================================================

  /**
   * 将任意 4 个角点规范化为标准屏幕方向顺序 [TL, TR, BR, BL]
   * 保证:
   * 1. TL (Top-Left), TR (Top-Right), BR (Bottom-Right), BL (Bottom-Left)
   * 2. u 指向屏幕右方 (X 增加方向)
   * 3. v 指向屏幕下方 (Y 增加方向)
   * 4. 彻底杜绝上下/左右边或把手颠倒错位问题
   */
  canonicalizeCorners(corners) {
    if (!corners || corners.length < 4) return corners;
    const cx = (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4;
    const cy = (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4;

    const e0 = [corners[1][0] - corners[0][0], corners[1][1] - corners[0][1]];
    const e1 = [corners[2][0] - corners[1][0], corners[2][1] - corners[1][1]];
    const len0 = Math.hypot(e0[0], e0[1]);
    const len1 = Math.hypot(e1[0], e1[1]);
    if (len0 < 1e-4 || len1 < 1e-4) return corners;

    const u0 = [e0[0] / len0, e0[1] / len0];
    const u1 = [e1[0] / len1, e1[1] / len1];

    // 选择与水平方向更接近的轴作为主 X 轴 (u)
    let u = (Math.abs(u0[0]) >= Math.abs(u1[0])) ? u0 : u1;

    // 强制 u 指向右侧 (u.x > 0，若严格垂直则 u.y > 0)
    if (u[0] < 0 || (Math.abs(u[0]) < 1e-4 && u[1] < 0)) {
      u = [-u[0], -u[1]];
    }

    // 辅轴 v 垂直于 u，强制指向下方 (v.y > 0，若严格水平则 v.x < 0)
    const v0 = [-u[1], u[0]];
    let v;
    if (v0[1] > 0 || (Math.abs(v0[1]) < 1e-4 && v0[0] < 0)) {
      v = v0;
    } else {
      v = [u[1], -u[0]];
    }

    // 投影所有 4 个点到 (u, v) 坐标系获取绝对最大半长宽
    let maxLx = 0, maxLy = 0;
    for (let i = 0; i < 4; i++) {
      const rx = corners[i][0] - cx;
      const ry = corners[i][1] - cy;
      const lx = Math.abs(rx * u[0] + ry * u[1]);
      const ly = Math.abs(rx * v[0] + ry * v[1]);
      if (lx > maxLx) maxLx = lx;
      if (ly > maxLy) maxLy = ly;
    }

    // 严格按照 [TL, TR, BR, BL] 顺时针构建规范化角点
    return [
      [cx - maxLx * u[0] - maxLy * v[0], cy - maxLx * u[1] - maxLy * v[1]], // TL (左上)
      [cx + maxLx * u[0] - maxLy * v[0], cy + maxLx * u[1] - maxLy * v[1]], // TR (右上)
      [cx + maxLx * u[0] + maxLy * v[0], cy + maxLx * u[1] + maxLy * v[1]], // BR (右下)
      [cx - maxLx * u[0] + maxLy * v[0], cy - maxLx * u[1] + maxLy * v[1]]  // BL (左下)
    ];
  }

  /**
   * 将 4 个角点转换为规范化的几何度量指标
   */
  getBoxMetrics(corners) {
    const c = this.canonicalizeCorners(corners);
    const [tl, tr, br, bl] = c;
    const cx = (tl[0] + tr[0] + br[0] + bl[0]) / 4;
    const cy = (tl[1] + tr[1] + br[1] + bl[1]) / 4;

    const w = Math.hypot(tr[0] - tl[0], tr[1] - tl[1]);
    const h = Math.hypot(bl[0] - tl[0], bl[1] - tl[1]);
    const angle = Math.atan2(tr[1] - tl[1], tr[0] - tl[0]);

    const u = w > 1e-4 ? [(tr[0] - tl[0]) / w, (tr[1] - tl[1]) / w] : [1, 0];
    const v = h > 1e-4 ? [(bl[0] - tl[0]) / h, (bl[1] - tl[1]) / h] : [0, 1];

    return { center: [cx, cy], size: [w, h], angle, u, v, corners: c };
  }

  /**
   * 由 center, w, h, angle 生成 4 个规范角点
   */
  createCorners(cx, cy, w, h, angle) {
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    const halfW = w / 2;
    const halfH = h / 2;

    const localPts = [
      [-halfW, -halfH], // TL
      [halfW, -halfH],  // TR
      [halfW, halfH],   // BR
      [-halfW, halfH],  // BL
    ];

    return localPts.map(([lx, ly]) => [
      cx + lx * cosA - ly * sinA,
      cy + lx * sinA + ly * cosA
    ]);
  }

  pointInPolygon(px, py, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0], yi = poly[i][1];
      const xj = poly[j][0], yj = poly[j][1];
      const intersect = ((yi > py) !== (yj > py))
          && (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  getHandles(corners) {
    const metrics = this.getBoxMetrics(corners);
    const [tl, tr, br, bl] = metrics.corners;
    const { center, u, v } = metrics;

    const topMid = [(tl[0] + tr[0]) / 2, (tl[1] + tr[1]) / 2];
    const bottomMid = [(bl[0] + br[0]) / 2, (bl[1] + br[1]) / 2];
    const leftMid = [(tl[0] + bl[0]) / 2, (tl[1] + bl[1]) / 2];
    const rightMid = [(tr[0] + br[0]) / 2, (tr[1] + br[1]) / 2];

    // 旋转手柄: 位于 topMid 沿 -v 方向 (向上) 22px 处
    const rotateDist = 22 / this.scale;
    const rotateHandle = [
      topMid[0] - v[0] * rotateDist,
      topMid[1] - v[1] * rotateDist
    ];

    return {
      tl: tl,
      tr: tr,
      br: br,
      bl: bl,
      top: topMid,
      right: rightMid,
      bottom: bottomMid,
      left: leftMid,
      rotate: rotateHandle,
      topMid: topMid,
      center: center
    };
  }

  // =========================================================================
  // 鼠标交互 (精准屏幕像素容差与绝对锚点单边拉伸)
  // =========================================================================

  onMouseDown(e) {
    const rect = this.topCanvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const [wx, wy] = this.screenToWorld(sx, sy);

    // 1. 空格键漫游
    if (e.spaceKey || e.button === 1 || (e.button === 0 && e.altKey)) {
      this.isPanning = true;
      this.panStartX = sx - this.panX;
      this.panStartY = sy - this.panY;
      this.topCanvas.style.cursor = 'grabbing';
      return;
    }

    if (e.button !== 0) return;

    // 2. 绘制模式 (Draw)
    if (this.currentMode === 'draw') {
      this.drawStep = 1;
      this.drawStartPoint = [wx, wy];
      this.drawEndPoint = [wx, wy];
      return;
    }

    // 3. 选择/变换模式 (Select)
    // 优先在屏幕像素空间检测操作把手 (8.5px 容差，旋转 11px 容差)
    if (this.selectedBoxId) {
      const selBox = this.boxes.find(b => b.id === this.selectedBoxId);
      if (selBox) {
        const handles = this.getHandles(selBox.corners);

        // 旋转手柄
        const [rotSx, rotSy] = this.worldToScreen(handles.rotate[0], handles.rotate[1]);
        if (Math.hypot(sx - rotSx, sy - rotSy) <= 11) {
          this.activeHandle = 'rotate';
          this.dragStartScreen = [sx, sy];
          this.dragStartWorld = [wx, wy];
          this.dragInitialBox = JSON.parse(JSON.stringify(selBox));
          return;
        }

        // 4 角点与 4 边手柄
        for (const hKey of ['tl', 'tr', 'br', 'bl', 'top', 'right', 'bottom', 'left']) {
          const hp = handles[hKey];
          const [hSx, hSy] = this.worldToScreen(hp[0], hp[1]);
          if (Math.hypot(sx - hSx, sy - hSy) <= 8.5) {
            this.activeHandle = hKey;
            this.dragStartScreen = [sx, sy];
            this.dragStartWorld = [wx, wy];
            this.dragInitialBox = JSON.parse(JSON.stringify(selBox));
            return;
          }
        }
      }
    }

    // 检查点中哪个 OBB
    let hitBox = null;
    for (let i = this.boxes.length - 1; i >= 0; i--) {
      if (this.pointInPolygon(wx, wy, this.boxes[i].corners)) {
        hitBox = this.boxes[i];
        break;
      }
    }

    if (hitBox) {
      this.selectedBoxId = hitBox.id;
      this.activeHandle = 'body';
      this.dragStartScreen = [sx, sy];
      this.dragStartWorld = [wx, wy];
      this.dragInitialBox = JSON.parse(JSON.stringify(hitBox));
      this.renderTop();
      if (this.onSelectionChange) this.onSelectionChange(hitBox);
    } else {
      // 点击空白处取消选中
      if (this.selectedBoxId !== null) {
        this.selectedBoxId = null;
        this.activeHandle = null;
        this.renderTop();
        if (this.onSelectionChange) this.onSelectionChange(null);
      }
    }
  }

  onMouseMove(e) {
    const rect = this.topCanvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const [wx, wy] = this.screenToWorld(sx, sy);

    // 实时通知当前鼠标世界坐标
    if (this.onCoordsChange) {
      this.onCoordsChange(Math.round(wx), Math.round(wy));
    }

    // 漫游
    if (this.isPanning) {
      this.panX = sx - this.panStartX;
      this.panY = sy - this.panStartY;
      this.render();
      return;
    }

    // 绘制中
    if (this.currentMode === 'draw' && this.drawStep === 1) {
      this.drawEndPoint = [wx, wy];
      this.renderTop();
      return;
    }

    // 变换操作中
    if (this.activeHandle && this.selectedBoxId && this.dragInitialBox) {
      const box = this.boxes.find(b => b.id === this.selectedBoxId);
      if (!box) return;

      const initCorners = this.dragInitialBox.corners;
      const metrics = this.getBoxMetrics(initCorners);
      const [cx, cy] = metrics.center;
      const [w, h] = metrics.size;
      const u = metrics.u;
      const v = metrics.v;
      const halfW = w / 2;
      const halfH = h / 2;

      if (this.activeHandle === 'body') {
        // 平移
        const dx = wx - this.dragStartWorld[0];
        const dy = wy - this.dragStartWorld[1];
        box.corners = initCorners.map(([x, y]) => [x + dx, y + dy]);
      } else if (this.activeHandle === 'rotate') {
        // 旋转
        let newAngle = Math.atan2(wy - cy, wx - cx) + Math.PI / 2;
        if (e.shiftKey) {
          const step = Math.PI / 12; // 15 度吸附
          newAngle = Math.round(newAngle / step) * step;
        }
        const cosA = Math.cos(newAngle);
        const sinA = Math.sin(newAngle);
        box.corners = [
          [cx - halfW * cosA - halfH * (-sinA), cy - halfW * sinA - halfH * cosA], // TL
          [cx + halfW * cosA - halfH * (-sinA), cy + halfW * sinA - halfH * cosA], // TR
          [cx + halfW * cosA + halfH * (-sinA), cy + halfW * sinA + halfH * cosA], // BR
          [cx - halfW * cosA + halfH * (-sinA), cy - halfW * sinA + halfH * cosA]  // BL
        ];
      } else {
        // 单边 / 单角独立拉伸 (对边/对角绝对固定锚点 Anchor，绝不两头同时缩放，绝对方向锁定)
        const relX = wx - cx;
        const relY = wy - cy;
        // 投影到初始 OBB 局部规范坐标系 (u 严格指向右侧，v 严格指向下方)
        const lx = relX * u[0] + relY * u[1];
        const ly = relX * v[0] + relY * v[1];

        let newHalfW = halfW;
        let newHalfH = halfH;
        let newLocalCx = 0;
        let newLocalCy = 0;

        switch (this.activeHandle) {
          case 'right': {
            // 左边绝对固定 (lx = -halfW)
            const minRight = -halfW + 3;
            const clampedLx = Math.max(lx, minRight);
            newHalfW = (clampedLx - (-halfW)) / 2;
            newLocalCx = (-halfW + clampedLx) / 2;
            break;
          }
          case 'left': {
            // 右边绝对固定 (lx = +halfW)
            const maxLeft = halfW - 3;
            const clampedLx = Math.min(lx, maxLeft);
            newHalfW = (halfW - clampedLx) / 2;
            newLocalCx = (clampedLx + halfW) / 2;
            break;
          }
          case 'bottom': {
            // 顶边绝对固定 (ly = -halfH)
            const minBottom = -halfH + 3;
            const clampedLy = Math.max(ly, minBottom);
            newHalfH = (clampedLy - (-halfH)) / 2;
            newLocalCy = (-halfH + clampedLy) / 2;
            break;
          }
          case 'top': {
            // 底边绝对固定 (ly = +halfH)
            const maxTop = halfH - 3;
            const clampedLy = Math.min(ly, maxTop);
            newHalfH = (halfH - clampedLy) / 2;
            newLocalCy = (clampedLy + halfH) / 2;
            break;
          }
          case 'br': {
            // 左上角绝对固定 (-halfW, -halfH)
            const clampedLx = Math.max(lx, -halfW + 3);
            const clampedLy = Math.max(ly, -halfH + 3);
            newHalfW = (clampedLx - (-halfW)) / 2;
            newLocalCx = (-halfW + clampedLx) / 2;
            newHalfH = (clampedLy - (-halfH)) / 2;
            newLocalCy = (-halfH + clampedLy) / 2;
            break;
          }
          case 'bl': {
            // 右上角绝对固定 (+halfW, -halfH)
            const clampedLx = Math.min(lx, halfW - 3);
            const clampedLy = Math.max(ly, -halfH + 3);
            newHalfW = (halfW - clampedLx) / 2;
            newLocalCx = (clampedLx + halfW) / 2;
            newHalfH = (clampedLy - (-halfH)) / 2;
            newLocalCy = (-halfH + clampedLy) / 2;
            break;
          }
          case 'tr': {
            // 左下角绝对固定 (-halfW, +halfH)
            const clampedLx = Math.max(lx, -halfW + 3);
            const clampedLy = Math.min(ly, halfH - 3);
            newHalfW = (clampedLx - (-halfW)) / 2;
            newLocalCx = (-halfW + clampedLx) / 2;
            newHalfH = (halfH - clampedLy) / 2;
            newLocalCy = (clampedLy + halfH) / 2;
            break;
          }
          case 'tl': {
            // 右下角绝对固定 (+halfW, +halfH)
            const clampedLx = Math.min(lx, halfW - 3);
            const clampedLy = Math.min(ly, halfH - 3);
            newHalfW = (halfW - clampedLx) / 2;
            newLocalCx = (clampedLx + halfW) / 2;
            newHalfH = (halfH - clampedLy) / 2;
            newLocalCy = (clampedLy + halfH) / 2;
            break;
          }
        }

        const newCenterX = cx + newLocalCx * u[0] + newLocalCy * v[0];
        const newCenterY = cy + newLocalCx * u[1] + newLocalCy * v[1];

        box.corners = [
          [newCenterX - newHalfW * u[0] - newHalfH * v[0], newCenterY - newHalfW * u[1] - newHalfH * v[1]], // TL
          [newCenterX + newHalfW * u[0] - newHalfH * v[0], newCenterY + newHalfW * u[1] - newHalfH * v[1]], // TR
          [newCenterX + newHalfW * u[0] + newHalfH * v[0], newCenterY + newHalfW * u[1] + newHalfH * v[1]], // BR
          [newCenterX - newHalfW * u[0] + newHalfH * v[0], newCenterY - newHalfW * u[1] + newHalfH * v[1]]  // BL
        ];
      }

      this.renderTop();
      return;
    }

    // 更新鼠标悬停光标状态 (与屏幕像素容差完全一致)
    if (this.currentMode === 'select') {
      let hovered = null;
      let cursor = 'default';

      if (this.selectedBoxId) {
        const selBox = this.boxes.find(b => b.id === this.selectedBoxId);
        if (selBox) {
          const handles = this.getHandles(selBox.corners);
          const [rotSx, rotSy] = this.worldToScreen(handles.rotate[0], handles.rotate[1]);

          if (Math.hypot(sx - rotSx, sy - rotSy) <= 11) {
            cursor = 'grab';
          } else {
            for (const k of ['tl', 'br']) {
              const [hx, hy] = this.worldToScreen(handles[k][0], handles[k][1]);
              if (Math.hypot(sx - hx, sy - hy) <= 8.5) { cursor = 'nwse-resize'; break; }
            }
            if (cursor === 'default') {
              for (const k of ['tr', 'bl']) {
                const [hx, hy] = this.worldToScreen(handles[k][0], handles[k][1]);
                if (Math.hypot(sx - hx, sy - hy) <= 8.5) { cursor = 'nesw-resize'; break; }
              }
            }
            if (cursor === 'default') {
              for (const k of ['top', 'bottom']) {
                const [hx, hy] = this.worldToScreen(handles[k][0], handles[k][1]);
                if (Math.hypot(sx - hx, sy - hy) <= 8.5) { cursor = 'ns-resize'; break; }
              }
            }
            if (cursor === 'default') {
              for (const k of ['left', 'right']) {
                const [hx, hy] = this.worldToScreen(handles[k][0], handles[k][1]);
                if (Math.hypot(sx - hx, sy - hy) <= 8.5) { cursor = 'ew-resize'; break; }
              }
            }
          }
        }
      }

      if (cursor === 'default') {
        for (let i = this.boxes.length - 1; i >= 0; i--) {
          if (this.pointInPolygon(wx, wy, this.boxes[i].corners)) {
            hovered = this.boxes[i].id;
            cursor = 'move';
            break;
          }
        }
      }

      this.topCanvas.style.cursor = cursor;
      if (hovered !== this.hoveredBoxId) {
        this.hoveredBoxId = hovered;
        this.renderTop();
      }
    } else if (this.currentMode === 'draw') {
      this.topCanvas.style.cursor = 'crosshair';
    }
  }

  onMouseUp(e) {
    if (this.isPanning) {
      this.isPanning = false;
      this.topCanvas.style.cursor = this.currentMode === 'draw' ? 'crosshair' : 'default';
      return;
    }

    // 完成绘制
    if (this.currentMode === 'draw' && this.drawStep === 1) {
      this.drawStep = 0;
      if (this.drawStartPoint && this.drawEndPoint) {
        const [x0, y0] = this.drawStartPoint;
        const [x1, y1] = this.drawEndPoint;
        const dist = Math.hypot(x1 - x0, y1 - y0);

        if (dist > 6) {
          const newCorners = [
            [Math.min(x0, x1), Math.min(y0, y1)],
            [Math.max(x0, x1), Math.min(y0, y1)],
            [Math.max(x0, x1), Math.max(y0, y1)],
            [Math.min(x0, x1), Math.max(y0, y1)],
          ];

          const newBox = {
            id: `box_${Date.now()}`,
            class_id: this.currentClassId,
            class_name: this.classNames[this.currentClassId] || `cls_${this.currentClassId}`,
            corners: this.canonicalizeCorners(newCorners),
            conf: 1.0,
          };

          this.boxes.push(newBox);
          this.selectedBoxId = newBox.id;
          this.recordHistory();
          this.currentMode = 'select';
          this.activeHandle = null;
          this.topCanvas.style.cursor = 'default';
          if (this.onModeChange) this.onModeChange('select');
          this.renderTop();
          if (this.onBoxesChange) this.onBoxesChange(this.boxes);
          if (this.onSelectionChange) this.onSelectionChange(newBox);
        }
      }
      this.drawStartPoint = null;
      this.drawEndPoint = null;
      return;
    }

    // 结束变换并规范化角点
    if (this.activeHandle) {
      const box = this.boxes.find(b => b.id === this.selectedBoxId);
      if (box) {
        box.corners = this.canonicalizeCorners(box.corners);
      }
      this.activeHandle = null;
      this.dragInitialBox = null;
      this.recordHistory();
      this.renderTop();
      if (this.onBoxesChange) this.onBoxesChange(this.boxes);
    }
  }

  // =========================================================================
  // 渲染层
  // =========================================================================

  render() {
    this.renderBottom();
    this.renderTop();
  }

  renderBottom() {
    const ctx = this.ctxBottom;
    const w = this.bottomCanvas.width / this.dpr;
    const h = this.bottomCanvas.height / this.dpr;
    ctx.clearRect(0, 0, w, h);

    if (!this.image) {
      ctx.fillStyle = "rgba(15, 23, 42, 0.05)";
      ctx.fillRect(0, 0, w, h);
      return;
    }

    ctx.save();
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.scale, this.scale);

    // 滤镜 (反相 / 透明度)
    let filterStr = "";
    if (this.invertImage) filterStr += "invert(1) ";
    ctx.filter = filterStr.trim() || "none";
    ctx.globalAlpha = Math.max(Math.min(this.imageOpacity, 1.0), 0.05);

    const [imgW, imgH] = this.imageSize;
    ctx.drawImage(this.image, 0, 0, imgW, imgH);
    ctx.restore();
  }

  renderTop() {
    const ctx = this.ctxTop;
    const w = this.topCanvas.width / this.dpr;
    const h = this.topCanvas.height / this.dpr;
    ctx.clearRect(0, 0, w, h);

    ctx.save();
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.scale, this.scale);

    // 1. 绘制已有 OBB 框
    for (const box of this.boxes) {
      const isSelected = box.id === this.selectedBoxId;
      const isHovered = box.id === this.hoveredBoxId;
      this.renderOBB(ctx, box, isSelected, isHovered);
    }

    // 2. 绘制正在拉框的临时矩形
    if (this.currentMode === 'draw' && this.drawStep === 1 && this.drawStartPoint && this.drawEndPoint) {
      const [x0, y0] = this.drawStartPoint;
      const [x1, y1] = this.drawEndPoint;
      const minX = Math.min(x0, x1);
      const minY = Math.min(y0, y1);
      const bw = Math.abs(x1 - x0);
      const bh = Math.abs(y1 - y0);

      const color = this.classColors[this.currentClassId] || "#2563eb";
      ctx.fillStyle = this.hexToRgba(color, 0.25);
      ctx.fillRect(minX, minY, bw, bh);

      ctx.strokeStyle = color;
      ctx.lineWidth = 2 / this.scale;
      ctx.setLineDash([4 / this.scale, 4 / this.scale]);
      ctx.strokeRect(minX, minY, bw, bh);
      ctx.setLineDash([]);
    }

    ctx.restore();
  }

  renderOBB(ctx, box, isSelected, isHovered) {
    const corners = box.corners;
    if (!corners || corners.length < 4) return;

    const color = this.classColors[box.class_id] || "#2563eb";
    const fillAlpha = isSelected ? 0.35 : (isHovered ? 0.25 : 0.15);
    const strokeWidth = (isSelected ? 2.0 : (isHovered ? 1.6 : 1.2)) / this.scale;

    // 1. 填充多边形
    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    for (let i = 1; i < 4; i++) {
      ctx.lineTo(corners[i][0], corners[i][1]);
    }
    ctx.closePath();

    ctx.fillStyle = this.hexToRgba(color, fillAlpha);
    ctx.fill();

    // 2. 边框
    ctx.strokeStyle = color;
    ctx.lineWidth = strokeWidth;
    ctx.lineJoin = "round";
    ctx.stroke();

    // 3. 顶边方向指示器 (Direction Pip) - 屏幕绝对 6px 长度
    if (this.showOrientation) {
      const metrics = this.getBoxMetrics(corners);
      const topMid = [(corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2];
      const pipLen = 6 / this.scale;

      ctx.beginPath();
      ctx.moveTo(topMid[0], topMid[1]);
      ctx.lineTo(topMid[0] - metrics.v[0] * pipLen, topMid[1] - metrics.v[1] * pipLen);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6 / this.scale;
      ctx.stroke();
    }

    // 4. 标签徽章 (Class + Conf) - 切入屏幕空间绘制，保证字体与色块 1:1 绝对贴合，杜绝缩放字号失效与色块变细线问题
    if (this.showLabels) {
      ctx.save();
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); // 恢复至原生屏幕像素坐标系

      const text = `${box.class_name}${box.conf < 1.0 ? ` ${(box.conf * 100).toFixed(0)}%` : ''}`;
      ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
      const textWidth = ctx.measureText(text).width;
      const padX = 5;
      const rw = Math.round(textWidth + padX * 2);
      const rh = 16;

      // 找到外接盒最高点在屏幕坐标系中的位置
      const minY = Math.min(corners[0][1], corners[1][1], corners[2][1], corners[3][1]);
      const minX = Math.min(corners[0][0], corners[1][0], corners[2][0], corners[3][0]);
      const maxX = Math.max(corners[0][0], corners[1][0], corners[2][0], corners[3][0]);
      const topCenterX = (minX + maxX) / 2;
      const [sx, sy] = this.worldToScreen(topCenterX, minY);

      // 若处于选中状态，避开上方 22px 的旋转把手
      const extraOffsetScreen = isSelected ? 34 : 6;
      const rx = Math.round(sx - rw / 2);
      const ry = Math.round(sy - rh - extraOffsetScreen);

      ctx.fillStyle = this.hexToRgba(color, 0.92);
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(rx, ry, rw, rh, 3);
      } else {
        ctx.rect(rx, ry, rw, rh);
      }
      ctx.fill();

      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, rx + rw / 2, ry + rh / 2 + 0.5);
      ctx.restore();
    }

    // 5. 选中状态时的操作把手 Handles (严格恒定屏幕像素，绝不随放大变大遮挡线条，半透明可透视)
    if (isSelected) {
      const handles = this.getHandles(corners);

      // 连接旋转手柄的细虚线
      ctx.beginPath();
      ctx.moveTo(handles.topMid[0], handles.topMid[1]);
      ctx.lineTo(handles.rotate[0], handles.rotate[1]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.0 / this.scale;
      ctx.setLineDash([2 / this.scale, 2 / this.scale]);
      ctx.stroke();
      ctx.setLineDash([]);

      // 旋转手柄 (屏幕绝对 3.6px 半径，白色填充 + 彩色边框)
      const rotateR = 3.6 / this.scale;
      ctx.beginPath();
      ctx.arc(handles.rotate[0], handles.rotate[1], rotateR, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.2 / this.scale;
      ctx.stroke();

      // 4 角点手柄 (屏幕绝对 3.0px 半径，半透明白色填充 + 彩色细边，完全可透视看清内部黑线条)
      const cornerR = 3.0 / this.scale;
      for (const hKey of ['tl', 'tr', 'br', 'bl']) {
        const hp = handles[hKey];
        ctx.beginPath();
        ctx.arc(hp[0], hp[1], cornerR, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255, 255, 255, 0.45)"; // 半透明透视，不遮挡底层黑线！
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.0 / this.scale;
        ctx.stroke();
      }

      // 4 边中点手柄 (屏幕绝对 2.0px 实心细点，不抢占视觉焦点)
      const edgeR = 2.0 / this.scale;
      for (const eKey of ['top', 'right', 'bottom', 'left']) {
        const ep = handles[eKey];
        ctx.beginPath();
        ctx.arc(ep[0], ep[1], edgeR, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 0.6 / this.scale;
        ctx.stroke();
      }
    }
  }

  hexToRgba(hex, alpha) {
    hex = hex.replace("#", "");
    if (hex.length === 3) {
      hex = hex.split("").map(c => c + c).join("");
    }
    const r = parseInt(hex.substring(0, 2), 16) || 0;
    const g = parseInt(hex.substring(2, 4), 16) || 0;
    const b = parseInt(hex.substring(4, 6), 16) || 0;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
}

// 暴露至全局
window.OBBEditor = OBBEditor;
