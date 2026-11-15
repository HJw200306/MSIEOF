/**
 * Interactive Molecular Graph Editor & Modular Bracket Controller
 */

// CanvasRenderingContext2D.roundRect 兼容层 (旧浏览器 / 部分 WebView 不支持)
if (typeof CanvasRenderingContext2D !== 'undefined' && !CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
    if (typeof r === 'number') r = [r, r, r, r];
    const [tl, tr, br, bl] = r.map(v => Math.min(v, Math.min(Math.abs(w), Math.abs(h)) / 2));
    this.moveTo(x + tl, y);
    this.lineTo(x + w - tr, y);
    this.arcTo(x + w, y, x + w, y + tr, tr);
    this.lineTo(x + w, y + h - br);
    this.arcTo(x + w, y + h, x + w - br, y + h, br);
    this.lineTo(x + bl, y + h);
    this.arcTo(x, y + h, x, y + h - bl, bl);
    this.lineTo(x, y + tl);
    this.arcTo(x, y, x + tl, y, tl);
    this.closePath();
    return this;
  };
}

const GROUP_COLORS = [
  '#8b5cf6', // 1: 紫罗兰
  '#3b82f6', // 2: 科技蓝
  '#10b981', // 3: 翡翠绿
  '#f59e0b', // 4: 琥珀黄
  '#ec4899', // 5: 魅惑粉
  '#06b6d4', // 6: 青空蓝
  '#84cc16', // 7: 柠檬绿
  '#f97316', // 8: 活力橙
];

function getGroupColor(gid) {
  const g = parseInt(gid) || 1;
  return GROUP_COLORS[(g - 1) % GROUP_COLORS.length] || '#8b5cf6';
}

function hexToRgba(hex, alpha) {
  let c = hex.replace('#', '');
  if (c.length === 3) c = c.split('').map(x => x + x).join('');
  const num = parseInt(c, 16);
  return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`;
}

class MolEditor {
  constructor(canvasEngine, onSelectionChange, onDataChange) {
    this.engine = canvasEngine;
    this.onSelectionChange = onSelectionChange;
    this.onDataChange = onDataChange;

    // 分子结构与模块化括号数据
    this.atoms = [];
    this.bonds = [];
    this.sgroups = [];
    this.brackets = []; // [{id, group_id, type: 'left'|'right'|'subscript', polygon: [[x,y]...], text}]
    this.showRefBrackets = true; // 参考括号显示/隐藏开关
    this.showAtomLabels = true; // 原子元素文字标签显示/隐藏开关

    // 当前工具状态
    this.activeTool = 'select'; // 'select' | 'atom' | 'bond' | 'bracket-left' | 'bracket-right' | 'bracket-sub' | 'delete'
    this.activeSymbol = 'C';
    this.activeBondType = 1;

    // 选中状态
    this.selectedAtoms = new Set();
    this.selectedBonds = new Set();
    this.selectedBracket = null;
    this.selectedBracketVertex = null; // 调整多边形某个顶点索引

    // 悬停交互状态
    this.hoverAtomIdx = null;
    this.hoverBondIdx = null;
    this.hoverBracketIdx = null;
    this.hoverHandle = null;
    this.hoverBracketHandle = null;

    // 交互拖拽状态
    this.dragMode = null; // 'pan' | 'box_select' | 'atom_drag' | 'bond_draw' | 'bracket_draw' | 'bracket_drag' | 'bracket_rotate' | 'bracket_scale' | 'bracket_vertex_drag' | 'transform_translate' | 'transform_scale' | 'transform_rotate'
    this.dragStartScreen = { x: 0, y: 0 };
    this.dragStartWorld = { x: 0, y: 0 };
    this.currentWorld = { x: 0, y: 0 };
    this.bondStartAtom = null;
    this.singleDragAtom = null;
    this.singleDragBracket = null; // { idx, initialPolygon, cx, cy, initialBBox }
    this.transformInitialAtoms = [];
    this.transformBBox = null;
    this.activeHandle = null;

    // 撤销 / 重做栈
    this.history = [];
    this.historyIndex = -1;
    this.maxHistory = 40;

    this.spacePressed = false;

    this.initEventListeners();
  }

  setData(data) {
    this.atoms = JSON.parse(JSON.stringify(data.atoms || []));
    this.bonds = JSON.parse(JSON.stringify(data.bonds || []));
    this.sgroups = JSON.parse(JSON.stringify(data.sgroups || []));
    this.brackets = JSON.parse(JSON.stringify(data.brackets || []));

    // 规范化超原子与元素符号颗粒度 (超原子一律 symbol='*', label='...')
    for (const a of this.atoms) {
      ChemUtils.normalizeAtom(a);
    }

    // 规范化化学键结构: 确保为 [u, v, bond_type]
    for (let i = 0; i < this.bonds.length; i++) {
      const b = this.bonds[i];
      if (Array.isArray(b)) {
        if (b.length < 3) b[2] = 1;
        else b[2] = parseInt(b[2]) || 1;
      }
    }

    // 清空选区并重置历史栈
    this.selectedAtoms.clear();
    this.selectedBonds.clear();
    this.selectedBracket = null;
    this.selectedBracketVertex = null;
    this.hoverAtomIdx = null;
    this.hoverBondIdx = null;
    this.hoverBracketIdx = null;
    this.hoverHandle = null;
    this.hoverBracketHandle = null;

    this.history = [];
    this.historyIndex = -1;
    this.pushHistory();

    this.notifySelection();
    this.updateCursor();
    this.engine.render();
  }

  getData() {
    return {
      atoms: JSON.parse(JSON.stringify(this.atoms)),
      bonds: JSON.parse(JSON.stringify(this.bonds)),
      sgroups: JSON.parse(JSON.stringify(this.sgroups)),
      brackets: JSON.parse(JSON.stringify(this.brackets)),
    };
  }

  pushHistory() {
    if (this.historyIndex < this.history.length - 1) {
      this.history = this.history.slice(0, this.historyIndex + 1);
    }
    const snapshot = JSON.stringify({
      atoms: this.atoms,
      bonds: this.bonds,
      sgroups: this.sgroups,
      brackets: this.brackets,
    });
    this.history.push(snapshot);
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    } else {
      this.historyIndex++;
    }
    if (this.onDataChange) this.onDataChange();
  }

  undo() {
    if (this.historyIndex > 0) {
      this.historyIndex--;
      const snapshot = JSON.parse(this.history[this.historyIndex]);
      this.atoms = JSON.parse(JSON.stringify(snapshot.atoms));
      this.bonds = JSON.parse(JSON.stringify(snapshot.bonds));
      this.sgroups = JSON.parse(JSON.stringify(snapshot.sgroups || []));
      this.brackets = JSON.parse(JSON.stringify(snapshot.brackets || []));
      this.selectedAtoms.clear();
      this.selectedBonds.clear();
      this.selectedBracket = null;
      this.selectedBracketVertex = null;
      this.notifySelection();
      this.engine.render();
      if (this.onDataChange) this.onDataChange();
    }
  }

  redo() {
    if (this.historyIndex < this.history.length - 1) {
      this.historyIndex++;
      const snapshot = JSON.parse(this.history[this.historyIndex]);
      this.atoms = JSON.parse(JSON.stringify(snapshot.atoms));
      this.bonds = JSON.parse(JSON.stringify(snapshot.bonds));
      this.sgroups = JSON.parse(JSON.stringify(snapshot.sgroups || []));
      this.brackets = JSON.parse(JSON.stringify(snapshot.brackets || []));
      this.selectedAtoms.clear();
      this.selectedBonds.clear();
      this.selectedBracket = null;
      this.selectedBracketVertex = null;
      this.notifySelection();
      this.engine.render();
      if (this.onDataChange) this.onDataChange();
    }
  }

  notifySelection() {
    if (this.onSelectionChange) {
      this.onSelectionChange({
        selectedAtoms: Array.from(this.selectedAtoms),
        selectedBonds: Array.from(this.selectedBonds),
        selectedSgroup: null,
        selectedBracket: this.selectedBracket,
        atoms: this.atoms,
        bonds: this.bonds,
        sgroups: this.sgroups,
        brackets: this.brackets,
      });
    }
  }

  // =========================================================================
  // 事件监听与光标系统
  // =========================================================================

  initEventListeners() {
    const canvas = this.engine.topCanvas;

    canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
    window.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    window.addEventListener('mouseup', (e) => this.handleMouseUp(e));
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space' && !this.spacePressed && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        this.spacePressed = true;
        this.updateCursor();
      }
    });

    window.addEventListener('keyup', (e) => {
      if (e.code === 'Space') {
        this.spacePressed = false;
        this.updateCursor();
      }
    });
  }

  updateCursor() {
    const canvas = this.engine.topCanvas;
    if (this.spacePressed || this.dragMode === 'pan') {
      canvas.style.cursor = 'grab';
      return;
    }

    if (this.dragMode === 'atom_drag' || this.dragMode === 'bracket_drag' || this.dragMode === 'transform_translate') {
      canvas.style.cursor = 'grabbing';
      return;
    }

    if (this.dragMode === 'bracket_rotate' || this.dragMode === 'transform_rotate') {
      canvas.style.cursor = 'grab';
      return;
    }

    if (this.dragMode === 'bracket_scale' || this.dragMode === 'transform_scale') {
      return;
    }

    const handle = this.hoverHandle || this.hoverBracketHandle;
    if (handle) {
      if (handle === 'nw' || handle === 'se') canvas.style.cursor = 'nwse-resize';
      else if (handle === 'ne' || handle === 'sw') canvas.style.cursor = 'nesw-resize';
      else if (handle === 'n' || handle === 's') canvas.style.cursor = 'ns-resize';
      else if (handle === 'e' || handle === 'w') canvas.style.cursor = 'ew-resize';
      else if (handle === 'rot') canvas.style.cursor = 'grab';
      else if (handle.startsWith('vertex_')) canvas.style.cursor = 'crosshair';
      else if (handle === 'body') canvas.style.cursor = 'move';
      return;
    }

    if (this.hoverAtomIdx !== null) {
      canvas.style.cursor = 'grab';
      return;
    }

    if (this.hoverBondIdx !== null || this.hoverBracketIdx !== null) {
      canvas.style.cursor = 'pointer';
      return;
    }

    if (this.activeTool === 'atom') canvas.style.cursor = 'crosshair';
    else if (this.activeTool === 'bond') canvas.style.cursor = 'crosshair';
    else if (this.activeTool.startsWith('bracket')) canvas.style.cursor = 'crosshair';
    else if (this.activeTool === 'delete') canvas.style.cursor = 'not-allowed';
    else canvas.style.cursor = 'default';
  }

  // =========================================================================
  // 鼠标交互流转
  // =========================================================================

  handleMouseDown(e) {
    if (e.button === 1 || this.spacePressed) {
      this.dragMode = 'pan';
      this.dragStartScreen = { x: e.clientX, y: e.clientY };
      return;
    }

    if (e.button !== 0) return;

    // 隐藏图结构时仅允许平移/缩放，避免误改不可见的标注
    if (this.engine && this.engine.showMolLayer === false) return;

    const screenPos = this.getCanvasRelativePos(e);
    const worldPos = this.engine.screenToWorld(screenPos.x, screenPos.y);
    this.dragStartScreen = { x: e.clientX, y: e.clientY };
    this.dragStartWorld = { ...worldPos };
    this.currentWorld = { ...worldPos };

    // 1. 检查多选原子变换把手
    if (this.transformBBox && this.selectedAtoms.size > 1) {
      const handle = this.hitTestTransformHandles(worldPos.x, worldPos.y);
      if (handle) {
        if (handle === 'rot') {
          this.dragMode = 'transform_rotate';
        } else if (handle === 'body') {
          this.dragMode = 'transform_translate';
        } else {
          this.dragMode = 'transform_scale';
        }
        this.activeHandle = handle;
        this.transformInitialAtoms = Array.from(this.selectedAtoms).map(idx => ({
          idx,
          x: this.atoms[idx].x,
          y: this.atoms[idx].y,
        }));
        this.updateCursor();
        return;
      }
    }

    // 2. 检查单选括号旋转/缩放/顶点微调把手
    if (this.selectedBracket !== null && this.brackets[this.selectedBracket]) {
      const bHandle = this.hitTestBracketHandles(this.selectedBracket, worldPos.x, worldPos.y);
      if (bHandle) {
        const b = this.brackets[this.selectedBracket];
        const bbox = this.getPolygonBBox(b.polygon);
        this.singleDragBracket = {
          idx: this.selectedBracket,
          initialPolygon: JSON.parse(JSON.stringify(b.polygon)),
          cx: (bbox.minX + bbox.maxX) / 2,
          cy: (bbox.minY + bbox.maxY) / 2,
          initialBBox: bbox,
        };

        if (bHandle === 'rot') {
          this.dragMode = 'bracket_rotate';
        } else if (bHandle.startsWith('vertex_')) {
          this.selectedBracketVertex = parseInt(bHandle.split('_')[1]);
          this.dragMode = 'bracket_vertex_drag';
        } else if (bHandle === 'body') {
          this.dragMode = 'bracket_drag';
        } else {
          this.activeHandle = bHandle;
          this.dragMode = 'bracket_scale';
        }
        this.updateCursor();
        return;
      }
    }

    // 3. 检查原子命中
    const hitAtom = this.findAtomAt(worldPos.x, worldPos.y);

    // 4. 检查化学键命中
    const hitBond = hitAtom === null ? this.findBondAt(worldPos.x, worldPos.y) : null;

    // 5. 检查模块化括号命中
    const hitBracket = (hitAtom === null && hitBond === null) ? this.findBracketAt(worldPos.x, worldPos.y) : null;

    // 根据工具分发动作
    if (this.activeTool === 'select') {
      if (hitAtom !== null) {
        if (e.shiftKey) {
          if (this.selectedAtoms.has(hitAtom)) this.selectedAtoms.delete(hitAtom);
          else this.selectedAtoms.add(hitAtom);
          this.selectedBonds.clear();
          this.selectedBracket = null;
        } else {
          if (!this.selectedAtoms.has(hitAtom)) {
            this.selectedAtoms.clear();
            this.selectedAtoms.add(hitAtom);
            this.selectedBonds.clear();
            this.selectedBracket = null;
          }
        }
        this.singleDragAtom = hitAtom;
        this.dragMode = 'atom_drag';
        this.transformInitialAtoms = Array.from(this.selectedAtoms).map(idx => ({
          idx,
          x: this.atoms[idx].x,
          y: this.atoms[idx].y,
        }));
        this.notifySelection();
        this.engine.render();
      } else if (hitBond !== null) {
        this.selectedAtoms.clear();
        this.selectedBonds.clear();
        this.selectedBonds.add(hitBond);
        this.selectedBracket = null;
        this.notifySelection();
        this.engine.render();
      } else if (hitBracket !== null) {
        this.selectedAtoms.clear();
        this.selectedBonds.clear();
        this.selectedBracket = hitBracket;
        const b = this.brackets[hitBracket];
        const bbox = this.getPolygonBBox(b.polygon);
        this.singleDragBracket = {
          idx: hitBracket,
          initialPolygon: JSON.parse(JSON.stringify(b.polygon || [])),
          cx: (bbox.minX + bbox.maxX) / 2,
          cy: (bbox.minY + bbox.maxY) / 2,
          initialBBox: bbox,
        };
        this.dragMode = 'bracket_drag';
        this.notifySelection();
        this.engine.render();
      } else {
        if (!e.shiftKey) {
          this.selectedAtoms.clear();
          this.selectedBonds.clear();
          this.selectedBracket = null;
          this.notifySelection();
        }
        this.dragMode = 'box_select';
      }
    } else if (this.activeTool === 'atom') {
      if (hitAtom !== null) {
        // 选中该原子以便在右侧检查栏展示和修改
        this.selectedAtoms.clear();
        this.selectedAtoms.add(hitAtom);
        this.selectedBonds.clear();
        this.selectedBracket = null;

        const sym = this.activeSymbol;
        this.atoms[hitAtom].symbol = sym;
        this.atoms[hitAtom].label = null;
        ChemUtils.normalizeAtom(this.atoms[hitAtom]);

        this.pushHistory();
        this.notifySelection();
        this.engine.render();
      } else {
        const sym = this.activeSymbol;
        const newAtom = {
          x: Math.round(worldPos.x * 10) / 10,
          y: Math.round(worldPos.y * 10) / 10,
          symbol: sym,
          label: null,
        };
        ChemUtils.normalizeAtom(newAtom);
        this.atoms.push(newAtom);
        const newIdx = this.atoms.length - 1;
        this.selectedAtoms.clear();
        this.selectedAtoms.add(newIdx);
        this.selectedBonds.clear();
        this.selectedBracket = null;
        this.pushHistory();
        this.notifySelection();
        this.engine.render();
      }
    } else if (this.activeTool === 'bond') {
      if (hitAtom !== null) {
        this.bondStartAtom = hitAtom;
        this.dragMode = 'bond_draw';
      }
    } else if (this.activeTool.startsWith('bracket')) {
      this.dragMode = 'bracket_draw';
    } else if (this.activeTool === 'delete') {
      if (hitAtom !== null) this.deleteAtom(hitAtom);
      else if (hitBond !== null) this.deleteBond(hitBond);
      else if (hitBracket !== null) this.deleteBracket(hitBracket);
    }

    this.updateCursor();
  }

  handleMouseMove(e) {
    const screenPos = this.getCanvasRelativePos(e);
    const worldPos = this.engine.screenToWorld(screenPos.x, screenPos.y);
    this.currentWorld = { ...worldPos };

    // 悬停检测更新
    if (!this.dragMode) {
      const prevAtom = this.hoverAtomIdx;
      const prevBond = this.hoverBondIdx;
      const prevBracket = this.hoverBracketIdx;
      const prevHandle = this.hoverHandle;
      const prevBHandle = this.hoverBracketHandle;

      if (this.transformBBox && this.selectedAtoms.size > 1) {
        this.hoverHandle = this.hitTestTransformHandles(worldPos.x, worldPos.y);
      } else {
        this.hoverHandle = null;
      }

      if (this.selectedBracket !== null && this.brackets[this.selectedBracket]) {
        this.hoverBracketHandle = this.hitTestBracketHandles(this.selectedBracket, worldPos.x, worldPos.y);
      } else {
        this.hoverBracketHandle = null;
      }

      this.hoverAtomIdx = (this.hoverHandle || this.hoverBracketHandle) ? null : this.findAtomAt(worldPos.x, worldPos.y);
      this.hoverBondIdx = (this.hoverHandle || this.hoverBracketHandle || this.hoverAtomIdx !== null) ? null : this.findBondAt(worldPos.x, worldPos.y);
      this.hoverBracketIdx = (this.hoverHandle || this.hoverBracketHandle || this.hoverAtomIdx !== null || this.hoverBondIdx !== null) ? null : this.findBracketAt(worldPos.x, worldPos.y);

      if (prevAtom !== this.hoverAtomIdx || prevBond !== this.hoverBondIdx || prevBracket !== this.hoverBracketIdx || prevHandle !== this.hoverHandle || prevBHandle !== this.hoverBracketHandle) {
        this.updateCursor();
        this.engine.render();
      }
      return;
    }

    // 拖拽逻辑处理
    if (this.dragMode === 'pan') {
      const dx = e.clientX - this.dragStartScreen.x;
      const dy = e.clientY - this.dragStartScreen.y;
      this.engine.pan(dx, dy);
      this.dragStartScreen = { x: e.clientX, y: e.clientY };
      return;
    }

    if (this.dragMode === 'atom_drag') {
      const dx = worldPos.x - this.dragStartWorld.x;
      const dy = worldPos.y - this.dragStartWorld.y;
      for (const item of this.transformInitialAtoms) {
        this.atoms[item.idx].x = Math.round((item.x + dx) * 10) / 10;
        this.atoms[item.idx].y = Math.round((item.y + dy) * 10) / 10;
      }
      this.engine.render();
      return;
    }

    if (this.dragMode === 'bracket_drag' && this.singleDragBracket) {
      const dx = worldPos.x - this.dragStartWorld.x;
      const dy = worldPos.y - this.dragStartWorld.y;
      const b = this.brackets[this.singleDragBracket.idx];
      if (b && this.singleDragBracket.initialPolygon) {
        b.polygon = this.singleDragBracket.initialPolygon.map(pt => [
          Math.round((pt[0] + dx) * 10) / 10,
          Math.round((pt[1] + dy) * 10) / 10,
        ]);
      }
      this.engine.render();
      return;
    }

    if (this.dragMode === 'bracket_rotate' && this.singleDragBracket) {
      const cx = this.singleDragBracket.cx;
      const cy = this.singleDragBracket.cy;
      const initialAngle = Math.atan2(this.dragStartWorld.y - cy, this.dragStartWorld.x - cx);
      const currentAngle = Math.atan2(worldPos.y - cy, worldPos.x - cx);
      const angleDelta = currentAngle - initialAngle;

      const cos = Math.cos(angleDelta);
      const sin = Math.sin(angleDelta);
      const b = this.brackets[this.singleDragBracket.idx];

      if (b && this.singleDragBracket.initialPolygon) {
        b.polygon = this.singleDragBracket.initialPolygon.map(pt => {
          const rx = pt[0] - cx;
          const ry = pt[1] - cy;
          return [
            Math.round((cx + rx * cos - ry * sin) * 10) / 10,
            Math.round((cy + rx * sin + ry * cos) * 10) / 10,
          ];
        });
      }
      this.engine.render();
      return;
    }

    if (this.dragMode === 'bracket_scale' && this.singleDragBracket) {
      const cx = this.singleDragBracket.cx;
      const cy = this.singleDragBracket.cy;
      const bbox = this.singleDragBracket.initialBBox;
      const initW = Math.max(10, bbox.maxX - bbox.minX);
      const initH = Math.max(10, bbox.maxY - bbox.minY);

      let scaleX = 1.0;
      let scaleY = 1.0;
      const currDx = Math.abs(worldPos.x - cx);
      const currDy = Math.abs(worldPos.y - cy);
      const h = this.activeHandle || 'se';

      if (h.includes('e') || h.includes('w')) {
        scaleX = Math.max(0.2, currDx / (initW / 2));
      }
      if (h.includes('n') || h.includes('s')) {
        scaleY = Math.max(0.2, currDy / (initH / 2));
      }
      if (['nw', 'ne', 'sw', 'se'].includes(h) && !e.shiftKey) {
        const uni = (scaleX + scaleY) / 2;
        scaleX = uni;
        scaleY = uni;
      }

      const b = this.brackets[this.singleDragBracket.idx];
      if (b && this.singleDragBracket.initialPolygon) {
        b.polygon = this.singleDragBracket.initialPolygon.map(pt => [
          Math.round((cx + (pt[0] - cx) * scaleX) * 10) / 10,
          Math.round((cy + (pt[1] - cy) * scaleY) * 10) / 10,
        ]);
      }
      this.engine.render();
      return;
    }

    if (this.dragMode === 'bracket_vertex_drag' && this.singleDragBracket && this.selectedBracketVertex !== null) {
      const b = this.brackets[this.singleDragBracket.idx];
      if (b && b.polygon && b.polygon[this.selectedBracketVertex]) {
        b.polygon[this.selectedBracketVertex] = [
          Math.round(worldPos.x * 10) / 10,
          Math.round(worldPos.y * 10) / 10,
        ];
      }
      this.engine.render();
      return;
    }

    if (this.dragMode === 'transform_translate') {
      const dx = worldPos.x - this.dragStartWorld.x;
      const dy = worldPos.y - this.dragStartWorld.y;
      for (const item of this.transformInitialAtoms) {
        this.atoms[item.idx].x = Math.round((item.x + dx) * 10) / 10;
        this.atoms[item.idx].y = Math.round((item.y + dy) * 10) / 10;
      }
      this.engine.render();
      return;
    }

    if (this.dragMode === 'transform_scale' && this.transformBBox) {
      const bbox = this.transformBBox;
      const cx = bbox.cx;
      const cy = bbox.cy;
      const initW = Math.max(1, bbox.maxX - bbox.minX);
      const initH = Math.max(1, bbox.maxY - bbox.minY);

      let scaleX = 1.0;
      let scaleY = 1.0;
      const currDx = Math.abs(worldPos.x - cx);
      const currDy = Math.abs(worldPos.y - cy);
      const h = this.activeHandle;

      if (h.includes('e') || h.includes('w')) {
        scaleX = currDx / (initW / 2);
      }
      if (h.includes('n') || h.includes('s')) {
        scaleY = currDy / (initH / 2);
      }
      if (['nw', 'ne', 'sw', 'se'].includes(h) && !e.shiftKey) {
        const uni = (scaleX + scaleY) / 2;
        scaleX = uni;
        scaleY = uni;
      }

      for (const item of this.transformInitialAtoms) {
        this.atoms[item.idx].x = Math.round((cx + (item.x - cx) * scaleX) * 10) / 10;
        this.atoms[item.idx].y = Math.round((cy + (item.y - cy) * scaleY) * 10) / 10;
      }
      this.engine.render();
      return;
    }

    if (this.dragMode === 'transform_rotate' && this.transformBBox) {
      const cx = this.transformBBox.cx;
      const cy = this.transformBBox.cy;
      const initialAngle = Math.atan2(this.dragStartWorld.y - cy, this.dragStartWorld.x - cx);
      const currentAngle = Math.atan2(worldPos.y - cy, worldPos.x - cx);
      const angleDelta = currentAngle - initialAngle;

      const cos = Math.cos(angleDelta);
      const sin = Math.sin(angleDelta);

      for (const item of this.transformInitialAtoms) {
        const rx = item.x - cx;
        const ry = item.y - cy;
        this.atoms[item.idx].x = Math.round((cx + rx * cos - ry * sin) * 10) / 10;
        this.atoms[item.idx].y = Math.round((cy + rx * sin + ry * cos) * 10) / 10;
      }
      this.engine.render();
      return;
    }

    // 实时重绘
    if (this.dragMode === 'box_select' || this.dragMode === 'bond_draw' || this.dragMode === 'bracket_draw') {
      this.engine.render();
    }
  }

  handleMouseUp(e) {
    if (!this.dragMode) return;

    const screenPos = this.getCanvasRelativePos(e);
    const worldPos = this.engine.screenToWorld(screenPos.x, screenPos.y);

    if (this.dragMode === 'box_select') {
      const minX = Math.min(this.dragStartWorld.x, worldPos.x);
      const maxX = Math.max(this.dragStartWorld.x, worldPos.x);
      const minY = Math.min(this.dragStartWorld.y, worldPos.y);
      const maxY = Math.max(this.dragStartWorld.y, worldPos.y);

      for (let i = 0; i < this.atoms.length; i++) {
        const a = this.atoms[i];
        if (a.x >= minX && a.x <= maxX && a.y >= minY && a.y <= maxY) {
          this.selectedAtoms.add(i);
        }
      }
      this.notifySelection();
    } else if (this.dragMode === 'bond_draw') {
      const endAtom = this.findAtomAt(worldPos.x, worldPos.y);
      if (endAtom !== null && endAtom !== this.bondStartAtom) {
        this.addBond(this.bondStartAtom, endAtom, this.activeBondType);
      } else if (endAtom === null) {
        const newIdx = this.atoms.length;
        const newAtom = {
          x: Math.round(worldPos.x * 10) / 10,
          y: Math.round(worldPos.y * 10) / 10,
          symbol: this.activeSymbol,
          label: null,
        };
        ChemUtils.normalizeAtom(newAtom);
        this.atoms.push(newAtom);
        this.addBond(this.bondStartAtom, newIdx, this.activeBondType);
      }
      this.bondStartAtom = null;
    } else if (this.dragMode === 'bracket_draw') {
      const minX = Math.min(this.dragStartWorld.x, worldPos.x);
      const maxX = Math.max(this.dragStartWorld.x, worldPos.x);
      const minY = Math.min(this.dragStartWorld.y, worldPos.y);
      const maxY = Math.max(this.dragStartWorld.y, worldPos.y);

      if (maxX - minX > 5 && maxY - minY > 5) {
        let bType = 'left';
        if (this.activeTool === 'bracket-right') bType = 'right';
        else if (this.activeTool === 'bracket-sub') bType = 'subscript';

        // 计算当前 group_id
        let nextGid = 1;
        if (this.brackets.length > 0) {
          const maxGid = Math.max(...this.brackets.map(b => b.group_id || 1));
          if (bType === 'left' && this.selectedBracket === null) {
            nextGid = maxGid + (this.brackets.some(b => b.group_id === maxGid && b.type === 'left') ? 1 : 0);
          } else {
            nextGid = this.selectedBracket !== null ? (this.brackets[this.selectedBracket].group_id || maxGid) : maxGid;
          }
        }

        const newBracket = {
          id: this.brackets.length,
          group_id: nextGid,
          type: bType,
          polygon: [
            [Math.round(minX * 10) / 10, Math.round(minY * 10) / 10],
            [Math.round(maxX * 10) / 10, Math.round(minY * 10) / 10],
            [Math.round(maxX * 10) / 10, Math.round(maxY * 10) / 10],
            [Math.round(minX * 10) / 10, Math.round(maxY * 10) / 10],
          ],
          text: bType === 'subscript' ? 'n' : null,
        };

        this.brackets.push(newBracket);
        this.selectedBracket = this.brackets.length - 1;
        this.pushHistory();
        this.notifySelection();
      }
    } else if (['atom_drag', 'bracket_drag', 'bracket_rotate', 'bracket_scale', 'bracket_vertex_drag', 'transform_translate', 'transform_scale', 'transform_rotate'].includes(this.dragMode)) {
      this.pushHistory();
    }

    this.dragMode = null;
    this.activeHandle = null;
    this.singleDragAtom = null;
    this.singleDragBracket = null;
    this.selectedBracketVertex = null;
    this.updateCursor();
    this.engine.render();
  }

  getCanvasRelativePos(e) {
    const rect = this.engine.topCanvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }

  // =========================================================================
  // 查找与命中测试 (Hit-Testing)
  // =========================================================================

  findAtomAt(worldX, worldY, hitRadius = 56) {
    const r = hitRadius / this.engine.scale;
    const rSq = r * r;
    let closestIdx = null;
    let minDistSq = Infinity;

    for (let i = 0; i < this.atoms.length; i++) {
      const a = this.atoms[i];
      const dSq = (a.x - worldX) ** 2 + (a.y - worldY) ** 2;
      if (dSq <= rSq && dSq < minDistSq) {
        minDistSq = dSq;
        closestIdx = i;
      }
    }
    return closestIdx;
  }

  findBondAt(worldX, worldY, hitDist = 8) {
    const distLimit = hitDist / this.engine.scale;
    let closestIdx = null;
    let minDist = Infinity;

    for (let i = 0; i < this.bonds.length; i++) {
      const b = this.bonds[i];
      const a1 = this.atoms[b[0]];
      const a2 = this.atoms[b[1]];
      if (!a1 || !a2) continue;

      const d = this.distPointToSegment(worldX, worldY, a1.x, a1.y, a2.x, a2.y);
      if (d <= distLimit && d < minDist) {
        minDist = d;
        closestIdx = i;
      }
    }
    return closestIdx;
  }

  findBracketAt(worldX, worldY) {
    for (let i = this.brackets.length - 1; i >= 0; i--) {
      const b = this.brackets[i];
      if (!b.polygon || b.polygon.length < 3) continue;
      if (this.isPointInPolygon(worldX, worldY, b.polygon)) {
        return i;
      }
      const bbox = this.getPolygonBBox(b.polygon);
      const pad = 6 / this.engine.scale;
      if (worldX >= bbox.minX - pad && worldX <= bbox.maxX + pad &&
        worldY >= bbox.minY - pad && worldY <= bbox.maxY + pad) {
        return i;
      }
    }
    return null;
  }

  hitTestBracketHandles(bracketIdx, worldX, worldY) {
    const b = this.brackets[bracketIdx];
    if (!b || !b.polygon || b.polygon.length < 3) return null;

    const scale = this.engine.scale;
    const handleRadius = 8 / scale;
    const rSq = handleRadius * handleRadius;
    const bbox = this.getPolygonBBox(b.polygon);
    const cx = (bbox.minX + bbox.maxX) / 2;
    const cy = (bbox.minY + bbox.maxY) / 2;

    // 1. 旋转把手
    const rotY = bbox.minY - 22 / scale;
    if ((cx - worldX) ** 2 + (rotY - worldY) ** 2 <= rSq) {
      return 'rot';
    }

    // 2. 多边形各顶点把手
    for (let k = 0; k < b.polygon.length; k++) {
      const pt = b.polygon[k];
      if ((pt[0] - worldX) ** 2 + (pt[1] - worldY) ** 2 <= rSq) {
        return `vertex_${k}`;
      }
    }

    // 3. 8向外框缩放把手
    const handles = {
      nw: { x: bbox.minX, y: bbox.minY },
      ne: { x: bbox.maxX, y: bbox.minY },
      se: { x: bbox.maxX, y: bbox.maxY },
      sw: { x: bbox.minX, y: bbox.maxY },
      n: { x: cx, y: bbox.minY },
      s: { x: cx, y: bbox.maxY },
      w: { x: bbox.minX, y: cy },
      e: { x: bbox.maxX, y: cy },
    };

    for (const [key, pt] of Object.entries(handles)) {
      if ((pt.x - worldX) ** 2 + (pt.y - worldY) ** 2 <= rSq) {
        return key;
      }
    }

    if (this.isPointInPolygon(worldX, worldY, b.polygon)) {
      return 'body';
    }

    return null;
  }

  distPointToSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return Math.hypot(px - x1, py - y1);
    let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
  }

  isPointInPolygon(px, py, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i][0], yi = polygon[i][1];
      const xj = polygon[j][0], yj = polygon[j][1];
      const intersect = ((yi > py) !== (yj > py)) &&
        (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  getPolygonBBox(poly) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const pt of poly) {
      if (pt[0] < minX) minX = pt[0];
      if (pt[0] > maxX) maxX = pt[0];
      if (pt[1] < minY) minY = pt[1];
      if (pt[1] > maxY) maxY = pt[1];
    }
    return { minX, minY, maxX, maxY };
  }

  // =========================================================================
  // 变换包围盒与把手
  // =========================================================================

  hitTestTransformHandles(worldX, worldY) {
    if (!this.transformBBox) return null;
    const bbox = this.transformBBox;
    const handleRadius = 8 / this.engine.scale;
    const rSq = handleRadius * handleRadius;

    const handles = {
      rot: { x: bbox.cx, y: bbox.minY - 24 / this.engine.scale },
      nw: { x: bbox.minX, y: bbox.minY },
      ne: { x: bbox.maxX, y: bbox.minY },
      se: { x: bbox.maxX, y: bbox.maxY },
      sw: { x: bbox.minX, y: bbox.maxY },
      n: { x: bbox.cx, y: bbox.minY },
      s: { x: bbox.cx, y: bbox.maxY },
      w: { x: bbox.minX, y: bbox.cy },
      e: { x: bbox.maxX, y: bbox.cy },
    };

    for (const [key, pt] of Object.entries(handles)) {
      if ((pt.x - worldX) ** 2 + (pt.y - worldY) ** 2 <= rSq) {
        return key;
      }
    }

    if (worldX >= bbox.minX && worldX <= bbox.maxX && worldY >= bbox.minY && worldY <= bbox.maxY) {
      return 'body';
    }

    return null;
  }

  computeSelectionBBox() {
    if (this.selectedAtoms.size <= 1) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const idx of this.selectedAtoms) {
      const a = this.atoms[idx];
      if (!a) continue;
      if (a.x < minX) minX = a.x;
      if (a.x > maxX) maxX = a.x;
      if (a.y < minY) minY = a.y;
      if (a.y > maxY) maxY = a.y;
    }
    const pad = 14 / this.engine.scale;
    minX -= pad;
    minY -= pad;
    maxX += pad;
    maxY += pad;
    return {
      minX, minY, maxX, maxY,
      cx: (minX + maxX) / 2,
      cy: (minY + maxY) / 2,
      width: maxX - minX,
      height: maxY - minY,
    };
  }

  // =========================================================================
  // 增删改操作 API
  // =========================================================================

  addBond(u, v, type = 1) {
    if (u === v || !this.atoms[u] || !this.atoms[v]) return;
    const existing = this.bonds.findIndex(b => (b[0] === u && b[1] === v) || (b[0] === v && b[1] === u));
    if (existing !== -1) {
      this.bonds[existing][2] = type;
    } else {
      this.bonds.push([u, v, type]);
    }
    this.pushHistory();
    this.engine.render();
  }

  deleteAtom(idx) {
    this.bonds = this.bonds.filter(b => b[0] !== idx && b[1] !== idx);
    this.bonds = this.bonds.map(b => [
      b[0] > idx ? b[0] - 1 : b[0],
      b[1] > idx ? b[1] - 1 : b[1],
      b[2]
    ]);

    this.atoms.splice(idx, 1);
    this.selectedAtoms.clear();
    this.pushHistory();
    this.notifySelection();
    this.engine.render();
  }

  deleteBond(idx) {
    this.bonds.splice(idx, 1);
    this.selectedBonds.clear();
    this.pushHistory();
    this.notifySelection();
    this.engine.render();
  }

  deleteBracket(idx) {
    this.brackets.splice(idx, 1);
    this.selectedBracket = null;
    this.pushHistory();
    this.notifySelection();
    this.engine.render();
  }

  deleteSelected() {
    if (this.selectedBracket !== null) {
      this.deleteBracket(this.selectedBracket);
      return;
    }
    if (this.selectedBonds.size > 0) {
      const sorted = Array.from(this.selectedBonds).sort((a, b) => b - a);
      for (const idx of sorted) {
        this.bonds.splice(idx, 1);
      }
      this.selectedBonds.clear();
      this.pushHistory();
      this.notifySelection();
      this.engine.render();
      return;
    }
    if (this.selectedAtoms.size > 0) {
      const sorted = Array.from(this.selectedAtoms).sort((a, b) => b - a);
      for (const idx of sorted) {
        this.deleteAtom(idx);
      }
      return;
    }
  }

  // =========================================================================
  // Canvas 渲染主循环 (单一清晰图层体系)
  // =========================================================================

  render(ctx, scale) {
    try {
      // 1. 绘制模块化括号 (Modular Brackets - 精确多边形/旋转框)
      if (this.showRefBrackets !== false) {
        this.drawBrackets(ctx, scale);
      }

      // 2. 绘制化学键
      this.drawBonds(ctx, scale);

      // 3. 绘制原子
      this.drawAtoms(ctx, scale);

      // 4. 绘制正在连线的化学键
      if (this.dragMode === 'bond_draw' && this.bondStartAtom !== null) {
        const a = this.atoms[this.bondStartAtom];
        if (a) {
          ctx.save();
          ctx.strokeStyle = '#3b82f6';
          ctx.lineWidth = 2.0 / scale;
          ctx.setLineDash([4 / scale, 4 / scale]);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(this.currentWorld.x, this.currentWorld.y);
          ctx.stroke();
          ctx.restore();
        }
      }

      // 5. 绘制多选原子变换框
      this.transformBBox = this.computeSelectionBBox();
      if (this.transformBBox) {
        this.drawTransformBBox(ctx, scale, this.transformBBox);
      }

      // 6. 绘制框选矩形
      if (this.dragMode === 'box_select') {
        ctx.save();
        const minX = Math.min(this.dragStartWorld.x, this.currentWorld.x);
        const maxX = Math.max(this.dragStartWorld.x, this.currentWorld.x);
        const minY = Math.min(this.dragStartWorld.y, this.currentWorld.y);
        const maxY = Math.max(this.dragStartWorld.y, this.currentWorld.y);

        ctx.fillStyle = 'rgba(59, 130, 246, 0.12)';
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 1.5 / scale;
        ctx.setLineDash([4 / scale, 4 / scale]);
        ctx.fillRect(minX, minY, maxX - minX, maxY - minY);
        ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
        ctx.restore();
      }

      // 7. 绘制正在拉框的模块化括号
      if (this.dragMode === 'bracket_draw') {
        ctx.save();
        const minX = Math.min(this.dragStartWorld.x, this.currentWorld.x);
        const maxX = Math.max(this.dragStartWorld.x, this.currentWorld.x);
        const minY = Math.min(this.dragStartWorld.y, this.currentWorld.y);
        const maxY = Math.max(this.dragStartWorld.y, this.currentWorld.y);

        const color = this.activeTool === 'bracket-sub' ? '#f59e0b' : '#8b5cf6';
        ctx.fillStyle = this.activeTool === 'bracket-sub' ? 'rgba(245, 158, 11, 0.08)' : 'rgba(139, 92, 246, 0.15)';
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.0 / scale;
        ctx.fillRect(minX, minY, maxX - minX, maxY - minY);
        ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
        ctx.restore();
      }
    } catch (err) {
      console.error('[MolEditor Render Error]', err);
    }
  }

  // =========================================================================
  // 模块化括号渲染器 (Modular Brackets - 非侵入式透明下标与把手全支持)
  // =========================================================================

  drawBrackets(ctx, scale) {
    if (!this.brackets || this.brackets.length === 0) return;

    ctx.save();

    // 1. 绘制同组左右括号辅助连接线
    const groupMap = {};
    for (const b of this.brackets) {
      const gid = b.group_id || 1;
      if (!groupMap[gid]) groupMap[gid] = {};
      groupMap[gid][b.type] = b;
    }

    for (const [gid, pair] of Object.entries(groupMap)) {
      if (pair.left && pair.right && pair.left.polygon && pair.right.polygon) {
        const bboxL = this.getPolygonBBox(pair.left.polygon);
        const bboxR = this.getPolygonBBox(pair.right.polygon);
        const color = getGroupColor(gid);

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.0 / scale;
        ctx.setLineDash([4 / scale, 4 / scale]);
        ctx.globalAlpha = 0.4;
        ctx.beginPath();
        ctx.moveTo((bboxL.minX + bboxL.maxX) / 2, (bboxL.minY + bboxL.maxY) / 2);
        ctx.lineTo((bboxR.minX + bboxR.maxX) / 2, (bboxR.minY + bboxR.maxY) / 2);
        ctx.stroke();
        ctx.globalAlpha = 1.0;
        ctx.setLineDash([]);
      }
    }

    // 2. 绘制每个括号组件的多边形 / 旋转框 (Exact OBB Polygon)
    for (let i = 0; i < this.brackets.length; i++) {
      const b = this.brackets[i];
      if (!b.polygon || b.polygon.length < 3) continue;

      const isSel = (this.selectedBracket === i);
      const isHover = (this.hoverBracketIdx === i);
      const color = getGroupColor(b.group_id);
      const isSub = (b.type === 'subscript');

      ctx.save();

      // a. 绘制真实多边形区域 (对于下标组件，内部完全透明，保证原图字符 100% 清晰显示)
      ctx.beginPath();
      ctx.moveTo(b.polygon[0][0], b.polygon[0][1]);
      for (let k = 1; k < b.polygon.length; k++) {
        ctx.lineTo(b.polygon[k][0], b.polygon[k][1]);
      }
      ctx.closePath();

      if (isSub) {
        if (isSel || isHover) {
          ctx.fillStyle = hexToRgba('#d97706', isSel ? 0.12 : 0.05);
          ctx.fill();
        }
        ctx.strokeStyle = isSel ? '#2563eb' : (isHover ? '#f59e0b' : '#d97706');
        ctx.lineWidth = (isSel ? 2.0 : 1.2) / scale;
        if (!isSel && !isHover) {
          ctx.setLineDash([3 / scale, 2 / scale]);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      } else {
        const fillAlpha = isSel ? 0.22 : (isHover ? 0.16 : 0.08);
        ctx.fillStyle = hexToRgba(color, fillAlpha);
        ctx.fill();

        ctx.strokeStyle = isSel ? '#2563eb' : color;
        ctx.lineWidth = (isSel ? 2.5 : 1.8) / scale;
        ctx.stroke();
      }

      // 计算中心与特征点
      const bbox = this.getPolygonBBox(b.polygon);
      const cx = (bbox.minX + bbox.maxX) / 2;
      const cy = (bbox.minY + bbox.maxY) / 2;

      // b. 绘制标签徽章（避障位置设计：对于下标，默认状态下绝不绘制任何标签或背景，仅在鼠标悬停或选中时在框外侧展示标签）
      if (b.type === 'left') {
        this.drawBadge(ctx, scale, `[L${b.group_id || 1}]`, bbox.minX, bbox.minY - 4 / scale, color);
      } else if (b.type === 'right') {
        this.drawBadge(ctx, scale, `[R${b.group_id || 1}]`, bbox.maxX, bbox.minY - 4 / scale, color);
      } else {
        // 下标组件：默认不画任何遮挡标签，仅选中或悬停时显示
        const subTxt = b.text || 'n';
        const badgeX = bbox.maxX + 4 / scale;
        const badgeY = bbox.minY - 2 / scale;
        this.drawSubscriptBadge(ctx, scale, subTxt, b.group_id || 1, badgeX, badgeY, isSel);
      }

      // 如果选中，绘制完整的旋转/缩放/顶点微调把手
      if (isSel) {
        // a. 顶点把手 (用于微调多边形顶点)
        const vSize = 6 / scale;
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 1.5 / scale;
        for (let k = 0; k < b.polygon.length; k++) {
          const pt = b.polygon[k];
          ctx.fillRect(pt[0] - vSize / 2, pt[1] - vSize / 2, vSize, vSize);
          ctx.strokeRect(pt[0] - vSize / 2, pt[1] - vSize / 2, vSize, vSize);
        }

        // b. 变换外框虚线
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 1.0 / scale;
        ctx.setLineDash([3 / scale, 3 / scale]);
        ctx.strokeRect(bbox.minX, bbox.minY, bbox.maxX - bbox.minX, bbox.maxY - bbox.minY);
        ctx.setLineDash([]);

        // c. 顶部旋转杆与旋转圆点
        const rotY = bbox.minY - 22 / scale;
        ctx.beginPath();
        ctx.moveTo(cx, bbox.minY);
        ctx.lineTo(cx, rotY);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(cx, rotY, 4.5 / scale, 0, Math.PI * 2);
        ctx.fillStyle = '#2563eb';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5 / scale;
        ctx.stroke();

        // d. 8 向外边框缩放把手
        const hSize = 5 / scale;
        const handles = [
          { x: bbox.minX, y: bbox.minY },
          { x: bbox.maxX, y: bbox.minY },
          { x: bbox.maxX, y: bbox.maxY },
          { x: bbox.minX, y: bbox.maxY },
          { x: cx, y: bbox.minY },
          { x: cx, y: bbox.maxY },
          { x: bbox.minX, y: cy },
          { x: bbox.maxX, y: cy },
        ];

        ctx.fillStyle = '#2563eb';
        for (const h of handles) {
          ctx.fillRect(h.x - hSize / 2, h.y - hSize / 2, hSize, hSize);
        }
      }

      ctx.restore();
    }

    ctx.restore();
  }

  drawBadge(ctx, scale, text, x, y, bgColor, centered = false) {
    ctx.save();
    const fontSize = Math.max(9, Math.round(11 / Math.sqrt(scale)));
    ctx.font = `bold ${fontSize}px sans-serif`;
    const metrics = ctx.measureText(text);
    const padX = 4 / scale;
    const padY = 2 / scale;
    const w = metrics.width + padX * 2;
    const h = fontSize + padY * 2;

    const bx = centered ? x - w / 2 : x;
    const by = centered ? y - h / 2 : y - h;

    ctx.fillStyle = bgColor;
    ctx.beginPath();
    ctx.roundRect(bx, by, w, h, 3 / scale);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, bx + w / 2, by + h / 2 + 0.5);
    ctx.restore();
  }

  drawSubscriptBadge(ctx, scale, text, groupId, x, y, isSelected = false) {
    ctx.save();
    const fontSize = Math.max(9, Math.round(11 / Math.sqrt(scale)));
    ctx.font = `bold ${fontSize}px sans-serif`;

    const displayTxt = `[Sub${groupId}: ${text}]`;
    const metrics = ctx.measureText(displayTxt);
    const padX = 3.5 / scale;
    const padY = 1.5 / scale;
    const w = metrics.width + padX * 2;
    const h = fontSize + padY * 2;

    const bx = x;
    const by = y;

    ctx.fillStyle = isSelected ? 'rgba(37, 99, 235, 0.92)' : 'rgba(217, 119, 6, 0.92)';
    ctx.beginPath();
    ctx.roundRect(bx, by, w, h, 3 / scale);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(displayTxt, bx + w / 2, by + h / 2 + 0.5);
    ctx.restore();
  }

  // =========================================================================
  // 化学键与原子渲染器
  // =========================================================================

  drawBonds(ctx, scale) {
    for (let i = 0; i < this.bonds.length; i++) {
      const b = this.bonds[i];
      const a1 = this.atoms[b[0]];
      const a2 = this.atoms[b[1]];
      if (!a1 || !a2) continue;

      const isSel = this.selectedBonds.has(i);
      const isHover = (this.hoverBondIdx === i);
      const bType = parseInt(b[2]) || 1;

      ctx.save();
      const bondColor = ChemUtils.getBondColor(bType, isSel, isHover);
      ctx.strokeStyle = bondColor;
      ctx.fillStyle = bondColor;

      if (isSel) {
        ctx.lineWidth = 4.2 / scale;
        ctx.shadowColor = 'rgba(37, 99, 235, 0.45)';
        ctx.shadowBlur = 8 / scale;
      } else if (isHover) {
        ctx.lineWidth = 3.6 / scale;
        ctx.shadowColor = 'rgba(59, 130, 246, 0.35)';
        ctx.shadowBlur = 5 / scale;
      } else {
        const baseWidth = (ChemUtils.BOND_TYPES[bType] && ChemUtils.BOND_TYPES[bType].width) || 2.2;
        ctx.lineWidth = baseWidth / scale;
      }

      const dx = a2.x - a1.x;
      const dy = a2.y - a1.y;
      const len = Math.hypot(dx, dy);
      if (len === 0) { ctx.restore(); continue; }

      const nx = -dy / len;
      const ny = dx / len;
      const bondDist = 3.2 / scale;

      if (bType === 1) {
        // 1: 单键 (Single) - 鲜明实线
        ctx.beginPath();
        ctx.moveTo(a1.x, a1.y);
        ctx.lineTo(a2.x, a2.y);
        ctx.stroke();
      } else if (bType === 2) {
        // 2: 双键 (Double) - 平行双线
        ctx.beginPath();
        ctx.moveTo(a1.x + nx * bondDist, a1.y + ny * bondDist);
        ctx.lineTo(a2.x + nx * bondDist, a2.y + ny * bondDist);
        ctx.moveTo(a1.x - nx * bondDist, a1.y - ny * bondDist);
        ctx.lineTo(a2.x - nx * bondDist, a2.y - ny * bondDist);
        ctx.stroke();
      } else if (bType === 3) {
        // 3: 三键 (Triple) - 紧凑三线
        ctx.beginPath();
        ctx.moveTo(a1.x, a1.y);
        ctx.lineTo(a2.x, a2.y);
        ctx.moveTo(a1.x + nx * bondDist * 1.4, a1.y + ny * bondDist * 1.4);
        ctx.lineTo(a2.x + nx * bondDist * 1.4, a2.y + ny * bondDist * 1.4);
        ctx.moveTo(a1.x - nx * bondDist * 1.4, a1.y - ny * bondDist * 1.4);
        ctx.lineTo(a2.x - nx * bondDist * 1.4, a2.y - ny * bondDist * 1.4);
        ctx.stroke();
      } else if (bType === 4) {
        // 4: 芳香键 (Aromatic) - 精细虚线
        ctx.setLineDash([4 / scale, 3 / scale]);
        ctx.beginPath();
        ctx.moveTo(a1.x, a1.y);
        ctx.lineTo(a2.x, a2.y);
        ctx.stroke();
      } else if (bType === 5) {
        // 5: 实心楔形键 (Solid Wedge) - 立体前向实心三角形
        const w = 4.8 / scale;
        ctx.beginPath();
        ctx.moveTo(a1.x, a1.y);
        ctx.lineTo(a2.x + nx * w, a2.y + ny * w);
        ctx.lineTo(a2.x - nx * w, a2.y - ny * w);
        ctx.closePath();
        ctx.fill();
      } else if (bType === 6) {
        // 6: 虚线楔形键 (Hash Wedge) - 立体后向阶梯虚线
        const steps = 7;
        const wMax = 4.8 / scale;
        for (let s = 1; s <= steps; s++) {
          const t = s / steps;
          const px = a1.x + dx * t;
          const py = a1.y + dy * t;
          const cw = wMax * t;
          ctx.beginPath();
          ctx.moveTo(px - nx * cw, py - ny * cw);
          ctx.lineTo(px + nx * cw, py + ny * cw);
          ctx.stroke();
        }
      } else if (bType === 7) {
        // 7: 波浪键 (Wave / 不定连接点) - 沿键轴正弦波, 振幅/波数随键长自适应
        const amp = Math.max(2.0 / scale, Math.min(len * 0.06, 9 / scale));
        const cycles = Math.max(2, Math.min(6, Math.round(len / (18 / scale))));
        const segs = cycles * 10;
        ctx.beginPath();
        for (let s = 0; s <= segs; s++) {
          const t = s / segs;
          const off = Math.sin(t * Math.PI * 2 * cycles) * amp;
          const px = a1.x + dx * t + nx * off;
          const py = a1.y + dy * t + ny * off;
          if (s === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
      ctx.restore();
    }
  }

  drawAtoms(ctx, scale) {
    for (let i = 0; i < this.atoms.length; i++) {
      const a = this.atoms[i];
      const isSel = this.selectedAtoms.has(i);
      const isHover = (this.hoverAtomIdx === i);

      const hasCustomLabel = !!a.label;
      const isAP = (a.label === '_AP' || (a.symbol === '*' && !a.label));
      const isSuper = hasCustomLabel && a.label !== '_AP';

      ctx.save();

      // 1. 选中 / 悬停发光外环
      if (isSel || isHover) {
        ctx.beginPath();
        ctx.arc(a.x, a.y, (isSel ? 12 : 9) / scale, 0, Math.PI * 2);
        ctx.fillStyle = isSel ? 'rgba(37, 99, 235, 0.25)' : 'rgba(59, 130, 246, 0.18)';
        ctx.fill();
        ctx.strokeStyle = isSel ? '#2563eb' : '#3b82f6';
        ctx.lineWidth = 1.5 / scale;
        ctx.stroke();
      }

      // 2. 节点圆圈
      const nodeRadius = 5.5 / scale;
      ctx.beginPath();
      ctx.arc(a.x, a.y, nodeRadius, 0, Math.PI * 2);

      let fillColor = '#10b981';
      if (isAP) fillColor = '#ef4444';
      else if (isSuper) fillColor = '#8b5cf6';
      else if (a.symbol === 'C') fillColor = '#0f172a';
      else fillColor = ChemUtils.getAtomColor(a.symbol, a.label);

      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5 / scale;
      ctx.stroke();

      // 3. 标签文字 (仅非 C 原子或带超原子标签展示)
      const displayLabel = a.label || (a.symbol !== 'C' ? a.symbol : null);
      if (this.showAtomLabels !== false && displayLabel) {
        const isDark = document.body.classList.contains('theme-dark');
        const fontSize = Math.max(8, Math.round(10 / Math.sqrt(scale)));
        ctx.font = `bold ${fontSize}px sans-serif`;
        const metrics = ctx.measureText(displayLabel);

        const lx = a.x + 8 / scale;
        const ly = a.y - 8 / scale;

        // 背景微胶囊
        ctx.fillStyle = isDark ? 'rgba(17, 24, 39, 0.85)' : 'rgba(255, 255, 255, 0.9)';
        ctx.beginPath();
        ctx.roundRect(lx - 2 / scale, ly - fontSize, metrics.width + 4 / scale, fontSize + 3 / scale, 3 / scale);
        ctx.fill();

        ctx.fillStyle = isAP ? '#ef4444' : (isSuper ? '#7c3aed' : (isDark ? '#f3f4f6' : '#0f172a'));
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText(displayLabel, lx, ly + 2 / scale);
      }

      ctx.restore();
    }
  }

  // =========================================================================
  // 多选包围盒 (8 向调节把手 + 旋转把手)
  // =========================================================================

  drawTransformBBox(ctx, scale, bbox) {
    ctx.save();
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 1.2 / scale;
    ctx.setLineDash([4 / scale, 3 / scale]);
    ctx.strokeRect(bbox.minX, bbox.minY, bbox.width, bbox.height);
    ctx.setLineDash([]);

    // 旋转杆与顶部把手
    const rotHandleY = bbox.minY - 24 / scale;
    ctx.beginPath();
    ctx.moveTo(bbox.cx, bbox.minY);
    ctx.lineTo(bbox.cx, rotHandleY);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(bbox.cx, rotHandleY, 4.5 / scale, 0, Math.PI * 2);
    ctx.fillStyle = '#2563eb';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5 / scale;
    ctx.stroke();

    // 8 个方形把手
    const handleSize = 7 / scale;
    const handles = [
      { x: bbox.minX, y: bbox.minY },
      { x: bbox.maxX, y: bbox.minY },
      { x: bbox.maxX, y: bbox.maxY },
      { x: bbox.minX, y: bbox.maxY },
      { x: bbox.cx, y: bbox.minY },
      { x: bbox.cx, y: bbox.maxY },
      { x: bbox.minX, y: bbox.cy },
      { x: bbox.maxX, y: bbox.cy },
    ];

    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 1.5 / scale;
    for (const h of handles) {
      ctx.fillRect(h.x - handleSize / 2, h.y - handleSize / 2, handleSize, handleSize);
      ctx.strokeRect(h.x - handleSize / 2, h.y - handleSize / 2, handleSize, handleSize);
    }

    ctx.restore();
  }
}
