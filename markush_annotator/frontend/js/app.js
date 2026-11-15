/**
 * Markush Visual Annotator - Main Application Controller
 */

class AppController {
  constructor() {
    this.currentSampleIndex = null;
    this.samplesList = [];
    this.currentSampleMeta = null;
    this.customElements = [];
    this.customSuperatoms = [];

    // OBB 标注模式状态
    this.currentAppMode = 'mol'; // 'mol' | 'obb'
    this.obbEditor = null;
    this.obbSamplesList = [];
    this.currentObbSampleId = null;
    this.currentObbPos = 0;

    // 自动保存与防竞态锁
    this.isSwitchingSample = false;
    this.lastSavedMolIndex = null;
    this.lastSavedObbSampleId = null;

    this.initDOM();
    this.initCanvasAndEditor();
    this.initTheme();
    this.initDropdownMenu();
    this.initFloatingToolbar();
    this.initCustomElements();
    this.initCustomSuperatoms();
    this.initPeriodicTableModal();
    this.initAppModeTabs();
    this.bindEvents();
    this.loadDatasetInfo();
  }

  initDOM() {
    // 顶部模式选项卡
    this.tabModeMol = document.getElementById('tab-mode-mol');
    this.tabModeObb = document.getElementById('tab-mode-obb');
    this.molWorkbench = document.getElementById('mol-workbench');
    this.obbWorkbench = document.getElementById('obb-workbench');
    this.molSampleNavigator = document.getElementById('mol-sample-navigator');
    this.obbSampleNavigator = document.getElementById('obb-sample-navigator');

    // 导航元素 (顺序位置与样本ID独立)
    this.btnPrev = document.getElementById('btn-prev');
    this.btnNext = document.getElementById('btn-next');
    this.inputSequencePos = document.getElementById('input-sequence-pos');
    this.inputSampleId = document.getElementById('input-sample-id');
    this.labelTotalCount = document.getElementById('label-total-count');
    this.badgeReason = document.getElementById('badge-reason');
    this.badgeStatus = document.getElementById('badge-status');

    this.selectReasonFilter = document.getElementById('select-reason-filter');
    this.selectStatusFilter = document.getElementById('select-status-filter');

    // 顶部审核分类标签 (总体 / 审核 / 待处理 / 废弃)
    this.statusFilterTabs = document.querySelectorAll('.status-filter-tab');
    this.countAll = document.getElementById('count-all');
    this.countApproved = document.getElementById('count-approved');
    this.countPending = document.getElementById('count-pending');
    this.countDiscarded = document.getElementById('count-discarded');

    // 核心操作按钮
    this.btnSave = document.getElementById('btn-save');
    this.btnSaveAndNext = document.getElementById('btn-save-and-next');
    this.btnSaveAndNextTop = document.getElementById('btn-save-and-next-top');
    this.btnDiscard = document.getElementById('btn-discard');
    this.btnHelp = document.getElementById('btn-help');
    this.helpModal = document.getElementById('help-modal');
    this.btnCloseModal = document.getElementById('btn-close-modal');

    // 下拉菜单与分类筛选
    this.dropdownMore = document.getElementById('dropdown-more');
    this.btnMoreMenu = document.getElementById('btn-more-menu');
    this.moreMenuDropdown = document.getElementById('more-menu-dropdown');
    this.moreFilterMolSection = document.getElementById('more-filter-mol-section');
    this.moreFilterObbSection = document.getElementById('more-filter-obb-section');
    this.moreSelectObbSource = document.getElementById('more-select-obb-source');
    this.moreSelectObbFeature = document.getElementById('more-select-obb-feature');
    this.moreSelectObbStatus = document.getElementById('more-select-obb-status');
    this.moreBtnObbPredict = document.getElementById('more-btn-obb-predict');
    this.moreBtnObbClear = document.getElementById('more-btn-obb-clear');

    // 主题切换
    this.btnThemeToggle = document.getElementById('btn-theme-toggle');
    this.themeIcon = document.getElementById('theme-icon');
    this.themeText = document.getElementById('theme-text');

    // 左侧工具与选板
    this.toolButtons = document.querySelectorAll('.tool-btn');
    this.elemButtons = document.querySelectorAll('#element-grid-main .elem-btn');
    this.customElemList = document.getElementById('custom-elem-list');
    this.btnOpenElemModal = document.getElementById('btn-open-elem-modal');
    
    // 超原子与取代基
    this.superatomGridMain = document.getElementById('superatom-grid-main');
    this.customSuperatomList = document.getElementById('custom-superatom-list');
    this.inputCustomSuperatom = document.getElementById('input-custom-superatom');
    this.btnAddCustomSuperatom = document.getElementById('btn-add-custom-superatom');

    this.bondTypeButtons = document.querySelectorAll('.bond-type-btn');
    this.btnUndo = document.getElementById('btn-undo');
    this.btnRedo = document.getElementById('btn-redo');

    // 视图浮动控制
    this.floatingToolbar = document.getElementById('floating-toolbar');
    this.btnCollapseFloating = document.getElementById('btn-collapse-floating');
    this.btnZoomIn = document.getElementById('btn-zoom-in');
    this.btnZoomOut = document.getElementById('btn-zoom-out');
    this.btnZoomFit = document.getElementById('btn-zoom-fit');
    this.btnZoomReset = document.getElementById('btn-zoom-reset');
    this.sliderImgOpacity = document.getElementById('slider-img-opacity');
    this.labelImgOpacity = document.getElementById('label-img-opacity');
    this.sliderMolOpacity = document.getElementById('slider-mol-opacity');
    this.labelMolOpacity = document.getElementById('label-mol-opacity');
    this.btnToggleMolLayer = document.getElementById('btn-toggle-mol-layer');
    this.btnToggleRefBrackets = document.getElementById('btn-toggle-ref-brackets');
    this.btnToggleAtomLabels = document.getElementById('btn-toggle-atom-labels');
    this.btnToggleInvert = document.getElementById('btn-toggle-invert');
    this.btnToggleGrid = document.getElementById('btn-toggle-grid');

    // 底部状态栏
    this.statusCoords = document.getElementById('status-coords');
    this.statusStats = document.getElementById('status-stats');
    this.statusZoom = document.getElementById('status-zoom');
    this.statusSmiles = document.getElementById('status-smiles');

    // 右侧检查面板
    this.infoIndex = document.getElementById('info-index');
    this.infoReason = document.getElementById('info-reason');
    this.infoResolution = document.getElementById('info-resolution');
    this.infoSource = document.getElementById('info-source');
    this.infoSmilesInput = document.getElementById('info-smiles-input');
    this.btnSyncSmiles = document.getElementById('btn-sync-smiles');
    this.selectionBody = document.getElementById('selection-body');
    this.selectionTypeBadge = document.getElementById('selection-type-badge');
    this.selectReviewStatus = document.getElementById('select-review-status');
    this.inputReviewNotes = document.getElementById('input-review-notes');

    // 元素周期表模态框 DOM
    this.elementModal = document.getElementById('element-modal');
    this.btnCloseElemModal = document.getElementById('btn-close-elem-modal');
    this.periodicTableGrid = document.getElementById('periodic-table-grid');

    this.toastContainer = document.getElementById('toast-container');
  }

  initCanvasAndEditor() {
    const bottomCanvas = document.getElementById('canvas-bottom');
    const topCanvas = document.getElementById('canvas-top');

    this.canvasEngine = new CanvasEngine(bottomCanvas, topCanvas, (ctx, scale) => {
      if (this.editor) {
        this.editor.render(ctx, scale);
      }
    });

    this.canvasEngine.onZoomChange = (scale) => {
      if (this.statusZoom) {
        this.statusZoom.textContent = `缩放: ${Math.round(scale * 100)}%`;
      }
    };

    this.editor = new MolEditor(
      this.canvasEngine,
      (sel) => this.handleSelectionChange(sel),
      () => this.handleDataChange()
    );

    // 确保透明度正确初始化
    this.canvasEngine.imageOpacity = 0.85;
    this.canvasEngine.molOpacity = 1.0;
    console.log('[App] Canvas引擎初始化完成，imageOpacity:', this.canvasEngine.imageOpacity, 'molOpacity:', this.canvasEngine.molOpacity);
  }

  initTheme() {
    const savedTheme = localStorage.getItem('markush_theme') || 'light';
    this.setTheme(savedTheme);

    if (this.btnThemeToggle) {
      this.btnThemeToggle.addEventListener('click', () => {
        const isCurrentDark = document.body.classList.contains('theme-dark');
        this.setTheme(isCurrentDark ? 'light' : 'dark');
        if (this.moreMenuDropdown) this.moreMenuDropdown.classList.remove('show');
      });
    }
  }

  setTheme(theme) {
    if (theme === 'dark') {
      document.body.classList.add('theme-dark');
      document.body.classList.remove('theme-light');
      if (this.themeIcon) this.themeIcon.textContent = '☀️';
      if (this.themeText) this.themeText.textContent = '亮色模式';
    } else {
      document.body.classList.add('theme-light');
      document.body.classList.remove('theme-dark');
      if (this.themeIcon) this.themeIcon.textContent = '🌙';
      if (this.themeText) this.themeText.textContent = '暗色模式';
    }
    localStorage.setItem('markush_theme', theme);
    if (this.canvasEngine) {
      this.canvasEngine.render();
    }
  }

  initDropdownMenu() {
    if (!this.btnMoreMenu || !this.moreMenuDropdown) return;

    this.btnMoreMenu.addEventListener('click', (e) => {
      e.stopPropagation();
      this.moreMenuDropdown.classList.toggle('show');
    });

    window.addEventListener('click', (e) => {
      if (this.moreMenuDropdown && !this.dropdownMore.contains(e.target)) {
        this.moreMenuDropdown.classList.remove('show');
      }
    });
  }

  initFloatingToolbar() {
    const toolbar = this.floatingToolbar;
    if (!toolbar) return;

    if (this.btnCollapseFloating) {
      this.btnCollapseFloating.addEventListener('click', () => {
        const isCollapsed = toolbar.classList.toggle('collapsed');
        this.btnCollapseFloating.textContent = isCollapsed ? '▼' : '▲';
        this.btnCollapseFloating.title = isCollapsed ? '展开视图工具' : '收起视图工具';
      });
    }

    const dragHandle = toolbar.querySelector('.drag-handle');
    if (dragHandle) {
      let isDragging = false;
      let startX, startY, initLeft, initTop;

      dragHandle.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        const rect = toolbar.getBoundingClientRect();
        const parentRect = toolbar.parentElement.getBoundingClientRect();
        initLeft = rect.left - parentRect.left;
        initTop = rect.top - parentRect.top;
        toolbar.style.left = `${initLeft}px`;
        toolbar.style.top = `${initTop}px`;
        toolbar.style.transform = 'none';
        e.preventDefault();
      });

      window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        toolbar.style.left = `${Math.max(10, initLeft + dx)}px`;
        toolbar.style.top = `${Math.max(10, initTop + dy)}px`;
      });

      window.addEventListener('mouseup', () => {
        if (isDragging) isDragging = false;
      });
    }
  }

  bindEvents() {
    // 样本流转
    this.btnPrev.addEventListener('click', () => this.navigateSample(-1));
    this.btnNext.addEventListener('click', () => this.navigateSample(1));

    // 1. 顺序号 (1 ~ Total) 跳转
    this.inputSequencePos.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const pos = parseInt(this.inputSequencePos.value.trim());
        if (!isNaN(pos) && pos >= 1 && pos <= this.samplesList.length) {
          this.loadSample(this.samplesList[pos - 1].index);
        } else {
          this.showToast(`请输入有效的序号 (1 ~ ${this.samplesList.length})`, 'warn');
        }
      }
    });

    // 2. 精确样本 ID (如 #67 或 67) 跳转
    this.inputSampleId.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        let valStr = this.inputSampleId.value.trim().replace(/^#/, '');
        const targetId = parseInt(valStr);
        if (!isNaN(targetId)) {
          this.loadSample(targetId);
        } else {
          this.showToast('请输入有效的样本编号', 'warn');
        }
      }
    });

    // 过滤器切换
    this.selectReasonFilter.addEventListener('change', () => this.reloadSamplesList());
    this.selectStatusFilter.addEventListener('change', () => {
      this.syncStatusFilterTabs(this.selectStatusFilter.value);
      this.reloadSamplesList();
    });

    // 顶部审核分类标签: 点击只显示对应分类的样本
    this.statusFilterTabs.forEach(tab => {
      tab.addEventListener('click', () => this.setStatusFilter(tab.dataset.status));
    });

    // 动作按钮
    this.btnSave.addEventListener('click', () => this.saveCurrentAnnotation(false));
    if (this.btnSaveAndNext) {
      this.btnSaveAndNext.addEventListener('click', () => this.saveCurrentAnnotation(true));
    }
    if (this.btnSyncSmiles) {
      this.btnSyncSmiles.addEventListener('click', () => this.syncSmilesFromGraph());
    }
    if (this.infoSmilesInput) {
      this.infoSmilesInput.addEventListener('input', (e) => {
        if (this.statusSmiles) {
          this.statusSmiles.textContent = e.target.value.trim() || '-';
        }
      });
    }
    this.btnDiscard.addEventListener('click', () => {
      if (this.moreMenuDropdown) this.moreMenuDropdown.classList.remove('show');
      this.markDiscard();
    });

    // 模态框
    this.btnHelp.addEventListener('click', () => {
      if (this.moreMenuDropdown) this.moreMenuDropdown.classList.remove('show');
      this.helpModal.classList.remove('hidden');
    });
    this.btnCloseModal.addEventListener('click', () => this.helpModal.classList.add('hidden'));
    this.helpModal.addEventListener('click', (e) => {
      if (e.target === this.helpModal) this.helpModal.classList.add('hidden');
    });

    // 工具栏切换
    this.toolButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        this.toolButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const tool = btn.dataset.tool;
        this.editor.activeTool = tool;
        this.editor.updateCursor();
      });
    });

    // 常用标准元素选板
    this.elemButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const sym = btn.dataset.symbol;
        this.selectElement(sym);
      });
    });

    // 常用超原子选板 (在 #superatom-grid-main 内)
    if (this.superatomGridMain) {
      this.superatomGridMain.querySelectorAll('.elem-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const sym = btn.dataset.symbol;
          this.selectSuperatom(sym);
        });
      });
    }

    // 自定义超原子添加与输入
    if (this.btnAddCustomSuperatom && this.inputCustomSuperatom) {
      const doAddSuperatom = () => {
        const sa = this.inputCustomSuperatom.value.trim();
        if (sa) {
          this.addCustomSuperatom(sa);
          this.inputCustomSuperatom.value = '';
        }
      };
      this.btnAddCustomSuperatom.addEventListener('click', doAddSuperatom);
      this.inputCustomSuperatom.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doAddSuperatom();
      });
    }

    // 元素周期表模态框打开与关闭
    if (this.btnOpenElemModal && this.elementModal) {
      this.btnOpenElemModal.addEventListener('click', () => {
        this.elementModal.classList.remove('hidden');
      });
    }
    if (this.btnCloseElemModal && this.elementModal) {
      this.btnCloseElemModal.addEventListener('click', () => this.elementModal.classList.add('hidden'));
      this.elementModal.addEventListener('click', (e) => {
        if (e.target === this.elementModal) this.elementModal.classList.add('hidden');
      });
    }

    // 键类型选板
    this.bondTypeButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        this.bondTypeButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const bType = parseInt(btn.dataset.bondType);
        this.editor.activeBondType = bType;
        if (this.editor.selectedBonds.size > 0) {
          for (const idx of this.editor.selectedBonds) {
            this.editor.bonds[idx][2] = bType;
          }
          this.editor.pushHistory();
          this.canvasEngine.renderTop();
          this.editor.notifySelection();
        }
      });
    });

    // 撤销 / 重做
    this.btnUndo.addEventListener('click', () => this.editor.undo());
    this.btnRedo.addEventListener('click', () => this.editor.redo());

    // 视图缩放与透明度调节
    this.btnZoomIn.addEventListener('click', () => {
      const rect = this.canvasEngine.topCanvas.getBoundingClientRect();
      this.canvasEngine.zoomAt(rect.width / 2, rect.height / 2, 1.25);
    });
    this.btnZoomOut.addEventListener('click', () => {
      const rect = this.canvasEngine.topCanvas.getBoundingClientRect();
      this.canvasEngine.zoomAt(rect.width / 2, rect.height / 2, 0.8);
    });
    this.btnZoomFit.addEventListener('click', () => this.canvasEngine.fitToScreen());
    this.btnZoomReset.addEventListener('click', () => this.canvasEngine.resetZoom());

    this.sliderImgOpacity.addEventListener('input', (e) => {
      const val = parseInt(e.target.value);
      this.canvasEngine.imageOpacity = val / 100;
      this.labelImgOpacity.textContent = `${val}%`;
      this.canvasEngine.renderBottom();
    });

    this.sliderMolOpacity.addEventListener('input', (e) => {
      const val = parseInt(e.target.value);
      this.canvasEngine.molOpacity = val / 100;
      this.labelMolOpacity.textContent = `${val}%`;
      this.canvasEngine.renderTop();
    });

    if (this.btnToggleMolLayer) {
      this.btnToggleMolLayer.addEventListener('click', () => {
        this.canvasEngine.showMolLayer = !this.canvasEngine.showMolLayer;
        this.btnToggleMolLayer.classList.toggle('active', this.canvasEngine.showMolLayer);
        this.btnToggleMolLayer.title = this.canvasEngine.showMolLayer
          ? '显示/隐藏 MolScribe 预标注的分子图结构 (只看原图)'
          : '当前已隐藏图结构，点击重新显示';
        this.canvasEngine.renderTop();
        this.showToast(
          this.canvasEngine.showMolLayer ? '已显示 MolScribe 图结构' : '已隐藏 MolScribe 图结构 (仅显示原图)',
          'info',
          1500
        );
      });
    }

    if (this.btnToggleRefBrackets) {
      this.btnToggleRefBrackets.addEventListener('click', () => {
        this.editor.showRefBrackets = !this.editor.showRefBrackets;
        this.btnToggleRefBrackets.classList.toggle('active', this.editor.showRefBrackets);
        this.canvasEngine.renderTop();
      });
    }

    if (this.btnToggleAtomLabels) {
      this.btnToggleAtomLabels.addEventListener('click', () => {
        this.editor.showAtomLabels = !this.editor.showAtomLabels;
        this.btnToggleAtomLabels.classList.toggle('active', this.editor.showAtomLabels);
        this.canvasEngine.renderTop();
      });
    }

    this.btnToggleInvert.addEventListener('click', () => {
      this.canvasEngine.invertImage = !this.canvasEngine.invertImage;
      this.btnToggleInvert.classList.toggle('active', this.canvasEngine.invertImage);
      this.canvasEngine.renderBottom();
    });

    this.btnToggleGrid.addEventListener('click', () => {
      this.canvasEngine.showGrid = !this.canvasEngine.showGrid;
      this.btnToggleGrid.classList.toggle('active', this.canvasEngine.showGrid);
      this.canvasEngine.renderBottom();
    });

    // 全局快捷键
    window.addEventListener('keydown', (e) => this.handleGlobalShortcuts(e));
  }

  handleGlobalShortcuts(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
      if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        this.saveCurrentAnnotation(true);
      }
      return;
    }

    if (e.ctrlKey && (e.key === 's' || e.key === 'S')) {
      e.preventDefault();
      this.saveCurrentAnnotation(false);
      return;
    }

    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      this.saveCurrentAnnotation(true);
      return;
    }

    // OBB 模式下的快捷键响应
    if (this.currentAppMode === 'obb') {
      if (e.ctrlKey && (e.key === 'z' || e.key === 'Z')) {
        e.preventDefault();
        if (this.obbEditor) this.obbEditor.undo();
        return;
      }
      if (e.ctrlKey && (e.key === 'y' || e.key === 'Y')) {
        e.preventDefault();
        if (this.obbEditor) this.obbEditor.redo();
        return;
      }

      const key = e.key.toLowerCase();
      if (key === 'v') this.setObbTool('select');
      else if (key === 'w') this.setObbTool('draw');
      else if (key === 'd' || e.key === 'Delete' || e.key === 'Backspace') {
        if (this.obbEditor) this.obbEditor.deleteSelected();
      } else if (key === 'q') this.navigateObbSample(-1);
      else if (key === 'e') this.navigateObbSample(1);
      else if (key === 'f') {
        if (this.obbEditor) this.obbEditor.fitToScreen();
      } else if (['1', '2', '3', '4', '5', '6'].includes(key)) {
        const clsMap = { '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 0 };
        this.setObbClass(clsMap[key]);
      }
      return;
    }

    if (e.ctrlKey && (e.key === 'z' || e.key === 'Z')) {
      e.preventDefault();
      this.editor.undo();
      return;
    }

    if (e.ctrlKey && (e.key === 'y' || e.key === 'Y')) {
      e.preventDefault();
      this.editor.redo();
      return;
    }

    const key = e.key.toLowerCase();
    if (key === 'v') this.setTool('select');
    else if (key === 'a') this.setTool('atom');
    else if (key === 'b') this.setTool('bond');
    else if (key === 'l' || key === '[') this.setTool('bracket-left');
    else if (key === 'r' || key === ']') this.setTool('bracket-right');
    else if (key === 's') this.setTool('bracket-sub');
    else if (key === 'd' || e.key === 'Delete' || e.key === 'Backspace') {
      this.editor.deleteSelected();
    } else if (key === 'q') this.navigateSample(-1);
    else if (key === 'e') this.navigateSample(1);
    else if (key === 'f') this.canvasEngine.fitToScreen();
    else if (key === '?' || key === '¿') this.helpModal.classList.toggle('hidden');
    else if (['1', '2', '3', '4', '5', '6'].includes(key)) {
      this.setBondType(parseInt(key));
    }
  }

  setTool(tool) {
    const btn = document.querySelector(`.tool-btn[data-tool="${tool}"]`);
    if (btn) btn.click();
  }

  setBondType(type) {
    const btn = document.querySelector(`.bond-type-btn[data-bond-type="${type}"]`);
    if (btn) btn.click();
  }

  // =========================================================================
  // 标准元素选择与周期表
  // =========================================================================

  selectElement(sym) {
    if (!sym) return;
    if (ChemUtils.isSuperatom(sym)) {
      this.selectSuperatom(sym);
      return;
    }
    this.editor.activeSymbol = sym;

    // 1. 同步常用元素按钮高亮
    this.elemButtons.forEach(b => {
      b.classList.toggle('active', b.dataset.symbol === sym);
    });
    if (this.customElemList) {
      this.customElemList.querySelectorAll('.custom-elem-badge').forEach(b => {
        b.classList.toggle('active', b.dataset.symbol === sym);
      });
    }

    // 2. 取消超原子选板高亮
    if (this.superatomGridMain) {
      this.superatomGridMain.querySelectorAll('.elem-btn').forEach(b => b.classList.remove('active'));
    }
    if (this.customSuperatomList) {
      this.customSuperatomList.querySelectorAll('.custom-elem-badge').forEach(b => b.classList.remove('active'));
    }

    // 3. 若当前有选中的原子，立即批量修改为该标准元素 (清空超原子标签)
    if (this.editor.selectedAtoms.size > 0) {
      for (const idx of this.editor.selectedAtoms) {
        this.editor.atoms[idx].symbol = sym;
        this.editor.atoms[idx].label = null;
        ChemUtils.normalizeAtom(this.editor.atoms[idx]);
      }
      this.editor.pushHistory();
      this.canvasEngine.renderTop();
      this.editor.notifySelection();
    }
  }

  initCustomElements() {
    try {
      const saved = localStorage.getItem('markush_custom_elements');
      if (saved) {
        this.customElements = JSON.parse(saved);
      } else {
        this.customElements = [];
      }
    } catch (e) {
      this.customElements = [];
    }
    this.renderCustomElements();
  }

  saveCustomElements() {
    try {
      localStorage.setItem('markush_custom_elements', JSON.stringify(this.customElements));
    } catch (e) {}
  }

  renderCustomElements() {
    if (!this.customElemList) return;
    this.customElemList.innerHTML = '';
    for (const sym of this.customElements) {
      const badge = document.createElement('div');
      badge.className = `custom-elem-badge ${this.editor.activeSymbol === sym ? 'active' : ''}`;
      badge.dataset.symbol = sym;
      badge.innerHTML = `
        <span>${sym}</span>
        <span class="del-btn" title="从常用元素栏移除">&times;</span>
      `;
      badge.addEventListener('click', (e) => {
        if (e.target.classList.contains('del-btn')) {
          e.stopPropagation();
          this.removeCustomElement(sym);
        } else {
          this.selectElement(sym);
        }
      });
      this.customElemList.appendChild(badge);
    }
  }

  addCustomElement(sym) {
    if (!sym) return;
    sym = sym.trim();
    if (ChemUtils.isSuperatom(sym)) {
      this.addCustomSuperatom(sym);
      return;
    }
    const defaultElements = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H', 'B', 'Si'];
    if (!defaultElements.includes(sym) && !this.customElements.includes(sym)) {
      this.customElements.push(sym);
      this.saveCustomElements();
      this.renderCustomElements();
    }
  }

  removeCustomElement(sym) {
    this.customElements = this.customElements.filter(s => s !== sym);
    this.saveCustomElements();
    this.renderCustomElements();
  }

  // =========================================================================
  // 超原子 / 取代基管理 (Symbol 恒为 *)
  // =========================================================================

  selectSuperatom(label) {
    if (!label) return;
    this.editor.activeSymbol = label;

    // 1. 同步超原子按钮高亮
    if (this.superatomGridMain) {
      this.superatomGridMain.querySelectorAll('.elem-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.symbol === label);
      });
    }
    if (this.customSuperatomList) {
      this.customSuperatomList.querySelectorAll('.custom-elem-badge').forEach(b => {
        b.classList.toggle('active', b.dataset.symbol === label);
      });
    }

    // 2. 取消常用标准元素高亮
    this.elemButtons.forEach(b => b.classList.remove('active'));
    if (this.customElemList) {
      this.customElemList.querySelectorAll('.custom-elem-badge').forEach(b => b.classList.remove('active'));
    }

    // 3. 若当前有选中的原子，立即批量修改为该超原子 (symbol 恒为 *)
    if (this.editor.selectedAtoms.size > 0) {
      for (const idx of this.editor.selectedAtoms) {
        this.editor.atoms[idx].symbol = '*';
        this.editor.atoms[idx].label = (label === '*' ? '_AP' : label);
        ChemUtils.normalizeAtom(this.editor.atoms[idx]);
      }
      this.editor.pushHistory();
      this.canvasEngine.renderTop();
      this.editor.notifySelection();
    }
  }

  initCustomSuperatoms() {
    try {
      const saved = localStorage.getItem('markush_custom_superatoms');
      if (saved) {
        this.customSuperatoms = JSON.parse(saved);
      } else {
        this.customSuperatoms = ['Ts', 'Fmoc', 'CF3', 'Tf', 'TBDMS'];
      }
    } catch (e) {
      this.customSuperatoms = ['Ts', 'Fmoc', 'CF3', 'Tf', 'TBDMS'];
    }
    this.renderCustomSuperatoms();
  }

  saveCustomSuperatoms() {
    try {
      localStorage.setItem('markush_custom_superatoms', JSON.stringify(this.customSuperatoms));
    } catch (e) {}
  }

  renderCustomSuperatoms() {
    if (!this.customSuperatomList) return;
    this.customSuperatomList.innerHTML = '';
    for (const sa of this.customSuperatoms) {
      const badge = document.createElement('div');
      badge.className = `custom-elem-badge`;
      badge.dataset.symbol = sa;
      badge.innerHTML = `
        <span>${sa}</span>
        <span class="del-btn" title="从快捷超原子栏移除">&times;</span>
      `;
      badge.addEventListener('click', (e) => {
        if (e.target.classList.contains('del-btn')) {
          e.stopPropagation();
          this.removeCustomSuperatom(sa);
        } else {
          this.selectSuperatom(sa);
        }
      });
      this.customSuperatomList.appendChild(badge);
    }
  }

  addCustomSuperatom(sa) {
    if (!sa) return;
    sa = sa.trim();
    if (!this.customSuperatoms.includes(sa)) {
      this.customSuperatoms.push(sa);
      this.saveCustomSuperatoms();
      this.renderCustomSuperatoms();
    }
    this.selectSuperatom(sa);
    this.showToast(`已添加快捷超原子/取代基: ${sa}`, 'success');
  }

  removeCustomSuperatom(sa) {
    this.customSuperatoms = this.customSuperatoms.filter(s => s !== sa);
    this.saveCustomSuperatoms();
    this.renderCustomSuperatoms();
  }

  // =========================================================================
  // 化学元素周期表模态框 (118 种元素)
  // =========================================================================

  initPeriodicTableModal() {
    if (!this.elementModal) return;
    this.renderPeriodicTable();
  }

  renderPeriodicTable() {
    if (!this.periodicTableGrid || !ChemUtils.PERIODIC_TABLE) return;
    this.periodicTableGrid.innerHTML = '';

    for (const elem of ChemUtils.PERIODIC_TABLE) {
      const card = document.createElement('div');
      card.className = `elem-card ${elem.category}`;

      let row = elem.period;
      let col = elem.group;
      if (elem.category === 'lanthanide') {
        row = 8;
        col = (elem.number - 57) + 4;
      } else if (elem.category === 'actinide') {
        row = 9;
        col = (elem.number - 89) + 4;
      }
      card.style.gridColumn = `${col}`;
      card.style.gridRow = `${row}`;
      card.title = `#${elem.number} ${elem.symbol} - ${elem.name} (${elem.en})`;

      card.innerHTML = `
        <span class="elem-num">${elem.number}</span>
        <span class="elem-sym">${elem.symbol}</span>
        <span class="elem-zh">${elem.name}</span>
      `;

      card.addEventListener('click', () => {
        this.addCustomElement(elem.symbol);
        this.selectElement(elem.symbol);
        this.elementModal.classList.add('hidden');
        this.showToast(`已选用元素: ${elem.symbol} (${elem.name})`, 'success');
      });

      this.periodicTableGrid.appendChild(card);
    }
  }

  // =========================================================================
  // 数据加载与流转 API
  // =========================================================================

  async loadDatasetInfo() {
    try {
      const res = await fetch('/api/info');
      const info = await res.json();
      
      this.labelTotalCount.textContent = `/ ${info.total_count}`;

      // 填充 Reason 过滤下拉 (按大类分类聚合)
      this.selectReasonFilter.innerHTML = '<option value="all">🌐 全部原因 (全部样本)</option>';
      if (info.categories && info.categories.length > 0) {
        for (const cat of info.categories) {
          const opt = document.createElement('option');
          opt.value = cat.key;
          opt.textContent = `${cat.icon} ${cat.label} (${cat.count})`;
          opt.title = `${cat.en}: ${cat.description}`;
          this.selectReasonFilter.appendChild(opt);
        }
      } else if (info.reasons) {
        for (const r of info.reasons) {
          const opt = document.createElement('option');
          opt.value = r;
          opt.textContent = r;
          this.selectReasonFilter.appendChild(opt);
        }
      }

      this.renderStatusCounts(info.status_counts, info.total_count);
      this.syncStatusFilterTabs(this.selectStatusFilter.value);

      await this.reloadSamplesList();
    } catch (err) {
      this.showToast(`无法连接后端服务: ${err.message}`, 'error');
    }
  }

  /** 切换审核分类 (审核 / 待处理 / 废弃) */
  setStatusFilter(status) {
    if (!status) return;
    this.syncStatusFilterTabs(status);
    if (this.selectStatusFilter.value !== status) {
      this.selectStatusFilter.value = status;
      this.reloadSamplesList();
    }
  }

  /** 同步顶部分类标签的高亮状态 */
  syncStatusFilterTabs(status) {
    if (!this.statusFilterTabs) return;
    this.statusFilterTabs.forEach(tab => {
      tab.classList.toggle('active', tab.dataset.status === status);
    });
  }

  /** 更新各分类的样本数量 (总体 = 审核 + 待处理 + 废弃) */
  renderStatusCounts(counts = {}, total = null) {
    const fallbackTotal = (counts.approved || 0) + (counts.pending || 0) + (counts.discarded || 0);
    const totalCount = (total === null || total === undefined) ? fallbackTotal : total;
    const items = [
      [this.countAll, totalCount],
      [this.countApproved, counts.approved],
      [this.countPending, counts.pending],
      [this.countDiscarded, counts.discarded],
    ];
    for (const [el, count] of items) {
      if (el) el.textContent = count || 0;
    }
  }

  /** 保存后刷新分类计数 */
  async refreshStatusCounts() {
    try {
      const res = await fetch('/api/info');
      if (!res.ok) return;
      const info = await res.json();
      this.renderStatusCounts(info.status_counts || {}, info.total_count);
    } catch (e) {
      console.warn('refreshStatusCounts failed:', e);
    }
  }

  async reloadSamplesList(options = {}) {
    const { preferredPos = null, autoLoad = true } = options;
    const reason = this.selectReasonFilter.value;
    const status = this.selectStatusFilter.value;

    try {
      const res = await fetch(`/api/samples?reason=${encodeURIComponent(reason)}&status=${encodeURIComponent(status)}&limit=15000`);
      if (!res.ok) throw new Error('获取样本清单失败');
      const data = await res.json();
      this.samplesList = data.items || [];
      this.labelTotalCount.textContent = `/ ${this.samplesList.length}`;

      if (this.samplesList.length === 0) {
        if (this.inputSequencePos) this.inputSequencePos.value = '';
        this.showToast('该分类下暂无样本', 'warn');
        return;
      }

      const currPos = this.samplesList.findIndex(s => s.index === this.currentSampleIndex);
      if (currPos !== -1) {
        this.inputSequencePos.value = currPos + 1;
        return;
      }

      if (!autoLoad) return;

      const targetPos = preferredPos === null
        ? 0
        : Math.min(Math.max(preferredPos, 0), this.samplesList.length - 1);
      await this.loadSample(this.samplesList[targetPos].index);
    } catch (err) {
      this.showToast(`加载样本清单失败: ${err.message}`, 'error');
    }
  }

  async navigateSample(direction) {
    if (this.samplesList.length === 0 || this.isSwitchingSample) return;
    let currPos = this.samplesList.findIndex(s => s.index === this.currentSampleIndex);
    if (currPos === -1) currPos = 0;
    let nextPos = currPos + direction;
    if (nextPos < 0) nextPos = 0;
    if (nextPos >= this.samplesList.length) nextPos = this.samplesList.length - 1;
    const targetIndex = this.samplesList[nextPos].index;
    if (targetIndex === this.currentSampleIndex) {
      this.showToast('已到达当前筛选样本的边界', 'info', 1200);
      return;
    }
    await this.loadSample(targetIndex);
  }

  async loadSample(index) {
    if (this.isSwitchingSample) return;
    this.isSwitchingSample = true;

    try {
      // 切换前自动保存当前样本
      if (this.currentSampleIndex !== null && this.currentSampleIndex !== index && this.editor) {
        if (this.lastSavedMolIndex !== this.currentSampleIndex) {
          await this.saveMolAnnotation(false, true);
        }
      }

      this.currentSampleIndex = index;

      // 1. 获取样本详情 (加时间戳避免浏览器缓存，确保读到最新的纠错文件)
      const res = await fetch(`/api/sample/${index}?t=${Date.now()}`);
      if (!res.ok) throw new Error(`样本 #${index} 加载失败`);
      const detail = await res.json();
      this.currentSampleMeta = detail;

      // 2. 更新编号栏 (顺序位置与唯一ID)
      const currPos = this.samplesList.findIndex(s => s.index === index);
      if (currPos !== -1) {
        this.inputSequencePos.value = currPos + 1;
      }
      this.inputSampleId.value = `#${String(index).padStart(6, '0')}`;

      // 3. 加载原图
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = `/api/image/${index}?t=${Date.now()}`;
      await new Promise((resolve) => {
        img.onload = resolve;
        img.onerror = () => {
          this.showToast('原图加载失败，请检查图像文件路径', 'error');
          resolve();
        };
      });

      // 4. 注入 Canvas 与分子编辑器
      this.canvasEngine.setImage(img, detail.image_size);
      this.editor.setData(detail);

      // 4.5. 强制重新渲染确保图结构显示
      setTimeout(() => {
        this.canvasEngine.render();
        console.log('[App] 强制重新渲染完成');
      }, 100);

      // 5. 更新 UI 检查面板
      this.updateSampleInfoPanel(detail);
      
      // 6. 双重保障：若初始 SMILES 未包含 CXSMILES 扩展段且图中存在超原子/连接点，自动触发从图生成权威 CXSMILES
      if (detail.atoms && detail.atoms.length > 0 && (!detail.smi || (!detail.smi.includes('|') && detail.atoms.some(a => a.label || a.symbol === '*')))) {
        this.syncSmilesFromGraph(true);
      }

      this.showToast(`已载入样本 #${index} (${detail.source === 'corrected' ? '已纠错' : '待人工复核'})`, 'success');
    } catch (err) {
      this.showToast(err.message, 'error');
    } finally {
      this.isSwitchingSample = false;
    }
  }

  updateSampleInfoPanel(detail) {
    this.infoIndex.textContent = `#${String(detail.index).padStart(6, '0')}`;
    
    // 大类与细粒度原因展示
    const rInfo = detail.reason_info;
    if (rInfo) {
      this.infoReason.innerHTML = `<span>${rInfo.category_icon} ${rInfo.category_label}</span>` + 
        (rInfo.detail ? `<span class="text-muted" style="margin-left: 6px; font-size: 11px;">(${rInfo.detail})</span>` : '');
      this.badgeReason.textContent = `${rInfo.category_icon} ${rInfo.category_label}`;
      this.badgeReason.className = `badge badge-reason ${rInfo.category}`;
      this.badgeReason.title = `详细原因: ${detail.reason || '-'}`;
    } else {
      this.infoReason.textContent = detail.reason || '-';
      this.badgeReason.textContent = detail.reason || 'manual_review';
      this.badgeReason.className = 'badge badge-reason';
    }

    this.infoResolution.textContent = `${detail.image_size[0]} × ${detail.image_size[1]}`;
    this.infoSource.textContent = detail.source === 'corrected' ? '已纠错 (corrected)' : '自动对齐草稿 (annotations)';
    this.infoSmilesInput.value = detail.smi || '';

    this.badgeStatus.textContent = detail.review_status === 'approved' ? '已纠错通过' : (detail.review_status === 'discarded' ? '已废弃' : '待处理');
    this.badgeStatus.className = `badge badge-status ${detail.review_status || 'pending'}`;

    this.selectReviewStatus.value = detail.review_status || 'approved';
    this.inputReviewNotes.value = detail.review_notes || '';

    if (this.statusSmiles) {
      this.statusSmiles.textContent = detail.smi || '-';
    }
  }

  // =========================================================================
  // 属性面板与交互反馈
  // =========================================================================

  handleSelectionChange(sel) {
    const { selectedAtoms, selectedBonds, selectedBracket, atoms, bonds, brackets } = sel;

    if (selectedAtoms.length === 1) {
      const idx = selectedAtoms[0];
      const a = atoms[idx];
      ChemUtils.normalizeAtom(a);
      const isSuper = !!a.label || a.symbol === '*';
      this.selectionTypeBadge.textContent = isSuper ? `超原子 #${idx}` : `原子 #${idx}`;
      this.selectionTypeBadge.className = isSuper ? 'badge badge-reason' : 'badge badge-status';

      this.selectionBody.innerHTML = `
        <div class="info-row">
          <span class="info-label">原子序号:</span>
          <span class="info-val font-mono">#${idx}</span>
        </div>
        <div class="form-group">
          <label>元素符号 (Element Symbol):</label>
          <div style="display: flex; gap: 6px; align-items: center;">
            <input type="text" id="edit-atom-sym" class="form-input font-mono" value="${a.symbol || 'C'}" maxlength="4" style="flex: 1; font-weight: 700;">
            <button id="btn-inspector-open-ptable" class="btn btn-secondary btn-xs" title="打开元素周期表选择元素">⚛️ 周期表</button>
          </div>
        </div>
        <div class="form-group">
          <label>超原子/取代基标签 (Label / R基团):</label>
          <input type="text" id="edit-atom-label" class="form-input font-mono" value="${a.label || ''}" placeholder="例如: SO2, SO3H, OMe, Et, Ph, R1, _AP">
          <span class="help-hint" style="font-size: 10px; color: var(--text-dim); display: block; margin-top: 2px;">注: 超原子规范为 元素符号 * + 对应 Label</span>
        </div>
        <div class="info-row">
          <span class="info-label">像素坐标 (X, Y):</span>
          <span class="info-val font-mono">${a.x}, ${a.y}</span>
        </div>
        <div class="btn-group" style="display: flex; gap: 4px; margin-top: 6px;">
          <button id="btn-quick-ap" class="btn btn-secondary btn-block" style="font-size: 11px;">设为连接点 (*)</button>
          <button id="btn-quick-del-atom" class="btn btn-danger btn-block" style="font-size: 11px;">删除原子</button>
        </div>
      `;

      const inputSym = document.getElementById('edit-atom-sym');
      const inputLabel = document.getElementById('edit-atom-label');
      const btnOpenPTable = document.getElementById('btn-inspector-open-ptable');

      if (btnOpenPTable) {
        btnOpenPTable.addEventListener('click', () => {
          if (this.elementModal) this.elementModal.classList.remove('hidden');
        });
      }

      if (inputSym) {
        inputSym.addEventListener('input', (e) => {
          const val = e.target.value.trim();
          if (val) {
            if (ChemUtils.isSuperatom(val)) {
              a.symbol = '*';
              a.label = (val === '*' ? '_AP' : val);
              inputSym.value = '*';
              inputLabel.value = a.label;
            } else {
              a.symbol = val.length <= 2 ? (val.charAt(0).toUpperCase() + val.slice(1).toLowerCase()) : val;
              a.label = null;
              inputLabel.value = '';
            }
            ChemUtils.normalizeAtom(a);
            this.editor.pushHistory();
            this.canvasEngine.renderTop();
          }
        });
      }

      if (inputLabel) {
        inputLabel.addEventListener('input', (e) => {
          const val = e.target.value.trim();
          if (val) {
            a.symbol = '*';
            a.label = val;
            inputSym.value = '*';
          } else {
            a.label = null;
            if (a.symbol === '*') a.symbol = 'C';
            inputSym.value = a.symbol;
          }
          ChemUtils.normalizeAtom(a);
          this.editor.pushHistory();
          this.canvasEngine.renderTop();
        });
      }

      document.getElementById('btn-quick-ap').addEventListener('click', () => {
        a.symbol = '*';
        a.label = '_AP';
        inputSym.value = '*';
        inputLabel.value = '_AP';
        this.editor.pushHistory();
        this.canvasEngine.renderTop();
        this.editor.notifySelection();
      });

      document.getElementById('btn-quick-del-atom').addEventListener('click', () => {
        this.editor.deleteAtom(idx);
        this.editor.pushHistory();
        this.canvasEngine.render();
      });
    } else if (selectedAtoms.length > 1) {
      this.selectionTypeBadge.textContent = `多选 (${selectedAtoms.length} 个原子)`;
      this.selectionTypeBadge.className = 'badge badge-reason';

      this.selectionBody.innerHTML = `
        <div class="info-row">
          <span class="info-label">选中原子数:</span>
          <span class="info-val font-mono">${selectedAtoms.length}</span>
        </div>
        <div class="empty-hint" style="padding: 6px 0;">
          💡 拖动画布上的<strong>虚线变换框</strong>可整体平移，拖动<strong>8向把手</strong>缩放，拖动<strong>顶部圆点</strong>旋转。
        </div>
        <div class="btn-group-stacked">
          <button id="btn-batch-del" class="btn btn-danger btn-block" style="font-size: 12px;">批量删除所选节点 (Delete)</button>
        </div>
      `;
      document.getElementById('btn-batch-del').addEventListener('click', () => {
        this.editor.deleteSelected();
      });
    } else if (selectedBonds.length === 1) {
      const bIdx = selectedBonds[0];
      const b = bonds[bIdx];
      const bType = parseInt(b[2]) || 1;
      const bondName = ChemUtils.getBondName(bType);
      const bondColor = ChemUtils.getBondColor(bType);
      this.selectionTypeBadge.innerHTML = `<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${bondColor}; margin-right:4px;"></span>${bondName} #${bIdx}`;
      this.selectionTypeBadge.className = 'badge badge-status';

      this.selectionBody.innerHTML = `
        <div class="info-row">
          <span class="info-label">连接原子:</span>
          <span class="info-val font-mono">Atom #${b[0]} ↔ Atom #${b[1]}</span>
        </div>
        <div class="form-group">
          <label>键类型 (Bond Type):</label>
          <select id="edit-bond-type" class="form-select">
            <option value="1" ${bType === 1 ? 'selected' : ''}>1: 单键 (Single) - 翡翠绿</option>
            <option value="2" ${bType === 2 ? 'selected' : ''}>2: 双键 (Double) - 科技蓝</option>
            <option value="3" ${bType === 3 ? 'selected' : ''}>3: 三键 (Triple) - 亮橙色</option>
            <option value="4" ${bType === 4 ? 'selected' : ''}>4: 芳香键 (Aromatic) - 紫罗兰</option>
            <option value="5" ${bType === 5 ? 'selected' : ''}>5: 楔形实键 (Wedge) - 极光靛蓝</option>
            <option value="6" ${bType === 6 ? 'selected' : ''}>6: 楔形虚键 (Hash) - 玫瑰红</option>
          </select>
        </div>
        <div class="btn-group" style="display: flex; gap: 4px; margin-top: 6px;">
          <button id="btn-swap-bond" class="btn btn-secondary btn-block" style="font-size: 11px;">反转楔形方向</button>
          <button id="btn-del-bond" class="btn btn-danger btn-block" style="font-size: 11px;">删除键</button>
        </div>
      `;

      document.getElementById('edit-bond-type').addEventListener('change', (e) => {
        b[2] = parseInt(e.target.value);
        this.editor.pushHistory();
        this.canvasEngine.renderTop();
        this.editor.notifySelection();
      });
      document.getElementById('btn-swap-bond').addEventListener('click', () => {
        const tmp = b[0];
        b[0] = b[1];
        b[1] = tmp;
        this.editor.pushHistory();
        this.canvasEngine.renderTop();
      });
      document.getElementById('btn-del-bond').addEventListener('click', () => {
        this.editor.deleteBond(bIdx);
        this.editor.pushHistory();
        this.canvasEngine.renderTop();
      });
    } else if (selectedBracket !== null && brackets && brackets[selectedBracket]) {
      const b = brackets[selectedBracket];
      const typeLabel = b.type === 'left' ? '左括号 [L]' : (b.type === 'right' ? '右括号 [R]' : '括号下标 [Sub]');
      this.selectionTypeBadge.textContent = `${typeLabel} (组 #${b.group_id || 1})`;
      this.selectionTypeBadge.className = 'badge badge-reason';

      this.selectionBody.innerHTML = `
        <div class="form-group">
          <label>所属组号 (Group ID):</label>
          <input type="number" id="edit-bracket-gid" class="form-input font-mono" min="1" max="20" value="${b.group_id || 1}">
        </div>
        <div class="form-group">
          <label>组件类型 (Type):</label>
          <select id="edit-bracket-type" class="form-select">
            <option value="left" ${b.type === 'left' ? 'selected' : ''}>左括号 [ (Left)</option>
            <option value="right" ${b.type === 'right' ? 'selected' : ''}>右括号 ] (Right)</option>
            <option value="subscript" ${b.type === 'subscript' ? 'selected' : ''}>括号下标 n (Subscript)</option>
          </select>
        </div>
        ${b.type === 'subscript' ? `
        <div class="form-group">
          <label>下标内容 (Text):</label>
          <input type="text" id="edit-bracket-text" class="form-input font-mono" value="${b.text || 'n'}" placeholder="例如: n, m, p, 2">
        </div>
        ` : ''}
        <div class="empty-hint" style="padding: 6px 0;">
          💡 拖动<strong>顶部圆点</strong>旋转框，拖动<strong>边框把手</strong>缩放，拖动<strong>4个顶点</strong>精确微调多边形。
        </div>
        <button id="btn-del-bracket" class="btn btn-danger btn-block" style="font-size: 12px; margin-top: 6px;">删除此括号组件</button>
      `;

      document.getElementById('edit-bracket-gid').addEventListener('change', (e) => {
        b.group_id = Math.max(1, parseInt(e.target.value) || 1);
        this.editor.pushHistory();
        this.canvasEngine.renderTop();
        this.editor.notifySelection();
      });

      document.getElementById('edit-bracket-type').addEventListener('change', (e) => {
        b.type = e.target.value;
        if (b.type === 'subscript' && !b.text) b.text = 'n';
        this.editor.pushHistory();
        this.canvasEngine.renderTop();
        this.editor.notifySelection();
      });

      const textInput = document.getElementById('edit-bracket-text');
      if (textInput) {
        textInput.addEventListener('input', (e) => {
          b.text = e.target.value.trim() || 'n';
          this.canvasEngine.renderTop();
        });
      }

      document.getElementById('btn-del-bracket').addEventListener('click', () => {
        this.editor.deleteBracket(selectedBracket);
      });
    } else {
      this.selectionTypeBadge.textContent = '未选中';
      this.selectionTypeBadge.className = 'badge';
      this.selectionBody.innerHTML = `
        <div class="empty-hint">在画布上点击原子、化学键、括号或框选多个节点进行编辑</div>
      `;
    }

    this.updateStatusBar();
  }

  handleDataChange() {
    this.updateStatusBar();
  }

  updateStatusBar() {
    const data = this.editor.getData();
    const bracketCount = (data.brackets || []).length;
    this.statusStats.textContent = `原子: ${data.atoms.length}, 键: ${data.bonds.length}, 括号组件: ${bracketCount}`;
    this.statusZoom.textContent = `缩放: ${Math.round(this.canvasEngine.scale * 100)}%`;
    this.statusSmiles.textContent = this.infoSmilesInput.value || '-';
  }

  // =========================================================================
  // 保存、自动对齐与 AI OCR 补全
  // =========================================================================

  async syncSmilesFromGraph(silent = false) {
    if (this.currentSampleIndex === null) return;
    try {
      if (!silent && this.btnSyncSmiles) {
        this.btnSyncSmiles.disabled = true;
        this.btnSyncSmiles.textContent = '⏳ 计算中...';
      }
      const data = this.editor.getData();
      const payload = {
        atoms: data.atoms,
        bonds: data.bonds,
        brackets: data.brackets || [],
      };

      const res = await fetch('/api/calc_smiles', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error('生成 SMILES 请求失败');
      const resData = await res.json();
      const newSmiles = resData.smiles || '';

      if (newSmiles) {
        this.infoSmilesInput.value = newSmiles;
        if (this.statusSmiles) {
          this.statusSmiles.textContent = newSmiles;
        }
        if (this.currentSampleMeta) {
          this.currentSampleMeta.smi = newSmiles;
        }
        if (!silent) {
          this.showToast('✅ 已根据当前画布分子图重新生成 SMILES/CXSMILES', 'success');
        }
      } else if (!silent) {
        this.showToast('未能从当前图生成有效 SMILES（请检查孤立节点或连接键）', 'warn');
      }
    } catch (err) {
      if (!silent) {
        this.showToast(`生成 SMILES 异常: ${err.message}`, 'error');
      }
    } finally {
      if (!silent && this.btnSyncSmiles) {
        this.btnSyncSmiles.disabled = false;
        this.btnSyncSmiles.innerHTML = '<span>🔄 从图更新</span>';
      }
    }
  }

  async saveCurrentAnnotation(gotoNext = false, silent = false) {
    if (this.currentAppMode === 'obb') {
      return this.saveObbAnnotation(gotoNext, silent);
    }
    return this.saveMolAnnotation(gotoNext, silent);
  }

  async saveMolAnnotation(gotoNext = false, silent = false) {
    if (this.currentSampleIndex === null) return false;
    try {
      const data = this.editor.getData();

      // 规范化所有原子
      for (const a of data.atoms) {
        ChemUtils.normalizeAtom(a);
      }

      // 1. 保存前自动根据当前图结构重新生成并同步最新 CXSMILES
      let currentSmiles = this.infoSmilesInput.value.trim() || (this.currentSampleMeta ? this.currentSampleMeta.smi : '');
      try {
        const calcRes = await fetch('/api/calc_smiles', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            atoms: data.atoms,
            bonds: data.bonds,
            brackets: data.brackets || [],
          }),
        });
        if (calcRes.ok) {
          const calcData = await calcRes.json();
          if (calcData.smiles) {
            currentSmiles = calcData.smiles;
            this.infoSmilesInput.value = currentSmiles;
            if (this.statusSmiles) {
              this.statusSmiles.textContent = currentSmiles;
            }
            if (this.currentSampleMeta) {
              this.currentSampleMeta.smi = currentSmiles;
            }
          }
        }
      } catch (calcErr) {
        console.warn('Auto sync CXSMILES before save warning:', calcErr);
      }

      const payload = {
        index: this.currentSampleIndex,
        esmiles: currentSmiles,
        image_size: this.currentSampleMeta ? this.currentSampleMeta.image_size : [800, 600],
        atoms: data.atoms,
        bonds: data.bonds,
        sgroups: data.sgroups,
        brackets: data.brackets,
        review_status: this.selectReviewStatus.value,
        review_notes: this.inputReviewNotes.value.trim(),
      };

      const res = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error('保存请求失败');

      this.lastSavedMolIndex = this.currentSampleIndex;

      // 更新前端缓存列表中的状态与 SMILES
      const sampleInList = this.samplesList.find(s => s.index === this.currentSampleIndex);
      if (sampleInList) {
        sampleInList.review_status = payload.review_status;
        sampleInList.has_corrected = true;
        sampleInList.smi = currentSmiles;
      }

      // 刷新顶部分类标签上的计数
      this.refreshStatusCounts();

      if (!silent) {
        this.showToast(`✅ 样本 #${this.currentSampleIndex} 纠错结果已保存！(CXSMILES 已同步)`, 'success');
      } else {
        this.showToast(`💾 样本 #${this.currentSampleIndex} 已自动保存`, 'info', 1500);
      }
      
      if (gotoNext) {
        const statusFilter = this.selectStatusFilter.value;
        if (statusFilter !== 'all' && payload.review_status !== statusFilter) {
          // 当前样本保存后已不属于当前分类: 重新拉取分类列表并停在原本的下一张
          const prevPos = this.samplesList.findIndex(s => s.index === this.currentSampleIndex);
          const hasNext = prevPos >= 0 && prevPos + 1 < this.samplesList.length;
          await this.reloadSamplesList({
            preferredPos: prevPos >= 0 ? prevPos : 0,
            autoLoad: hasNext,
          });
          if (!hasNext) {
            this.showToast('当前分类已全部处理完成，可切换其他分类查看', 'info', 3200);
          }
        } else {
          await this.navigateSample(1);
        }
      }
      return true;
    } catch (err) {
      this.showToast(`保存失败: ${err.message}`, 'error');
      return false;
    }
  }

  // =========================================================================
  // 模式切换与 OBB 标注页面管理
  // =========================================================================

  initAppModeTabs() {
    if (!this.tabModeMol || !this.tabModeObb) return;

    this.tabModeMol.addEventListener('click', () => this.switchAppMode('mol'));
    this.tabModeObb.addEventListener('click', () => this.switchAppMode('obb'));

    if (this.btnSaveAndNextTop) {
      this.btnSaveAndNextTop.addEventListener('click', () => this.saveCurrentAnnotation(true));
    }
  }

  async switchAppMode(mode) {
    if (this.currentAppMode === mode) return;
    const prevMode = this.currentAppMode;

    // 模式切换前自动保存上一模式的当前样本
    if (prevMode === 'obb' && this.currentObbSampleId && this.obbEditor && this.obbEditor.image) {
      if (this.lastSavedObbSampleId !== this.currentObbSampleId) {
        await this.saveObbAnnotation(false, true);
      }
    } else if (prevMode === 'mol' && this.currentSampleIndex !== null && this.editor) {
      if (this.lastSavedMolIndex !== this.currentSampleIndex) {
        await this.saveMolAnnotation(false, true);
      }
    }

    this.currentAppMode = mode;

    if (mode === 'mol') {
      this.tabModeMol.classList.add('active');
      this.tabModeObb.classList.remove('active');
      this.molWorkbench.style.display = 'flex';
      this.obbWorkbench.style.display = 'none';
      if (this.molSampleNavigator) this.molSampleNavigator.style.display = 'flex';
      if (this.obbSampleNavigator) this.obbSampleNavigator.style.display = 'none';
      if (this.moreFilterMolSection) this.moreFilterMolSection.style.display = 'block';
      if (this.moreFilterObbSection) this.moreFilterObbSection.style.display = 'none';
      setTimeout(() => this.canvasEngine.fitToScreen(), 50);
      this.showToast('已切换至 [分子图结构纠错] 工作台', 'info');
    } else {
      this.tabModeMol.classList.remove('active');
      this.tabModeObb.classList.add('active');
      this.molWorkbench.style.display = 'none';
      this.obbWorkbench.style.display = 'flex';
      if (this.molSampleNavigator) this.molSampleNavigator.style.display = 'none';
      if (this.obbSampleNavigator) this.obbSampleNavigator.style.display = 'flex';
      if (this.moreFilterMolSection) this.moreFilterMolSection.style.display = 'none';
      if (this.moreFilterObbSection) this.moreFilterObbSection.style.display = 'block';

      // 同步右侧面板的值到“更多”菜单中
      const selSource = document.getElementById('select-obb-source');
      const selFeature = document.getElementById('select-obb-feature');
      const selStatus = document.getElementById('select-obb-status');
      if (selSource && this.moreSelectObbSource) this.moreSelectObbSource.value = selSource.value;
      if (selFeature && this.moreSelectObbFeature) this.moreSelectObbFeature.value = selFeature.value;
      if (selStatus && this.moreSelectObbStatus) this.moreSelectObbStatus.value = selStatus.value;

      if (!this.obbEditor) {
        this.initOBBEditor();
      }
      requestAnimationFrame(() => {
        if (this.obbEditor) {
          this.obbEditor.handleResize();
          this.obbEditor.fitToScreen();
          this.updateObbStatusBar();
        }
      });
      this.showToast('已切换至 [多目标 OBB 目标框标注] 工作台', 'info');
    }
  }

  initOBBEditor() {
    const bottomCanvas = document.getElementById('obb-canvas-bottom');
    const topCanvas = document.getElementById('obb-canvas-top');
    if (!bottomCanvas || !topCanvas) return;

    this.obbEditor = new OBBEditor({
      bottomCanvas: bottomCanvas,
      topCanvas: topCanvas,
      onSelectionChange: (box) => this.handleObbSelectionChange(box),
      onBoxesChange: (boxes) => this.handleObbBoxesChange(boxes),
      onStatusChange: () => this.updateObbStatusBar(),
      onModeChange: (tool) => this.setObbTool(tool),
      onCoordsChange: (x, y) => {
        const el = document.getElementById('obb-status-coords');
        if (el) el.textContent = `X: ${x}, Y: ${y}`;
      }
    });

    this.bindOBBEvents();
    this.loadObbDatasetInfo();
    this.loadObbSamplesList();
  }

  bindOBBEvents() {
    // 1. 标注模式切换 (Select / Draw / Delete)
    const toolSelect = document.getElementById('obb-tool-select');
    const toolDraw = document.getElementById('obb-tool-draw');
    const toolDelete = document.getElementById('obb-tool-delete');

    if (toolSelect) toolSelect.addEventListener('click', () => this.setObbTool('select'));
    if (toolDraw) toolDraw.addEventListener('click', () => this.setObbTool('draw'));
    if (toolDelete) toolDelete.addEventListener('click', () => {
      if (this.obbEditor) this.obbEditor.deleteSelected();
    });

    // 2. 类别快捷按钮
    document.querySelectorAll('.obb-class-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const cId = parseInt(btn.dataset.classId);
        this.setObbClass(cId);
      });
    });

    // 3. AI 预标注与清空
    const btnAi = document.getElementById('btn-obb-ai-predict');
    if (btnAi) btnAi.addEventListener('click', () => this.runObbAiPredict());

    const btnClear = document.getElementById('btn-obb-clear-all');
    if (btnClear) {
      btnClear.addEventListener('click', () => {
        if (confirm('确认清空当前图上的所有 OBB 标注框吗？')) {
          if (this.obbEditor) this.obbEditor.clearAll();
        }
      });
    }

    // 4. 撤销 / 重做
    const btnUndo = document.getElementById('btn-obb-undo');
    if (btnUndo) btnUndo.addEventListener('click', () => {
      if (this.obbEditor) this.obbEditor.undo();
    });
    const btnRedo = document.getElementById('btn-obb-redo');
    if (btnRedo) btnRedo.addEventListener('click', () => {
      if (this.obbEditor) this.obbEditor.redo();
    });

    // 5. 视图浮动控制器
    const btnZoomIn = document.getElementById('btn-obb-zoom-in');
    if (btnZoomIn) btnZoomIn.addEventListener('click', () => {
      if (this.obbEditor) {
        this.obbEditor.scale = Math.min(this.obbEditor.scale * 1.2, 15.0);
        this.obbEditor.render();
        this.updateObbStatusBar();
      }
    });
    const btnZoomOut = document.getElementById('btn-obb-zoom-out');
    if (btnZoomOut) btnZoomOut.addEventListener('click', () => {
      if (this.obbEditor) {
        this.obbEditor.scale = Math.max(this.obbEditor.scale / 1.2, 0.1);
        this.obbEditor.render();
        this.updateObbStatusBar();
      }
    });
    const btnZoomFit = document.getElementById('btn-obb-zoom-fit');
    if (btnZoomFit) btnZoomFit.addEventListener('click', () => {
      if (this.obbEditor) {
        this.obbEditor.fitToScreen();
        this.updateObbStatusBar();
      }
    });
    const btnZoomReset = document.getElementById('btn-obb-zoom-reset');
    if (btnZoomReset) btnZoomReset.addEventListener('click', () => {
      if (this.obbEditor) {
        this.obbEditor.scale = 1.0;
        this.obbEditor.render();
        this.updateObbStatusBar();
      }
    });

    const sliderOpacity = document.getElementById('slider-obb-img-opacity');
    if (sliderOpacity) {
      sliderOpacity.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        if (this.obbEditor) {
          this.obbEditor.imageOpacity = val / 100;
          const lbl = document.getElementById('label-obb-img-opacity');
          if (lbl) lbl.textContent = `${val}%`;
          this.obbEditor.renderBottom();
        }
      });
    }

    const btnInvert = document.getElementById('btn-obb-toggle-invert');
    if (btnInvert) {
      btnInvert.addEventListener('click', () => {
        if (this.obbEditor) {
          this.obbEditor.invertImage = !this.obbEditor.invertImage;
          btnInvert.classList.toggle('active', this.obbEditor.invertImage);
          this.obbEditor.renderBottom();
        }
      });
    }

    const btnLabels = document.getElementById('btn-obb-toggle-labels');
    if (btnLabels) {
      btnLabels.addEventListener('click', () => {
        if (this.obbEditor) {
          this.obbEditor.showLabels = !this.obbEditor.showLabels;
          btnLabels.classList.toggle('active', this.obbEditor.showLabels);
          this.obbEditor.renderTop();
        }
      });
    }

    const btnDir = document.getElementById('btn-obb-toggle-direction');
    if (btnDir) {
      btnDir.addEventListener('click', () => {
        if (this.obbEditor) {
          this.obbEditor.showOrientation = !this.obbEditor.showOrientation;
          btnDir.classList.toggle('active', this.obbEditor.showOrientation);
          this.obbEditor.renderTop();
        }
      });
    }

    // 6. OBB 样本流转导航
    const btnPrev = document.getElementById('btn-obb-prev');
    const btnNext = document.getElementById('btn-obb-next');
    if (btnPrev) btnPrev.addEventListener('click', () => this.navigateObbSample(-1));
    if (btnNext) btnNext.addEventListener('click', () => this.navigateObbSample(1));

    const inputPos = document.getElementById('input-obb-sequence-pos');
    if (inputPos) {
      inputPos.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const pos = parseInt(inputPos.value.trim());
          if (!isNaN(pos) && pos >= 1 && pos <= this.obbSamplesList.length) {
            this.loadObbSample(this.obbSamplesList[pos - 1].id);
          } else {
            this.showToast(`请输入有效序号 (1 ~ ${this.obbSamplesList.length})`, 'warn');
          }
        }
      });
    }

    const inputId = document.getElementById('input-obb-sample-id');
    if (inputId) {
      inputId.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const sid = inputId.value.trim();
          if (sid) this.loadObbSample(sid);
        }
      });
    }

    // 7. 筛选器 (右侧边栏与顶部“更多”菜单双向联动)
    const selSource = document.getElementById('select-obb-source');
    const selFeature = document.getElementById('select-obb-feature');
    const selStatus = document.getElementById('select-obb-status');

    // 右侧边栏改变时同步到“更多”菜单
    if (selSource) {
      selSource.addEventListener('change', () => {
        if (this.moreSelectObbSource) this.moreSelectObbSource.value = selSource.value;
        this.loadObbSamplesList();
      });
    }
    if (selFeature) {
      selFeature.addEventListener('change', () => {
        if (this.moreSelectObbFeature) this.moreSelectObbFeature.value = selFeature.value;
        this.loadObbSamplesList();
      });
    }
    if (selStatus) {
      selStatus.addEventListener('change', () => {
        if (this.moreSelectObbStatus) this.moreSelectObbStatus.value = selStatus.value;
        this.loadObbSamplesList();
      });
    }

    // “更多”菜单改变时同步到右侧边栏
    if (this.moreSelectObbSource) {
      this.moreSelectObbSource.addEventListener('change', () => {
        if (selSource) selSource.value = this.moreSelectObbSource.value;
        this.loadObbSamplesList();
      });
    }
    if (this.moreSelectObbFeature) {
      this.moreSelectObbFeature.addEventListener('change', () => {
        if (selFeature) selFeature.value = this.moreSelectObbFeature.value;
        this.loadObbSamplesList();
      });
    }
    if (this.moreSelectObbStatus) {
      this.moreSelectObbStatus.addEventListener('change', () => {
        if (selStatus) selStatus.value = this.moreSelectObbStatus.value;
        this.loadObbSamplesList();
      });
    }

    // “更多”菜单中的快捷操作按钮
    if (this.moreBtnObbPredict) {
      this.moreBtnObbPredict.addEventListener('click', () => {
        if (this.dropdownMore) this.dropdownMore.classList.remove('open');
        const btnPredict = document.getElementById('btn-obb-ai-predict');
        if (btnPredict) btnPredict.click();
      });
    }
    if (this.moreBtnObbClear) {
      this.moreBtnObbClear.addEventListener('click', () => {
        if (this.dropdownMore) this.dropdownMore.classList.remove('open');
        const btnClear = document.getElementById('btn-obb-clear-all');
        if (btnClear) btnClear.click();
      });
    }

    // 8. 右侧目标列表删除
    const btnDelSel = document.getElementById('btn-obb-del-selected');
    if (btnDelSel) btnDelSel.addEventListener('click', () => {
      if (this.obbEditor) this.obbEditor.deleteSelected();
    });
  }

  setObbTool(tool) {
    if (!this.obbEditor) return;
    this.obbEditor.setMode(tool);
    document.querySelectorAll('#obb-left-toolbar .tool-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tool === tool);
    });
  }

  setObbClass(classId) {
    if (!this.obbEditor) return;
    this.obbEditor.setClass(classId);
    document.querySelectorAll('.obb-class-btn').forEach(btn => {
      btn.classList.toggle('active', parseInt(btn.dataset.classId) === classId);
    });
  }

  async loadObbDatasetInfo() {
    try {
      const res = await fetch('/api/obb/info');
      if (!res.ok) return;
      const data = await res.json();
      const selSource = document.getElementById('select-obb-source');
      if (data.sources) {
        const optHtml = `
          <option value="all">全部来源 (${data.total})</option>
          <option value="molparser">MolParser (${data.sources.molparser || 0})</option>
          <option value="uspto">USPTO (${data.sources.uspto || 0})</option>
        `;
        if (selSource) selSource.innerHTML = optHtml;
        if (this.moreSelectObbSource) this.moreSelectObbSource.innerHTML = optHtml;
      }
    } catch (e) {
      console.warn('loadObbDatasetInfo failed:', e);
    }
  }

  async loadObbSamplesList() {
    try {
      const source = document.getElementById('select-obb-source')?.value || 'all';
      const feature = document.getElementById('select-obb-feature')?.value || 'all';
      const status = document.getElementById('select-obb-status')?.value || 'all';

      const res = await fetch(`/api/obb/samples?source=${source}&feature=${feature}&status=${status}&limit=2000`);
      if (!res.ok) throw new Error('获取 OBB 样本清单失败');
      const data = await res.json();
      this.obbSamplesList = data.items || [];

      const lblTotal = document.getElementById('label-obb-total-count');
      if (lblTotal) lblTotal.textContent = `/ ${data.total}`;

      if (this.obbSamplesList.length > 0) {
        this.loadObbSample(this.obbSamplesList[0].id);
      } else {
        this.showToast('筛选条件下未找到匹配的 OBB 样本', 'warn');
      }
    } catch (err) {
      this.showToast(`加载 OBB 样本列表失败: ${err.message}`, 'error');
    }
  }

  async loadObbSample(sampleId) {
    if (this.isSwitchingSample) return;
    this.isSwitchingSample = true;

    try {
      // 切换前自动保存当前样本
      if (this.currentObbSampleId && this.currentObbSampleId !== sampleId && this.obbEditor && this.obbEditor.image) {
        if (this.lastSavedObbSampleId !== this.currentObbSampleId) {
          await this.saveObbAnnotation(false, true);
        }
      }

      this.currentObbSampleId = sampleId;

      // 1. 获取样本详情 (若无标注自动预测)
      const res = await fetch(`/api/obb/sample/${sampleId}?auto_predict=true`);
      if (!res.ok) throw new Error(`OBB 样本 ${sampleId} 加载失败`);
      const detail = await res.json();

      // 2. 更新顶部导航序号与 ID
      const pos = this.obbSamplesList.findIndex(s => s.id === sampleId);
      this.currentObbPos = pos >= 0 ? pos : 0;
      const inputPos = document.getElementById('input-obb-sequence-pos');
      if (inputPos) inputPos.value = this.currentObbPos + 1;
      const inputId = document.getElementById('input-obb-sample-id');
      if (inputId) inputId.value = sampleId;

      const badgeSource = document.getElementById('badge-obb-source');
      if (badgeSource) badgeSource.textContent = detail.source === 'molparser' ? 'MolParser' : 'USPTO';
      const badgeStatus = document.getElementById('badge-obb-status');
      if (badgeStatus) {
        badgeStatus.textContent = detail.status === 'reviewed' ? '已完成' : '待审核';
        badgeStatus.className = `badge badge-status ${detail.status === 'reviewed' ? 'approved' : ''}`;
      }

      // 3. 加载图像
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = `/api/obb/image/${sampleId}?t=${Date.now()}`;
      await new Promise(resolve => {
        img.onload = resolve;
        img.onerror = () => {
          this.showToast('原图加载失败，请检查文件', 'error');
          resolve();
        };
      });

      // 4. 载入 OBB 编辑器
      this.obbEditor.loadData(img, detail.image_size, detail.boxes || []);

      // 5. 更新右侧信息栏
      const infoId = document.getElementById('obb-info-id');
      if (infoId) infoId.textContent = detail.id;
      const infoSrc = document.getElementById('obb-info-source');
      if (infoSrc) infoSrc.textContent = detail.source;
      const infoFeat = document.getElementById('obb-info-features');
      if (infoFeat) infoFeat.textContent = (detail.features || []).join(', ') || '-';
      const infoDims = document.getElementById('obb-info-dims');
      if (infoDims) infoDims.textContent = `${detail.image_size[0]} × ${detail.image_size[1]}`;
      const infoStat = document.getElementById('obb-info-status');
      if (infoStat) {
        infoStat.textContent = detail.status === 'reviewed' ? '已完成' : '待审核';
        infoStat.className = `badge badge-status ${detail.status === 'reviewed' ? 'approved' : ''}`;
      }

      const statusSmiles = document.getElementById('obb-status-smiles');
      if (statusSmiles) statusSmiles.textContent = detail.smiles || '-';

      this.updateObbBoxList(detail.boxes || []);
      this.updateObbStatusBar();

      this.showToast(`已载入 OBB 样本 ${sampleId} (共 ${detail.boxes ? detail.boxes.length : 0} 目标框)`, 'success');
    } catch (err) {
      this.showToast(`载入 OBB 样本失败: ${err.message}`, 'error');
    } finally {
      this.isSwitchingSample = false;
    }
  }

  async navigateObbSample(direction) {
    if (!this.obbSamplesList.length || this.isSwitchingSample) return;
    let nextPos = this.currentObbPos + direction;
    if (nextPos < 0) nextPos = 0;
    if (nextPos >= this.obbSamplesList.length) nextPos = this.obbSamplesList.length - 1;
    const targetId = this.obbSamplesList[nextPos].id;
    if (targetId === this.currentObbSampleId) {
      this.showToast('已到达当前筛选样本的边界', 'info', 1200);
      return;
    }
    await this.loadObbSample(targetId);
  }

  async runObbAiPredict() {
    if (!this.currentObbSampleId) return;
    const btnAi = document.getElementById('btn-obb-ai-predict');
    try {
      if (btnAi) {
        btnAi.disabled = true;
        btnAi.textContent = '⏳ AI 推理中...';
      }
      const res = await fetch(`/api/obb/predict/${this.currentObbSampleId}?conf=0.18`, { method: 'POST' });
      if (!res.ok) throw new Error('AI 预标注请求失败');
      const data = await res.json();
      const boxes = data.boxes || [];

      this.obbEditor.loadData(this.obbEditor.image, this.obbEditor.imageSize, boxes);
      this.showToast(`🤖 AI 预标注完成！检测到 ${boxes.length} 个目标框`, 'success');
    } catch (err) {
      this.showToast(`AI 预标注异常: ${err.message}`, 'error');
    } finally {
      if (btnAi) {
        btnAi.disabled = false;
        btnAi.innerHTML = '<span>🤖 AI 预标注</span>';
      }
    }
  }

  async saveObbAnnotation(gotoNext = false, silent = false) {
    if (!this.currentObbSampleId || !this.obbEditor || !this.obbEditor.image) return false;
    try {
      const boxes = this.obbEditor.getBoxes();
      const payload = {
        id: this.currentObbSampleId,
        boxes: boxes,
        image_size: this.obbEditor.imageSize,
        status: 'reviewed',
        annotator: 'human_review',
      };

      const res = await fetch('/api/obb/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error('保存请求失败');

      this.lastSavedObbSampleId = this.currentObbSampleId;

      const sInList = this.obbSamplesList.find(s => s.id === this.currentObbSampleId);
      if (sInList) {
        sInList.status = 'reviewed';
        sInList.box_count = boxes.length;
      }

      const badgeStatus = document.getElementById('badge-obb-status');
      if (badgeStatus) {
        badgeStatus.textContent = '已完成';
        badgeStatus.className = 'badge badge-status approved';
      }
      const obbInfoStatus = document.getElementById('obb-info-status');
      if (obbInfoStatus) {
        obbInfoStatus.textContent = '已完成';
        obbInfoStatus.className = 'badge badge-status approved';
      }

      if (!silent) {
        this.showToast(`✅ OBB 样本 ${this.currentObbSampleId} 标注已保存！(YOLO OBB 标签已同步导出)`, 'success');
      } else {
        this.showToast(`💾 样本 ${this.currentObbSampleId} 已自动保存`, 'info', 1500);
      }

      if (gotoNext) {
        await this.navigateObbSample(1);
      }
      return true;
    } catch (err) {
      this.showToast(`保存 OBB 标注失败: ${err.message}`, 'error');
      return false;
    }
  }

  handleObbSelectionChange(box) {
    const listItems = document.querySelectorAll('.obb-box-item');
    listItems.forEach(item => {
      item.classList.toggle('active', box && item.dataset.boxId === box.id);
      if (box && item.dataset.boxId === box.id) {
        item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    });

    if (box) {
      document.querySelectorAll('.obb-class-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.classId) === box.class_id);
      });
    }
  }

  handleObbBoxesChange(boxes) {
    this.updateObbBoxList(boxes);
    this.updateObbStatusBar();
  }

  updateObbBoxList(boxes = []) {
    const countEl = document.getElementById('obb-box-count');
    if (countEl) countEl.textContent = boxes.length;

    const container = document.getElementById('obb-box-list');
    if (!container) return;

    if (!boxes.length) {
      container.innerHTML = `<div class="empty-hint" style="padding: 16px; text-align: center; color: var(--text-dim); font-size: 12px;">当前图像暂无标注框</div>`;
      return;
    }

    let html = '';
    boxes.forEach((b, idx) => {
      const color = this.obbEditor ? this.obbEditor.classColors[b.class_id] : '#2563eb';
      const isSelected = this.obbEditor && this.obbEditor.selectedBoxId === b.id;
      const confStr = b.conf < 1.0 ? `${(b.conf * 100).toFixed(0)}%` : 'Manual';

      html += `
        <div class="obb-box-item ${isSelected ? 'active' : ''}" data-box-id="${b.id}">
          <div class="obb-box-info">
            <span class="obb-box-badge" style="background: ${color};">#${idx + 1} ${b.class_name}</span>
            <span class="obb-box-conf">${confStr}</span>
          </div>
          <button class="obb-box-del-btn" data-box-id="${b.id}" title="删除该框">&times;</button>
        </div>
      `;
    });

    container.innerHTML = html;

    // 绑定列表点击事件
    container.querySelectorAll('.obb-box-item').forEach(item => {
      item.addEventListener('click', (e) => {
        if (e.target.classList.contains('obb-box-del-btn')) return;
        const bid = item.dataset.boxId;
        if (this.obbEditor) this.obbEditor.selectBox(bid);
      });
    });

    container.querySelectorAll('.obb-box-del-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const bid = btn.dataset.boxId;
        if (this.obbEditor) {
          this.obbEditor.selectBox(bid);
          this.obbEditor.deleteSelected();
        }
      });
    });
  }

  updateObbStatusBar() {
    if (!this.obbEditor) return;
    const statsEl = document.getElementById('obb-status-stats');
    if (statsEl) {
      const boxes = this.obbEditor.getBoxes();
      const counts = {};
      boxes.forEach(b => { counts[b.class_name] = (counts[b.class_name] || 0) + 1; });
      const summary = Object.entries(counts).map(([k, v]) => `${k}: ${v}`).join(', ');
      statsEl.textContent = `总计: ${boxes.length} 框${summary ? ` (${summary})` : ''}`;
    }

    const zoomEl = document.getElementById('obb-status-zoom');
    if (zoomEl) {
      zoomEl.textContent = `缩放: ${Math.round(this.obbEditor.scale * 100)}%`;
    }
  }

  async markDiscard() {
    this.selectReviewStatus.value = 'discarded';
    if (!this.inputReviewNotes.value) {
      this.inputReviewNotes.value = '原图模糊或损坏';
    }
    await this.saveCurrentAnnotation(true);
  }

  showToast(message, type = 'info', duration = 2800) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    this.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }
}

// 启动应用
window.addEventListener('DOMContentLoaded', () => {
  window.app = new AppController();
});
