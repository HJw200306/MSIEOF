/**
 * Chemistry Visualization & Geometric Utilities
 * 包含全量 118 种元素周期表与常用超原子/保护基/Markush 基团库
 */

const ChemUtils = {
  isDarkTheme() {
    return document.body.classList.contains('theme-dark');
  },

  // 118 种元素周期表完整数据集 (原子序数, 符号, 中文名, 英文名, 类别, 周期, 族)
  PERIODIC_TABLE: [
    { number: 1, symbol: 'H', name: '氢', en: 'Hydrogen', category: 'nonmetal', period: 1, group: 1 },
    { number: 2, symbol: 'He', name: '氦', en: 'Helium', category: 'noble_gas', period: 1, group: 18 },
    { number: 3, symbol: 'Li', name: '锂', en: 'Lithium', category: 'alkali', period: 2, group: 1 },
    { number: 4, symbol: 'Be', name: '铍', en: 'Beryllium', category: 'alkaline_earth', period: 2, group: 2 },
    { number: 5, symbol: 'B', name: '硼', en: 'Boron', category: 'metalloid', period: 2, group: 13 },
    { number: 6, symbol: 'C', name: '碳', en: 'Carbon', category: 'nonmetal', period: 2, group: 14 },
    { number: 7, symbol: 'N', name: '氮', en: 'Nitrogen', category: 'nonmetal', period: 2, group: 15 },
    { number: 8, symbol: 'O', name: '氧', en: 'Oxygen', category: 'nonmetal', period: 2, group: 16 },
    { number: 9, symbol: 'F', name: '氟', en: 'Fluorine', category: 'halogen', period: 2, group: 17 },
    { number: 10, symbol: 'Ne', name: '氖', en: 'Neon', category: 'noble_gas', period: 2, group: 18 },
    { number: 11, symbol: 'Na', name: '钠', en: 'Sodium', category: 'alkali', period: 3, group: 1 },
    { number: 12, symbol: 'Mg', name: '镁', en: 'Magnesium', category: 'alkaline_earth', period: 3, group: 2 },
    { number: 13, symbol: 'Al', name: '铝', en: 'Aluminium', category: 'post_transition', period: 3, group: 13 },
    { number: 14, symbol: 'Si', name: '硅', en: 'Silicon', category: 'metalloid', period: 3, group: 14 },
    { number: 15, symbol: 'P', name: '磷', en: 'Phosphorus', category: 'nonmetal', period: 3, group: 15 },
    { number: 16, symbol: 'S', name: '硫', en: 'Sulfur', category: 'nonmetal', period: 3, group: 16 },
    { number: 17, symbol: 'Cl', name: '氯', en: 'Chlorine', category: 'halogen', period: 3, group: 17 },
    { number: 18, symbol: 'Ar', name: '氩', en: 'Argon', category: 'noble_gas', period: 3, group: 18 },
    { number: 19, symbol: 'K', name: '钾', en: 'Potassium', category: 'alkali', period: 4, group: 1 },
    { number: 20, symbol: 'Ca', name: '钙', en: 'Calcium', category: 'alkaline_earth', period: 4, group: 2 },
    { number: 21, symbol: 'Sc', name: '钪', en: 'Scandium', category: 'transition_metal', period: 4, group: 3 },
    { number: 22, symbol: 'Ti', name: '钛', en: 'Titanium', category: 'transition_metal', period: 4, group: 4 },
    { number: 23, symbol: 'V', name: '钒', en: 'Vanadium', category: 'transition_metal', period: 4, group: 5 },
    { number: 24, symbol: 'Cr', name: '铬', en: 'Chromium', category: 'transition_metal', period: 4, group: 6 },
    { number: 25, symbol: 'Mn', name: '锰', en: 'Manganese', category: 'transition_metal', period: 4, group: 7 },
    { number: 26, symbol: 'Fe', name: '铁', en: 'Iron', category: 'transition_metal', period: 4, group: 8 },
    { number: 27, symbol: 'Co', name: '钴', en: 'Cobalt', category: 'transition_metal', period: 4, group: 9 },
    { number: 28, symbol: 'Ni', name: '镍', en: 'Nickel', category: 'transition_metal', period: 4, group: 10 },
    { number: 29, symbol: 'Cu', name: '铜', en: 'Copper', category: 'transition_metal', period: 4, group: 11 },
    { number: 30, symbol: 'Zn', name: '锌', en: 'Zinc', category: 'transition_metal', period: 4, group: 12 },
    { number: 31, symbol: 'Ga', name: '镓', en: 'Gallium', category: 'post_transition', period: 4, group: 13 },
    { number: 32, symbol: 'Ge', name: '锗', en: 'Germanium', category: 'metalloid', period: 4, group: 14 },
    { number: 33, symbol: 'As', name: '砷', en: 'Arsenic', category: 'metalloid', period: 4, group: 15 },
    { number: 34, symbol: 'Se', name: '硒', en: 'Selenium', category: 'nonmetal', period: 4, group: 16 },
    { number: 35, symbol: 'Br', name: '溴', en: 'Bromine', category: 'halogen', period: 4, group: 17 },
    { number: 36, symbol: 'Kr', name: '氪', en: 'Krypton', category: 'noble_gas', period: 4, group: 18 },
    { number: 37, symbol: 'Rb', name: '铷', en: 'Rubidium', category: 'alkali', period: 5, group: 1 },
    { number: 38, symbol: 'Sr', name: '锶', en: 'Strontium', category: 'alkaline_earth', period: 5, group: 2 },
    { number: 39, symbol: 'Y', name: '钇', en: 'Yttrium', category: 'transition_metal', period: 5, group: 3 },
    { number: 40, symbol: 'Zr', name: '锆', en: 'Zirconium', category: 'transition_metal', period: 5, group: 4 },
    { number: 41, symbol: 'Nb', name: '铌', en: 'Niobium', category: 'transition_metal', period: 5, group: 5 },
    { number: 42, symbol: 'Mo', name: '钼', en: 'Molybdenum', category: 'transition_metal', period: 5, group: 6 },
    { number: 43, symbol: 'Tc', name: '锝', en: 'Technetium', category: 'transition_metal', period: 5, group: 7 },
    { number: 44, symbol: 'Ru', name: '钌', en: 'Ruthenium', category: 'transition_metal', period: 5, group: 8 },
    { number: 45, symbol: 'Rh', name: '铑', en: 'Rhodium', category: 'transition_metal', period: 5, group: 9 },
    { number: 46, symbol: 'Pd', name: '钯', en: 'Palladium', category: 'transition_metal', period: 5, group: 10 },
    { number: 47, symbol: 'Ag', name: '银', en: 'Silver', category: 'transition_metal', period: 5, group: 11 },
    { number: 48, symbol: 'Cd', name: '镉', en: 'Cadmium', category: 'transition_metal', period: 5, group: 12 },
    { number: 49, symbol: 'In', name: '铟', en: 'Indium', category: 'post_transition', period: 5, group: 13 },
    { number: 50, symbol: 'Sn', name: '锡', en: 'Tin', category: 'post_transition', period: 5, group: 14 },
    { number: 51, symbol: 'Sb', name: '锑', en: 'Antimony', category: 'metalloid', period: 5, group: 15 },
    { number: 52, symbol: 'Te', name: '碲', en: 'Tellurium', category: 'metalloid', period: 5, group: 16 },
    { number: 53, symbol: 'I', name: '碘', en: 'Iodine', category: 'halogen', period: 5, group: 17 },
    { number: 54, symbol: 'Xe', name: '氙', en: 'Xenon', category: 'noble_gas', period: 5, group: 18 },
    { number: 55, symbol: 'Cs', name: '铯', en: 'Caesium', category: 'alkali', period: 6, group: 1 },
    { number: 56, symbol: 'Ba', name: '钡', en: 'Barium', category: 'alkaline_earth', period: 6, group: 2 },
    { number: 57, symbol: 'La', name: '镧', en: 'Lanthanum', category: 'lanthanide', period: 6, group: 3 },
    { number: 58, symbol: 'Ce', name: '铈', en: 'Cerium', category: 'lanthanide', period: 6, group: 3 },
    { number: 59, symbol: 'Pr', name: '镨', en: 'Praseodymium', category: 'lanthanide', period: 6, group: 3 },
    { number: 60, symbol: 'Nd', name: '钕', en: 'Neodymium', category: 'lanthanide', period: 6, group: 3 },
    { number: 61, symbol: 'Pm', name: ' promethium', en: 'Promethium', category: 'lanthanide', period: 6, group: 3 },
    { number: 62, symbol: 'Sm', name: '钐', en: 'Samarium', category: 'lanthanide', period: 6, group: 3 },
    { number: 63, symbol: 'Eu', name: '铕', en: 'Europium', category: 'lanthanide', period: 6, group: 3 },
    { number: 64, symbol: 'Gd', name: ' Gadolinium', en: 'Gadolinium', category: 'lanthanide', period: 6, group: 3 },
    { number: 65, symbol: 'Tb', name: '铽', en: 'Terbium', category: 'lanthanide', period: 6, group: 3 },
    { number: 66, symbol: 'Dy', name: ' dysprosium', en: 'Dysprosium', category: 'lanthanide', period: 6, group: 3 },
    { number: 67, symbol: 'Ho', name: '钬', en: 'Holmium', category: 'lanthanide', period: 6, group: 3 },
    { number: 68, symbol: 'Er', name: '铒', en: 'Erbium', category: 'lanthanide', period: 6, group: 3 },
    { number: 69, symbol: 'Tm', name: '铥', en: 'Thulium', category: 'lanthanide', period: 6, group: 3 },
    { number: 70, symbol: 'Yb', name: '镱', en: 'Ytterbium', category: 'lanthanide', period: 6, group: 3 },
    { number: 71, symbol: 'Lu', name: '镥', en: 'Lutetium', category: 'lanthanide', period: 6, group: 3 },
    { number: 72, symbol: 'Hf', name: '铪', en: 'Hafnium', category: 'transition_metal', period: 6, group: 4 },
    { number: 73, symbol: 'Ta', name: '钽', en: 'Tantalum', category: 'transition_metal', period: 6, group: 5 },
    { number: 74, symbol: 'W', name: '钨', en: 'Tungsten', category: 'transition_metal', period: 6, group: 6 },
    { number: 75, symbol: 'Re', name: '铼', en: 'Rhenium', category: 'transition_metal', period: 6, group: 7 },
    { number: 76, symbol: 'Os', name: '锇', en: 'Osmium', category: 'transition_metal', period: 6, group: 8 },
    { number: 77, symbol: 'Ir', name: '铱', en: 'Iridium', category: 'transition_metal', period: 6, group: 9 },
    { number: 78, symbol: 'Pt', name: '铂', en: 'Platinum', category: 'transition_metal', period: 6, group: 10 },
    { number: 79, symbol: 'Au', name: '金', en: 'Gold', category: 'transition_metal', period: 6, group: 11 },
    { number: 80, symbol: 'Hg', name: '汞', en: 'Mercury', category: 'transition_metal', period: 6, group: 12 },
    { number: 81, symbol: 'Tl', name: '铊', en: 'Thallium', category: 'post_transition', period: 6, group: 13 },
    { number: 82, symbol: 'Pb', name: '铅', en: 'Lead', category: 'post_transition', period: 6, group: 14 },
    { number: 83, symbol: 'Bi', name: '铋', en: 'Bismuth', category: 'post_transition', period: 6, group: 15 },
    { number: 84, symbol: 'Po', name: '钋', en: 'Polonium', category: 'post_transition', period: 6, group: 16 },
    { number: 85, symbol: 'At', name: '砹', en: 'Astatine', category: 'halogen', period: 6, group: 17 },
    { number: 86, symbol: 'Rn', name: '氡', en: 'Radon', category: 'noble_gas', period: 6, group: 18 },
    { number: 87, symbol: 'Fr', name: '钫', en: 'Francium', category: 'alkali', period: 7, group: 1 },
    { number: 88, symbol: 'Ra', name: '镭', en: 'Radium', category: 'alkaline_earth', period: 7, group: 2 },
    { number: 89, symbol: 'Ac', name: '锕', en: 'Actinium', category: 'actinide', period: 7, group: 3 },
    { number: 90, symbol: 'Th', name: ' thorium', en: 'Thorium', category: 'actinide', period: 7, group: 3 },
    { number: 91, symbol: 'Pa', name: ' protactinium', en: 'Protactinium', category: 'actinide', period: 7, group: 3 },
    { number: 92, symbol: 'U', name: '铀', en: 'Uranium', category: 'actinide', period: 7, group: 3 },
    { number: 93, symbol: 'Np', name: ' neptunium', en: 'Neptunium', category: 'actinide', period: 7, group: 3 },
    { number: 94, symbol: 'Pu', name: '钚', en: 'Plutonium', category: 'actinide', period: 7, group: 3 },
    { number: 95, symbol: 'Am', name: ' americium', en: 'Americium', category: 'actinide', period: 7, group: 3 },
    { number: 96, symbol: 'Cm', name: ' curium', en: 'Curium', category: 'actinide', period: 7, group: 3 },
    { number: 97, symbol: 'Bk', name: ' berkelium', en: 'Berkelium', category: 'actinide', period: 7, group: 3 },
    { number: 98, symbol: 'Cf', name: ' californium', en: 'Californium', category: 'actinide', period: 7, group: 3 },
    { number: 99, symbol: 'Es', name: ' einsteinium', en: 'Einsteinium', category: 'actinide', period: 7, group: 3 },
    { number: 100, symbol: 'Fm', name: ' fermium', en: 'Fermium', category: 'actinide', period: 7, group: 3 },
    { number: 101, symbol: 'Md', name: ' mendelevium', en: 'Mendelevium', category: 'actinide', period: 7, group: 3 },
    { number: 102, symbol: 'No', name: ' nobelium', en: 'Nobelium', category: 'actinide', period: 7, group: 3 },
    { number: 103, symbol: 'Lr', name: ' lawrencium', en: 'Lawrencium', category: 'actinide', period: 7, group: 3 },
    { number: 104, symbol: 'Rf', name: ' Rutherfordium', en: 'Rutherfordium', category: 'transition_metal', period: 7, group: 4 },
    { number: 105, symbol: 'Db', name: ' dubnium', en: 'Dubnium', category: 'transition_metal', period: 7, group: 5 },
    { number: 106, symbol: 'Sg', name: ' seaborgium', en: 'Seaborgium', category: 'transition_metal', period: 7, group: 6 },
    { number: 107, symbol: 'Bh', name: ' bohrium', en: 'Bohrium', category: 'transition_metal', period: 7, group: 7 },
    { number: 108, symbol: 'Hs', name: ' hassium', en: 'Hassium', category: 'transition_metal', period: 7, group: 8 },
    { number: 109, symbol: 'Mt', name: ' meitnerium', en: 'Meitnerium', category: 'transition_metal', period: 7, group: 9 },
    { number: 110, symbol: 'Ds', name: ' darmstadtium', en: 'Darmstadtium', category: 'transition_metal', period: 7, group: 10 },
    { number: 111, symbol: 'Rg', name: ' roentgenium', en: 'Roentgenium', category: 'transition_metal', period: 7, group: 11 },
    { number: 112, symbol: 'Cn', name: ' copernicium', en: 'Copernicium', category: 'transition_metal', period: 7, group: 12 },
    { number: 113, symbol: 'Nh', name: ' nihonium', en: 'Nihonium', category: 'post_transition', period: 7, group: 13 },
    { number: 114, symbol: 'Fl', name: ' flerovium', en: 'Flerovium', category: 'post_transition', period: 7, group: 14 },
    { number: 115, symbol: 'Mc', name: ' moscovium', en: 'Moscovium', category: 'post_transition', period: 7, group: 15 },
    { number: 116, symbol: 'Lv', name: ' livermorium', en: 'Livermorium', category: 'post_transition', period: 7, group: 16 },
    { number: 117, symbol: 'Ts', name: ' tennessine', en: 'Tennessine', category: 'halogen', period: 7, group: 17 },
    { number: 118, symbol: 'Og', name: ' oganesson', en: 'Oganesson', category: 'noble_gas', period: 7, group: 18 },
  ],

  // 常用保护基、超原子与 Markush 基团分类库
  COMMON_SUPERATOMS: {
    '烷基与芳基 (Alkyl/Aryl)': [
      { symbol: 'Me', name: '甲基 (Methyl)' },
      { symbol: 'Et', name: '乙基 (Ethyl)' },
      { symbol: 'nPr', name: '正丙基 (n-Propyl)' },
      { symbol: 'iPr', name: '异丙基 (Isopropyl)' },
      { symbol: 'cPr', name: '环丙基 (Cyclopropyl)' },
      { symbol: 'nBu', name: '正丁基 (n-Butyl)' },
      { symbol: 'iBu', name: '异丁基 (Isobutyl)' },
      { symbol: 'tBu', name: '叔丁基 (tert-Butyl)' },
      { symbol: 'cHex', name: '环己基 (Cyclohexyl)' },
      { symbol: 'Ph', name: '苯基 (Phenyl)' },
      { symbol: 'Bn', name: '苄基 (Benzyl)' },
      { symbol: 'Tol', name: '甲苯基 (Tolyl)' },
      { symbol: 'Mes', name: '均三甲苯基 (Mesityl)' },
      { symbol: 'Naph', name: '萘基 (Naphthyl)' },
    ],
    '保护基 (Protecting Groups)': [
      { symbol: 'Boc', name: '叔丁氧羰基 (tert-Butoxycarbonyl)' },
      { symbol: 'Fmoc', name: '芴甲氧羰基 (Fluorenylmethyloxycarbonyl)' },
      { symbol: 'Cbz', name: '苄氧羰基 (Carboxybenzyl)' },
      { symbol: 'Ac', name: '乙酰基 (Acetyl)' },
      { symbol: 'Bz', name: '苯甲酰基 (Benzoyl)' },
      { symbol: 'Piv', name: '特戊酰基 (Pivaloyl)' },
      { symbol: 'Ts', name: '对甲苯磺酰基 (Tosyl)' },
      { symbol: 'Ms', name: '甲磺酰基 (Mesyl)' },
      { symbol: 'Tf', name: '三氟甲磺酰基 (Triflyl)' },
      { symbol: 'TBDMS', name: '叔丁基二甲基硅基 (TBS)' },
      { symbol: 'TMS', name: '三甲基硅基 (Trimethylsilyl)' },
      { symbol: 'TIPS', name: '三异丙基硅基 (Triisopropylsilyl)' },
      { symbol: 'THP', name: '四氢吡喃基 (Tetrahydropyranyl)' },
      { symbol: 'Tr', name: '三苯甲基 (Trityl)' },
      { symbol: 'PMB', name: '对甲氧基苄基 (p-Methoxybenzyl)' },
    ],
    '官能团 (Functional Groups)': [
      { symbol: 'OMe', name: '甲氧基 (Methoxy)' },
      { symbol: 'OEt', name: '乙氧基 (Ethoxy)' },
      { symbol: 'OPh', name: '苯氧基 (Phenoxy)' },
      { symbol: 'SMe', name: '甲硫基 (Methylthio)' },
      { symbol: 'NH2', name: '氨基 (Amino)' },
      { symbol: 'NMe2', name: '二甲氨基 (Dimethylamino)' },
      { symbol: 'NO2', name: '硝基 (Nitro)' },
      { symbol: 'SO2', name: '磺酰基 (Sulfonyl)' },
      { symbol: 'SO3H', name: '磺酸基 (Sulfonic acid)' },
      { symbol: 'COOH', name: '羧基 (Carboxyl)' },
      { symbol: 'COOMe', name: '甲酯基 (Methyl ester)' },
      { symbol: 'COOEt', name: '乙酯基 (Ethyl ester)' },
      { symbol: 'CONH2', name: '氨基甲酰基 (Carbamoyl)' },
      { symbol: 'CHO', name: '醛基 (Formyl)' },
      { symbol: 'CN', name: '氰基 (Cyano)' },
      { symbol: 'CF3', name: '三氟甲基 (Trifluoromethyl)' },
      { symbol: 'OCF3', name: '三氟甲氧基 (Trifluoromethoxy)' },
      { symbol: 'SF5', name: '五氟硫基 (Pentafluorosulfanyl)' },
    ],
    'Markush 变量基团 (Markush & R-Groups)': [
      { symbol: '*', name: '连接点 / 开放端 (*)' },
      { symbol: '_AP', name: '附着点基团 (_AP)' },
      { symbol: 'R', name: '通用取代基 (R)' },
      { symbol: 'R1', name: '取代基 1 (R1)' },
      { symbol: 'R2', name: '取代基 2 (R2)' },
      { symbol: 'R3', name: '取代基 3 (R3)' },
      { symbol: 'R4', name: '取代基 4 (R4)' },
      { symbol: 'R5', name: '取代基 5 (R5)' },
      { symbol: 'R6', name: '取代基 6 (R6)' },
      { symbol: 'R7', name: '取代基 7 (R7)' },
      { symbol: 'R8', name: '取代基 8 (R8)' },
      { symbol: 'X', name: '可变杂原子/卤素 (X)' },
      { symbol: 'Y', name: '可变连接基团 (Y)' },
      { symbol: 'Z', name: '可变杂原子 (Z)' },
      { symbol: 'Ar', name: '可变芳基 (Aryl)' },
      { symbol: 'Het', name: '可变杂环基 (Heteroaryl)' },
      { symbol: 'Hal', name: '通用卤素 (Halogen)' },
    ]
  },

  ELEMENT_COLORS: {
    C: { light: '#1e293b', dark: '#94a3b8' },
    N: { light: '#2563eb', dark: '#3b82f6' },
    O: { light: '#dc2626', dark: '#ef4444' },
    S: { light: '#d97706', dark: '#eab308' },
    P: { light: '#ea580c', dark: '#f97316' },
    F: { light: '#16a34a', dark: '#22c55e' },
    Cl: { light: '#059669', dark: '#10b981' },
    Br: { light: '#0d9488', dark: '#14b8a6' },
    I: { light: '#0284c7', dark: '#06b6d4' },
    H: { light: '#64748b', dark: '#cbd5e1' },
    B: { light: '#9333ea', dark: '#d946ef' },
    Si: { light: '#65a30d', dark: '#84cc16' },
    Na: { light: '#7c3aed', dark: '#a855f7' },
    K: { light: '#7c3aed', dark: '#a855f7' },
    Fe: { light: '#ea580c', dark: '#f97316' },
    Cu: { light: '#b45309', dark: '#d97706' },
    Zn: { light: '#475569', dark: '#94a3b8' },
    Pd: { light: '#0284c7', dark: '#38bdf8' },
    Pt: { light: '#0f766e', dark: '#14b8a6' },
    '*': { light: '#ea580c', dark: '#f97316' },
    _AP: { light: '#ea580c', dark: '#f97316' },
    R1: { light: '#db2777', dark: '#ec4899' },
    R2: { light: '#c026d3', dark: '#d946ef' },
    R3: { light: '#9333ea', dark: '#a855f7' },
    X: { light: '#7c3aed', dark: '#8b5cf6' },
  },

  BOND_NAMES: {
    1: '单键 (Single)',
    2: '双键 (Double)',
    3: '三键 (Triple)',
    4: '芳香键 (Aromatic)',
    5: '实心楔形键 (Wedge)',
    6: '虚线楔形键 (Hash Wedge)',
    7: '配位/波浪键 (Coordinate/Wave)',
  },

  STANDARD_ELEMENTS: new Set([
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Th', 'Pa', 'U', 'Np', 'Pu'
  ]),

  FORCED_SUPERATOMS: new Set([
    'Ac', 'Ts', 'Ms', 'Tf', 'Ph', 'Bn', 'Bz', 'Tol', 'Mes', 'Naph',
    'Boc', 'Fmoc', 'Cbz', 'Piv', 'TBDMS', 'TMS', 'TIPS', 'THP', 'Tr', 'PMB',
    'Me', 'Et', 'Pr', 'iPr', 'nPr', 'cPr', 'Bu', 'iBu', 'nBu', 'tBu', 'cHex',
    'OMe', 'OEt', 'OPh', 'SMe', 'NH2', 'NMe2', 'NO2', 'SO2', 'SO3H', 'COOH',
    'CO2H', 'COOMe', 'COOEt', 'CONH2', 'CHO', 'CN', 'CF3', 'OCF3', 'SF5',
    'CH3', 'H3C', 'CH2', 'C2H5', 'CH2CH3', 'HO3S', 'SO2NH2', 'O2S', 'SOOO',
    'R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
    'X', 'Y', 'Z', 'Ar', 'Het', 'Hal', '_AP', '*'
  ]),

  isStandardPeriodicElement(sym) {
    if (!sym || typeof sym !== 'string') return false;
    const clean = sym.trim();
    if (this.FORCED_SUPERATOMS.has(clean)) return false;
    return this.STANDARD_ELEMENTS.has(clean) || this.STANDARD_ELEMENTS.has(clean.toUpperCase());
  },

  isSuperatom(sym) {
    if (!sym || typeof sym !== 'string') return false;
    const clean = sym.trim();
    if (clean === '*' || clean === '_AP') return true;
    if (this.FORCED_SUPERATOMS.has(clean)) return true;
    if (/^R\d*$/i.test(clean)) return true;
    if (/^\d+\*$/.test(clean)) return true;
    if (/\d/.test(clean)) return true; // 含数字 (如 SO2, SO3H, CF3, CH3)
    if (clean.length > 2) return true;
    return !this.isStandardPeriodicElement(clean);
  },

  normalizeAtom(atom) {
    if (!atom) return atom;
    const rawSym = String(atom.symbol || 'C').trim();
    let rawLab = atom.label;
    if (typeof rawLab === 'string') rawLab = rawLab.trim();
    if (!rawLab) rawLab = null;

    if (rawLab) {
      atom.symbol = '*';
      atom.label = rawLab;
      return atom;
    }

    if (rawSym === '*' || rawSym === '_AP') {
      atom.symbol = '*';
      atom.label = '_AP';
      return atom;
    }

    if (this.isSuperatom(rawSym)) {
      let normLab = rawSym;
      if (/^\d+\*$/.test(rawSym)) {
        normLab = 'R' + rawSym.slice(0, -1);
      }
      atom.symbol = '*';
      atom.label = normLab;
    } else {
      atom.symbol = rawSym.length <= 2 ? (rawSym.charAt(0).toUpperCase() + rawSym.slice(1).toLowerCase()) : rawSym;
      atom.label = null;
    }
    return atom;
  },

  getPeriodicElement(sym) {
    if (!sym || typeof sym !== 'string') return null;
    const clean = sym.trim();
    return this.PERIODIC_TABLE.find(e => e.symbol.toLowerCase() === clean.toLowerCase()) || null;
  },

  getAtomColor(symbol, label) {
    const isDark = this.isDarkTheme();
    if (label && (label.startsWith('R') || label.startsWith('r'))) {
      return isDark ? '#ec4899' : '#db2777';
    }
    if (label && label === '_AP') {
      return isDark ? '#f97316' : '#ea580c';
    }
    if (symbol === '*' || symbol === '_AP') {
      return isDark ? '#f97316' : '#ea580c';
    }
    if (label && label.length > 0) {
      return isDark ? '#a855f7' : '#7c3aed';
    }
    const colorPair = this.ELEMENT_COLORS[symbol];
    if (colorPair) {
      return isDark ? colorPair.dark : colorPair.light;
    }
    // 检查元素类别颜色
    const pElem = this.getPeriodicElement(symbol);
    if (pElem) {
      const catColors = {
        alkali: isDark ? '#c084fc' : '#9333ea',
        alkaline_earth: isDark ? '#a855f7' : '#7c3aed',
        transition_metal: isDark ? '#f59e0b' : '#d97706',
        post_transition: isDark ? '#38bdf8' : '#0284c7',
        metalloid: isDark ? '#4ade80' : '#16a34a',
        nonmetal: isDark ? '#94a3b8' : '#334155',
        halogen: isDark ? '#34d399' : '#059669',
        noble_gas: isDark ? '#818cf8' : '#4f46e5',
        lanthanide: isDark ? '#fb7185' : '#e11d48',
        actinide: isDark ? '#f43f5e' : '#be123c',
      };
      return catColors[pElem.category] || (isDark ? '#38bdf8' : '#0284c7');
    }
    return isDark ? '#38bdf8' : '#0284c7';
  },

  BOND_TYPES: {
    1: { name: '单键 (Single)', code: 'single', light: '#10b981', dark: '#34d399', width: 2.2 },        // 生机翡翠绿
    2: { name: '双键 (Double)', code: 'double', light: '#0284c7', dark: '#38bdf8', width: 2.2 },        // 蔚蓝/科技蓝
    3: { name: '三键 (Triple)', code: 'triple', light: '#ea580c', dark: '#fb923c', width: 2.0 },        // 活力亮橙
    4: { name: '芳香键 (Aromatic)', code: 'aromatic', light: '#7c3aed', dark: '#c084fc', width: 2.2 },  // 雅致紫罗兰
    5: { name: '楔形实键 (Wedge)', code: 'wedge', light: '#4f46e5', dark: '#818cf8', width: 2.5 },      // 极光靛蓝
    6: { name: '楔形虚键 (Hash)', code: 'dash-wedge', light: '#e11d48', dark: '#fb7185', width: 2.5 },  // 绯红/玫瑰粉
    7: { name: '波浪键 (Wave)', code: 'wave', light: '#d946ef', dark: '#e879f9', width: 2.2 },          // 品红/波浪
  },

  getBondColor(bType = 1, isSelected = false, isHovered = false) {
    const isDark = this.isDarkTheme();
    if (isSelected) return isDark ? '#60a5fa' : '#2563eb';
    if (isHovered) return isDark ? '#93c5fd' : '#3b82f6';
    const cfg = this.BOND_TYPES[bType] || this.BOND_TYPES[1];
    return isDark ? cfg.dark : cfg.light;
  },

  getBondName(bType = 1) {
    const cfg = this.BOND_TYPES[bType] || this.BOND_TYPES[1];
    return cfg.name;
  },

  dist(p1, p2) {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    return Math.hypot(dx, dy);
  },

  distToSegment(p, a, b) {
    const l2 = (b.x - a.x) ** 2 + (b.y - a.y) ** 2;
    if (l2 === 0) return this.dist(p, a);
    let t = ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / l2;
    t = Math.max(0, Math.min(1, t));
    return this.dist(p, {
      x: a.x + t * (b.x - a.x),
      y: a.y + t * (b.y - a.y),
    });
  },

  calcBoundingBox(points) {
    if (!points || points.length === 0) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    return {
      minX,
      minY,
      maxX,
      maxY,
      cx: (minX + maxX) / 2,
      cy: (minY + maxY) / 2,
      width: Math.max(maxX - minX, 10),
      height: Math.max(maxY - minY, 10),
    };
  }
};

// 全局函数便捷导出
function getAtomColor(symbol, label) {
  return ChemUtils.getAtomColor(symbol, label);
}

function getBondColor(bondType, isSelected, isHovered) {
  if (typeof bondType === 'boolean') {
    return ChemUtils.getBondColor(bondType, isSelected);
  }
  return ChemUtils.getBondColor(isSelected, isHovered);
}

if (typeof window !== 'undefined') {
  window.ChemUtils = ChemUtils;
  window.getAtomColor = getAtomColor;
  window.getBondColor = getBondColor;
}
