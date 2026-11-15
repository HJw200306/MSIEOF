"""一次性迁移旧分辨率分类器 pickle 的工具。

现行流程直接使用 model/classifier_fixed.pkl。只有另行取得旧 pickle 时，才需要
显式传入旧文件路径进行转换；项目不再保留或默认引用旧模型文件。
"""
import pickle
import sys
from pathlib import Path

from utils.base.paths import PROJECT_ROOT
CLASSIFIER_FIXED_PATH = PROJECT_ROOT / "model" / "classifier_fixed.pkl"


class CompatUnpickler(pickle.Unpickler):
    """自定义 Unpickler 来处理 numpy.random 兼容性"""

    def persistent_load(self, pid):
        # 处理持久化 ID
        return None

    def find_class(self, module, name):
        # 拦截 numpy.random._pickle 函数
        if module == 'numpy.random._pickle':
            if name == '__generator_ctor':
                # 返回一个假的构造函数
                def fake_ctor(*args, **kwargs):
                    import numpy as np
                    return np.random.default_rng()
                return fake_ctor
            elif name == '__bit_generator_ctor':
                def fake_ctor(*args, **kwargs):
                    import numpy as np
                    return np.random.MT19937()
                return fake_ctor

        # 拦截 Generator 和 BitGenerator 的 __reduce__ 调用
        if module == 'numpy.random._generator' and name == 'Generator':
            import numpy as np

            class FakeGenerator:
                """假的 Generator 类，忽略状态恢复"""
                def __new__(cls, *args, **kwargs):
                    return np.random.default_rng()

                def __reduce__(self):
                    import numpy as np
                    return (np.random.default_rng, ())

            return FakeGenerator

        return super().find_class(module, name)


def fix_pickle_file(input_path, output_path=None) -> bool:
    """用 CompatUnpickler 加载 pickle 并重新 dump 成当前环境可直接读的格式。

    Args:
        input_path: 原始 pickle 路径。
        output_path: 输出路径，默认在原文件名后加 _fixed。
    """
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return False

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"
    else:
        output_path = Path(output_path)

    print(f"正在读取: {input_path}")
    try:
        with open(input_path, 'rb') as f:
            data = CompatUnpickler(f).load()
        print("✅ 成功加载")
        if isinstance(data, dict):
            print(f"   键: {list(data.keys())}")
            if 'model' in data:
                print(f"   模型类型: {type(data['model']).__name__}")

        print(f"\n保存到: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        print("✅ 修复完成")
        return True
    except Exception as e:
        print(f"❌ 修复失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python -m utils.maintenance.fix_classifier_pickle <旧pickle路径> [输出路径]")
        print("现行流程已使用 model/classifier_fixed.pkl；转换旧文件时必须显式给出路径。")
        ok = False
    else:
        ok = fix_pickle_file(sys.argv[1],
                             sys.argv[2] if len(sys.argv) > 2 else None)
    sys.exit(0 if ok else 1)
