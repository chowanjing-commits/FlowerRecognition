# 手动指定 yolopy39 环境的 site-packages 路径（请根据你的实际路径修改！）
# 你的路径通常是：D:\Anaconda3\envs\yolopy39\lib\site-packages
import sys
sys.path.insert(0, "D:\\Anaconda3\\envs\\yolopy39\\lib\\site-packages")

# 检查 supervision 版本和类
import supervision
print("1. supervision 版本:", supervision.__version__)
print("2. 安装路径:", supervision.__file__)
print("3. 是否有 BoundingBoxAnnotator:", hasattr(supervision, "BoundingBoxAnnotator"))

# 尝试直接导入
try:
    from supervision import BoundingBoxAnnotator
    print("4. ✅ 成功导入 BoundingBoxAnnotator")
except Exception as e:
    print(f"4. ❌ 导入失败: {e}")