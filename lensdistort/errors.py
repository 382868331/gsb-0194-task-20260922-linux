"""错误类型：所有本库抛出的异常都可定位到具体参数。

公开错误语义：
- InvalidInputError: 输入契约违反（NaN/Inf、bool 冒充数值、点数超限、
  非法搜索盒、非法预算参数等）。消息中必含出错的参数名。
  继承 ValueError，便于调用方按标准语义捕获。
- 算法搜索预算耗尽不是异常：undistort 返回 status="exhausted"，
  与 "proven_no_solution"（径向证书证明无解）严格分开。
"""


class LensDistortError(Exception):
    """本库所有异常的公共基类。"""


class InvalidInputError(LensDistortError, ValueError):
    """输入违反契约：NaN/Inf、类型错误、越界、点数超限等。"""
