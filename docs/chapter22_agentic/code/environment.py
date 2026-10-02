"""Agent 环境接口：代码执行必须由调用方提供隔离后端。"""


class SandboxEnv:
    """将代码交给隔离后端；默认禁止在宿主机执行模型输出。

    executor 必须是调用方配置的隔离执行函数，接收 code、timeout 和
    max_memory 关键字参数并返回文本。后端必须自行限制文件系统、网络、
    进程、内存和运行时间；普通 subprocess 不构成安全隔离。
    """

    def __init__(self, timeout=10, max_memory=256 * 1024 * 1024, *, executor=None):
        self.timeout = timeout
        self.max_memory = max_memory
        self.executor = executor

    def step(self, action_type: str, action_args: dict) -> dict:
        if action_type == "execute_code":
            return self._exec_code(action_args["code"])
        if action_type == "finish":
            return {"observation": "", "done": True}
        return {"observation": f"Unknown action: {action_type}", "done": False}

    def _exec_code(self, code: str) -> dict:
        if self.executor is None:
            return {
                "observation": "ERROR: isolated code executor is not configured; execution is disabled",
                "done": True,
            }
        try:
            output = self.executor(code=code, timeout=self.timeout, max_memory=self.max_memory)
            return {"observation": str(output)[-500:], "done": False}
        except TimeoutError:
            return {"observation": "TIMEOUT", "done": True}
        except Exception as error:
            return {"observation": f"ERROR: {error}", "done": True}

    def reset(self):
        """本接口不保存回合状态；隔离后端应为每次执行创建独立环境。"""
