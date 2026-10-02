# run.py
from environment import SandboxEnv

# 在下载模型之前检查隔离执行后端。将 executor 替换为已配置的隔离服务。
executor = None
if executor is None:
    raise SystemExit("请先配置隔离执行后端 executor；本示例不会在宿主机执行模型生成的代码。")

from transformers import AutoModelForCausalLM, AutoTokenizer

from policy import Policy
from trainer import GRPOAgentTrainer

# 加载一个小模型
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 初始化各组件
env = SandboxEnv(timeout=10, executor=executor)
policy = Policy(model, tokenizer, lr=5e-5)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)
policy.set_ref_model(ref_model)


# 定义 reward：代码执行结果是否正确
def code_reward(trajectory):
    """如果最终答案包含正确的执行结果，reward = 1，否则 = 0。"""
    for interaction in trajectory["interactions"]:
        obs = interaction.get("observation", "")
        if obs and "ERROR" not in obs and "TIMEOUT" not in obs:
            return 1.0
    return 0.0


# 训练 prompts
prompts = [
    "写一段 Python 代码计算斐波那契数列的第 10 项并输出结果。",
    "写一段代码检查字符串是否是回文。",
    "写一段代码对列表进行冒泡排序。",
]

# 开始训练
trainer = GRPOAgentTrainer(
    policy=policy,
    env=env,
    reward_fn=code_reward,
    group_size=4,
    max_turns=3,
)
history = trainer.fit(prompts, n_steps=30)
