# policy.py
import torch
import torch.nn.functional as F


class Policy:
    """语言模型策略：生成回答，并按组内优势更新回答 token 的概率。"""

    def __init__(self, model, tokenizer, lr=1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.ref_model = None

    def set_ref_model(self, ref_model):
        """保存冻结的参考模型，用于 KL 正则项。"""
        self.ref_model = ref_model
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=128) -> str:
        """只返回新生成的文本，避免把输入 prompt 解析成 Agent 动作。"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
        prompt_length = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(outputs[0, prompt_length:], skip_special_tokens=True)

    def _model_logprobs(self, model, prompt: str, response: str) -> torch.Tensor:
        # 分别编码后拼接，使 BPE 不会跨 prompt/response 边界合并 token。
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        response_ids = self.tokenizer(
            response, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        if prompt_ids.shape[1] == 0 or response_ids.shape[1] == 0:
            raise ValueError("prompt and response must each contain at least one token")
        input_ids = torch.cat((prompt_ids, response_ids), dim=1).to(model.device)
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
        prompt_len = prompt_ids.shape[1]
        response_logits = logits[:, prompt_len - 1:-1, :]
        response_ids = input_ids[:, prompt_len:]
        return F.log_softmax(response_logits, dim=-1).gather(
            2, response_ids.unsqueeze(-1)
        ).squeeze(-1)

    def get_logprobs(self, prompt: str, response: str) -> torch.Tensor:
        """保留梯度，以便 train_step 更新策略。"""
        return self._model_logprobs(self.model, prompt, response)

    @torch.no_grad()
    def _get_ref_logprobs(self, prompt: str, response: str) -> torch.Tensor:
        return self._model_logprobs(self.ref_model, prompt, response)

    def train_step_with_advantage(self, trajectories: list):
        """按 (prompt, response, advantage) 样本进行策略梯度更新。"""
        if not trajectories:
            raise ValueError("trajectories must not be empty")
        losses = []
        for prompt, response, advantage in trajectories:
            new_logprobs = self.get_logprobs(prompt, response)
            if self.ref_model is not None:
                ref_logprobs = self._get_ref_logprobs(prompt, response)
                log_ratio = ref_logprobs - new_logprobs
                kl = (log_ratio.exp() - log_ratio - 1).sum()
            else:
                kl = 0.0
            pg_loss = -(new_logprobs.sum() * advantage)
            losses.append(pg_loss + 0.1 * kl)

        total_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
