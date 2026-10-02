"""CPU-only regressions: no API requests, model downloads, or generated-code execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "docs/chapter22_agentic/code"


def load(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EnvironmentTests(unittest.TestCase):
    def test_default_disables_code_execution(self):
        env = load("environment").SandboxEnv()
        result = env.step("execute_code", {"code": "print('test')"})
        self.assertTrue(result["done"])
        self.assertIn("execution is disabled", result["observation"])

    def test_executor_receives_resource_limits(self):
        executor = Mock(return_value="x" * 600)
        env = load("environment").SandboxEnv(timeout=2, max_memory=123, executor=executor)
        result = env.step("execute_code", {"code": "print('test')"})
        executor.assert_called_once_with(code="print('test')", timeout=2, max_memory=123)
        self.assertEqual(result, {"observation": "x" * 500, "done": False})

    def test_executor_timeout_ends_episode(self):
        env = load("environment").SandboxEnv(executor=Mock(side_effect=TimeoutError))
        self.assertEqual(env.step("execute_code", {"code": "pass"}),
                         {"observation": "TIMEOUT", "done": True})

    def test_run_refuses_before_importing_model_dependencies(self):
        result = subprocess.run([sys.executable, str(CODE / "run.py")],
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("executor", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_deep_research_import_paths_exist(self):
        for filename in ("mini_deep_research_grpo.py", "generate_synthetic_data.py"):
            source = ROOT / "code/chapter10_agentic_rl" / filename
            self.assertIn("../../docs/chapter22_agentic/code", source.read_text())
            self.assertTrue((source.parent / "../../docs/chapter22_agentic/code/deep_research_rl_benchmark.py").is_file())


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for policy CPU regression tests")
class PolicyTests(unittest.TestCase):
    def setUp(self):
        class Batch(dict):
            def to(self, device):
                return Batch({key: value.to(device) for key, value in self.items()})

        class Tokenizer:
            def __call__(self, text, return_tensors="pt", add_special_tokens=True):
                # Model a BPE merge across the boundary between 'a' and 'b'.
                ids = [7] if text == "ab" else [ord(c) % 7 for c in text]
                return Batch(input_ids=torch.tensor([ids], dtype=torch.long))

            def decode(self, ids, skip_special_tokens=True):
                return ",".join(str(i.item()) for i in ids)

        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 8)
                self.projection = torch.nn.Linear(8, 8)
                self.last_ids = None

            @property
            def device(self):
                return self.embedding.weight.device

            def forward(self, input_ids, **kwargs):
                self.last_ids = input_ids.detach().clone()
                return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))

            def generate(self, input_ids, **kwargs):
                return torch.cat((input_ids, torch.tensor([[1, 2]])), dim=1)

        torch.manual_seed(7)
        self.model = TinyLM()
        self.policy = load("policy").Policy(self.model, Tokenizer(), lr=.01)

    def test_response_only_generation(self):
        self.assertEqual(self.policy.generate("prompt"), "1,2")

    def test_token_boundary_is_consistent(self):
        logprobs = self.policy.get_logprobs("a", "b")
        self.assertEqual(tuple(logprobs.shape), (1, 1))
        self.assertEqual(self.model.last_ids.tolist(), [[6, 0]])
        self.assertTrue(logprobs.requires_grad)
        for prompt, response in [("", "b"), ("a", "")]:
            with self.assertRaises(ValueError):
                self.policy.get_logprobs(prompt, response)

    def test_training_updates_policy_and_keeps_reference_frozen(self):
        reference = deepcopy(self.model)
        self.policy.set_ref_model(reference)
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        ref_before = {name: p.detach().clone() for name, p in reference.named_parameters()}
        loss = self.policy.train_step_with_advantage([("a", "b", 1.0), ("c", "d", -.5)])
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(any(not torch.equal(before[n], p) for n, p in self.model.named_parameters()))
        for name, p in reference.named_parameters():
            self.assertTrue(torch.equal(ref_before[name], p))
            self.assertIsNone(p.grad)


if __name__ == "__main__":
    unittest.main()
