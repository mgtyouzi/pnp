import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODULE = ROOT / "models" / "discriminative_dual_proxy.py"


def test_dual_proxy_module_contract_exists():
    assert MODULE.is_file(), "dual-proxy module is missing"
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    assert "DiscriminativeDualProxy" in classes
    methods = {
        node.name
        for node in classes["DiscriminativeDualProxy"].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "forward",
        "sample_candidates",
        "compute_loss_and_cache",
        "begin_epoch",
        "maybe_refresh_prototypes",
        "finalize_epoch",
    }
    assert required.issubset(methods), sorted(required.difference(methods))


def test_dual_proxy_declares_independent_class_state():
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    registered = {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "register_buffer"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }
    required = {
        "foreground_prototypes",
        "background_prototypes",
        "foreground_queue",
        "background_queue",
        "foreground_queue_priorities",
        "background_queue_priorities",
        "prototype_ready",
        "sampling_step",
    }
    assert required.issubset(registered), sorted(required.difference(registered))
    assert 'DISCRIMINATIVE_DUAL_PROXY_VERSION' in source
    assert 'discriminative_dual_proxy_v2_20260831' in source
    assert 'F.relu(self.separation_margin - foreground_gap_tensor).mean()' in source
    assert 'dualproxy_separation_fraction' in source
    assert '"fused_logits": raw_logits' in source


def main():
    test_dual_proxy_module_contract_exists()
    test_dual_proxy_declares_independent_class_state()
    print("Discriminative dual-proxy structure tests passed")


if __name__ == "__main__":
    main()
