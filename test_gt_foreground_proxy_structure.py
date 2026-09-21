import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODULE = ROOT / "models" / "gt_foreground_proxy.py"


def test_module_contract_exists():
    assert MODULE.is_file(), "GT foreground proxy module is missing"
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    assert "GTForegroundProxy" in classes
    methods = {
        node.name
        for node in classes["GTForegroundProxy"].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "forward",
        "compute_loss_and_cache",
        "begin_epoch",
        "maybe_refresh_prototypes",
        "finalize_epoch",
    }
    assert required.issubset(methods), sorted(required.difference(methods))


def test_module_declares_foreground_only_state():
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
    assert "prototypes" in registered
    assert "prototype_ready" in registered
    assert "support_queue_priorities" in registered
    assert "support_seen_count" in registered
    assert "bg_prototypes" not in source
    assert "fused_logits" in source


def main():
    test_module_contract_exists()
    test_module_declares_foreground_only_state()
    print("GT foreground proxy structure tests passed")


if __name__ == "__main__":
    main()
