import ast
import re
import unittest
from pathlib import Path
from typing import Iterable, List, Set, Tuple


_CORE_MODULE_PATHS_V129 = [
    "atos_core/arc_solver_v129.py",
    "atos_core/arc_inverse_v129.py",
    "atos_core/arc_selector_v129.py",
    "atos_core/arc_delta_v129.py",
    "atos_core/arc_dsl_v129.py",
]


def _iter_name_nodes(tree: ast.AST) -> Iterable[ast.Name]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node


def _collect_names(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.add(str(n.id))
    return names


def _find_forbidden_task_id_in_conditions(tree: ast.AST) -> List[Tuple[int, str]]:
    violations: List[Tuple[int, str]] = []
    conditional_nodes: List[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            conditional_nodes.append(node.test)
        elif isinstance(node, ast.IfExp):
            conditional_nodes.append(node.test)
        elif isinstance(node, ast.While):
            conditional_nodes.append(node.test)
        elif isinstance(node, ast.Assert):
            if node.test is not None:
                conditional_nodes.append(node.test)

    for test in conditional_nodes:
        names = _collect_names(test)
        for bad in ("task_id", "arc_task_id", "taskid"):
            if bad in names:
                lineno = int(getattr(test, "lineno", 0) or 0)
                violations.append((lineno, bad))
    violations.sort(key=lambda t: (int(t[0]), str(t[1])))
    return violations


def _find_hardcoded_arc_task_ids_in_literals(tree: ast.AST) -> List[Tuple[int, str]]:
    # ARC tasks are commonly addressed by an 8-hex filename, e.g. "00d62c1b" or "00d62c1b.json".
    # Flag any string literal that looks like that inside core solver modules.
    pat = re.compile(r"^[0-9a-f]{8}(?:\\.json)?$")
    found: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value.strip()
            if pat.match(s):
                lineno = int(getattr(node, "lineno", 0) or 0)
                found.append((lineno, s))
    found.sort(key=lambda t: (int(t[0]), str(t[1])))
    return found


def _find_forbidden_imports(tree: ast.AST) -> List[Tuple[int, str]]:
    forbidden_roots = {"os", "pathlib", "glob"}
    found: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = str(alias.name).split(".")[0]
                if root in forbidden_roots:
                    found.append((int(getattr(node, "lineno", 0) or 0), root))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = str(node.module).split(".")[0]
            if root in forbidden_roots:
                found.append((int(getattr(node, "lineno", 0) or 0), root))
    found.sort(key=lambda t: (int(t[0]), str(t[1])))
    return found


class TestArcAntiCheatV129(unittest.TestCase):
    def test_arc_v129_no_task_id_heuristics_in_core_modules(self) -> None:
        for rel in _CORE_MODULE_PATHS_V129:
            path = Path(rel)
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(rel))

            # 1) Disallow branching on task_id (or variants) inside core modules.
            violations = _find_forbidden_task_id_in_conditions(tree)
            self.assertFalse(violations, f"forbidden_task_id_condition:{rel}:{violations}")

            # 2) Disallow hardcoded ARC task ids in core modules.
            literals = _find_hardcoded_arc_task_ids_in_literals(tree)
            self.assertFalse(literals, f"hardcoded_arc_task_id_literal:{rel}:{literals}")

            # 3) Disallow reading filesystem paths inside core solver/proposer code.
            # Loaders/harnesses may use pathlib/os, but core should not.
            imports = _find_forbidden_imports(tree)
            self.assertFalse(imports, f"forbidden_import_in_core:{rel}:{imports}")
