import ast
import importlib.metadata
import sys
from pathlib import Path

from ai_hub.tools import file_tools


WORKSPACE_REQUIREMENTS_FILE = "requirements.txt"
WORKSPACE_VENV_DIR = ".venv"

MODULE_TO_PACKAGE = {
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "Crypto": "pycryptodome",
    "dotenv": "python-dotenv",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
}


def workspace_requirements_path(thread_id: str) -> Path:
    return file_tools.ensure_thread_workspace(thread_id) / WORKSPACE_REQUIREMENTS_FILE


def workspace_venv_path(thread_id: str) -> Path:
    return file_tools.ensure_thread_workspace(thread_id) / WORKSPACE_VENV_DIR


def list_workspace_python_files(thread_id: str) -> list[str]:
    workspace = file_tools.ensure_thread_workspace(thread_id)
    return sorted(
        str(path.relative_to(workspace))
        for path in workspace.rglob("*.py")
        if ".venv" not in path.parts
    )


def read_workspace_requirements(thread_id: str) -> list[str]:
    requirements_path = workspace_requirements_path(thread_id)
    if not requirements_path.exists():
        return []
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    requirements: list[str] = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#") or cleaned.startswith("-r "):
            continue
        requirements.append(cleaned)
    return requirements


def _requirement_name(spec: str) -> str:
    name = spec
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if marker in name:
            name = name.split(marker, 1)[0]
            break
    if "[" in name:
        name = name.split("[", 1)[0]
    return name.strip().lower().replace("_", "-")


def _locate_workspace_site_packages(thread_id: str) -> Path | None:
    venv_path = workspace_venv_path(thread_id)
    posix_lib = venv_path / "lib"
    if posix_lib.exists():
        for candidate in sorted(posix_lib.glob("python*/site-packages")):
            if candidate.exists():
                return candidate
    windows_site_packages = venv_path / "Lib" / "site-packages"
    if windows_site_packages.exists():
        return windows_site_packages
    return None


def read_workspace_installed_packages(thread_id: str) -> list[str]:
    site_packages = _locate_workspace_site_packages(thread_id)
    if site_packages is None or not site_packages.exists():
        return []

    installed_names: set[str] = set()
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        name = distribution.metadata.get("Name") or distribution.name
        if not name:
            continue
        installed_names.add(_requirement_name(name))
    return sorted(installed_names)


def _workspace_local_modules(thread_id: str) -> set[str]:
    workspace = file_tools.ensure_thread_workspace(thread_id)
    local_modules: set[str] = set()
    for path in workspace.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        relative_path = path.relative_to(workspace)
        if len(relative_path.parts) > 1:
            local_modules.add(relative_path.parts[0])
        if path.name == "__init__.py":
            if path.parent != workspace:
                local_modules.add(path.parent.name)
            continue
        local_modules.add(path.stem)
    for path in workspace.rglob("__init__.py"):
        if ".venv" in path.parts:
            continue
        if path.parent != workspace:
            local_modules.add(path.parent.name)
    return {name for name in local_modules if name}


def _top_level_imports(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                imports.add(node.module.split(".", 1)[0])
    return imports


def _module_to_package_name(module_name: str) -> str:
    mapped = MODULE_TO_PACKAGE.get(module_name, module_name)
    return mapped.replace("_", "-")


def analyze_workspace_dependencies(thread_id: str, relative_paths: list[str] | None = None) -> dict:
    workspace = file_tools.ensure_thread_workspace(thread_id)
    python_files = relative_paths or list_workspace_python_files(thread_id)
    declared_requirements = read_workspace_requirements(thread_id)
    declared_names = {_requirement_name(spec) for spec in declared_requirements}
    installed_names = set(read_workspace_installed_packages(thread_id))
    local_modules = _workspace_local_modules(thread_id)

    imported_modules: set[str] = set()
    file_imports: dict[str, list[str]] = {}
    for relative_path in python_files:
        path = workspace / relative_path
        if not path.exists() or not path.is_file() or path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8")
        imports = sorted(_top_level_imports(source))
        if imports:
            file_imports[relative_path] = imports
            imported_modules.update(imports)

    stdlib_modules = set(getattr(sys, "stdlib_module_names", set()))
    external_modules = sorted(
        module
        for module in imported_modules
        if module not in stdlib_modules and module not in local_modules and module != "__future__"
    )
    missing_requirements = []
    for module in external_modules:
        package_name = _module_to_package_name(module)
        normalized = _requirement_name(package_name)
        if normalized not in declared_names:
            missing_requirements.append(package_name)

    missing_installations = [
        requirement
        for requirement in declared_requirements
        if _requirement_name(requirement) not in installed_names
    ]

    packages_to_install = sorted(
        dict.fromkeys([*missing_installations, *missing_requirements])
    )

    return {
        "python_files": python_files,
        "file_imports": file_imports,
        "imported_modules": sorted(imported_modules),
        "local_modules": sorted(local_modules),
        "declared_requirements": declared_requirements,
        "installed_requirements": sorted(installed_names),
        "missing_requirements": sorted(dict.fromkeys(missing_requirements)),
        "missing_installations": missing_installations,
        "packages_to_install": packages_to_install,
        "workspace_requirements_file": WORKSPACE_REQUIREMENTS_FILE,
        "workspace_venv_dir": WORKSPACE_VENV_DIR,
    }


def write_workspace_requirements(thread_id: str, requirements: list[str]) -> dict:
    content = "".join(f"{line}\n" for line in requirements)
    return file_tools.write_file(thread_id, WORKSPACE_REQUIREMENTS_FILE, content)
