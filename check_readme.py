import sys


def check_file(path, is_python=False):
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  MISSING  {path}")
        return False

    lines = content.splitlines()
    issues = []

    if is_python:
        import ast

        try:
            ast.parse(content)
            print(f"  OK       {path}  ({len(lines)} lines, syntax valid)")
        except SyntaxError as e:
            issues.append(f"SyntaxError at line {e.lineno}: {e.msg}")
    else:
        # Markdown checks
        fence_count = content.count("```")
        if fence_count % 2 != 0:
            issues.append(f"Unmatched code fences ({fence_count} backtick-triples)")
        if "````" in content:
            issues.append("Quadruple backtick found")
        if len(content.strip()) < 500:
            issues.append("File suspiciously short")
        print(f"  OK       {path}  ({len(lines)} lines, {len(content)} chars)")

    for issue in issues:
        print(f"  WARN     {path} -> {issue}")

    return len(issues) == 0


def check_app_logic():
    import ast
    import re

    path = "app.py"
    with open(path, encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    required_funcs = [
        "run_generation",
        "shap_e_text",
        "shap_e_image",
        "triposr_image",
        "point_e_text",
        "_save_mesh",
        "_post_process",
        "_cleanup",
        "get_memory_info",
        "unload_models",
        "preview_bg_removal",
        "apply_potato_preset",
        "toggle_inputs",
        "update_info",
    ]

    all_ok = True
    for fn in required_funcs:
        if fn in funcs:
            print(f"  OK       fn:{fn}")
        else:
            print(f"  MISSING  fn:{fn}")
            all_ok = False

    checks = [
        ("torch.no_grad() x4", src.count("with torch.no_grad()") == 4),
        ("USE_FP16 used x2", src.count("use_fp16=USE_FP16") == 2),
        ("Point-E R channel fix", 'vertex_channels["R"]' in src),
        ("Point-E G channel fix", 'vertex_channels["G"]' in src),
        ("Point-E B channel fix", 'vertex_channels["B"]' in src),
        ("_cleanup in error path", src.count("_cleanup()") >= 3),
        ("GLTFLoader in viewer", "GLTFLoader" in src),
        ("loadGLBInViewer defined", "loadGLBInViewer" in src),
        ("OBJLoader removed", "OBJLoader" not in src),
        ("glb_path_out elem_id", "glb_path_out" in src),
        ("allowed_paths in launch", "allowed_paths" in src),
        ("Taubin smoothing", "filter_taubin" in src),
        ("Quadric decimation", "simplify_quadric_decimation" in src),
        ("Mesh repair fix_winding", "fix_winding" in src),
        ("Mesh repair fill_holes", "fill_holes" in src),
        ("TRIPOSR_CHUNK auto-tune", "TRIPOSR_CHUNK" in src),
        ("ON_CPU flag", "ON_CPU" in src),
        ("USE_FP16 flag", "USE_FP16" in src),
        ("psutil import", "import psutil" in src),
        ("PROMPT_PRESETS", "PROMPT_PRESETS" in src),
        ("potato preset btn", "potato_btn" in src),
        ("unload btn wired", src.count("unload_btn") >= 2),
        (
            "6 outputs in gen_btn",
            "out_obj, out_glb, out_stl, status_md, stats_html, hidden_glb_path" in src,
        ),
        ("9 inputs to run_gen", src.count("decimate_sl") >= 2),
        ("early returns have 6 vals", src.count("return None, None, None,") >= 6),
        ("TripoSR vertex color fix", "visual.vertex_colors" in src),
        ("CSS mem-bar defined", "mem-bar" in src),
        ("CPU badge CSS", "cpu-badge" in src),
        ("COLOR viewer button", "toggleColors" in src),
        ("GLB polling interval", "setInterval" in src),
    ]

    for desc, result in checks:
        status = "OK  " if result else "FAIL"
        if not result:
            all_ok = False
        print(f"  {status}     check:{desc}")

    return all_ok


def main():
    sep = "─" * 56
    print(sep)
    print("  3D Generator — Health Check")
    print(sep)

    print("\n[1] File syntax & size")
    r1 = check_file("app.py", is_python=True)
    r2 = check_file("requirements.txt", is_python=False)
    r3 = check_file("README.md", is_python=False)
    r4 = check_file("run_tests.py", is_python=True)
    r5 = check_file("install.sh", is_python=False)
    r6 = check_file("run.ps1", is_python=False)

    print("\n[2] app.py logic checks")
    r7 = check_app_logic()

    print("\n[3] requirements.txt checks")
    with open("requirements.txt", encoding="utf-8") as f:
        req = f.read()
    req_checks = [
        ("trimesh present", "trimesh" in req),
        ("gradio present", "gradio" in req),
        ("Pillow present", "Pillow" in req),
        ("numpy present", "numpy" in req),
        ("rembg present", "rembg" in req),
        ("psutil present", "psutil" in req),
        ("scipy present", "scipy" in req),
        ("huggingface_hub", "huggingface_hub" in req),
        ("transformers", "transformers" in req),
    ]
    r8 = True
    for desc, result in req_checks:
        status = "OK  " if result else "FAIL"
        if not result:
            r8 = False
        print(f"  {status}     {desc}")

    print(f"\n{sep}")
    all_passed = all([r1, r2, r3, r4, r5, r6, r7, r8])
    if all_passed:
        print("  RESULT   All checks passed.")
    else:
        print("  RESULT   Some checks FAILED — review output above.")
    print(sep)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
