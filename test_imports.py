import sys
import time


def test(label, fn):
    t = time.time()
    try:
        fn()
        print(f"  [OK]  {label}  ({time.time() - t:.1f}s)")
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")


print("=== Import tests ===")

test("torch", lambda: __import__("torch"))
test("gradio", lambda: __import__("gradio"))
test("trimesh", lambda: __import__("trimesh"))
test("numpy", lambda: __import__("numpy"))
test("PIL", lambda: __import__("PIL"))
test("shap_e", lambda: __import__("shap_e"))
test("point_e", lambda: __import__("point_e"))
test("transformers", lambda: __import__("transformers"))
test("huggingface_hub", lambda: __import__("huggingface_hub"))
test("omegaconf", lambda: __import__("omegaconf"))
test("einops", lambda: __import__("einops"))
test("rembg", lambda: __import__("rembg"))

# TripoSR — needs TripoSR/ on the path
sys.path.insert(0, "TripoSR")
test("tsr.system (TripoSR)", lambda: __import__("tsr.system"))

print("\n=== Done ===")
