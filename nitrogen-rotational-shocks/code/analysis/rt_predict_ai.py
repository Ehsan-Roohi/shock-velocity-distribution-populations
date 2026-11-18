"""Level populations of the shock runs with the kernel from the ab initio N2-N2 potential (tag v19ai), as
rt_predict_rin.py does for the main runs: all molecules kept as they are (the stream enters with the measured
distribution), with the one-collision shape of the ab initio trajectory ensembles for the kernel labels.

Usage (in the data folder, next to rt_predict.py): python rt_predict_ai.py
Writes rt_predict_v19ai.json and rt_predict_v19ai_seeds.json."""
import sys, json, subprocess

TAG = "v19ai"
EXC_LJ = {"7": "ctc_cluster/exc_M7tm.npz", "12.9": "ctc_cluster/exc_M12.9tm.npz"}
EXC_AI = {"7": "ctc_cluster/exc_M7ai.npz", "12.9": "ctc_cluster/exc_M12.9ai.npz"}
for out, npz, labs in ((f"rt_predict_{TAG}.json", f"erh_{TAG}.npz", ("k", "zi10", "c", "pu5")),
                       (f"rt_predict_{TAG}_seeds.json", f"erh_{TAG}_seeds.npz", ("s11", "s12", "s13", "s14"))):
    spec = [f"M{m}{l}:{m}:{(EXC_LJ if l in ('zi10', 'c', 'pu5') else EXC_AI)[m]}" for l in labs for m in ("7", "12.9")]
    subprocess.run([sys.executable, "rt_predict.py", out, npz] + spec, check=True, stdout=subprocess.DEVNULL)
    d = json.load(open(out))
    for lab, recs in d.items():
        for r in recs:
            r["pred_replaced"] = r["pred"]; r["pred"] = r["pred_raw"]; r["res_stream"] = r["res_raw"]
    json.dump(d, open(out, "w"), indent=1); print("written", out, "labels", list(d))
