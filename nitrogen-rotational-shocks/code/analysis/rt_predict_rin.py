"""Level populations of the runs whose stream enters with the measured free-stream rotational distribution (main
comparison of the letter): runs rt_predict.py and keeps the populations of all molecules as they are (no replacement
of the unscattered molecules, which already carry the measured distribution), i.e. pred := pred_raw and
res_stream := res_raw, so that the table and figure scripts read them unchanged.

Usage (in the n2 folder): python rt_predict_rin.py [tag]   (tag rin by default; e.g. v19 reads erh_v19.npz, writes rt_predict_v19.json)"""
import sys, json, subprocess

TAG = sys.argv[1] if len(sys.argv) > 1 else "rin"
EXC = {"7": "ctc_cluster/exc_M7tm.npz", "12.9": "ctc_cluster/exc_M12.9tm.npz"}
for out, npz, labs in ((f"rt_predict_{TAG}.json", f"erh_{TAG}.npz", ("k", "zi10", "c", "pu5")),
                       (f"rt_predict_{TAG}_seeds.json", f"erh_{TAG}_seeds.npz", ("s11", "s12", "s13", "s14"))):
    spec = [f"M{m}{l}:{m}:{EXC[m]}" for l in labs for m in ("7", "12.9")]
    subprocess.run([sys.executable, "rt_predict.py", out, npz] + spec, check=True, stdout=subprocess.DEVNULL)
    d = json.load(open(out))
    for lab, recs in d.items():
        for r in recs:
            r["pred_replaced"] = r["pred"]; r["pred"] = r["pred_raw"]; r["res_stream"] = r["res_raw"]
    json.dump(d, open(out, "w"), indent=1); print("written", out, "labels", list(d))
