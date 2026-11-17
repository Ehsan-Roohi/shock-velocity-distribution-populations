"""Grid of ctc_kcell.py (total energies, impact-parameter limits, bin edges of the rotor shares, source cells), read
from that file so that there is one definition."""
import os, re

_t = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ctc_kcell.py"), encoding="utf-8").read()
_ns = {}
exec(_t[_t.index("ETOT = "):_t.index("if sys.argv[1] == \"list\"")], _ns)
ETOT, BMAX, XE, XTMIN, NB, SRC, JOBS = (_ns[k] for k in ("ETOT", "BMAX", "XE", "XTMIN", "NB", "SRC", "JOBS"))
