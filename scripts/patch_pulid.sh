#!/bin/bash
# Applies the required **kwargs patch to the PuLID-Flux node so it works with
# newer ComfyUI (fixes: pulid_forward_orig() unexpected kwarg 'timestep_zero_index').
# Run this in the cog build after custom nodes are cloned.
set -e
F="ComfyUI/custom_nodes/ComfyUI_PuLID_Flux_ll/PulidFluxHook.py"
if [ -f "$F" ] && ! grep -q "    \*\*kwargs," "$F"; then
  python3 - "$F" <<'PY'
import sys
p=sys.argv[1]; lines=open(p).readlines(); out=[]; patched=False
for i,l in enumerate(lines):
    out.append(l)
    if not patched and l.strip()=="attn_mask: Tensor = None," and i<170:
        out.append("    **kwargs,\n"); patched=True
open(p,"w").writelines(out)
print("PuLID patched:",patched)
PY
else
  echo "PuLID patch skipped (already applied or file missing)"
fi
