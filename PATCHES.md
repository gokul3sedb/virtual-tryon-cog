# Required patch — PuLID-Flux node vs newer ComfyUI

The `ComfyUI_PuLID_Flux_ll` node patches Flux's forward pass, but newer ComfyUI
passes an extra kwarg (`timestep_zero_index`) that the fork's function doesn't
accept, causing:

```
KSampler: pulid_forward_orig() got an unexpected keyword argument 'timestep_zero_index'
```

## Fix
In `custom_nodes/ComfyUI_PuLID_Flux_ll/PulidFluxHook.py`, the function
`pulid_forward_orig(...)` (~line 139) ends its signature with:

```python
    attn_mask: Tensor = None,
) -> Tensor:
```

Add `**kwargs,` before the closing paren:

```python
    attn_mask: Tensor = None,
    **kwargs,
) -> Tensor:
```

Apply this in the cog build (a `run:` sed step in cog.yaml, or a post-clone
patch script) so it survives image rebuilds.

Also required: `facenet-pytorch` must be installed (the node imports it), and it
must NOT be allowed to downgrade torch below 2.6 — pin torch==2.6.0 after
installing facenet-pytorch.
