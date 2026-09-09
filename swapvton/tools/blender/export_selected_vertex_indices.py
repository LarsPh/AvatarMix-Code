from __future__ import annotations

import bpy
from pathlib import Path


def export_selected_vertex_indices(
    *,
    output_path: str,
    active_object_name: str | None = None,
) -> int:
    if active_object_name is None:
        obj = bpy.context.object
    else:
        obj = bpy.data.objects.get(active_object_name)

    if obj is None:
        raise RuntimeError("No active object (or object name not found).")
    if obj.type != "MESH":
        raise RuntimeError(f"Active object is not a mesh: {obj.name} ({obj.type})")


    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    mesh = obj.data
    indices = sorted({v.index for v in mesh.vertices if v.select})

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{i}\n" for i in indices))
    print(f"[export_selected_vertex_indices] wrote {len(indices)} indices -> {out}")
    return len(indices)


if __name__ == "__main__":


    export_selected_vertex_indices(
        output_path="/tmp/feet_indices_manual.txt",
        active_object_name=None,
    )
