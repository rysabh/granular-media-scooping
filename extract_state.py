import numpy as np
import omni.usd
from pxr import UsdGeom, Gf
from pxr import Usd

stage = omni.usd.get_context().get_stage()

def local_to_world_points(points_np, world_transform):
    out = []
    for p in points_np:
        pw = world_transform.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
        out.append([pw[0], pw[1], pw[2]])
    return np.asarray(out, dtype=float)

def summarize_points(points_world):
    if points_world is None or len(points_world) == 0:
        return None

    centroid = points_world.mean(axis=0)
    mins = points_world.min(axis=0)
    maxs = points_world.max(axis=0)
    extents = maxs - mins

    z = points_world[:, 2]
    xy = points_world[:, :2]
    radial_dist = np.linalg.norm(xy - xy.mean(axis=0), axis=1)

    return {
        "num_points": int(len(points_world)),
        "centroid_world": centroid,
        "bbox_min_world": mins,
        "bbox_max_world": maxs,
        "extents_world": extents,
        "pile_height": float(z.max() - z.min()),
        "mean_radius_xy": float(radial_dist.mean()),
        "std_radius_xy": float(radial_dist.std()),
    }

print("Searching stage for particle-like prims...")
for prim in stage.Traverse():
    attrs = [a.GetName() for a in prim.GetAttributes()]
    if prim.GetTypeName() in ["Points", "PointInstancer", "Mesh"] or "points" in attrs or "positions" in attrs:
        print("\nPrim:", prim.GetPath(), "type:", prim.GetTypeName())

        xform = UsdGeom.Xformable(prim)
        world_tf = xform.ComputeLocalToWorldTransform(0)

        if prim.IsA(UsdGeom.Points):
            pts = UsdGeom.Points(prim).GetPointsAttr().Get()
            if pts:
                pts_local = np.array([[p[0], p[1], p[2]] for p in pts], dtype=float)
                pts_world = local_to_world_points(pts_local, world_tf)
                print(summarize_points(pts_world))

        elif prim.GetTypeName() == "PointInstancer":
            pos_attr = prim.GetAttribute("positions")
            if pos_attr and pos_attr.HasValue():
                   pos = pos_attr.Get(Usd.TimeCode.Default())
                   if pos is None:
                       print("positions is None for:", prim.GetPath())
                   else:
                       pts_local = np.array([[p[0], p[1], p[2]] for p in pos], dtype=float)
                       pts_world = local_to_world_points(pts_local, world_tf)
                       print(summarize_points(pts_world))

        elif prim.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if pts:
                pts_local = np.array([[p[0], p[1], p[2]] for p in pts], dtype=float)
                pts_world = local_to_world_points(pts_local, world_tf)
                print(summarize_points(pts_world))
