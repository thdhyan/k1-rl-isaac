"""Isaac Sim RTX replay of episodes recorded by mjlab_k1/scripts_kick_record.py: chase camera beside head camera.
No physics runs here. Each link's world pose is written from the recording, so the motion is exactly the mjlab run.
Usage (python3 of the Isaac container): replay_kick_rtx.py <in.npz> <robot.usd> <out.mp4> [subframes]
"""
import sys

from isaacsim import SimulationApp

in_npz, usd_path, out_mp4 = sys.argv[1:4]
SUB = int(sys.argv[4]) if len(sys.argv) > 4 else 12
W, H = 960, 540
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": W, "height": H})

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402

import carb  # noqa: E402

settings = carb.settings.get_settings()  # same denoiser settings as panel_video.set_clean_render
settings.set_int("/rtx/rtpt/spp", 16)
settings.set_bool("/rtx/newDenoiser/enabled", True)
settings.set_bool("/rtx/rtpt/fireflyFilter/enabled", True)
CAM_OFFSET = np.array((0.06, 0.0, 0.05))  # mjlab_k1.tasks_head.CAM_OFFSET, Head_2 frame, x forward
rec = np.load(in_npz)
names = [str(n) for n in rec["names"]]
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def material(path, rgb, rough=0.7):
  mat = UsdShade.Material.Define(stage, path)
  sh = UsdShade.Shader.Define(stage, path + "/S")
  sh.CreateIdAttr("UsdPreviewSurface")
  sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
  sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
  mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
  return mat


def bind(prim, mat):
  UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)


def box(path, center, half, mat):
  c = UsdGeom.Cube.Define(stage, path)
  c.CreateSizeAttr(2.0)
  c.AddTranslateOp().Set(Gf.Vec3d(*center))
  c.AddScaleOp().Set(Gf.Vec3f(*half))
  bind(c, mat)


green, white, line = material("/World/M/g", (0.13, 0.36, 0.14), 0.9), material("/World/M/w", (0.92, 0.92, 0.9)), material("/World/M/l", (0.85, 0.85, 0.82), 0.9)
box("/World/ground", (0, 0, -0.05), (30, 30, 0.05), green)
gx, gw, gh = float(rec["goal"][0]), float(rec["goal_w"]), float(rec["goal_h"])
has_target = "target_0" in rec  # pass recordings: per-episode target marker instead of the goal frame
if not has_target:
  for name, c, h in (("post_l", (gx, gw / 2, gh / 2), (0.04, 0.04, gh / 2)), ("post_r", (gx, -gw / 2, gh / 2), (0.04, 0.04, gh / 2)),
                     ("bar", (gx, 0, gh), (0.04, gw / 2 + 0.04, 0.04)), ("line", (gx, 0, 0.002), (0.03, 6, 0.002))):
    box(f"/World/{name}", c, h, white if name != "line" else line)
else:
  marker = UsdGeom.Cylinder.Define(stage, "/World/target")
  marker.CreateRadiusAttr(0.2)
  marker.CreateHeightAttr(0.6)
  marker_t = marker.AddTranslateOp()
  bind(marker, material("/World/M/t", (0.85, 0.1, 0.1), 0.6))
ball = UsdGeom.Sphere.Define(stage, "/World/ball")
ball.CreateRadiusAttr(float(rec["ball_r"]))
ball_t = ball.AddTranslateOp()
bind(ball, material("/World/M/b", (0.95, 0.95, 0.95), 0.4))
UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(350.0)
sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(1200.0)
sun.CreateAngleAttr(1.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(50.0, 15.0, 30.0))

robot = stage.DefinePrim("/World/K1", "Xform")
robot.GetReferences().AddReference(usd_path)
for _ in range(10):
  app.update()
by_name = {}
for p in stage.Traverse():
  if p.GetName() in names and str(p.GetPath()).startswith("/World/K1") and p.IsA(UsdGeom.Xformable):
    by_name.setdefault(p.GetName(), p)
missing = [n for n in names if n not in by_name]
print("LINKS matched", len(by_name), "missing", missing, flush=True)
assert "Trunk" in by_name and "Head_2" in by_name, "robot USD lacks the mjlab link names"
order = sorted(by_name, key=lambda n: str(by_name[n].GetPath()).count("/"))  # parents first
ops = {}
for n in order:
  x = UsdGeom.Xformable(by_name[n])
  x.ClearXformOpOrder()
  ops[n] = x.AddTransformOp()


def mat4(pos, q):  # q = wxyz; USD row-vector convention
  r = Gf.Matrix3d(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
  m = Gf.Matrix4d()
  m.SetRotate(r)
  m.SetTranslateOnly(Gf.Vec3d(*map(float, pos)))
  return m


def set_cam(cam_op, eye, target, up=(0, 0, 1)):
  cam_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(*map(float, up))).GetInverse())


def camera(path, focal, aperture):
  c = UsdGeom.Camera.Define(stage, path)
  c.CreateFocalLengthAttr(focal)
  c.CreateHorizontalApertureAttr(aperture)
  c.CreateVerticalApertureAttr(aperture * H / W)
  c.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
  return c, UsdGeom.Xformable(c).AddTransformOp()


chase, chase_op = camera("/World/cam_chase", 24.0, 36.0)
pov, pov_op = camera("/World/cam_pov", 52.76, 128.0)  # ZED 2i-class optics, as in tasks/head/head_env_cfg.py
ann = []
for c in ("/World/cam_chase", "/World/cam_pov"):
  rp = rep.create.render_product(c, (W, H))
  a = rep.AnnotatorRegistry.get_annotator("rgb")
  a.attach([rp])
  ann.append(a)

font = ImageFont.load_default(size=26)
writer = imageio.get_writer(out_mp4, fps=25, codec="libx264", quality=None, ffmpeg_params=["-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p"])
ti, hi = names.index("Trunk"), names.index("Head_2")
step = 2
for ep in range(int(rec["n"])):
  P, Q, B = rec[f"pos_{ep}"], rec[f"quat_{ep}"], rec[f"ball_{ep}"]
  smooth, frames = None, range(0, len(P), step)
  if has_target:
    marker_t.Set(Gf.Vec3d(float(rec[f"target_{ep}"][0]), float(rec[f"target_{ep}"][1]), 0.3))
  tail = list(frames) + [len(P) - 1] * 25  # hold the last pose for 1 s
  for f, t in enumerate(tail):
    cache = UsdGeom.XformCache()
    world = {}
    for n in order:
      i = names.index(n)
      w = mat4(P[t, i], Q[t, i])
      world[n] = w
      parent = by_name[n].GetParent()
      pname = parent.GetName()
      pw = world[pname] if pname in world else cache.GetLocalToWorldTransform(parent)
      ops[n].Set(w * pw.GetInverse())
    ball_t.Set(Gf.Vec3d(*map(float, B[t])))
    trunk = P[t, ti]
    d = B[t, :2] - trunk[:2]
    d = d / max(np.linalg.norm(d), 1e-3)
    smooth = d if smooth is None else 0.95 * smooth + 0.05 * d
    u = smooth / max(np.linalg.norm(smooth), 1e-3)
    set_cam(chase_op, (trunk[0] - 2.6 * u[0], trunk[1] - 2.6 * u[1], 1.25), (trunk[0] + 0.8 * u[0], trunk[1] + 0.8 * u[1], 0.35))
    hp, hq = P[t, hi], Q[t, hi]
    w_, x_, y_, z_ = hq
    R = np.array([[1 - 2 * (y_**2 + z_**2), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)],
                  [2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_**2 + z_**2), 2 * (y_ * z_ - w_ * x_)],
                  [2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_**2 + y_**2)]])
    eye = hp + R @ CAM_OFFSET
    set_cam(pov_op, eye, eye + R[:, 0], R[:, 2])
    # The denoiser needs NGX, unavailable in this container, so average SUB passes of the path tracer per frame.
    acc = [np.zeros((H, W, 3)) for _ in ann]
    for _ in range(SUB):
      app.update()
      rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
      for k, a in enumerate(ann):
        acc[k] += np.asarray(a.get_data())[..., :3]
    imgs = []
    for k, label in enumerate(("Isaac Sim RTX replay, chase", "head camera (ZED 2i optics)")):
      im = Image.fromarray((acc[k] / SUB).astype(np.uint8))
      ImageDraw.Draw(im).text((14, 10), f"{label}, ep {ep + 1}, {t * float(rec['dt']):.1f} s", font=font, fill="white", stroke_width=3, stroke_fill="black")
      imgs.append(np.asarray(im))
    writer.append_data(np.concatenate(imgs, 1))
    if f % 50 == 0:
      print("FRAME", ep, f, len(tail), "mean", float(imgs[0].mean()), flush=True)
writer.close()
print("REPLAY_DONE", out_mp4, flush=True)
app.close()
