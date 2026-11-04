"""Run COMPASS run.py with the K1 embodiment registered.

Used as K1_TRAIN_SCRIPT inside the K1 training container, e.g.:

    K1_TRAIN_SCRIPT=compass_k1/launch.py ~/run_k1_train.sh \
        -c configs/train_config.gin -o /workspace/mounts/logs/compass/k1_xxx \
        -b assets/x_mobility.ckpt --embodiment k1 --enable_cameras --headless

COMPASS lives at $COMPASS_ROOT (default /tmp/COMPASS, i.e. $S/k1/tmp/COMPASS on zz-bw). Its
Python deps go to a separate --target dir on first use, without dependencies, so the
container's torch / numpy / isaaclab are never replaced.
"""
import os
import subprocess
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("COMPASS_ROOT", "/tmp/COMPASS")
LIB = os.environ.get("COMPASS_PYLIB", os.path.expanduser("~/compass_pylib"))
# requirements.txt minus what the container already ships (numpy, pandas, onnx*, h5py, ...)
DEPS = ["diffusers==0.29.2", "einops==0.8.2", "gin-config==0.5.0", "pytorch-lightning==2.6.1",
        "lightning-utilities", "torchmetrics", "timm==1.0.26", "torcheval==0.0.7",
        "transformers==4.57.6", "tokenizers", "safetensors", "huggingface_hub", "regex",
        "msgpack==1.1.2", "moviepy==2.2.1", "imageio-ffmpeg", "proglog", "decorator",
        "python-dotenv", "toml", "wandb==0.25.1", "zmq==0.0.0", "pyzmq"]

if not os.path.isdir(os.path.join(LIB, "x_mobility")):
    os.makedirs(LIB, exist_ok=True)
    whl = os.path.join(ROOT, "x_mobility/x_mobility-0.1.0-py3-none-any.whl")
    for args in (DEPS, [whl]):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                               "--target", LIB, *args])

# LIB last: never shadow the container's packages.
sys.path[:0] = [ROOT, os.path.join(ROOT, "compass/rl_env/exts/mobility_es"), WS]
sys.path.append(LIB)

run_py = os.path.join(ROOT, "run.py")
src = open(run_py).read()
anchor = "EmbodimentEnvCfgMap = {"
assert anchor in src, "run.py layout changed: EmbodimentEnvCfgMap not found"
src = src.replace(anchor, "from compass_k1.k1_compass_env_cfg import K1GoalReachingEnvCfg\n"
                  f"{anchor}\n    'k1': K1GoalReachingEnvCfg,", 1)
os.chdir(ROOT)
sys.argv[0] = run_py
exec(compile(src, run_py, "exec"), {"__name__": "__main__", "__file__": run_py})
