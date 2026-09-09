import os
import re
from setuptools import find_packages, setup

def get_rocm_version():
    """Detect the ROCm version installed on this system."""
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    version_file = os.path.join(rocm_path, '.info', 'version')

    if os.path.exists(version_file):
        with open(version_file) as f:
            # Extract major.minor.patch
            match = re.match(r'\d+\.\d+\.\d+', f.read().strip())
            if match:
                return match.group(0)

    return None

def amdsmi_requirement():
    """Best-effort amdsmi pin for the [rocm-auto] extra.

    Pins as close to this machine's ROCm release as PyPI allows. amdsmi's
    release history has gaps (e.g. no 6.2.3, and 6.2.2 exists only as
    6.2.2.post0) and lags GitHub/ROCm (nothing past 7.0.x while ROCm 7.2
    ships), so an exact ``==`` pin can be unsatisfiable even for a real
    ROCm release. Instead:

    - ROCm < 7: ``~=X.Y.Z`` -- the newest release of the system's
      major.minor at or above its patch level (post-releases included).
    - ROCm >= 7: ``>=X,<X.(minor+1)`` -- the newest release in the major
      that doesn't exceed the system's minor.

    Returns an unpinned ``amdsmi`` when no system ROCm is found.
    """
    rocm_version = get_rocm_version()
    if not rocm_version:
        return "amdsmi"

    major, minor, _ = (int(p) for p in rocm_version.split('.'))
    if major >= 7:
        return f"amdsmi>={major},<{major}.{minor + 1}"
    return f"amdsmi~={rocm_version}"

with open("README.md", "r") as fp:
    long_description = fp.read()

with open(os.path.join("hpc_launcher", "version.py"), "r") as fp:
    version = fp.read().strip().split(" ")[-1][1:-1]

# GPU vendor libraries (amdsmi, nvidia-ml-py) are optional -- both call
# sites (hpc_launcher/systems/autodetect.py's find_AMD_gpus/find_NVIDIA_gpus)
# already guard the import and degrade gracefully at runtime. They must
# NOT be computed by probing *this* (build) machine's installed GPU
# libraries: doing so made the same commit produce a different wheel
# depending on whether it happened to be built on an AMD node, an NVIDIA
# node, or a CPU-only CI runner, which breaks build reproducibility and
# is a landmine for air-gapped/private-mirror installs that don't carry
# the hardware-mismatched package. They belong in extras_require, listed
# unconditionally, exactly like the torch/mpi/testing groups below --
# users opt in with `pip install hpc-launcher[rocm]` / `[cuda]`.
#
# [rocm-auto] is the one deliberate exception: it probes the machine
# running pip ($ROCM_PATH/.info/version) and pins amdsmi as close to
# that ROCm release as PyPI allows, because an amdsmi that doesn't match
# the ROCm runtime it talks to is broken at runtime. Users installing
# from source on the machine they'll run on can pick it for the right
# pin automatically; everyone else ([rocm], and any published wheel,
# whose metadata is frozen at build time) gets unpinned amdsmi and can
# pin by hand per the README.
setup(
    name="hpc-launcher",
    version=version,
    license="Apache-2.0",
    url="https://github.com/LBANN/HPC-launcher",
    author="Lawrence Livermore National Laboratory",
    author_email="lbann@llnl.gov",
    description="LBANN Launcher utilities for distributed jobs on HPC clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    entry_points={
        "console_scripts": [
            "torchrun-hpc = hpc_launcher.cli.torchrun_hpc:main",
            "launch = hpc_launcher.cli.launch:main",
            "hpc-launcher-cancel = hpc_launcher.cli.cancel:main",
        ],
    },
    install_requires=["psutil"],
    extras_require={
        "torch": ["torch", "numpy"],
        "mpi": ["mpi4py>=3.1.4", "mpi_rdv"],
        "testing": ["pytest"],
        "e2e_testing": ["accelerate"],
        "rocm": ["amdsmi"],
        "rocm-auto": [amdsmi_requirement()],
        "cuda": ["nvidia-ml-py"],
    },
)
