from glob import glob
from setuptools import setup

package_name = "monocular_vo"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sooryas",
    maintainer_email="sooryas@local.dev",
    description="Phase 3 scaled baseline for Task 8 dashcam monocular visual odometry.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "video_bridge = monocular_vo.video_bridge:main",
            "vo_node = monocular_vo.vo_node:main",
        ],
    },
)
