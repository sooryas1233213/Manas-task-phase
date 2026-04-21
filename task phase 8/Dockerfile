FROM arm64v8/ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    QT_X11_NO_MITSHM=1 \
    XDG_RUNTIME_DIR=/tmp/runtime-ros \
    NO_AT_BRIDGE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      ca-certificates \
      curl \
      gnupg \
      lsb-release && \
    add-apt-repository universe && \
    if ! grep -R -q 'packages.ros.org/ros2/ubuntu' /etc/apt /etc/apt/sources.list.d 2>/dev/null; then \
      curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | \
        gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
        > /etc/apt/sources.list.d/ros2.list; \
    fi && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-humble-rviz2 \
      ros-humble-rviz-default-plugins \
      ros-humble-cv-bridge \
      ros-humble-tf2-ros \
      ros-humble-demo-nodes-cpp \
      ros-humble-demo-nodes-py \
      python3-colcon-common-extensions \
      python3-numpy \
      python3-opencv \
      python3-yaml \
      python3-rosdep \
      build-essential \
      curl \
      dbus-x11 \
      fluxbox \
      git \
      mesa-utils \
      novnc \
      procps \
      supervisor \
      websockify \
      x11vnc \
      xterm \
      xvfb && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p \
      /opt/task8/rviz \
      /opt/task8/scripts \
      /tmp/runtime-ros \
      /var/log/supervisor \
      /ws/src && \
    chmod 700 /tmp/runtime-ros

COPY config/rviz/task8_humble.rviz /opt/task8/rviz/task8_humble.rviz
COPY scripts/task8_ros.sh /opt/task8/scripts/task8_ros.sh
COPY scripts/profile_task8_ros.sh /etc/profile.d/task8_ros.sh
COPY scripts/run_rviz.sh /opt/task8/scripts/run_rviz.sh
COPY scripts/task8-supervisord.conf /etc/supervisor/conf.d/task8-supervisord.conf

RUN chmod +x \
      /opt/task8/scripts/task8_ros.sh \
      /opt/task8/scripts/run_rviz.sh \
      /etc/profile.d/task8_ros.sh

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/task8-supervisord.conf"]
