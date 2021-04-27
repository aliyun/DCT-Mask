#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

[[ -d "dev/packaging" ]] || {
  echo "Please run this script at detectron2 root!"
  exit 1
}

build_one() {
  cu=$1
  pytorch_ver=$2

  case "$cu" in
    cu*)
      container_name=manylinux-cuda${cu/cu/}
      ;;
    cpu)
      container_name=manylinux-cuda101
      ;;
    *)
      echo "Unrecognized cu=$cu"
      exit 1
      ;;
  esac

  echo "Launching container $container_name ..."

  for py in 3.6 3.7 3.8; do
    docker run -itd \
      --name $container_name \
      --mount type=bind,source="$(pwd)",target=/detectron2 \
      pytorch/$container_name

    cat <<EOF | docker exec -i $container_name sh
      export CU_VERSION=$cu D2_VERSION_SUFFIX=+$cu PYTHON_VERSION=$py
      export PYTORCH_VERSION=$pytorch_ver
      cd /detectron2 && ./dev/packaging/build_wheel.sh
EOF

    docker container stop $container_name
    docker container rm $container_name
  done
}


if [[ -n "$1" ]] && [[ -n "$2" ]]; then
  build_one "$1" "$2"
else
  build_one cu102 1.5
  build_one cu101 1.5
  build_one cu92 1.5
  build_one cpu 1.5

  build_one cu101 1.4
  build_one cu100 1.4
  build_one cu92 1.4
  build_one cpu 1.4
fi
