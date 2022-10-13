#!/bin/sh
docker_image="pytorchlightning/pytorch_lightning"
gpu=0
shm_size="16g"

while getopts ":g:i:w:m:" opt; do
  case $opt in
    g) gpu="$OPTARG"
    ;;
    i) docker_image="$OPTARG"
    ;;
    m) shm_size="$OPTARG"
    ;;
    w) wandb_api_key="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

if docker image inspect ${docker_image} > /dev/null;
  then
    docker run -v"$(pwd):/workdir" --gpus device=$gpu --shm-size=$shm_size -it $docker_image
else
  if [ "$docker_image" = "pytorchlightning/pytorch_lightning:latest" ];
    then
      echo "Pulling $docker_image from Docker Hub..."
      docker pull $docker_image
  else 
    echo "Building $docker_image..."
    # docker build --build-arg WANDB_API_KEY=$wandb_api_key -t $docker_image .  
    docker build --build-arg -t $docker_image .  
  fi
  docker run -v"$(pwd):/workdir" --gpus device=$gpu --workdir="workdir/" --shm-size=$shm_size -it $docker_image
fi