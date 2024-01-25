
arg_tag=fight_detector:latest
arg_name=fight_detector
#debug
docker_args="--entrypoint /bin/bash --name $arg_name --restart unless-stopped --gpus all -v /opt/fight_detector:/opt/  --log-driver local --log-opt max-size=10m --net host -dt $arg_tag"

#prod
#docker_args="--name $arg_name --restart unless-stopped -v /edgar1/fight_detector:/opt/  --log-driver local --log-opt max-size=10m --net host -dt $arg_tag  bash"
echo "Launching container:"
echo "> docker run $docker_args"
docker run $docker_args
