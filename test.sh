SHARE=$(pwd)
IMAGE=$1
NAME=$(sudo docker run -d -v ${SHARE}:/host/Users -it ${IMAGE} /bin/bash)
echo '****************'
sudo docker exec -i $NAME ./ask /host/Users/data/set1/a1.txt 5 2>/dev/null
echo '****************'
sudo docker stop $NAME >/dev/null
