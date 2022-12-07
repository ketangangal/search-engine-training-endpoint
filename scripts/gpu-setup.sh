# Paper Space Machine cmds
sudo apt-get update
sudo apt-get upgrade -y

## Aws cli installation
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip
unzip awscliv2.zip
sudo ./aws/install

## Add Paperspace machine as runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.299.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.299.1/actions-runner-linux-x64-2.299.1.tar.gz
echo "147c14700c6cb997421b9a239c012197f11ea9854cd901ee88ead6fe73a72c74  actions-runner-linux-x64-2.299.1.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.299.1.tar.gz

./config.sh --url <url> --token <token>

## Add Runner as a service
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status

## To stop the service
sudo ./svc.sh stop
sudo ./svc.sh uninstall
