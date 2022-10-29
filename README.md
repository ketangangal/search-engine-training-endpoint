# SearchEngine-ModelTrainer
ModelTrainer endpoint is a continuous Training pipeline where I have added paperspace GPU Instance as a runner to
with high configuration. Since Training endpoint is expensive we cant keep it live all the time so, Instance will Always be in off state.
We have to manually trigger workflow to start the training.

# Architecture 
![image](https://user-images.githubusercontent.com/40850370/194861755-9e04c1ca-f33e-4fbf-8503-2ed5e6de887d.png)
# Infrastructure Needed 
1. Gpu Access on paperSpace 
2. Aws S3 bucket for model Registry and Data

# Project Setup
### Runner Setup
### Setup 1
```bash
sudo apt-get update
```
```bash
sudo apt-get upgrade
```
### Setup 2
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
```
```bash
sudo apt install unzip
```
```bash
unzip awscliv2.zip
```
```bash
sudo ./aws/install
```
### Step 3
1. Install Github runner and Configure runner.
2. Run github runner as a ubuntu service sudo `./svc.sh install`
3. Start the service sudo `./svc.sh start`

# Cost Involved
```Text
# Aws S3
s3 Storage: $0.025 per GB / First 50 TB / Month
s3 PUT : $0.005 (per 1,000 requests)
S3 GET : $0.0004 (per 1,000 requests)

# PaperSpace
Gpu Machine: 
    Ram : 30 GB
    Cpu's: 8
    Storage: 50 Gb
    Gpu: 8 GB 
    $0.462/ hour
```