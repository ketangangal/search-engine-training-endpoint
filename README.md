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
1. Update and upgrade the machine 
2. Install the paperspace cli
3. Register Gpu as a runner
4. Add secrets
5. Done 

### Env Variables
```bash

export ACCESS_KEY_ID=<access-key>
export AWS_SECRET_KEY=<secret-key>
export AWS_REGION=<aws-region>

export DATABASE_USERNAME=<username>
export DATABASE_PASSWORD=<password>

export API_KEY=<api-key>
export MACHINE_ID=<machine-id>

```
### Errors

fatal error: Python.h: No such file or directory
```bash
sudo apt install libpython3.8-dev
```


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