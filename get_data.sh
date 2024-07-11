#!/usr/bin/env bash
set -eu

url='https://github.com/huaweicloud/HUAWEICloudPublicDataset/raw/main/HuaweiCloudVMSchedulingDataset/vm_online_scheduling.tar.gz?download='

rm -f /tmp/tmp.tar.gz
wget -O /tmp/tmp.tar.gz "${url}"
tar -xvzf /tmp/tmp.tar.gz > /dev/null
mv dataset2 dataset
rm -f /tmp/tmp.tar.gz

