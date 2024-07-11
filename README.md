**Introduction**

This is demo code for the article "Hotspot-Aware Scheduling of Virtual Machines with Overcommitment for Ultimate Utilization in Cloud Datacenters".


**Manual running:**

First you should install required packages with
```
pip install -r requirements.txt
```

Next, run the following command to download the dataset and unpack it into `dataset` directory (182 MiB to download, 2.2 GiB unpacked):
```
bash get_data.sh
```

Now you can evaluate the schedulers:
```
python run.py --trace dataset/50_1.json --placers CloseRadiusFit FirstFit RandomFit CloseRadiusLB --p=0.95
```

To perform your own experiments, take a look at `run_example` function in `run.py`.



**Running with Docker:**

The following commands will build container, download the dataset (182 MiB to download, 2.2 GiB unpacked) and perform evaluation of the scheduling algorithms for one of the 297 traces with default arguments.

```
$ sudo docker build -t g-robust .
  ...
$ sudo docker run g-robust
    placed         Placer
      2113  CloseRadiusLB
      2079 CloseRadiusFit
      1944       FirstFit
      1932      RandomFit
```

You can further connect to the container and play along with the parameters:
```
sudo docker run -it g-robust /bin/bash
```
