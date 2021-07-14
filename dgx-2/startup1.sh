#!/bin/bash

# mark system packages to prevent from upgrade
sudo apt list --installed |grep ^libnv | awk -F'/' '{print $1}'|xargs sudo apt-mark hold
sudo apt list --installed |grep ^dgx | awk -F'/' '{print $1}'|xargs sudo apt-mark hold
sudo apt list --installed |grep nvidia | awk -F'/' '{print $1}'|xargs sudo apt-mark hold
sudo apt list --installed |grep ^ib | awk -F'/' '{print $1}'|xargs sudo apt-mark hold

# then upgrade the rest (takes a lot of time)
sudo apt update
sudo apt upgrade -y
