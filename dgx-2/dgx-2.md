# Running model on DGX-2

## Preparation
Official documentation can be found [here](https://scilifelab.atlassian.net/wiki/spaces/aida/pages/1235189793/AIDA+DGX-2+Service).  
First book your GPU slot in the [booking sheet](https://docs.google.com/spreadsheets/d/1wA7H3Uh53ADVYptiQWXROnMD67HvPOAwSvW20EnzlFM/edit#gid=254337069).  
Setup your AIDA VPN using these [instructions](https://scilifelab.atlassian.net/wiki/spaces/aida/pages/1235189793/AIDA+DGX-2+Service#VPN). This might take a while so make sure you prepare in time.  
I recommend testing the VPN by connecting to the SFTP share as it is always available even if you have nothing booked.  

## Connecting
Once the VM is ready (usually before lunch on monday) you should get an email stating that the VM is ready together with the IP-address to connect to.  
Connect with ``ssh username@123.456.789.012``.  
Make sure nvidia-drivers and GPU's are correctly setup before you go further by running ``nvidia-smi``.  

## General info
The VM will automatically have access to a fileshare at ``/proj``. This is the same share that you connect to with SFTP.  
Files and folders in ``/proj`` will only be available to users in our project with one exception: ``/proj/datahub``.  
This folder is for datasets shared among all AIDA users and files and folders there can therefore be accessed by others.  
Other important folders are:  
``/proj/startup`` - folder containing startup scripts. The scripts can also be found in this git repo.  
``/proj/suef_data`` - folder containing data for this project.  
``/proj/deepcoronary`` - folder containing data for the deepcoronary project.  


## Setup

1. Run the script ``/proj/startup/startup1.sh``. 
This first excludes several packages that should not be installed (as they would interfere with the DGX-2 software and drivers). Then it updates and upgrades the VM.
Once the script has finished, reboot the VM with ``sudo reboot`` and connect again when its up.
2. Run the script ``/proj/startup/startup2.sh``.
This installs and configures privoxy which is used to setup an HTTP proxy that routes HTTP traffic through the SSH tunnel.
As the VM's cannot access internet on their own except for a few whitelisted addresses, we need this to be able to install pip packages and use git.
Once the script has finished, ``exit`` and connect again with ``ssh -R 3128 username@123.456.789.012`` where 3128 is the port number privoxy was configured to use.
3. Run the script ``/proj/startup/startup3_suef.sh``.
This installs system dependencies such as cuda-toolkit, cudnn and libraries required for pyenv. It then configures your bashrc for pyenv and installs the correct python version. 
4. Create a new virtual python enviroment with ``python -m venv dgx2`` and activate it with ``source ~/dgx2/bin/activate``
5. Clone the git repo with ``git clone https://github.com/Cardiobotics/SUEF.git``
6. Run ``pip install -r requirements.txt`` from the repo.
7. If you are logging with Neptune, you also need to set the environment variable NEPTUNE_API_TOKEN with ``export NEPTUNE_API_TOKEN='your personal api token here'``
8. Now you can run training on the cluster.

## Troubleshooting

One issue I have encountered two times is that I was unable to access the GPU's even though nvidia-smi listed them correctly. 
An easy test after your finished the setup is running torch.cuda.is_available() in an interactive python terminal and it then gives you an error message that states the device is unavailable.
This means something has gone wrong when the VM was created and you have to ask Joel to recreate the entire VM have have not found a simple fix for it yet.
In general, this can be a good idea if your VM is in a bad state. Instead of spending hours trying to troubleshooting it back to functional, just ask admins to delete and start from scratch.

Admins and help from other users can be found in this [slack channel](nbiisweden.slack.com)
