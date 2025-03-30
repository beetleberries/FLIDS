
#TEST



data
https://ieee-dataport.org/open-access/car-hacking-attack-defense-challenge-2020-dataset
car_hacking_challenge_dataset_rev20Mar2021
https://data.4tu.nl/articles/dataset/Automotive_Controller_Area_Network_CAN_Bus_Intrusion_Dataset/12696950/2


so if you cant pip3 install the requirements you need to open a virtual machine through python using venv.

1. in the terminal type: python3 -m venv .venv - create virtual environment
2. source .venv/bin/activate activate the Virtual environment
3. which python to check if it is using the virtual environment python interpreter 
4. now do the pip install -r requirements.txt
5. run: python3 preprocess.py

Also download and put in both the car datasets:

    in drive: the dataset and car_hacking_dataset
