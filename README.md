# ENDEAVR-AutoPilot
This project implements an end-to-end autonomous vehicle capable of completing routes and avoiding obstacles. The model processes input from a RGB camera, depth camera, high-level commands (HLC), current speed, and traffic light status to output steering, throttle, and brake values. Although the model was trained on a pre-existing dataset, additional data can be collected using the same methodology and format using the data collection scripts provided. Additionally, the provided evaluation script assesses trained models using CARLA leaderboard metrics.

## Model Architecture
The below diagram was generated using the torchview library:

![Model Diagram](train/Model%20Diagram.png)

From torchinfo, the total number of parameters comes out to 317,544.

## Training
All training was conducted on Google Colab using PyTorch v2.3.1+cu121. The model was trained for 30 epochs with a batch size of 32, using the AdamW optimizer with a learning rate of 0.001 and a weight decay value of 0.01. Cosine annealing was utilized to decay the learning rate, along with AutoClip for gradient clipping. The training notebook is attached in the train directory.

The following graph shows average loss during training for the current model:
![Loss Graph](train/Loss%20Graph.png)

## Setup
Install CARLA Simulator 0.9.15 from [here](https://carla.org/2023/11/10/release-0.9.15/ "CARLA 0.9.15").

Set up an environment using conda + pip or venv + pip, Python version 3.10.12 is required.

To install required packages run:
```
pip install -r requirements.txt
```

## Running the Code
### Data Collection
To run the data collection script:
1. Run ```./CarlaUE4.sh``` in your CARLA installation path if you're using Linux. If you're on Windows, run ```CarlaUE4.exe```
2. Run ```python data_collection.py ```

The following command line arguments can be used:
Argument      | Description   | Default Value
------------- | ------------- | -------------
--town | CARLA Town to run data collection on | Town01
--weather  | Weather condition for world | ClearNoon
--max_frames  | Max number of frames to run episode for | 8000
--episodes | Number of episodes to run data collection for | 16
--vehicles | Number of vehicles present in simulation | 80
--route_file | Filepath for route file | routes/Town01_Train.txt
--collect_steer | Only collect data with high steering angle | Off by default

To run the data collection for VLM finetuning script:
1. Run ```./CarlaUE4.sh``` in your CARLA installation path if you're using Linux. If you're on Windows, run ```CarlaUE4.exe```
2. Run ```python data_collection_vlm.py ```

The following command line arguments can be used:
Argument      | Description   | Default Value
------------- | ------------- | -------------
--town | CARLA Town to run data collection on | Town01
--weather  | Weather condition for world | ClearNoon
--max_frames  | Max number of frames to run episode for | 8000
--episodes | Number of episodes to run data collection for | 16
--vehicles | Number of vehicles present in simulation | 80
--pedestrians | Number of pedestrians present in simulation | 80
--route_file | Filepath for route file | routes/Town01_Train.txt
--image_path | Image filepath for each JSON entry | vlm_data/images/

### Evaluation
To run the evaluation script: 
1. Run ```./CarlaUE4.sh``` in your CARLA installation path if you're using Linux. If you're on Windows, run ```CarlaUE4.exe```
2. Run ```python evaluation.py ```

The following command line arguments can be used:
Argument      | Description   | Default Value
------------- | ------------- | -------------
--town | CARLA Town to run evaluation on | Town01
--weather  | Weather condition for world | ClearNoon
--max_frames  | Max number of frames to run episode for | 5000
--episodes | Number of episodes to run model for | 12
--vehicles | Number of vehicles present in simulation | 50
--route_file | Filepath for route file | routes/Town02_All.txt
--model | Filename of trained model | av_model.pt

When the script finishes running, the following metrics will be saved in evaluation.log:

* Episode Completion Percentage
* Driving Score
* Route Completion
* Infraction Score

## Citations
### Papers
The following papers were instrumental in guiding this project. In particular, paper #1 was the most influential, as its model architecture directly inspired my work. Paper #2 was also very influential, as the incorporation of HLC, speed, and traffic light during training was directly adapted from this paper.
1. [K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger, 2023, PAMI'23, TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving] (https://arxiv.org/pdf/2205.15997)
2. [H. Haavaldsen M. Aasbø and F. Lindseth, 2019, NAIS 2019, Autonomous Vehicle Control: End-to-end Learning in Simulated Urban Environments](https://arxiv.org/pdf/1905.06712)
3. [I. Vasiljević, J. Musić, J. Mendes & J. Lima, 2023, OL2A 2023, Adaptive Convolutional Neural Network for Predicting Steering Angle and Acceleration on Autonomous Driving Scenario](https://link.springer.com/chapter/10.1007/978-3-031-53036-4_10)
4. [Y. Wang, D. Liu, H. Jeon, Z. Chu and E. T. Matson, 2019, ICAART 2019, End-to-end Learning Approach for Autonomous Driving: A Convolutional Neural Network Model](https://pdfs.semanticscholar.org/8944/67dc8db83f1bb07563c1f0f24361e5e57115.pdf)
5. [NVIDIA Corporation, 2020, arXiv reprint, The NVIDIA PilotNet Experiments](https://arxiv.org/pdf/2010.08776)
6. [P. Viswanath, S. Nagori, M. Mody, M. Mathew, P. Swami, 2018, ICEE 2018, End to End Learning based Self-Driving using JacintoNet](https://ieeexplore.ieee.org/document/8576190)

### Repos
This project relies on the following repositories, with repos #1 and #2 being the most influential. Many of repo #2's methodologies were directly used in this project, as its data collection and evaluation methods proved to be optimal for this task. A massive thank you to [TheRoboticsClub](https://github.com/TheRoboticsClub), as their contributions were essential to the success of this project.
1. [transfuser] (https://github.com/autonomousvision/transfuser)
2. [gsoc2023-Meiqi_Zhao](https://github.com/TheRoboticsClub/gsoc2023-Meiqi_Zhao/)
3. [carla_dataset_tools](https://github.com/KevinLADLee/carla_dataset_tools)
4. [AutoClip](https://github.com/pseeth/autoclip)
5. [SeparableConv-Torch](https://github.com/reshalfahsi/separableconv-torch)
6. [torchview](https://github.com/mert-kurttutan/torchview)
7. [torchinfo](https://github.com/TylerYep/torchinfo)