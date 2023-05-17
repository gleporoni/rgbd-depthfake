In src/automation folder there are some script to automatize the training and testing process.
In this file I briefly summary how to use them.

Training phase:
- write in 'train_to_do.txt' all the train parameters that we want change in config files in this way:
ex.
model=rgb_xception data=faceforensicsplusplus_c40 data.attacks=['Face2Face']
model=depth_xception data=faceforensicsplusplus_c40_depth data.input_type="d" data.attacks=['Face2Face']
....

We have two modalities for starting the training:

1) from "python src/automatation/automated_train.py [gpu]" (inside the docker)
In this way we launch a single python process that reads the "train_to_do.txt" file one line at time each time a train ended.
If we have a multi-gpu hardware, launch several dockers and select the wanted gpu in the "gpu" flag 

2)Since after few hours of traning, the docker containers do not see anymore the gpus, they require to be restarted.
In this case put in the "running_process.json" file the informations in the following way:

{
    "status": "running/finished",
    "docker_id":{
        "status": "UP/DOWN",
        "gpu": "gpu_id
    },
    "docker_id":{
        "status": "UP/DOWN",
        "gpu": "gpu_id
    },
    ....
}


At first initialization set all dockers in "DOWN" state.
so, from outside the dockers, from host (be carefull),to start use

"nohup python3 src/automatation/dockers_revive.py > my.log 2>&1 &
echo $! > save_pid.txt"

to stop 

"kill -9 `cat save_pid.txt`
rm save_pid.txt"

or manually place "finished" in the "status" key

Testing phase:
- Since the training phase saves the model weights basing on three aspects (val_accuracy, val_loss, last_epoch)
if we want test all these weigths we can automatize this procedure.

Perform all these steps inside the docker.

1) launch "python src/automatation/test_maker.py [weights_folder]", the file "test_to_do.txt" is automatically filled
with the networks and weights given by "weights_folder"

2) change in "test.yaml" the "check_all" flag to "True"

3) launch "python src/automatation/automated_test.py [gpu]" that automatically fields the "results.txt" file.
