import os
import yaml
import sys

root_folder = sys.argv[1]

def get_file_names(root_folder):
    file_names = []
    for root, dirs, files in os.walk(root_folder):
        for file_name in files:
            file_names.append(os.path.join(root, file_name))
    return file_names

# root_folder = "/media/weights_2/raw/all_attacks"
file_names = get_file_names(root_folder)
with open("/workdir/test_to_do.txt", "a") as file:

    for file_name in file_names:
        if '.hydra/config.yaml' in file_name:
            with open(file_name, 'r') as f:
                doc = yaml.load(f, yaml.FullLoader)
                print(doc["model"]["model_name"])
                model_name = doc["model"]["model_name"]
                attacks = doc["data"]["attacks"]
                if doc["data"]["use_depth"]:
                    dataset="faceforensicsplusplus_"+doc["data"]["compression_level"][0]+"_depth"
                else:
                    dataset="faceforensicsplusplus_"+doc["data"]["compression_level"][0]
                    
                file.write("model="+model_name)
                try:
                    file.write(" data.input_type="+doc["data"]["input_type"])
                except:
                    None
                file.write(" data="+dataset)
                file.write(" data.attacks=[")

                len_attacks = len(attacks)
                i = 0
                for attack in attacks:
                    i+=1
                    file.write("'"+attack+"'")
                    if i != len_attacks:
                        file.write(",")
                file.write("]")

        if "experiments/accuracy" in file_name:
            file.write(" run.experiment.checkpoint_file_acc='"+file_name+"'")
        elif "experiments/last" in file_name:
            file.write(" run.experiment.checkpoint_file_last='"+file_name+"'")
        elif "experiments/loss" in file_name:
            file.write(" run.experiment.checkpoint_file_loss='"+file_name+"'\n")
            
