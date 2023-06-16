# A Guided-Based Approach for Deepfake Detection: RGB-Depth Integration via Features Fusion

## Introduction
Deep fake technology paves the way for a new generation of super realistic artificial content. While this opens the door to extraordinary new applications, the malicious use of deepfakes allows for far more realistic disinformation attacks than ever before. In this paper, we start from the intuition that generating fake content introduces possible inconsistencies in the depth of the generated images. This extra information provides valuable spatial and semantic cues that can reveal inconsistencies facial generative methods introduce. To test this idea, we evaluate different strategies for integrating depth information into an RGB detector. Our *Masked Depthfake Network* method is 3.2% more robust against common adversarial attacks on average than a typical RGB detector. Furthermore, we show how this technique allows the model to learn more discriminative features than RGB alone.

<img src="https://github.com/gleporoni/rgbd-depthfake/blob/5e2b9bab1af4a1d1bc60e4e123f6dd1e062eafac/doc/DepthFake_2-1.png" width="900">

## Main Results
### Detection capabilities

| Class   |  | RAW         |             | | C40         |             |
|---------|--|-------------|-------------|-|-------------|-------------|
|         |  | **RGB**     | **MDN**     | | **RGB**     | **MDN**     |
| **DF**  |  | 96,00 %     | **96,86** % | | 88,15 %     | **91,26** % |
| **F2F** |  | 95,35 %     | **95,85** % | | **82,57** % | 81,82 %     |
| **FS**  |  | 95,32 %     | **96,29** % | | 86,11 %     | **87,17** % |
| **NT**  |  | 92,01 %     | **92,65** % | | **70,77** % | 70,50 %     |
| **ALL** |  | **95,02** % | 94,87 %     | | 82,37 %     | **82,43** % |

### Feature analysis
<img src="https://github.com/gleporoni/rgbd-depthfake/blob/e0224b6f1fedeb277743276322425dfe477bcde4/doc/depthfake3-1.png" width="300">

### Robustness

| Attack | Model |    |     | RAW |    |     | |    |     | C40 |    |     |
|--------|-------|----|-----|-----|----|-----|-|----|-----|-----|----|-----|
|        |       | DF | F2F | FS  | NT | ALL | | DF | F2F | FS  | NT | ALL |
| BLR    | RGB   |    |     |     |    |     | |    |     |     |    |     |
|        | MDN   |    |     |     |    |     | |    |     |     |    |     |
|        |       |    |     |     |    |     | |    |     |     |    |     |
| NSE    | RGB   |    |     |     |    |     | |    |     |     |    |     |
|        | MDN   |    |     |     |    |     | |    |     |     |    |     |
|        |       |    |     |     |    |     | |    |     |     |    |     |
| RSC    | RGB   |    |     |     |    |     | |    |     |     |    |     |
|        | MDN   |    |     |     |    |     | |    |     |     |    |     |
|        |       |    |     |     |    |     | |    |     |     |    |     |
| TRN    | RGB   |    |     |     |    |     | |    |     |     |    |     |
|        | MDN   |    |     |     |    |     | |    |     |     |    |     |
|        |       |    |     |     |    |     | |    |     |     |    |     |
| CMB    | RGB   |    |     |     |    |     | |    |     |     |    |     |
|        | MDN   |    |     |     |    |     | |    |     |     |    |     |


Instruction to run the code:

1. Run the Docker Image.
```
./run.sh
```

2. Edit the ```config``` files accordingly to the experiment you want to execution. 

3. Run the ```train``` or ```test```.
```
python src/$select_a_file$.py
```

Pretrained models available at: [weigths](https://drive.google.com/drive/folders/193iP4iZiQivgZ7WOROoHkmQ1TAu-LaH_?usp=share_link)
# 
