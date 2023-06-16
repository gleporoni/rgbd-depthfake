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
Accuracy results obtained on deepfake detection task for RAW and C40 dataset settings when  Blur (BLR), Noise (NSE), Rescale (RSC), Translation (TRN), and all Combined (CMB) black box attacks are applied. The best results are in bold.

| Attack | Model |             |             | RAW         |             |             | |             |             | C40         |             |            |
|--------|-------|-------------|-------------|-------------|-------------|-------------|-|-------------|-------------|-------------|-------------|------------|
|        |       | DF          | F2F         | FS          | NT          | ALL         | | DF          | F2F         | FS          | NT          | ALL        |
| BLR    | RGB   | 50,32 %     | 54,30 %     | 50,45 %     | 49,00 %     | **79,74** % | | 66,03 %     | 67,82 %     | 58,02 %     | **56,00** % | 79,27 %    |
|        | MDN   | **50,98** % | **70,38** % | **50,62** % | **50,73** % | 79,61 %     | | **75,17** % | **74,60** % | **71,92** % | 49,70 %     | **79,90** %|
|        |       |             |             |             |             |             | |             |             |             |             |            |
| NSE    | RGB   | 85,80 %     | 88,73 %     | 93,83 %     | 76,21 %     | 92,06 %     | | 87,20 %     | 81,05 %     | 85,57 %     | 62,00 %     | 81,41 %    | 
|        | MDN   | **95,69** % | **95,43** % | **95,67** % | **91,61** % | **94,67** % | | **89,75** % | **81,69** % | **86,86** % | **70,65** % | **82,00** %|    
|        |       |             |             |             |             |             | |             |             |             |             |            |
| RSC    | RGB   | 60,63 %     | 60,60 %     | 50,58 %     | 53,89 %     | 74,48 %     | | 73,02 %     | 70,19 %     | 64,67 %     | **57,18** % | 80,11 %    | 
|        | MDN   | **61,62 %** | **75,64 %** | **50,62 %** | **60,80 %** | **78,76 %** | | **78,37 %** | **74,96 %** | **76,59 %** | 50,46 %     | **80,38** %|   
|        |       |             |             |             |             |             | |             |             |             |             |            |
| TRN    | RGB   | 95,87 %     | 95,11 %     | 95,19 %     | 91,63 %     | **94,89** % | | 87,63 %     | 81,23 %     | 84,83 %     | **70,13** % | 81,07 %    |   
|        | MDN   | **96,75** % | **95,49** % | **96,23** % | **92,13** % | 94,51 %     | | **89,76** % | **81,46** % | **86,24** % | 69,81 %     | **81,82** %|   
|        |       |             |             |             |             |             | |             |             |             |             |            |
| CMB    | RGB   | **50,31** % | 55,22 %     | 50,33 %     | 49,47 %     | 77,95 %     | | 58,73 %     | 66,32 %     | 55,75 %     | **50,71** % | 79,67 %    |    
|        | MDN   | 50,27 %     | **64,64** % | **50,61** % | **50,46** % | **79,80** % | | **71,96** % | **73,22** % | **65,96** % | 49,52 %     | **79,80** %|   


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
