# WAY: Estimation of Vessel Destination in Worldwide AIS Trajectory</br>(IEEE Transactions on Aerospace and Electronic Systems) 
This is a Pytorch Implementation of [WAY: Estimation of Vessel Destination in Worldwide AIS Trajectory](https://ieeexplore.ieee.org/document/10107762).


<p align="center">
<img src="/img/Fig-Annotation_Framework_Overview.png" width="900" height="250">   
  <br>Visualization of annotation framework for extracting trajectories from raw AIS data
</p>
<br>
<p align="center">
<img src="/img/Fig-Model_Overview.png" width="900" height="410">   
  <br>Overall model architecture of WAY
</p>

## Environment & Python Requirements
![Windows](https://img.shields.io/badge/Windows-10&11-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-18.04+-E95420?style=for-the-badge&logo=ubuntu&logoColor=E95420)
![Python](https://img.shields.io/badge/Python-3.8.8-3776AB?style=for-the-badge&logo=python&logoColor=FFEE73)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=EE4C2C)   
* We recommend you to visit [Previous Versions (v1.7.1)](https://pytorch.org/get-started/previous-versions/#v171) for instructions to install **PyTorch** with torchvision==0.8.2.

Use the [requirements.txt](/requirements.txt) to install the rest of Python dependencies.   
```bash
$ pip install -r requirements.txt
```
## Experimental Results
The trajectory begins from the red marker 🔴 (departure port), and progresses to the blue marker 🔵 (destination port).
Spatial grids are colored green 🟩 if the model estimated the correct destination at the corresponding phase of the ship operation, else orange 🟧.
<p align="center">
<img src="/img/Fig-model_estimation_example.png" width="900" height="370">   
  <br>The visualization of estimation correctness on a single test example between each comparison model along the ship trajectory progression
</p>

## Citation
```
@article{kim2023way,
  title={WAY: Estimation of Vessel Destination in Worldwide AIS Trajectory},
  author={Kim, Jin Sob and Park, Hyun Joon and Shin, Wooseok and Han, Sung Won},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  year={2023}
}
```

## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This repository is released under the [MIT](https://choosealicense.com/licenses/mit/) license.
