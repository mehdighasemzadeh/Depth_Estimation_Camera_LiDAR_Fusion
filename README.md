# Depth Estimation by Camera and LiDAR Fusion

This work proposes a Recurrent-CNN-based network for depth estimation using an event camera, a frame camera, and a LiDAR sensor, we evaluate our model on [The MVSEC Dataset](https://daniilidis-group.github.io/mvsec/)  and [The Event-Scape Dataset](https://rpg.ifi.uzh.ch/RAMNet.html) .

## Abstract
<p align="justify">
Depth estimation is one of the most important tasks in autonomous driving, and LiDAR sensors could accomplish this task accurately but their data is sparse. On the other hand, monocular methods for this task are not as accurate and robust as LiDAR sensors, therefore, combining LiDAR data with cameras could boost the robustness and precision of the task, and also solve the sparsity of LiDAR data. Three sensors are used in this work, two types of cameras (a frame camera and an event camera) and a LiDAR sensor, so this network uses three types of sensors as the input, and generates depth information on the output.
</p>

**videos from results:**

Day1 on The MVSEC Dataset
[![Watch the video](pic/day1.png)](https://youtu.be/qXXOawgLiWM?si=lunNW49saccludHj)
P: Our network prediction, GT: Ground Truth




Night1 on The MVSEC Dataset
[![Watch the video](pic/night1.png)](https://youtu.be/QkP9GXNrDvw?si=FX6sAdvqsLnr1Eq6)
P: Our network prediction, GT: Ground Truth




Night2 on The MVSEC Dataset
[![Watch the video](pic/night2.png)](https://youtu.be/2xh3fDA7ifs?si=duBZOW8rWzrtZXgC)
P: Our network prediction, GT: Ground Truth




Night3 on The MVSEC Dataset
[![Watch the video](pic/night3.png)](https://youtu.be/pswhx0yAZKo?si=_1DzNFz7QHKUoljJ)
P: Our network prediction, GT: Ground Truth



