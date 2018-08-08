---
updated: "2018-07-10"
layout: post
list_title: 自动驾驶入门（一）| Udacity Autonomous Driving Part 1 | Bayers Rule
title: 贝叶斯概率模型
categories: [AI,Autonomous-Driving]
mathjax: true
---

> Course Notes from Udacity Self-Driving Car Nanodegree Program



## 附录

- Solution python code for Project: Joy Ride

```python
car_parameters = {"throttle": 0, "steer": 0, "brake": 0}

def control(pos_x, pos_y, time, velocity):
    """ Controls the simulated car"""
    global car_parameters
    if(time < 3):
        car_parameters["throttle"] = -1.0
        car_parameters["steer"] = 25
        car_parameters["brake"] = 0
    elif(pos_y > 32):
        car_parameters["throttle"] = -1.0
        car_parameters["steer"] = -25
        car_parameters["brake"] = 0
    else:
        car_parameters["throttle"] = 0
        car_parameters["steer"] = 0
        car_parameters["brake"] = 1
    
    return car_parameters
    
import src.simulate as sim
sim.run(control)
```
- Video on [Youtube](https://www.youtube.com/watch?v=pYvCvNFZFMw)