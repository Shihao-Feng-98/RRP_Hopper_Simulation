# RRP Hopper
The pybullet simulation of RRP hopper based on the Raibert decoupled controller.

(Reference: [https://github.com/YuXianYuan/Hopping-in-Three-Dimensions](https://github.com/YuXianYuan/Hopping-in-Three-Dimensions))

## Controller paremeter adjustment
1.  Fixed the base. Selecting the joint PD gain for the FIGHT phase that can track the given foot trajectory (Add the damping gain on the P joint is recommended).
2.  Fixed the base on the vertical dimensional. Adding the thrust, and adjusting the stiffness gain of the P joint that keeps stable hopping.
3.  Without constrained. Setting a small body velocity gain, and adjusting the PD gain for body attitude control that keeps the body stable.
4.  Adjusting the body velocity gain that makes the body track the given velocity command.

## Performances [demo](https://www.bilibili.com/video/BV1WQ4y1q7sa?from=search&seid=2053570693863558656&spm_id_from=333.337.0.0)
Hopping speed up to 2m/s

In the case of hopping in place, resist the 0.5 roll/pitch disturbance

Horizontal velocity  tracking accuracy ±0.2m/2

Body attitude change within 0.1 while hopping stability

Hard to track a low speed < 0.3m/s, because the body can not go through the neutral point and resulting in taking a step back

(Note:  You can  greatly improve the dynamic performance by adjusting the parameter)

## Dependency
**Pinocchio** - 2.6.3 [https://github.com/stack-of-tasks/pinocchio](https://github.com/stack-of-tasks/pinocchio)

**Pybullet** - 3.1.7 [https://pybullet.org/wordpress/](https://pybullet.org/wordpress/)

## Author: 
Shihao Feng

13247344844@163.com
