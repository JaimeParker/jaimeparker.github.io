---
title: "T-Motor F60Pro Kv2550 Parameters"
categories: tech
tags: [Aerial Robotics]
use_math: true
---

T-Motor F60Pro Kv2550 建模所需参数。

影响四旋翼偏航力矩的电机力矩系数，偏航力矩可以表示为：

$$
\tau_z = k_m F
$$

这应该是一个与电机有关的参数，See [交流电机力矩系数计算方法详解](https://www.sy-motor.net/news/2234.html)，[电机扭矩的计算和推导](https://motor.eetrend.com/content/2020/100050279.html)

现在用的电机是 **T-Motor F60PRO V KV2550**，see [F60PRO V 参数](https://uav-cn.tmotor.com/html/2022/Motor_0415/925.html)

通过与 **T-Motor** 交流，获得电机参数如下：

| 参数                   | 值                      |
| ---------------------- | ----------------------- |
| 槽极                   | 12N14P                  |
| 轴径                   | 4mm                     |
| 电机硅胶线规格         | 20# 150mm               |
| 电机重量（含线）       | 33.3g                   |
| 测试电压               | 16.8V（4s）             |
| 空载电流（10V）        | 1.6A                    |
| 最大功率（10s）        | 825W                    |
| 最大电流（10s）        | 50.7A                   |
| 定子绕组电感 $L_s$     | 6-10 $\mu H$            |
| 定子绕组电阻 $R_s$     | 30 +- 5 $\Omega$        |
| 电机极对数 p           | 7                       |
| 电机粘滞阻尼系数 b     | $4.9*10^{-5}$ $N.m/rpm$ |
| 电机转子转动惯量 J     | $6.47*10^{-7}$          |
| 电机反电动势系数 $K_e$ | 0.00375 $V/(rad/s)$     |

感谢T-Motor公司（江西新拓实业有限公司） Kiko & 蒋女士提供 P60 Pro Kv2550 详细参数。

感谢朱洪正同学帮助电机建模与仿真。