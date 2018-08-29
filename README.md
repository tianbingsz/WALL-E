## WALL-E
* Contributors: Tianbing Xu, Andrew Zhang
* An Efficient yet simple Reinforcement Learning Research Framework codebase with potential applications in Robotics and so on.

## Dependencies

* Python 3.6
* The Usual Suspects: NumPy, matplotlib, scipy
* TensorFlow
* gym - [installation instructions](https://gym.openai.com/docs)
* [MuJoCo](http://www.mujoco.org/) (30-day trial available and free to students)

### Running Command

#### Single Process
```
cd ./src
python main.py HalfCheetah-v2 -it 1000 -b 10000
```

#### Multi-Process, Parallel Sampler
```
cd ./src
python run_parallel_main.py HalfCheetah-v2 -it 1000 -b 1000 -n 10
```

#### plot curve


## Reference
* Kevin Frans, Danijar Hafner, "Speeding Up TRPO Through Parallelization and Parameter Adaptation"
* Hao Liu, Yihao Feng, Yi Mao, Dengyong Zhou, Jian Peng, Qiang Liu,
"Action-depedent Control Variates for Policy Optimization via Stein's Identity"
