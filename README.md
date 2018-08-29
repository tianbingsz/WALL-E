## WALL-E
* Contributors: Tianbing Xu (Baidu Research, CA), Andrew Zhang (Stanford University, CA), Liang Zhao (Baidu Research, CA)
* An Efficient, Fast, yet Simple Reinforcement Learning Research Framework codebase with potential applications in Robotics and beyond.

## Dependencies

* Python 3.6
* The Usual Suspects: NumPy, matplotlib, scipy
* TensorFlow
* gym - [installation instructions](https://gym.openai.com/docs)
* [MuJoCo](http://www.mujoco.org/) (30-day trial available and free to students)
* Pickle
* Refer to requirements.txt for details

### Running Command

#### Single Process
```
cd ./src
python main.py HalfCheetah-v2 -it 1000 -b 10000
```

#### Multi-Process, Parallel Sampler (10 Processes)
```
cd ./src
python run_parallel_main.py HalfCheetah-v2 -it 1000 -b 1000 -n 10
```

#### Plot Curve
```
cd ./src/experiment
python plotcurve.py -x xvariable -i /path-to-log/ -o fig.png
```

## Reference
* Danijar Hafner, James Davidson, Vincent Vanhoucke, "TensorFlow Agents: Efficient Batched Reinforcement Learning in TensorFlow"
* Kevin Frans, Danijar Hafner, "Speeding Up TRPO Through Parallelization and Parameter Adaptation"
* Hao Liu, Yihao Feng, Yi Mao, Dengyong Zhou, Jian Peng, Qiang Liu,
"Action-depedent Control Variates for Policy Optimization via Stein's Identity"
