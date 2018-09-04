## WALL-E
* Contributors: Tianbing Xu (Baidu Research, CA), Andrew Zhang (Stanford University), Liang Zhao (Baidu Research, CA), Lenjoy (Houzz), Shunan Zhang(Apple).
* An Efficient, Fast, yet Simple Reinforcement Learning Research Framework codebase with potential applications in Robotics and beyond.

## Motivations:
This is a long term Reinfocement Learning project focusing on developping an efficient yet simple RL framework to support
the ongoing RL research related to systems, methodologies and so on.
The first completed milestone is to speedup RL with multi-process architecture support. In RL, the time to collect expereince
by running policy on the environment MDP is a bottleneck, it takes much more time compared to the computaions of policy learning on GPU.
With the multi-process support, we are able to collect experience parallelly and thus to reduce the data collection time nearly linearly.

## Dependencies

* Python 3.6
* The Usual Suspects: NumPy, matplotlib, scipy
* TensorFlow
* gym - [installation instructions](https://gym.openai.com/docs)
* [MuJoCo](http://www.mujoco.org/) (30-day trial available and free to students)
* Pickle

Refer to requirements.txt for more details.

If you are using `conda`
```
conda env create -f conda_walle.yml --prefix=`which conda`/../../envs/walle
source activate walle
```

### Running Command

#### Single Process, using one CPU to collect experience
```
cd ./src
CUDA_VISIBLE_DEVICES=0
python main.py HalfCheetah-v2 -it 1000 -b 10000
```

#### Multi-Process, Parallel Sampler (10 CPU Processes to collect experience)
```
cd ./src
CUDA_VISIBLE_DEVICES=0
python run_parallel_main.py HalfCheetah-v2 -it 1000 -b 1000 -n 10
```

#### Plot Curve
```
cd ./src/experiment
python plotcurve.py -x xvariable -i /path-to-log/ -o fig.png
python plotcurve_cmp.py -x xvariable -i /path-to-log/ -b /path-to-baseline-log/ -o fig.png
```

#### Results on Half-Cheetah-V2
![Compare the performance between number of process=10 vs 1](https://github.com/tianbingsz/WALL-E/tree/master/Doc/cmp-cheetah-08-30-n=10-vs-n=1.png)

## Reference
* Danijar Hafner, James Davidson, Vincent Vanhoucke, "TensorFlow Agents: Efficient Batched Reinforcement Learning in TensorFlow"
* Kevin Frans, Danijar Hafner, "Speeding Up TRPO Through Parallelization and Parameter Adaptation"
* Hao Liu, Yihao Feng, Yi Mao, Dengyong Zhou, Jian Peng, Qiang Liu,
"Action-depedent Control Variates for Policy Optimization via Stein's Identity"
