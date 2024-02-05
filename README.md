# Optical Cavity Control for Lock Maintaining using Q-Learning

After having configured the Red Pitaya as described in the paper, 
run the following line for the software relock:

```console
python src/redpitaya.py
```

To train or test Q-Learning:
```console
python src/qlearning.py
```

Training and testing can be changed by simply passing to
the `RedPitayaQLearning` constructor `load=True` and `test=True`
```python
rpql = RedPitayaQLearning('169.254.167.128', load=True, test=True)
```

During the experiments we conducted the used also a LabJack 
to which two tmp35 sensors are connected to monitor the temperature
in the optical cavity. In the case you do not have it, just comment
all the lines were `u3` and `tmp35` are invoked in `RedPitayaQLearning`.