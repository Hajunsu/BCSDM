from franka_panda import PandaRobot

root = 'results/SE3'
identifier = 'SE3_bc-deepovec'
config_file = 'SE3_bc-deepovec.yml'
ckpt_file = 'model_best.pkl'

import time

t1 = time.time()
robot = PandaRobot(holder=True, device='cuda:0') # holder_move=T, cpu -> lieflow error, device="cpu"
t2 = time.time()
print('load panda done :', t2 - t1)
robot.load_demo('datasets/SE3_demos.pt', vis_demo=False)
t3 = time.time()
print("load demo done :", t3 - t2)
robot.load_model(root, identifier, config_file, ckpt_file)
print("load model done :", time.time() - t3)
robot.simulate_VF(vis_path=False)