import json
import pydart2 as pydart

import numpy as np
from math import exp, pi, log, acos, sqrt
import copy
import glob

def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    for people in data['people']:
        kp = np.asarray(people['pose_keypoints_2d']).reshape((-1,3))
        kps.append(kp)
        return kps

def main():
    cur_path = '/'.join(__file__.split('/')[:-1])
    world = pydart.World(1./150., cur_path+'/../../data/skel/human_mass_limited_dof_v2.skel')
    world.control_skel = world.skeletons[1]
    skel = world.skeletons[1]

    pelvis = skel.body(0)
    p_pelvis = pelvis.world_transform()[:3, 3]
    R_pelvis = pelvis.world_transform()[:3, :3]

    print(R_pelvis.T)

main()