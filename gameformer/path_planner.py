import scipy
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
try:
    from common_utils import *
except:
    from .common_utils import *
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

class PathOptmizer:
    def __init__(self, path_len, ds=0.1):
        self.N = int(path_len/ds)
        self._init_optimization()

        # Set default solver options (quiet)
        options = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.max_iter": 50}
        self._optimizer.solver("ipopt", options)
    
    def _init_optimization(self):
        self._optimizer = cs.Opti()  # Optimization problem
        self._create_decision_variables()
        self._create_parameters()
        self._create_constraints()
        self._set_objective()

    def _create_constraints(self):
        self._optimizer.subject_to(self.l[0] == 0)

    def _create_decision_variables(self):
        self.l = self._optimizer.variable(self.N)

    def _create_parameters(self):
        self.l0 = self._optimizer.parameter()
        self.corridor_ub = self._optimizer.parameter(self.N)
        self.corridor_lb = self._optimizer.parameter(self.N)
    
    def _set_objective(self):
        ref_l = (self.corridor_lb + self.corridor_ub) / 2
        cost = 0.1 * cs.sumsqr(self.l[1:] - ref_l[1:]) + cs.sumsqr(cs.diff(self.l)) + 10 * cs.sumsqr(cs.diff(self.l, 2))
        self._optimizer.minimize(cost)

    def set_corridor(self, corridor_ub, corridor_lb):
        self._optimizer.set_value(self.corridor_ub, cs.DM(corridor_ub))
        self._optimizer.set_value(self.corridor_lb, cs.DM(corridor_lb))

    def solve(self, corridor_ub, corridor_lb):
        self.set_corridor(corridor_ub, corridor_lb)
      
        try:
            solution = self._optimizer.solve()
            path = solution.value(self.l)
        except:
            path = self._optimizer.debug.value(self.l)

        return path

path_optmizer = PathOptmizer(50, 0.1)

class PathPlanner:
    def __init__(self):
        self.path_len = 50 # [m]
        self.ds = 0.1 # [m]
        self.N = int(self.path_len/self.ds)
        self.ego_width = WIDTH

    def plan(self, ego_state, objects, ref_path):
        corridor_lb, corridor_ub = self.get_corridor(objects, ego_state, ref_path)
        path = path_optmizer.solve(corridor_ub, corridor_lb)

        if len(ref_path) > self.N:
            path = np.append(path, path[-1] * np.ones(len(ref_path) - self.N))
        else:
            path = path[:len(ref_path)]

        s = np.arange(0, len(ref_path)) * 0.1
        path = np.column_stack([s, path])

        return path

    def get_corridor(self, objects, ego_state, ref_path):
        corridor_ub = np.ones(self.N) * 3
        corridor_lb = np.ones(self.N) * -3
        ego_state = (ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading)

        for obj in objects:
            x, y = obj.box.center.x, obj.box.center.y
            length, width = obj.box.length, obj.box.width
            length = np.clip(length, 3, 20)
            t = obj.tracked_object_type
            s, l = self.transform_to_ego_Frenet(x, y, ego_state, ref_path)

            if t == TrackedObjectType.VEHICLE:
                v = obj.velocity.magnitude()
                if np.isclose(v, 0) and s > 0 and s < self.path_len and abs(l) > 0.6 and abs(l) < 1.8:
                    start = int(((s - length)*10).clip(0, self.N))
                    end = int(((s + length)*10).clip(start+1, self.N))
                    if l > 0.6:
                        corridor_ub[start:end] = np.minimum(corridor_ub[start:end], l - width - self.ego_width)
                    else:
                        corridor_lb[start:end] = np.maximum(corridor_lb[start:end], l + width + self.ego_width) 
            else:
                if s > 0 and s < self.path_len and abs(l) < 1.8:
                    start = int(((s - length)*10).clip(0, self.N))
                    end = int(((s + length)*10).clip(start+1, self.N))
                    if l > 0:
                        corridor_ub[start:end] = np.minimum(corridor_ub[start:end], l - width/2 - self.ego_width/2)
                    else:
                        corridor_lb[start:end] = np.maximum(corridor_lb[start:end], l + width/2 + self.ego_width/2)

        return corridor_lb, corridor_ub

    @staticmethod
    def transform_to_ego_Frenet(x, y, ego, ref_path):
        def rotate(x, y, theta):
            x_rot = x * np.cos(theta) - y * np.sin(theta)
            y_rot = x * np.sin(theta) + y * np.cos(theta)
            
            return x_rot, y_rot

        def transform_to_ego_frame(x, y, ego):
            x_ego = x - ego[0]
            y_ego = y - ego[1]
            x_ego, y_ego = rotate(x_ego, y_ego, -ego[2])
        
            return np.array([x_ego, y_ego])

        point = transform_to_ego_frame(x, y, ego)
        distance_to_ref_path = scipy.spatial.distance.cdist(point[None, :], ref_path[:, :2])
        frenet_idx = np.argmin(distance_to_ref_path[0])
        ref_point = ref_path[frenet_idx]

        frenet_s = frenet_idx * 0.1
        e = np.sign((point[1] - ref_point[1]) * np.cos(ref_point[2]) - (point[0] - ref_point[0]) * np.sin(ref_point[2]))
        frenet_l = np.linalg.norm(point - ref_point[:2]) * e 

        return frenet_s, frenet_l
    