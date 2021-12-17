import numpy as np
from elastica import *
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
class LegSimulator(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks
):
    pass
# Callback functions
# Add call backs
class RodCallback(CallBackBaseClass):
    """
    Call back function for testing joints
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            return
def run_leg_sim(leg_angle, foot_angle, dt=1e-4):
    leg_sim = LegSimulator()

    # setting up test params
    n_elem = 2
    start_rod_1 = np.array([0.0, 0.0, 1.5])
    direction1 = np.array([0.0, 0.0, -1.0])
    # direction2 = np.array([1.0, 0.0, 0.0])
    
#     leg_angle = 0
    direction2 = np.array([np.cos(leg_angle), 0.0, np.sin(leg_angle)])
    normal1 = np.array([1.0, 0.0, 0.0])
    # normal2 = np.array([0.0, 0.0, 1.0])
    normal2 = np.array([np.cos(leg_angle+np.pi/2), 0.0, np.sin(leg_angle+np.pi/2)])

    roll_direction = np.array([0.0, 1.0, 0.0])#np.cross(direction1, normal1)
    base_length = 0.5
    base_radius = 0.25
    base_area = np.pi * base_radius ** 2
    density = 5000
    nu = 0.1
    E = 1e6
    # For shear modulus of 1e4, nu is 99!
    poisson_ratio = 0
    shear_modulus = E / (poisson_ratio + 1.0)
    shear_modulus_unshearable = E / (-0.7 + 1.0)

    # Create rod 1
    torso = CosseratRod.straight_rod(
        n_elem,
        start_rod_1,
        direction1,
        normal1,
        base_length,
        base_radius,
        density,
        nu,
        E,
        shear_modulus=shear_modulus_unshearable,
        poisson_ratio=poisson_ratio
    )
    leg_sim.append(torso)
    start_rod_2 = start_rod_1 + direction1 * base_length

    straight_leg = CosseratRod.straight_rod(
        n_elem,
        start_rod_2,
        direction2,
        normal2,
        base_length,
        base_radius/2,
        density,
        nu,
        E,
        shear_modulus=shear_modulus_unshearable,
        poisson_ratio=poisson_ratio
    )
    leg_sim.append(straight_leg)
    start_rod_3 = start_rod_2 + direction2 * base_length
#     foot_angle = -np.pi/4
    direction3 = np.array([np.cos(foot_angle), 0.0, np.sin(foot_angle)])
    normal3 = np.array([np.cos(foot_angle+np.pi/2), 0.0, np.sin(foot_angle+np.pi/2)])

    foot = CosseratRod.straight_rod(
        n_elem,
        start_rod_3,
        direction3,
        normal3,
        base_length/3,
        base_radius,
        density,
        nu,
        E,
        shear_modulus=shear_modulus_unshearable,
        poisson_ratio=poisson_ratio
    )
    leg_sim.append(foot)

    leg_sim.constrain(torso).using(
        OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    # Connect rod 1 and rod 2
    leg_sim.connect(
        first_rod=torso, second_rod=straight_leg, first_connect_idx=-1, second_connect_idx=0
    ).using(
        HingeJoint, k=1e6, nu=0, kt=5e3, normal_direction=roll_direction
    )  # 1e-2

    leg_sim.connect(
        first_rod=straight_leg, second_rod=foot, first_connect_idx=-1, second_connect_idx=0
    ).using(
        FixedJoint, k=1e5, nu=0, kt=1e4
    )  # 1e-2
    final_time = 1
    # leg_sim.add_forcing_to(rod1).using(
    #     GravityForces
    # )
    # leg_sim.add_forcing_to(straight_leg).using(
    #     GravityForces, acc_gravity=np.array([0, 0, -9.8])
    # )
    leg_sim.add_forcing_to(foot).using(
        GravityForces, acc_gravity=np.array([0, 0, -9.8])
    )
    pp_list_torso = defaultdict(list)
    pp_list_straight_leg = defaultdict(list)
    pp_list_foot = defaultdict(list)
    leg_sim.collect_diagnostics(torso).using(
        RodCallback, step_skip=1000, callback_params=pp_list_torso
    )
    leg_sim.collect_diagnostics(straight_leg).using(
        RodCallback, step_skip=1000, callback_params=pp_list_straight_leg
    )
    leg_sim.collect_diagnostics(foot).using(
        RodCallback, step_skip=1000, callback_params=pp_list_foot
    )
    leg_sim.finalize()

    timestepper = PositionVerlet()
    # timestepper = PEFRL()

    dl = base_length / n_elem
#     dt = 1e-4
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    integrate(timestepper, leg_sim, final_time, total_steps)
    max_timesteps = len(pp_list_torso['time'])
    return pp_list_torso, pp_list_straight_leg, pp_list_foot
    
def leg_sim_fitness_func(leg_angle, foot_angle):
    is_leg_angle_valid = (leg_angle <= 0 and leg_angle >= -np.pi/2)
    is_foot_angle_valid = (foot_angle <= 0 and foot_angle >= -np.pi/2)
    if not is_leg_angle_valid or not is_foot_angle_valid:
        return 10
    _, _, pp_list_foot = run_leg_sim(leg_angle, foot_angle)
    end_node_z = []
    for timestep in range(max_timesteps):
        end_node_z.append(pp_list_foot['position'][timestep][2][-1])
    low_timestep = np.argmin(np.array(end_node_z))
    max_foot_vel = np.linalg.norm(pp_list_foot['velocity'][low_timestep][:,-1])
    return -max_foot_vel