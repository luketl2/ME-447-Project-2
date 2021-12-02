from elastica import *
from numpy import record

class StretchingBeamSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass

stretch_sim = StretchingBeamSimulator()

# Simulation Parameters
final_time = 20
n_elem = 19
start = np.zeros((3,))
direction = np.array([1.0, 0., 0.]) # rod direction x
normal = np.array([0., 1.0, 0.]) # rod normal in y dir
base_length = 1.0 # m
base_radius = 0.025 # m
density = 1000 # kg/m^3
youngs_modulus = 1E4 # 10 Kpa
poisson_ratio = 0.5
nu = 2.0 # internal dissipation of the rod

strechable_rod = CosseratRod.straight_rod(
    n_elem, 
    start, 
    direction,
    normal, 
    base_length, 
    base_radius,
    density, 
    nu, 
    youngs_modulus,
    poisson_ratio
)
stretch_sim.append(strechable_rod) # Rod now added to the simulator

# Boundary Conditions
# OneEndFixedRod class constrains one end of the rod.
stretch_sim.constrain(strechable_rod).using(
    OneEndFixedRod, 
    constrained_position_idx=(0,), 
    constrained_director_idx=(0,)
)
# External Forces
# We will add tip force at the end of the rod.
# EndpointForces class
end_force = np.array([1.0, 0., 0.]) # End force in x direction, 1N
stretch_sim.add_forcing_to(strechable_rod).using(
    EndpointForces,
    0.0*end_force, # base force
    end_force, #tip force
    ramp_up_time=1E-2 # ramp up force from base to tip in seconds
)

# CallBacks for diagnostics
class AxialStretchingCallBack(CallBackBaseClass):
    def __init__(self, step_skip, callback_postprocessing_dict: dict):
        CallBackBaseClass.__init__(self)

        self.step_skip = step_skip
        self.callback_postprocessing_dict = callback_postprocessing_dict

    def make_callback(self, system, time, current_step: int):
        if current_step % self.step_skip == 0:
            self.callback_postprocessing_dict["time"].append(time)
            # We are collecting only x position of the tip
            # position_collection (3, n_nodes)
            self.callback_postprocessing_dict["position"].append(system.position_collection[0, - 1].copy())

            self.callback_postprocessing_dict["velocity"].append(system.velocity_collection[0, -1].copy())
        return

# Dict for collecting rod data, we will use this in callback
recorded_rod_history = defaultdict(list)

stretch_sim.collect_diagnostics(strechable_rod).using(
    AxialStretchingCallBack, 
    step_skip=200,
    callback_postprocessing_dict=recorded_rod_history
)

# Final step before time integration
stretch_sim.finalize()

timestepper = PositionVerlet()
dl = base_length/n_elem
dt = 0.01* dl
total_steps = int(final_time/dt)

integrate(timestepper, stretch_sim, final_time, total_steps)


from matplotlib import pyplot as plt

# First order approximation for the final length rod
base_area = np.pi * base_radius**2
expected_tip_disp = end_force[0] * base_length / base_area / youngs_modulus
# Improved approximation
expected_tip_disp_improved = end_force[0] * base_length / (base_area * youngs_modulus - end_force[0])

# Plot the results
fig = plt.figure(figsize=(10,0), frameon=True, dpi=150)
ax = fig.add_subplot(111)
ax.plot(recorded_rod_history["time"], recorded_rod_history["position"], lw=2.0, label="Simulation")
ax.hlines(base_length + expected_tip_disp, 0.0, final_time, "k", "dashdot", lw=1.0, label="first order")
ax.hlines(base_length + expected_tip_disp_improved, 0.0, final_time, "k", "dashed", lw=1.0, label="second order")
fig.legend(prop={'size': 15}, loc = "upper center")
plt.show()
