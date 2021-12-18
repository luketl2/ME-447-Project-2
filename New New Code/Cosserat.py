#!/usr/bin/env python
# coding: utf-8

# In[318]:


import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
from matplotlib import animation as an

# This class is not completely general: only supports uniform mass and single radius
class cosserat_rod:
    # initialize the rod
    def __init__(self, r_o, Q_o, v_o, w_o, n_elements, rad, rho, E, G, lock_e: bool, damping: bool, milestone, ext_forces, ext_couples):
        self.tracker = 0 # used to track variables during testing
        self.milestone = milestone
        self.lock_e = lock_e
        self.damping = damping
        
        # forces
        self.ext_forces = ext_forces
        self.ext_couples = ext_couples
            
        # state vars
        self.r = r_o
        self.Q = Q_o
        self.v = v_o
        self.w = w_o
        
        self.Q_ref = Q_o
        
        # describe rod       
        self.L_ref, self.L_ref_norm = self.calc_L(self.r) 
        dL = np.sum(self.L_ref_norm)/n_elements
        self.n_elements = n_elements
        self.n_nodes = self.n_elements + 1
        self.D_ref = self.calc_D(self.L_ref_norm) # voronoi lengths
 
        # material props
        self.A = np.pi * rad * rad
        dm = rho * self.A * dL
        self.masses = dm*np.ones(self.n_nodes,)
        self.masses[0] *= 0.5
        self.masses[-1] *= 0.5
        self.G = G
        self.E = E
        self.S_ref = np.zeros((3, 3, n_elements)) # S
        alpha_c = 4.0 / 3.0
        self.S_ref[0, 0, :] = alpha_c * self.G * self.A # S1
        self.S_ref[1, 1, :] = alpha_c * self.G * self.A # S2
        self.S_ref[2, 2, :] = self.E * self.A # S2
        
        # moments of inertia
        self.I = np.zeros(3,)
        self.I[0:2] = 0.25 * np.pi * rad**4
        self.I[2] = 0.5 * np.pi * rad**4
        self.B_ref = np.zeros((3, 3, n_elements))
        self.J_ref = np.zeros((3, n_elements))
        for idx in range(n_elements):
            self.B_ref[:, :, idx] = np.diag(self.I * np.array([self.E, self.E, self.G]))
            self.J_ref[:, idx] = rho*self.L_ref_norm[:, 0]*self.I
            
        # Need B in voronoi domain  
        B_next = np.roll(self.B_ref, -1, -1)
        self.B_ref = (B_next[:, :, :-1] * self.L_ref_norm[:, 1:] + self.B_ref[:, :, :-1] * self.L_ref_norm[:, :-1])/ (2 * self.D_ref)  
    
    def calc_L(self, r):
        L = r[:, 1:] - r[:, :-1]
        L_norm = la.norm(L, axis=0, keepdims=True)
        return L, L_norm
    
    def calc_D(self, L_norm):
        return 0.5 * (L_norm[0, 1:] + L_norm[0, :-1])

    def inverse_rotate(self, t_frameone, t_frametwo):
        # this einsum is Q*R.T for two matrices Q and R
        R = np.einsum('ijk, ljk -> ilk', t_frametwo, t_frameone)
        
        angle = np.arccos((np.trace(R)-1.0)/2.0)
        angle = np.nan_to_num(angle, 0)
        K = (R-np.transpose(R, (1,0,2)))/(2.0*np.sin(angle))
        about = np.array([-K[1,2,:], K[0,2,:], -K[0,1,:]])        
        about = np.nan_to_num(about, 0)
        
        return about*angle
    
    def rotate_rodrigues(self, t_frame, t_angle, about):
        # Convert about to np.array and normalize it
        about = np.array(about)
        norm = np.linalg.norm(about, None, 0) # along directions
        about = about / (norm + 1E-14)

        I = np.zeros((3, 3, self.n_elements))
        K = np.zeros((3, 3, self.n_elements))
        
        for elem in range(self.n_elements):
            I[:, :, elem] = np.eye(3)
            K[:, :, elem] = [[0.0, -about[2, elem], about[1, elem]], 
                             [about[2, elem], 0.0, -about[0, elem]], 
                             [-about[1, elem], about[0, elem], 0.0]]  
            
        # rot_matrix = I + K @ (s_angle * I + (1-c_angle)* K_mat)
        K_mat = np.einsum('ijk, lik-> ljk', K,  K)
        R = I + (np.sin(t_angle) * K) + ((1.0 - np.cos(t_angle)) * K_mat)
        
        R_T = np.transpose(R, (1,0,2))
        
        rot_frame = np.einsum('ijk, jlk-> ilk', R_T, t_frame)

        return rot_frame

    def deltaH_operator(self, t_x): 
        n_pad = [(0,0)] * t_x.ndim
        n_pad[t_x.ndim - 1] = (0,1)
        temp = np.pad(t_x, n_pad, 'constant', constant_values=(0,0))
        return temp - np.roll(temp, 1, t_x.ndim - 1)
    
    def Ah_operator(self, t_x):        
        n_pad = [(0,0)] * t_x.ndim
        n_pad[t_x.ndim - 1] = (0,1)
        temp = np.pad(t_x, n_pad, 'constant', constant_values=(0,0))
        return (temp + np.roll(temp, 1, t_x.ndim - 1)) / 2
        
    # F = ma type calculations - key is to use position at half time step to recalc internal forces, then a
    def Newton2(self, x, Q):        
        '''Update lengths at half positions'''
        L, L_norm = self.calc_L(x) 
        D = self.calc_D(L_norm)
                
        if self.lock_e: # if dilatations is constrained to 1
            e = np.ones((1, self.n_elements))
            e_v = np.ones((1, self.n_elements - 1))
        else:
            e = L_norm / self.L_ref_norm 
            e_v = D / self.D_ref # voronoi dilatations
           
        tangents = L / L_norm
        
        '''Linear accelerations'''
        # damping
        if self.damping:
            gamma = 100
        else:
            gamma = 0
        
        linear_damping = -self.v*gamma
        
        # this einsum is matrix-vector multiplication for 3 x 3 x k and 3 x k
        sigma = np.einsum('ijk, jk-> ik', Q, e * tangents - Q[2]) # in material frame 
        internal_forces = np.einsum('jil, jkl, kl -> il', Q, self.S_ref, sigma)  
        node_forces = self.deltaH_operator(internal_forces / e)
        dvdt = (node_forces + linear_damping + self.ext_forces) / self.masses 
        
        "Angular accelerations"
        if self.n_elements == 1: # for beam stretching validation
            return dvdt, np.zeros((3, 1))
        
        # damping
        rotational_damping = -self.w*gamma
        
        # calculate kappa, 3 x n_elem-1 
        self.kappa_ref = -self.inverse_rotate(Q[:, :, :-1], Q[:, :, 1:]) / self.D_ref 
        
        # bend internal couple, 3 x n_elem 
        BK_product = np.einsum('ijk, jk -> ik', self.B_ref, self.kappa_ref) # 3 x n_elements - 1
        bend_couple = self.deltaH_operator(BK_product / e_v**3)
        
        # twist internal couple, 3 x n_elem 
        KB_cross = np.cross(self.kappa_ref, BK_product, axis = 0)
        twist_couple = self.Ah_operator(KB_cross*self.D_ref / e_v**3)
        
        # shear stretch internal couple, 3 x n_elem
        Qt_product = np.einsum('ijk, jk-> ik', Q, tangents)
        Ssigma_product = np.einsum('ijk, jk-> ik', self.S_ref, sigma)
        shear_stretch_couple = np.cross(Qt_product, Ssigma_product, axis = 0) * self.L_ref_norm
        couples = rotational_damping + bend_couple + twist_couple + shear_stretch_couple + self.ext_couples
        dwdt = couples / (self.J_ref / e)
        
        return dvdt, dwdt

    def position_verlet(self, dt, x, v, Q, w):
        '''Half positions update'''
        x_half = x + 0.5*dt*v
        Q_half = self.rotate_rodrigues(Q, np.linalg.norm(0.5*dt*w), 0.5*dt*w)
        
        '''Accelerations --> velocities update'''
        dvdt, dwdt = self.Newton2(x_half, Q_half)
        self.v = v + dt * dvdt
        self.w = w + dt * dwdt
        
        '''Dirichlet condition'''
        self.set_BC(self.milestone)
        
        '''Final positions update'''
        self.r = x_half + 0.5 * dt * self.v
        self.Q = self.rotate_rodrigues(Q_half, np.linalg.norm(self.w*0.5*dt), self.w*0.5*dt)

    def set_BC(self, milestone):
        if milestone == 1:
            self.r[:, 0] = np.zeros(3,)
            self.v[:, 0] = np.zeros(3,)
        elif milestone == 2:
            self.v[:, 1] = np.zeros(3,)
        elif milestone == 3:
            self.r[:, 0] = np.zeros(3,)
            self.v[:, 0] = np.zeros(3,)
            self.Q[:, :, 0] = self.Q_ref[:, :, 0]
            self.w[:, 0] = np.zeros(3,)
            
    def get_state(self, keys):
        return_vals = []
        state_vars = {'r': self.r, 'v': self.v, 'Q': self.Q, 'w': self.w, 'tracker': self.tracker}
        for key in keys:
            if key in state_vars.keys():
                return_vals.append(state_vars[key])
                            
        return tuple(return_vals)
    
    # run the simulation, specifying external conditions ** for now it's just for first benchmark
    def step(self, dt):
        self.position_verlet(dt, self.r, self.v, self.Q, self.w)




