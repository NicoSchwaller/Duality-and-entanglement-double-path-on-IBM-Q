#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Needed for functions
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Import Qiskit classes
import qiskit 
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, IBMQ, execute
from qiskit.providers.aer import noise
from qiskit.quantum_info import state_fidelity
from qiskit.tools.visualization import plot_histogram

# Tomography functions
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
import qiskit.ignis.mitigation.measurement as mc

# Import measurement calibration functions
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, CompleteMeasFitter, MeasurementFilter)


# In[2]:


# Load IBMQ account
provider = IBMQ.load_account()


# In[3]:


qr = qiskit.QuantumRegister(2)
meas_calibs, state_labels = complete_meas_cal(qubit_list=[0,1], qr=qr, circlabel='mcal')

# Execute the calibration circuits without noise
backend = qiskit.Aer.get_backend('qasm_simulator')
job = qiskit.execute(meas_calibs, backend=backend, shots=1000)
cal_results = job.result()

# The calibration matrix without noise is the identity matrix
meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')


# Creation of the noise model
# 
# 
# *   Set real_device to 1 and create the noise model from any real device
# 
# or
# *   Set real_device to 0 and add noise sources manually to the noise model
# 
# 

# In[4]:


#Create a noise model from a real device or specified by user? (0=manual, 1=real)
real_device=1

#User-specified noise model
if real_device==0:
   noise_model = noise.NoiseModel()
   device = Aer.get_backend('qasm_simulator')
  
   for qi in range(2):
      read_err = noise.errors.readout_error.ReadoutError([[0.9, 0.1],[0.25,0.75]])
      noise_model.add_readout_error(read_err, [qi])

#Noise model from real device
elif real_device==1:
   #Select backend to extract noise model from
   device = provider.get_backend('ibmqx2')

   #Create noise model
   properties = device.properties()
   coupling_map = device.configuration().coupling_map
   noise_model = noise.device.basic_device_noise_model(properties)
   basis_gates = noise_model.basis_gates

# Execute the calibration circuits using the noise model
backend = qiskit.Aer.get_backend('qasm_simulator')
noisy_job = qiskit.execute(meas_calibs, backend=backend, shots=1000, noise_model=noise_model)
results = noisy_job.result()

# Calculate the calibration matrix
meas_fitter = CompleteMeasFitter(results, state_labels, circlabel='mcal')
meas_filter = meas_fitter.filter

# What is the measurement fidelity?
print("Average Measurement Fidelity: %f" % meas_fitter.readout_fidelity())


# **Setting up the experiment**
# 
# The four following parameters can be set before to run the experiment:
# 
# *   Simulate noise using the Aer Qasm simulator from the noise model
# *   Apply error mitigation (from the noise model) to the computation
# *   Run in random mode, reaching any random values of V_k, P_k and C (otherwise experimental angles will be set to compute 13 points reaching all extremal values for V_A, P_A and C
# *   Choice of the backend to use for the experiment (has no effect if noise_simulation is set to 1)
# 
# Default values are all set to 0, using the Aer Qasm simulator.
# 
# You can also specify the path where the data will be registered (.txt files useful for further analysis)

# In[5]:


# Simulate noise with the noise model? (the simulator will be used)
noise_simulation=0
# Apply error mitigation from the noise model to the result?
correction=0
# Pick random points? (otherwise 13 points will be selected to reach equally the 1/8 of sphere)
rand=0

backend_1 = provider.get_backend('ibmqx2')
simulator = qiskit.Aer.get_backend('qasm_simulator')
# Choose the backend to use
backend = simulator

# Set path to store data
path="drive/My Drive/C_VDC/"


# In[6]:


# Preparing 13 experimental states lying on the V,P,C sphere
for j in range(1,14):
   j=14-j

   # number of measurements to be done for each point = 10, useful to analyse dispersion and errors
   for f in range (1,11):
 
      delta_xi=np.pi/8
   
      xi_inf_1=0.5
      xi_sup_1=4

      xi_inf_1=4
      xi_sup_1=7
  
      # Loop on BS2_bit: first compute the circuit with the beam splitter BS2, then without it (to compute P and C)
      for BS2_bit in range(0,2):

         if BS2_bit==0:
            BS2=1
         if BS2_bit==1:
            BS2=0

         str_BS2='_state'
         if BS2==1:
            str_BS2='_interference_BS2'
         
         # Parameters choosed for the 13 states lying on the sphere
         if j==1:
            phi_D=np.pi/4
            theta_sphere=np.pi/2
         if j==2:
            phi_D=0.523599
            theta_sphere=np.pi/2
         if j==3:
            phi_D=0.261825
            theta_sphere=np.pi/2
         if j==4:
            phi_D=0
            theta_sphere=0
         if j==5:
            phi_D=np.pi/4
            theta_sphere=0.9817
         if j==6:
            phi_D=0.561489
            theta_sphere=0.989
         if j==7:
            phi_D=0.36
            theta_sphere=0.65
         if j==8:
            phi_D=0.261825
            theta_sphere=0
         if j==9:
            phi_D=np.pi/4
            theta_sphere=0.47
         if j==10:
            phi_D=0.659058
            theta_sphere=0.43
         if j==11:
            phi_D=0.561489
            theta_sphere=0.24
         if j==12:
            phi_D=0.523599
            theta_sphere=0
         if j==13:
            phi_D=np.pi/4
            theta_sphere=0
    
         if BS2==1:
            # Compute the experimental values of the phase xi for which we get maximum and minimum intensity due to interference (as those values slightly vary from the analytical ones)
            for extremum in range(1,3):
               # Range of variation of the phase xi
               if extremum==1:
                  xi_inf_1=0.5
                  xi_sup_1=4
               if extremum==2:
                  xi_inf_1=4
                  xi_sup_1=7
 
               qr = QuantumRegister(2)
               cr = ClassicalRegister(2)
               
               # Intensity
               I = []
               # Phase
               Xi=[]
               circuits_xi=[]

               nb_circuits_xi=-1
               # Varies the phase of the beam
               for k in range(round(xi_inf_1/delta_xi),round(xi_sup_1/delta_xi)+1):
                   circuit = QuantumCircuit(qr,cr)                       
                   circuit.u3(phi_D*2,0,0,qr[0])            
                   circuit.cx(qr[0],qr[1]) #---                            
                   circuit.cu3(theta_sphere*2,0,0,qr[0],qr[1])         
                   circuit.x(qr[1])
                   circuit.cu1(k*delta_xi,qr[1],qr[0])      
                   circuit.x(qr[1])

                   # Applying BS2
                   circuit.u3(np.pi/2,np.pi/2,np.pi/2,qr[0])

                   circuit.measure(qr[0], cr[0])
                   circuit.measure(qr[1], cr[1])
                   nb_circuits_xi=nb_circuits_xi+1
                   circuits_xi.append(circuit)
   
               shots=1000
            
               if noise_simulation==1:
                  job = execute(circuits_xi, simulator, noise_model=noise_model,coupling_map=coupling_map,basis_gates=basis_gates)
               elif noise_simulation==0:
                  job = execute(circuits_xi, backend=backend, shots=shots)
            
               result = job.result()
               status = job.status()

               nb_circuits_xi=-1
               for k in range(round(xi_inf_1/delta_xi),round(xi_sup_1/delta_xi)+1):
                   nb_circuits_xi=nb_circuits_xi+1

                   res=result.get_counts(circuits_xi[nb_circuits_xi])
                
                   if correction==1:
                      mitigated_results = meas_filter.apply(res)
                      res=mitigated_results

                   counts=np.array([0,0,0,0])
                   for state, events in res.items():
                      if float(state)==0:
                         counts[0]=float(events)
                      if float(state)==10:
                         counts[1]=float(events)
                      if float(state)==1:
                         counts[2]=float(events)
                      if float(state)==11:
                         counts[3]=float(events)
    
                   # Number of events is converted into probabilities [0,1]
                   counts=counts/shots
                   Zbasis_00=counts[0]
                   Zbasis_01=counts[1]
                   Zbasis_10=counts[2]
                   Zbasis_11=counts[3]

                   # Compute the intensity in one of the output ports of BS2
                   Intensity=Zbasis_00+Zbasis_01

                   I.append(Intensity)
                   Xi.append(k*delta_xi)

               x_data=Xi
               y_data=I
               
               # Recording the extremums of the intensities
               def test_func(x, a, b, c):
                   arg = np.array(x, dtype=float)*b
                   return a * np.sin(arg) + c
          
               params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=None)

               plt.figure(figsize=(6, 4))
               plt.scatter(x_data, y_data, label='Measurements')
               plt.plot(x_data, test_func(x_data, params[0], params[1], params[2]), label='Sinusoidal fit')
               plt.legend(loc='best')
               plt.xlabel('ξ [rad]')
               plt.ylabel('Intensity')
               plt.show()

               def derivative(x):
                   return params[0]*np.cos(params[1]*x)
         
               if np.sign(derivative(xi_inf_1)) == np.sign(derivative(xi_sup_1)):
                  # If no extremum is detected, choose analytical angles
                  if extremum==1:
                     x_min=np.pi/2
                     I_min=test_func(x_min, params[0], params[1], params[2])
                  if extremum==2:
                     x_max=3*np.pi/2
                     I_max=test_func(x_max, params[0], params[1], params[2])

               elif np.sign(derivative(xi_inf_1)) != np.sign(derivative(xi_sup_1)):
                  # If extremums are detected, record associated angles and intensities
                  if extremum==1:
                     x_min=optimize.brentq(derivative, xi_inf_1, xi_sup_1)
                     I_min=test_func(x_min, params[0], params[1], params[2])
                  if extremum==2:
                     x_max=optimize.brentq(derivative, xi_inf_1, xi_sup_1)
                     I_max=test_func(x_max, params[0], params[1], params[2])

               if extremum==1:
                  print('ξ_min : ', x_min, ', I_min : ', I_min)
                  xi_1=x_min
         
               if extremum==2:
                  print('ξ_max : ', x_max, ', I_max : ', I_max)
                  xi_2=x_max

         log=open(path+"/Classical_VDC_point_"+str(j)+str_BS2+"_"+str(f)+".txt","w")

         # Varying φ (HWP2):
         for n in range(1,2):

            for k in range(1,2):
                V_fit=abs(I_max-I_min)
                circuits = []
                if k==1:
                   xi_extrema=xi_1
                if k==2:
                   xi_extrema=xi_2
                
                print('Point ',j,': phi =',phi_D,', theta =',theta_sphere,', xi =',xi_extrema)

                # Density matrix
                rho = np.array([[0+0j, 0+0j, 0+0j, 0+0j], [0+0j, 0+0j, 0+0j, 0+0j], [0+0j, 0+0j, 0+0j, 0+0j], [0+0j, 0+0j, 0+0j, 0+0j]])
                
                # Tomography step
                for i in range(1,2):
                    
                    qr = QuantumRegister(2)
                    if BS2==1:
                       circuit = QuantumCircuit(qr)
                    elif BS2==0:
                       circuit = QuantumCircuit(qr,cr)
                 
                    # Prepare state
                    circuit.u3(phi_D*2,0,0,qr[0]) #------------------------|ψ> = cosφ|00> + sinφ|10>
                    circuit.cx(qr[0],qr[1]) #--------------------------------|ψ> = cosφ|00> + sinφ|11>
                    circuit.cu3(theta_sphere*2,np.pi,np.pi,qr[0],qr[1]) #------|ψ> = cosφ|00> + sinφsinθ|10> + sinφcosθ|11>
                    circuit.x(qr[1])
                    circuit.cu1(xi_extrema,qr[1],qr[0]) #----------------|ψ> = cosφ|00> + e^(ξ)sinφsinθ|10> + sinφcosθ|11>
                    circuit.x(qr[1])
               
                    # Apply BS2
                    if BS2==1:
                       circuit.u3(np.pi/2,np.pi/2,np.pi/2,qr[0])
         
                    # Perform Tomography
                    if BS2==1:
                    
                       qst_VDC = state_tomography_circuits(circuit,[qr[0], qr[1]])
                        
                       if noise_simulation==1:
                          job = execute(qst_VDC, simulator, noise_model=noise_model,coupling_map=coupling_map,basis_gates=basis_gates)
                       elif noise_simulation==0:
                          job = execute(qst_VDC, backend, shots=1000)
                    
                       results=job.result()
                       res=results
                    
                       if correction==1:
                          mitigated_results = meas_filter.apply(results)
                          res=mitigated_results
            
                       tomo_VDC = StateTomographyFitter(res, qst_VDC)

                       rho = tomo_VDC.fit()
                       
                    if BS2==0:
                       circuit.measure(qr[0], cr[0])
                       circuit.measure(qr[1], cr[1])

                       shots=1000
                  
                       if noise_simulation==1:
                          job = execute(circuit, simulator, noise_model=noise_model,coupling_map=coupling_map,basis_gates=basis_gates)
                       elif noise_simulation==0:
                          job = execute(circuit, backend=backend, shots=shots)
                       
                       result = job.result()
                       
                if BS2==0:
                 
                    res=result.get_counts()
                    if correction==1:
                       mitigated_results = meas_filter.apply(result)
                       res=mitigated_results.get_counts()

                    counts=np.array([0,0,0,0])
                    for state, events in res.items():
                        if float(state)==0:
                            counts[0]=float(events)
                        if float(state)==10:
                            counts[1]=float(events)
                        if float(state)==1:
                           counts[2]=float(events)
                        if float(state)==11:
                           counts[3]=float(events)
    
                    counts=counts/shots
                    if i==1:

                        string_Zbasis_counts = str(counts[0]) + ' ' +str(counts[1]) + ' ' +str(counts[2]) + ' ' +str(counts[3]) + '\n'
                        Zbasis_00=counts[0]
                        Zbasis_01=counts[1]
                        Zbasis_10=counts[2]
                        Zbasis_11=counts[3]

                if BS2==1:        
                   # Compute C
                   Sigma = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
                   rho_transpose = np.transpose(rho)
                   V_p_rho=np.linalg.eigvals(rho)
                   R = (rho.dot(Sigma)).dot(rho_transpose.dot(Sigma))
                   V_p=np.linalg.eigvals(R)
                   V_p.sort()
                   arg1=0
                   arg2=np.sqrt(float(abs(V_p[3])))-np.sqrt(float(abs(V_p[2])))-np.sqrt(float(abs(V_p[1])))-np.sqrt(float(abs(V_p[0])))
                   C=max(arg1,arg2)

                # Compute D
                if BS2==0:

                   I=Zbasis_10+Zbasis_11
            
                   D=(max(Zbasis_00+Zbasis_01,Zbasis_10+Zbasis_11)-0.5)*2

                # Output string data
                if BS2==1:
                   string_C_V_D_theta_xi = str(0) + ' ' +str(V_fit) + ' ' +str(0) + '\n'
                if BS2==0:
                   string_C_V_D_theta_xi = str(C) + ' ' +str(0) + ' ' +str(D) + ' ' + '\n'
                log.write(string_C_V_D_theta_xi)
              
         log.close()

