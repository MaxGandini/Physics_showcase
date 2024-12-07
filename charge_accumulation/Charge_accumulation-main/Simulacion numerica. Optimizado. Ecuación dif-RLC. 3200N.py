# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 23:52:06 2022

@author: Maximiliano Gandini
"""
import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.ticker import MultipleLocator
import sympy as smp
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.vector import CoordSys3D
from numpy import *

z=smp.symbols('z',real=True)
d=smp.symbols('d',real=True, positive=True)
m=smp.symbols('m',real=True, positive=True)
K=smp.symbols('K',real=True, positive=True)
l=smp.symbols('l',real=True, positive=True)
t=smp.symbols('t',real=True)
a=smp.symbols('a',real=True, positive=True)
x=smp.symbols('x',real=True)
y=smp.symbols('y',real=True)
theta=smp.symbols('θ',real=True,cls=smp.Function)
theta=theta(t)
theta

R = CoordSys3D('R')

#---------Definido el SR

m_vec=0*R.i + m*smp.sin(theta)*R.j + -m*smp.cos(theta)*R.k 
rp_vec=0*R.i + (smp.sin(theta)*l)*R.j + (d-smp.cos(theta)*l)*R.k

#-------------Definidos los vectores que caracterizan al dipolo

r_vec=x*R.i + y*R.j + z*R.k
Δ=r_vec-rp_vec

#---------------Defino al SR sobre la espira y calculo el vector "Δ" que va a medir el campo:

Δ
Pint=Δ & m_vec
mΔ=smp.sqrt(Δ & Δ )

B= K*((3*Pint*Δ)/ mΔ**2 - m_vec)/(mΔ)**3
v= 0*R.i + 0*R.j + 1*R.k 
Bz=B & v
Bz

dBz=smp.diff(Bz,t)
omega=smp.diff(theta,t)

B1z=K*m*smp.cos(theta)/ mΔ**3

#Separo en un primer término la parte que tiene el denominador a la 3/2

Int1_x=smp.integrate(B1z,(x,-a,smp.sqrt(a**2-y**2)), conds="separate")
Int1_x

# En esta próxima calculo lo que tiene el denominador a la 5/2 y omito la parte del numerador que tiene la variable y, que la calculo en la integral 3:

B2z=K*3*(((Pint-(m*y*smp.sin(theta))))*(z+l*smp.cos(theta)-d))/mΔ**5
B2z
Int2_x=smp.integrate(B2z,(x,-a,smp.sqrt(a**2-y**2)))
Int2_x

#Separo en un tercer término la parte que tiene el denominador a la 5/2 y un y en el numerador:

B3z=K*3*((m*y*smp.sin(theta))*(z+l*smp.cos(theta)-d))/mΔ**5
B3z
Int3_x=smp.integrate(B3z,(x,-a,smp.sqrt(a**2-y**2)))
B3z

p= (-l*smp.sin(theta)+y)**2 + (-d+l*smp.cos(theta)+z)**2
Int1_x= x/(p**2 *((p**2 + x**2)**(smp.Rational(1,2) )))

B2z=simplify(B2z)
B2z

Int2_x= (2*(x**3) + 3*(p**2)*x)/(3*p**4 *(x**2 + p**2)**smp.Rational(3,2))
Int2_x
Num1=K*m*smp.cos(theta)
Term1=Num1*Int1_x

Num2=K*3*((((Pint-(m*y*smp.sin(theta)))))*(z+l*smp.cos(theta)-d))
simplify(Num2)
Term2=Num2*Int2_x

Num3=K*3*((m*y*smp.sin(theta))*(z+l*smp.cos(theta)-d))

#La integral era la misma que en el segundo término:
Term3=Num3*Int2_x

dT1=smp.diff(Term1,t)
dT1_dt=dT1.subs([(Term1,Term1)]).doit()
dT2=smp.diff(Term2,t)
dT2_dt=dT2.subs([(Term2,Term2)]).doit()
dT3=smp.diff(Term3,t)
dT3_dt=dT3.subs([(Term3,Term3)]).doit()

dT1_dtsp=dT1_dt.subs(x,smp.sqrt(a**2-y**2))
dT1_dtsp
dT1_dtin=dT1_dt.subs(x,-smp.sqrt(a**2-y**2))
dT1_dtin

dT2_dtsp=dT1_dt.subs(x,smp.sqrt(a**2-y**2))
dT2_dtsp
dT2_dtin=dT1_dt.subs(x,-smp.sqrt(a**2-y**2))
dT2_dtin

dT3_dtsp=dT1_dt.subs(x,smp.sqrt(a**2-y**2))
dT3_dtsp
dT3_dtin=dT1_dt.subs(x,-smp.sqrt(a**2-y**2))
dT3_dtin

dT1_dt_fsup=smp.lambdify([y,z,theta,omega,d,l,m,K,a],dT1_dtsp)
dT1_dt_finf=smp.lambdify([y,z,theta,omega,d,l,m,K,a],dT1_dtin)

dT2_dt_fsup=smp.lambdify([y,z,theta,omega,d,l,m,K,a],dT2_dtsp)
dT2_dt_finf=smp.lambdify([y,z,theta,omega,d,l,m,K,a],dT2_dtin)

dT3_dt_fsup=smp.lambdify([y,z,theta,omega,d,l,m,K,a],dT3_dtsp)
dT3_dt_finf=smp.lambdify([y,z,theta,omega,d,l,m,K,a],dT3_dtin)

#--------------------------

fig, ax = plt.subplots(figsize=(7, 5))
plt.title('Pendulo armonico')
plt.rcParams['xtick.direction'] = 'inout'
plt.rcParams['ytick.direction'] = 'inout'
plt.rcParams['axes.linewidth'] = 0.2
plt.rcParams["figure.figsize"] = (9,5)
plt.xlabel('Angulo(rad)')
plt.ylabel('Tiempo(s)')
plt.rcParams.update({'font.size': 10})
plt.rcParams['legend.handlelength'] = 5.0
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 5
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.yaxis.set_minor_locator(MultipleLocator(12.5))
plt.grid(linestyle='--',which='both')


l=0.69
def dSdt(S,t):
  thetan, omegan = S
  g=9.8
  return [omegan,(-g/l)*np.sin(thetan)]

#Con esto controlan el ángulo inicial:
    
theta0=np.pi/6
omega0=0
S0=[theta0,omega0]

#---------- Con esto controlan la resolución y el largo del intervalo de tiempo:    
L=0.7
res=1000
#______________________________________________________________

t=np.linspace(0,L,res)
sol=odeint(dSdt, S0, t)
thetand, omegand = sol.T

plt.legend()

omeganp=np.asarray(omegand)
Tp=np.asarray(t)
thetanp=np.asarray(thetand)

#------- Con esto controlan el intervalo que quieren mirar y aproximar.
#(Se recorta y se convierte en numpy arrays para optimizarlo, sino tarda un huevo):
    
Inf=0.35
Sup=0.58

#-----------------------------------------------------------------------

T= Tp[(Tp>Inf) & (Tp < Sup)]-Inf
omegan= omeganp[(Tp>Inf) & (Tp < Sup)]
thetan= thetanp[(Tp>Inf) & (Tp < Sup)]

print(len(thetan))

plt.scatter(T,thetan,s=2,label='theta')
plt.show()

#------------- Parámetros del problema:
    
d=0.7
R=0.02
K=8.9875517873681764*(10**9)
m=2*10**(-10)  
z0=-0.03

#--------------------------
Term1=[]

for i in range(len(thetan)):
    integrando = lambda y: dT1_dt_fsup(y,z0,thetan[i],omegan[i],d,l,m,K,R) - dT1_dt_finf(y,z0,thetan[i],omegan[i],d,l,m,K,R)
    integral, integral_error = quad(integrando, -R, R)
    Term1.append(integral)
    
Term2=[]

for i in range(len(thetan)):
    integrando = lambda y: dT2_dt_fsup(y,z0,thetan[i],omegan[i],d,l,m,K,R) - dT2_dt_finf(y,z0,thetan[i],omegan[i],d,l,m,K,R)
    integral, integral_error = quad(integrando, -R, R)
    Term2.append(integral)
    
Term3=[]

for i in range(len(thetan)):
    integrando = lambda y: dT3_dt_fsup(y,z0,thetan[i],omegan[i],d,l,m,K,R) - dT3_dt_finf(y,z0,thetan[i],omegan[i],d,l,m,K,R)
    integral, integral_error = quad(integrando, -R, R)
    Term3.append(integral)
    
    
VarFd=[]
for i in range(len(thetan)):
    VarFd.append(Term3[i]+Term2[i]+Term1[i])
    
VarF=np.asarray(VarFd)   
print(len(VarF))

plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(figsize=(7, 5))
plt.title(r'$\mathrm{Corriente \quad Sol-Num}$')
plt.rcParams['xtick.direction'] = 'inout'
plt.rcParams['ytick.direction'] = 'inout'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams["figure.figsize"] = (9,5)
plt.rcParams.update({'font.size': 8})
plt.xlabel(r'$\mathrm{Tiempo(s)}$')
plt.ylabel(r'$\mathrm{Tension(mV)}$')
plt.rcParams['legend.handlelength'] = 5.0
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 5
#plt.xlim(Inf, Sup)

t__=[]
bn=[]
an=[]
L= 0.7

for i in range(len(T)):
    t__.append((L/res)*i)
    
t_=np.asarray(t__)

a0=0
integrando=[]

for i in range(len(thetan)):
    integrando.append(VarF[i]/L)

b0= trapz(integrando,t_)

Faprox=-b0/2

for i in range(len(thetan)):
    integrando=[]
    for j in range(len(thetan)):
        integrandoi = np.sin(t_[j]*i*np.pi/L)*VarF[j]/L
        integrando.append(integrandoi)
    integral = trapz(integrando,t_)
    an.append(integral)

for i in range(len(thetan)):
    integrando=[]
    for j in range(len(thetan)):
        integrandoi = np.cos(t_[j]*i*np.pi/L)*VarF[j]/L
        integrando.append(integrandoi)
    integral = trapz(integrando,t_)
    bn.append(integral)


for n in range(len(thetan)): 
    Faprox = Faprox + an[n]*np.sin(n*(np.pi)*T/L)+ bn[n]*np.cos(n*(np.pi)*T/L)

plt.grid()
#ax.plot(T,Faprox,label='Solucion Fourier')
#ax.scatter(T,VarF,s=4,label='Variacion de flujo',color='k',marker='s')
plt.legend()

def dSist(S,t):
    Q, I = S
    integrando=[]
    for i in range(len(thetan)):
        integrando.append(VarF[i]/L)
    b0= trapz(integrando,t_)
    Faprox=-b0/2
    for n in range(len(thetan)): 
      Faprox = Faprox + an[n]*np.sin(n*(np.pi)*t/L)+ bn[n]*np.cos(n*(np.pi)*t/L)
    inductancia=0.19738
    resistencia=146.25
    capacitancia=0.00012769
    return [I,(Faprox-Q/capacitancia -resistencia*I)/inductancia]

theta0=0
omega0=0
S0=[theta0,omega0]
sol=odeint(dSist, S0, T)
Q, I = sol.T
ax.scatter(T,I,s=2,label='Corriente')
plt.legend()

a0=0
integrando=[]

for i in range(len(I)):
    integrando.append(I[i]/L)

b0= trapz(integrando,t_)
Faprox=-b0/2

aj=[]
bj=[]

for i in range(len(I)):
    integrando=[]
    for j in range(len(thetan)):
        integrandoi = np.sin(t_[j]*i*np.pi/L)*I[j]/L
        integrando.append(integrandoi)
    integral = trapz(integrando,t_)
    aj.append(integral)


for i in range(len(I)):
    integrando=[]
    for j in range(len(thetan)):
        integrandoi = np.cos(t_[j]*i*np.pi/L)*I[j]/L
        integrando.append(integrandoi)
    integral = trapz(integrando,t_)
    bj.append(integral)


Faproxi=0
for n in range(len(I)): 
    Faproxi = Faproxi + aj[n]*np.sin(n*(np.pi)*T/L)+ bj[n]*np.cos(n*(np.pi)*T/L)


plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(figsize=(7, 5))
plt.title(r'$\mathrm{Variacion \quad Flujo}$')
plt.rcParams['xtick.direction'] = 'inout'
plt.rcParams['ytick.direction'] = 'inout'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams["figure.figsize"] = (9,5)
plt.rcParams.update({'font.size': 8})
plt.xlabel(r'$\mathrm{Tiempo(s)}$')
plt.ylabel(r'$\mathrm{Tension(mV)}$')
plt.rcParams['legend.handlelength'] = 5.0
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 5
plt.grid()

ax.plot(T,Faproxi,label='Solución Fourier-Corriente')
plt.legend()
