import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
import matplotlib.pyplot as plt
from matplotlib import cm

# Data for GeSbS is too sparse for proper spline fit.
# We instead fit the available values to a 2D plane and *extrapolate*
# https://stackoverflow.com/a/34746883


GeSbS = np.genfromtxt('GeSbS.csv', delimiter=',',dtype=float, skip_header=1)
GeSbSe = np.genfromtxt('GeSbSe.csv', delimiter=',',dtype=float, skip_header=1)

# GeSbS
Ge = 23
Sb = 7
data_points = GeSbS.shape[0]

A = np.column_stack([np.ones(data_points), GeSbS[:,0],GeSbS[:,1]])
density_S = GeSbS[:,3]
young_S = GeSbS[:,4]
shear_S = GeSbS[:,5]
poisson_S = GeSbS[:,6]

coefficients, _, _, _ = np.linalg.lstsq(A, density_S, rcond=None)
a, b, c = coefficients
print('GeSbS Density: ', a + b*Ge + c*Sb)
coefficients, _, _, _ = np.linalg.lstsq(A, young_S, rcond=None)
a, b, c = coefficients
print('GeSbS Youngs modulus: ', a + b*Ge + c*Sb)
coefficients, _, _, _ = np.linalg.lstsq(A, shear_S, rcond=None)
a, b, c = coefficients
print('GeSbS Shear modulus: ', a + b*Ge + c*Sb)
coefficients, _, _, _ = np.linalg.lstsq(A, poisson_S, rcond=None)
a, b, c = coefficients
print('GeSbS Poisson ratio: ', a + b*Ge + c*Sb)


# GeSbSe
Ge = 28
Sb = 12

density_Se = GeSbSe[:,3]
young_Se = GeSbSe[:,4]
shear_Se = GeSbSe[:,5]
poisson_Se = GeSbSe[:,6]

Interp2D_Se = CloughTocher2DInterpolator(list(zip(GeSbSe[:,0],GeSbSe[:,1])),density_Se)
print('GeSbSe Density: ', Interp2D_Se(Ge,Sb))
Interp2D_Se = CloughTocher2DInterpolator(list(zip(GeSbSe[:,0],GeSbSe[:,1])),young_Se)
print('GeSbSe Youngs Modulus: ', Interp2D_Se(Ge,Sb))
Interp2D_Se = CloughTocher2DInterpolator(list(zip(GeSbSe[:,0],GeSbSe[:,1])),shear_Se)
print('GeSbSe Shear Modulus: ', Interp2D_Se(Ge,Sb))
Interp2D_Se = CloughTocher2DInterpolator(list(zip(GeSbSe[:,0],GeSbSe[:,1])),poisson_Se)
print('GeSbSe Poisson Ratio: ', Interp2D_Se(Ge,Sb))


# 3D Plots
# Sulfide
theCM = cm.inferno
theCM._init()
alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
theCM._lut[:-3,-1] = alphas

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

X = np.linspace(6,25)
Y = np.linspace(6,35)
X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation

coefficients, _, _, _ = np.linalg.lstsq(A, young_S, rcond=None)
a, b, c = coefficients
Z = a + b*X + c*Y
ax.plot_surface(X, Y, Z,cmap=theCM)
ax.scatter3D(GeSbS[:,0],GeSbS[:,1], young_S, color = "green")
ax.scatter3D(23, 7, a + b*23 + c*7, color = "red")

plt.title("Young's Modulus for Sulfide")
plt.xlabel("Germanium")
plt.ylabel("Antimony")
plt.show()

# Selenide
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

X = np.linspace(np.min(GeSbSe[:,0]),np.max(GeSbSe[:,0]))
Y = np.linspace(np.min(GeSbSe[:,1]),np.max(GeSbSe[:,1]))
X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
Interp2D_Se = CloughTocher2DInterpolator(list(zip(GeSbSe[:,0],GeSbSe[:,1])),young_Se)
Z = Interp2D_Se(X,Y)
ax.plot_surface(X, Y, Z,cmap=theCM)
ax.scatter3D(GeSbSe[:,0],GeSbSe[:,1], young_Se, color = "green")
ax.scatter3D(28, 12, Interp2D_Se(28,12), color = "red")

plt.title("Young's Modulus for Selenide")
plt.xlabel("Germanium")
plt.ylabel("Antimony")
plt.show()


