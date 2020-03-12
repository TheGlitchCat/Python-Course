## GRAPHS

import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from matplotlib import ticker, cm



def graph(array, **kwargs):
    plt.plot(array)
    print(kwargs)
    if kwargs.keys():
        try:
            plt.xlabel(kwargs['x'])
            plt.ylabel(kwargs['y'])
        except:
            print()
    plt.show()
    

    

    
def koch_snowflake(order, scale=10):
    """
    Return two lists x, y of point coordinates of the Koch snowflake.

    Arguments
    ---------
    order : int
        The recursion depth.
    scale : float
        The extent of the snowflake (edge length of the base triangle).
    """
    def _koch_snowflake_complex(order):
        if order == 0:
            # initial triangle
            angles = np.array([0, 120, 240]) + 90
            return scale / np.sqrt(3) * np.exp(np.deg2rad(angles) * 1j)
        else:
            ZR = 0.5 - 0.5j * np.sqrt(3) / 3

            p1 = _koch_snowflake_complex(order - 1)  # start points
            p2 = np.roll(p1, shift=-1)  # end points
            dp = p2 - p1  # connection vectors

            new_points = np.empty(len(p1) * 4, dtype=np.complex128)
            new_points[::4] = p1
            new_points[1::4] = p1 + dp / 3
            new_points[2::4] = p1 + dp * ZR
            new_points[3::4] = p1 + dp / 3 * 2
            return new_points

    points = _koch_snowflake_complex(order)
    x, y = points.real, points.imag
    return x, y 

    
    
def graph_snowflake(order,size):
    x, y = koch_snowflake(order)
    plt.figure(figsize=size)
    plt.axis('equal')
    plt.fill(x, y)
    plt.show()
    
    
    
    
    
    
    
def warm_map(N,x,y,colormap):
    
    X, Y = np.meshgrid(x, y)

    # A low hump with a spike coming out.
    # Needs to have z/colour axis on a log scale so we see both hump and spike.
    # linear scale only shows the spike.
    Z1 = np.exp(-(X)**2 - (Y)**2)
    Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
    z = Z1 + 50 * Z2

    # Put in some negative values (lower left corner) to cause trouble with logs:
    z[:5, :5] = -1

    # The following is not strictly essential, but it will eliminate
    # a warning.  Comment it out to see the warning.
    z = ma.masked_where(z <= 0, z)


    # Automatic selection of levels works; setting the
    # log locator tells contourf to use a log scale:
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, z, locator=ticker.LogLocator(), cmap=colormap)

    # Alternatively, you can manually set the levels
    # and the norm:
    # lev_exp = np.arange(np.floor(np.log10(z.min())-1),
    #                    np.ceil(np.log10(z.max())+1))
    # levs = np.power(10, lev_exp)
    # cs = ax.contourf(X, Y, z, levs, norm=colors.LogNorm())

    cbar = fig.colorbar(cs)

    plt.show()

