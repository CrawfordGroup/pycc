"""
Test integrators with simple ODE
dx/dy = 3x^2y given x0 = 1, y0 = 2
ANALYTIC SOLUTION: 
y = e^{x^3 + c}, c = ln(2) - 1
y(1,1.1,1.2,1.3,1.4) = [2,2.78471958461639,4.141869187709196,6.6203429951303265,11.440356871885081]
"""

# Import package, test suite, and other packages as needed
import numpy as np
from pycc.rt import integrators as ints

def f(x,y):
    """dy/dx = f(x,y) = 3x^2y"""
    Y = 3.*x**2. * y
    return Y

def chk_ode(ode):
    h = 0.1
    ODE = ode(h)
    t0 = 1
    y0 = 2
    
    y1 = ODE(f,t0,y0)
    y2 = ODE(f,t0+h,y1)
    y3 = ODE(f,t0+2*h,y2)
    y4 = ODE(f,t0+3*h,y3)

    return np.array([y0,y1,y2,y3,y4])

def test_rk4():
    """Test 4th-order Runge-Kutta"""
    rk4 = chk_ode(ints.rk4)
    ref = np.array([2,2.7846419118859376,4.141490537335979,6.618844434974082,11.434686303979237])

    assert np.allclose(rk4,ref)

def test_rk38():
    """Test "corrected" 3rd-order Runge-Kutta"""
    rk38 = chk_ode(ints.rk38)
    ref = np.array([2,2.7846719015333337,4.141594947022453,6.619134913159302,11.435455703714204])

    assert np.allclose(rk38,ref)

def test_rk3():
    """Test 3rd-order Runge-Kutta"""
    rk3 = chk_ode(ints.rk3)
    ref = np.array([2,2.783897725,4.137908208354427,6.60545045860959,11.38808439342214])

    assert np.allclose(rk3,ref)

def test_rk2():
    """Test 2nd-order Runge-Kutta"""
    rk2 = chk_ode(ints.rk2)
    ref = np.array([2,2.7643999999999997,4.066743395,6.396857224546359,10.804576512405294])

    assert np.allclose(rk2,ref)

def test_gl6():
    """Test 6th-order Gauss-Legendre"""
    gl6 = chk_ode(ints.gl6)
    ref = np.array([2,2.78364923694925,4.1371512621094695,6.603613786914487,11.383853535021142])

    assert np.allclose(gl6,ref)
