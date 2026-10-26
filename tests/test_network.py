import numpy as np
import pytest
from notebook_helpers import definitions

f = definitions("Neural_network_scratch.ipynb", {"np": np})

def test_softmax_and_missing_classes():
    probabilities = f["softmax"](np.array([[1000., -1000.], [999., -999.]]))
    np.testing.assert_allclose(probabilities.sum(axis=0), 1.)
    encoded = f["one_hot"](np.array([0, 2, 2]))
    assert encoded.shape == (10, 3)
    np.testing.assert_allclose(encoded.sum(axis=0), 1.)
    with pytest.raises(ValueError): f["one_hot"](np.array([-1]))

def test_backprop_matches_numerical_gradient():
    rng = np.random.default_rng(12)
    x = rng.random((784, 3)); y = np.array([0, 1, 2])
    w1 = rng.normal(0,.01,(10,784)); b1 = np.ones((10,1))
    w2 = rng.normal(0,.01,(10,10)); b2 = np.zeros((10,1))
    z1,a1,z2,a2 = f["forward_prop"](w1,b1,w2,b2,x)
    dw1,db1,dw2,db2 = f["back_prop"](z1,a1,z2,a2,w2,x,y)
    def loss():
        a = f["forward_prop"](w1,b1,w2,b2,x)[3]
        return -np.log(a[y,np.arange(3)]).mean()
    for param,grad,index in [(w1,dw1,(0,0)),(b1,db1,(0,0)),(w2,dw2,(0,0)),(b2,db2,(0,0))]:
        original=param[index]; eps=1e-6
        param[index]=original+eps; plus=loss()
        param[index]=original-eps; minus=loss()
        param[index]=original
        assert grad[index] == pytest.approx((plus-minus)/(2*eps),abs=1e-7)
