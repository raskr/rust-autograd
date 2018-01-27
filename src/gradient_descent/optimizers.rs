use ndarray_ext::NdArray;
use std::collections::hash_map::HashMap;
use tensor::Tensor;


/// Vanilla SGD optimizer
pub struct SGD {
    pub lr: f32,
}

impl ::gradient_descent::Optimizer for SGD {
    #[inline]
    fn update(&mut self, target: &Tensor, grad: &NdArray)
    {
        if let Some(mut a) = unsafe { target.get_persistent_array_mut() } {
            a.scaled_add(-self.lr, grad);
        } else {
            panic!("Can't optimize non-variable.");
        }
    }
}


/// Adam optimizer
///
/// This implementation is based on http://arxiv.org/abs/1412.6980v8
pub struct Adam {
    // static params
    pub alpha: f32,
    pub eps: f32,
    pub b1: f32,
    pub b2: f32,
    // dynamic params
    pub states: HashMap<Tensor, AdamState>,
}

#[doc(hidden)]
pub struct AdamState {
    m: NdArray,
    v: NdArray,
    t: f32,
}

impl Default for Adam {
    fn default() -> Adam
    {
        Adam {
            alpha: 0.001,
            eps: 1e-08,
            b1: 0.9,
            b2: 0.999,
            states: HashMap::new(),
        }
    }
}

impl ::gradient_descent::Optimizer for Adam {
    #[inline]
    fn update(&mut self, target: &Tensor, grad: &NdArray)
    {
        let new_key = target.clone();
        // get current state
        if let Some(param) = unsafe { target.get_persistent_array_mut() } {
            let AdamState { mut m, mut v, t } = self.states.remove(&new_key).unwrap_or_else(|| {
                AdamState {
                    m: NdArray::zeros(param.shape()),
                    v: NdArray::zeros(param.shape()),
                    t: 1.,
                }
            });

            // make new m
            let b1 = self.b1;
            let tmp = 1. - self.b1;
            m.zip_mut_with(&grad, move |a, &g| *a = (*a) * b1 + tmp * g);
            let m_new = m;

            // make new v
            let b2 = self.b2;
            let tmp = 1. - self.b2;
            v.zip_mut_with(&grad, move |a, &g| *a = (*a) * b2 + tmp * g * g);
            let v_new = v;

            // make hat
            let mut m_hat = &m_new * (1. / (1. - b1.powf(t)));
            let v_hat = &v_new * (1. / (1. - b2.powf(t)));

            // update states
            self.states.insert(
                new_key,
                AdamState { m: m_new, v: v_new, t: t + 1. },
            );

            let eps = self.eps;
            m_hat.zip_mut_with(&v_hat, move |a, &b| (*a) /= b.sqrt() + eps);
            param.scaled_add(-self.alpha, &m_hat);
        }
    }
} // Adam end
