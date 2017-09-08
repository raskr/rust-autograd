extern crate ndarray;

use std::mem;
use std::collections::hash_map::HashMap;
use tensor::Tensor;
use ndarray_ext::NdArray;



/// Trait for any stochastic gradient descent optimizer
pub trait Optimizer {
    #[inline]
    /// Updates the variable tensor
    ///
    /// Updates `var` with `grad`
    fn update(&mut self, var: &Tensor, grad: NdArray);
}


/// Vanilla SGD optimizer
pub struct SGD {
    pub lr: f32,
}

impl Optimizer for SGD {
    #[inline]
    fn update(&mut self, var: &Tensor, grad: NdArray) {
        if let Some(ref mut data) = var.borrow_mut().param {
            data.scaled_add(-self.lr, &grad);
        } else {
            panic!("\"{}\" doesn't have parameter", var.borrow().op.name());
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

pub struct AdamState {
    m: NdArray,
    v: NdArray,
    t: f32,
}

impl Default for Adam {
    fn default() -> Adam {
        Adam {
            alpha: 0.001,
            eps: 1e-08,
            b1: 0.9,
            b2: 0.999,
            states: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    #[inline]
    fn update(&mut self, var: &Tensor, mut grad: NdArray) {
        if let Some(ref mut data) = var.borrow_mut().param {
            // get current state
            let AdamState { mut m, mut v, t } = self.states.remove(&var).unwrap_or_else(|| {
                AdamState {
                    m: NdArray::zeros(data.shape()),
                    v: NdArray::zeros(data.shape()),
                    t: 1.,
                }
            });

            // make new m
            m *= self.b1;
            m += &((1. - self.b1) * &grad);
            let m_new = m;

            // make new v
            let tmp = 1. - self.b2;
            grad.mapv_inplace(move |a| tmp * a * a);
            v *= self.b2;
            v += &grad;
            let v_new = v;

            // make hat
            let mut m_hat = &m_new / (1. - self.b1.powf(t));
            let mut v_hat = &v_new / (1. - self.b2.powf(t));

            // update states
            self.states.insert(
                var.clone(),
                AdamState {
                    m: m_new,
                    v: v_new,
                    t: t + 1.,
                },
            );

            let eps = self.eps;
            v_hat.mapv_inplace(move |a| a.sqrt() + eps);
            m_hat /= &v_hat;
            data.scaled_add(-self.alpha, &m_hat);
        } else {
            panic!("\"{}\" doesn't have parameter", var.borrow().op.name());
        }
    }
} // Adam end
