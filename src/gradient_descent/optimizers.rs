use ndarray_ext::NdArray;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use tensor::Tensor;


/// Vanilla SGD optimizer
pub struct SGD {
    pub lr: f32,
}

impl<'a> super::Optimizer<'a> for SGD {
    #[inline]
    fn update(&mut self, param: &'a Tensor, grad: NdArray)
    {
        if let Some(mut param_arr) = unsafe { param.get_persistent_array_mut() } {
            param_arr.scaled_add(-self.lr, &grad);
        } else {
            panic!("Can't optimize non-variable.");
        }
    }
}


/// Adam optimizer
///
/// This implementation is based on http://arxiv.org/abs/1412.6980v8
pub struct Adam<'a> {
    // static params
    pub alpha: f32,
    pub eps: f32,
    pub b1: f32,
    pub b2: f32,
    // dynamic params
    pub states: BTreeMap<super::StateKey<'a>, AdamState>,
}

pub struct AdamState {
    m: NdArray,
    v: NdArray,
    t: f32,
}

impl<'a> Default for Adam<'a> {
    fn default() -> Adam<'a>
    {
        Adam {
            alpha: 0.001,
            eps: 1e-08,
            b1: 0.9,
            b2: 0.999,
            states: BTreeMap::new(),
        }
    }
}

impl<'a> super::Optimizer<'a> for Adam<'a> {
    #[inline]
    fn update(&mut self, param: &'a Tensor, grad: NdArray)
    {
        // get current state
        if let Some(mut param_arr) = unsafe { param.get_persistent_array_mut() } {
            // If first time access => Insert new param
            match self.states.entry(param.as_state_key()) {
                Entry::Vacant(ent) => {
                    ent.insert(AdamState {
                        m: NdArray::zeros(param_arr.shape()),
                        v: NdArray::zeros(param_arr.shape()),
                        t: 1.,
                    });
                }
                _ => {}
            }

            // Get current state with aafe unwrap.
            let &mut AdamState { ref mut m, ref mut v, ref mut t } =
                self.states.get_mut(&param.as_state_key()).unwrap();

            // Make new m
            let b1 = self.b1;
            let tmp = 1. - self.b1;
            m.zip_mut_with(&grad, move |a, &g| *a = (*a) * b1 + tmp * g);

            // Make new v
            let b2 = self.b2;
            let tmp = 1. - self.b2;
            v.zip_mut_with(&grad, move |a, &g| *a = (*a) * b2 + tmp * g * g);

            // Make hat
            let new_m: &NdArray = m;
            let new_v: &NdArray = v;
            let mut m_hat = new_m * (1. / (1. - b1.powf(*t)));
            let v_hat = new_v * (1. / (1. - b2.powf(*t)));

            let eps = self.eps;
            m_hat.zip_mut_with(&v_hat, move |a, &b| (*a) /= b.sqrt() + eps);
            param_arr.scaled_add(-self.alpha, &m_hat);
            *t += 1.;
        }
    }
} // Adam end
