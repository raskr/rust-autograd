extern crate ndarray;

use ndarray_ext;
use ndarray_ext::NdArray;
use std::collections::hash_set::HashSet;
use std::mem;
use tensor::{Feed, Tensor};


#[inline]
/// Update params with gradients
pub fn apply_gradients<T: optimizers::Optimizer>(
    optimizer: &mut T,
    variables: &[&Tensor],
    gradients: &[Tensor],
    feed_dict: Feed,
)
{
    assert!(variables.len() == gradients.len());
    // run graph and get gradient arrays
    let mut grad_arrays = eval_gradients(gradients, feed_dict);
    for v in variables {
        let g = maybe_reduce_grad(grad_arrays.remove(0), v);
        optimizer.update(&v, g);
    }
}


#[inline]
pub fn eval_gradients(gradients: &[Tensor], feed_dict: Feed) -> Vec<NdArray>
{
    // move internal dict
    let mut memo = feed_dict.hash_map;

    // ** pre process **
    // collect variable tensors in the whole graph
    let mut variable_set = HashSet::new();
    for g in gradients.iter() {
        g.visit_once(&mut |arg: &Tensor| if let Some(v) = mem::replace(
            &mut arg.borrow_mut()
                .param,
            None,
        )
        {
            variable_set.insert(arg.clone());
            let k = arg.clone();
            memo.insert(k, v);
        });
    }

    // run graph
    for t in gradients.iter() {
        ::topology::perform_eval(t, &mut memo, true);
    }

    // extracts target arrays
    let mut gradient_arrays = Vec::with_capacity(gradients.len());
    for (i, t) in gradients.iter().enumerate() {
        // Need to handle cases where multiple gradient nodes
        // share an output array.
        // (Safe unwrapping is guaranteed by ::topology::symbolic_gradients())
        if gradients[i + 1..].contains(t) {
            // need to preserve the array for following nodes
            // => copy the array
            gradient_arrays.push(memo.get(t).unwrap().clone());
        } else {
            // do not need to preserve
            // => move the array from memo
            gradient_arrays.push(memo.remove(t).unwrap());
        }
    }

    // ** post process **
    // need to return param arrays to the original places
    for v in variable_set.iter() {
        mem::swap(&mut v.borrow_mut().param, &mut memo.remove(&v));
    }

    gradient_arrays
}

#[inline(always)]
/// Reduces gradient's each dim by summation.
/// This is used when parameter shape and
/// gradient shape are not same due to broadcast.
pub fn maybe_reduce_grad(mut grad: NdArray, variable: &Tensor) -> NdArray
{
    let variable = variable.borrow();
    let variable = variable.param.as_ref().expect(&format!(
        "{} is not variable",
        variable.op.name()
    ));
    let var_shape = variable.shape();
    let grad_shape = grad.shape().to_vec();
    // for each grad axis
    for (i, (g, v)) in grad_shape.iter().zip(var_shape).enumerate() {
        if g == v {
            continue; // do nothing
        } else if g < v {
            panic!("bad gradient")
        } else {
            grad = ndarray_ext::expand_dims(grad.sum(ndarray::Axis(i)), i);
        }
    }
    grad
}


pub mod optimizers {
    use ndarray_ext::NdArray;
    use std::collections::hash_map::HashMap;
    use tensor::Tensor;

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
        fn update(&mut self, var: &Tensor, grad: NdArray)
        {
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

    impl Optimizer for Adam {
        #[inline]
        fn update(&mut self, var: &Tensor, mut grad: NdArray)
        {
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
}
