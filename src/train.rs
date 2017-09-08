extern crate ndarray;

use std::mem;
use std::collections::hash_set::HashSet;
use tensor::{Tensor, Input};
use sgd;
use ndarray_ext;
use ndarray_ext::NdArray;


#[inline]
/// Update params with gradients
pub fn apply_gradients<T: sgd::Optimizer>(
    optimizer: &mut T,
    variables: &[&Tensor],
    gradients: &[Tensor],
    feed_dict: Input,
) {
    assert!(variables.len() == gradients.len());
    // run graph and get gradient arrays
    let mut grad_arrays = eval_gradients(gradients, feed_dict);
    for v in variables {
        let g = maybe_reduce_grad(grad_arrays.remove(0), v);
        optimizer.update(&v, g);
    }
}


#[inline]
pub fn eval_gradients(gradients: &[Tensor], feed_dict: Input) -> Vec<NdArray> {
    // move internal dict
    let mut memo = feed_dict.hash_map;

    // ** pre process **
    // collect variable tensors in the whole graph
    let mut variable_set = HashSet::new();
    for g in gradients.iter() {
        g.visit_once(&mut |arg: &Tensor| {
            if let Some(v) = mem::replace(&mut arg.borrow_mut().param, None) {
                variable_set.insert(arg.clone());
                let k = arg.clone();
                memo.insert(k, v);
            }
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
        if gradients[i+1..].contains(t) {
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
pub fn maybe_reduce_grad(mut grad: NdArray, variable: &Tensor) -> NdArray {
    let variable = variable.borrow();
    let variable = variable.param.as_ref().expect(
        &format!("{} is not variable", variable.op.name()));
    let var_shape = variable.shape();
    let grad_shape = grad.shape().to_vec();
    // for each grad axis
    for (i, (g, v)) in grad_shape.iter().zip(var_shape).enumerate() {
        if g == v {
            continue  // do nothing
        } else if g < v {
            panic!("bad gradient")
        } else {
            grad = ndarray_ext::expand_dims(grad.sum(ndarray::Axis(i)), i);
        }
    }
    grad
}
