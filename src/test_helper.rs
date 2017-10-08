extern crate ndarray;
extern crate rand;
extern crate fnv;

use self::fnv::FnvHashMap;
use graph;
use ndarray_ext::NdArray;
use sgd;
use std::mem;
use tensor;
use tensor::Tensor;


#[allow(mutable_transmutes)]
/// This computes partial derivatives of `objective` with `var_node` using
/// back propagation, and then checks those are close to numerically
/// computed gradients (uses finite difference trick).
pub fn gradient_check(
    objective: &Tensor,
    gradients: &[Tensor],
    variables: &[&Tensor],
    graph: graph::Graph,
    eps: f32,
    tol: f32,
)
{
    // back prop
    let gradients = gradients.iter().map(|a| a).collect::<Vec<_>>();
    let theoretical_grads = eval_keep_feeds(&graph, gradients.as_slice());

    // for each variable nodes
    for (v, th_grad) in variables.iter().zip(theoretical_grads) {
        let v_arr = graph.variables.get(v).expect("You passed non-variable");
        let mut v_arr = unsafe { mem::transmute::<&NdArray, &mut NdArray>(v_arr) };

        // reduce gradient if necessary
        let th_grad = sgd::maybe_reduce_grad(th_grad, v_arr.shape());

        let head_ptr: *mut f32 = v_arr.as_mut_ptr();

        // for each values
        for i in 0..v_arr.len() as isize {
            let evacuated;

            // perturbation (+)
            unsafe {
                evacuated = *head_ptr.offset(i);
                *head_ptr.offset(i) = evacuated + eps;
            }

            // eval
            let ref obj_pos = eval_keep_feeds(&graph, &[objective])[0];

            // perturbation (-)
            unsafe {
                *head_ptr.offset(i) = evacuated - eps;
            }

            // eval
            let ref obj_neg = eval_keep_feeds(&graph, &[objective])[0];

            // restore
            unsafe {
                *head_ptr.offset(i) = evacuated;
            }

            let g_num = (obj_pos - obj_neg).scalar_sum() / (2. * eps);
            let g_th = unsafe { *th_grad.as_ptr().offset(i) };

            // compare
            let diff = (g_num - g_th).abs();
            if diff > tol {
                panic!(
                    "Gradient checking failed with too large error: num={}, bp={}",
                    g_num,
                    g_th
                );
            }
        }
    }
}

#[allow(mutable_transmutes)]
/// Almost same as `graph.eval`, but feeds remains after calling this.
fn eval_keep_feeds(graph: &graph::Graph, xs: &[&Tensor])
    -> Vec<ndarray::Array<f32, ndarray::IxDyn>>
{
    let xs = xs.into_iter().map(|a| (*a).clone()).collect::<Vec<_>>();

    let mut memo = unsafe { mem::replace(mem::transmute(&graph.outputs), FnvHashMap::default()) };

    type M = FnvHashMap<Tensor, NdArray>;
    let ret = tensor::eval_tensors(
        xs.as_slice(),
        unsafe { &mut mem::transmute::<&M, &mut M>(&graph.variables) },
        &mut memo,
    );

    // Drain except for placeholder nodes and its feeds
    let mut memo = memo.into_iter()
        .filter(|&(ref k, _)| k.op.name() == "Placeholder")
        .collect::<FnvHashMap<Tensor, NdArray>>();

    mem::swap(&mut memo, unsafe { mem::transmute(&graph.outputs) });
    ret
}


#[doc(hidden)]
#[macro_export]
macro_rules! eval_with_time {
  ( $x:expr) => {
    {
      use std::time::{Duration, Instant};
      let start = Instant::now();
      let result = $x;
      let end = start.elapsed();
      println!("{}.{:03} sec", end.as_secs(), end.subsec_nanos() / 1_000_000);
      result
    }
  };
}
