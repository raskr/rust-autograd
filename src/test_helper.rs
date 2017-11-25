extern crate ndarray;

use context;
use ndarray_ext::NdArray;
use std::collections::hash_map::HashMap;
use std::mem;
use tensor::Tensor;


#[allow(mutable_transmutes)]
/// Checks the validity of `gradients` with finite difference trick.
/// For this test only, `variables` must be "shared" variables.
pub fn gradient_check(
    objective: &Tensor,
    gradients: &[Tensor],
    variables: &[&Tensor],
    ctx: context::Context, // cannot be mut ...
    eps: f32,
    tol: f32,
)
{
    // back prop
    let gradients = gradients.iter().map(|a| a).collect::<Vec<_>>();
    let theoretical_grads = eval_keep_feeds(&ctx, gradients.as_slice());

    // for each variable nodes
    for (v, th_grad) in variables.iter().zip(theoretical_grads) {
        let v_arr = ctx.variables.get(v).expect("You passed non-variable");
        let mut v_arr = unsafe { mem::transmute::<&NdArray, &mut NdArray>(v_arr) };

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
            let ref obj_pos = eval_keep_feeds(&ctx, &[objective])[0];

            // perturbation (-)
            unsafe {
                *head_ptr.offset(i) = evacuated - eps;
            }

            // eval
            let ref obj_neg = eval_keep_feeds(&ctx, &[objective])[0];

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

type Memo = HashMap<Tensor, NdArray>;

#[allow(mutable_transmutes)]
/// Almost same as `eval`, but feeds remains after calling this.
fn eval_keep_feeds(
    ctx: &context::Context,
    xs: &[&Tensor],
) -> Vec<ndarray::Array<f32, ndarray::IxDyn>>
{
    // Pull out `outputs` and `variables` from context
    let mut memo = unsafe { mem::replace(mem::transmute(&ctx.outputs), HashMap::new()) };
    let mut vars = unsafe { mem::transmute::<&Memo, &mut Memo>(&ctx.variables) };

    // Run eval with those
    let arrays = ::eval::eval_tensors(xs, vars, &mut memo);

    // Drain outputs except for placeholder nodes
    let mut memo = memo.into_iter()
        .filter(|&(ref k, _)| k.op.name() == "PH")
        .collect::<HashMap<_, _>>();

    // Return back memo to `ctx.outputs`
    mem::swap(&mut memo, unsafe { mem::transmute(&ctx.outputs) });
    arrays
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
