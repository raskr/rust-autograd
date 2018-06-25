extern crate ndarray;

use ndarray_ext::NdArray;
use std::cmp::Ordering;
use std::collections::btree_set::BTreeSet;
use tensor::Tensor;

/// Checks the validity of `gradients` with finite difference trick.
/// For this test only, `variables` must be "shared" variables.
pub fn gradient_check<'a, 'b, T>(
    objective: &Tensor,
    gradients: &[T],
    variables: &[&Tensor],
    feeds: &[(&Tensor, &NdArray)],
    eps: f32,
    tol: f32,
) where
    T: AsRef<Tensor>,
{
    // backprop
    let theoretical_grads = ::runtime::eval(gradients, feeds);

    // for each variable nodes
    for (var_node, th_grad) in variables.iter().zip(theoretical_grads) {
        let v_arr = unsafe {
            var_node
                .persistent_array
                .as_ref()
                .expect("This is not variable")
                .get_as_variable_mut()
        };
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
            let obj_pos = objective.eval(feeds).unwrap();

            // perturbation (-)
            unsafe {
                *head_ptr.offset(i) = evacuated - eps;
            }

            // eval
            let obj_neg = objective.eval(feeds).unwrap();

            // restore
            unsafe {
                *head_ptr.offset(i) = evacuated;
            }

            let g_num = (obj_pos - obj_neg).scalar_sum() / (2. * eps);
            let g_th = unsafe { *th_grad.as_ref().unwrap().as_ptr().offset(i) };

            // compare
            let diff = (g_num - g_th).abs();
            if diff > tol {
                panic!(
                    "Gradient checking failed with too large error: num={}, bp={}",
                    g_num, g_th
                );
            }
        }
    }
}

/// Traverse a graph from endpoint "t".
pub fn visit_once<F>(t: &Tensor, f: &mut F)
where
    F: FnMut(&Tensor) -> (),
{
    visit_once_internal(t, f, &mut BTreeSet::new())
}

fn visit_once_internal<'a, F>(t: &'a Tensor, f: &mut F, visited: &mut BTreeSet<&'a Tensor>)
where
    F: FnMut(&'a Tensor) -> (),
{
    if visited.contains(&t) {
        return; // exit early
    } else {
        visited.insert(t); // first visit
    }

    f(&t);

    for child in t.inputs.iter() {
        visit_once_internal(child, f, visited)
    }
}

impl<'a> Ord for &'a Tensor {
    #[inline]
    /// Compares addresses of the two tensors.
    /// This can be used for ordering-based data structures (e.g. BinaryTree).
    fn cmp(&self, other: &&'a Tensor) -> Ordering {
        let a = (*self) as *const Tensor;
        let b = (*other) as *const Tensor;
        a.cmp(&b)
    }
}

impl<'a> PartialOrd for &'a Tensor {
    #[inline]
    /// Compares addresses of the two tensors.
    /// This can be used for ordering-based data structures (e.g. BinaryTree).
    fn partial_cmp(&self, other: &&'a Tensor) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! eval_with_time {
    ($x:expr) => {{
        use std::time::{Duration, Instant};
        let start = Instant::now();
        let result = $x;
        let end = start.elapsed();
        println!(
            "{}.{:03} sec",
            end.as_secs(),
            end.subsec_nanos() / 1_000_000
        );
        result
    }};
}
