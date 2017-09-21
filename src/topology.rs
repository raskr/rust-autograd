extern crate ndarray;
extern crate fnv;

use self::fnv::FnvHashMap;
use ndarray_ext::NdArray;
use ops;
use std::collections::binary_heap::BinaryHeap;
use std::collections::hash_set::HashSet;
use tensor::Tensor;



/// Performs actual graph traversal and its evaluation
// TODO: loop-based rather than recursion (this would be difficult)
#[inline]
pub fn perform_eval(target: &Tensor, memo: &mut FnvHashMap<Tensor, NdArray>, train: bool)
{
    if memo.contains_key(target) {
        return;
    }

    let y = {
        let mut xs = vec![];
        {
            let ref inputs = target.borrow().inputs;
            // integrating loops below is impossible because of
            // "memo is already mutably borrowed"
            for input in inputs.iter() {
                perform_eval(input, memo, train);
            }
            for input in inputs.iter() {
                // unwrap is safe
                xs.push(memo.get(input).unwrap());
            }
        }
        // run op
        target.borrow_mut().op.compute(xs.as_slice(), train)
    };

    // cache output
    memo.insert(target.clone(), y);
}


#[inline]
// make mapping of {node => the node contributed to gradient or not}
// TODO: loop-based rather than recursion
pub fn contributed_to_grads(objective: &Tensor, variables: &[&Tensor]) -> FnvHashMap<Tensor, bool>
{
    fn rec(target: &Tensor, vars: &[&Tensor], memo: &mut FnvHashMap<Tensor, bool>)
    {
        if memo.contains_key(target) {
            return;
        }

        let mut contrib = false;

        if vars.contains(&target) {
            contrib = true;
        } else {
            for input in target.borrow().inputs.iter() {
                // recurse
                rec(input, vars, memo);
                // unwrap is always safe
                contrib |= *memo.get(input).unwrap();
            }
        }

        memo.insert(target.clone(), contrib);
    }

    let mut memo = FnvHashMap::default();
    rec(objective, variables, &mut memo);
    memo
}


#[inline]
/// Returns symbolic gradient tensors.
///
/// This computes partial derivatives of `objective` with `variables` and returns the
/// gradients. This is achieved by building the subgraph between `objective` and
/// `variables` in reverse order from user's graph definition.
///
/// NOTE: Nodes that do not contribute to the gradient won't be included to avoid
/// unnecessary computation.
pub fn symbolic_gradients(
    objective: &Tensor,
    variables: &[&Tensor],
    initial_grad: Option<&Tensor>,
) -> Vec<Tensor>
{
    let initial_grad = initial_grad.map(|a| a.clone()).unwrap_or_else(
        || ops::scalar(1.),
    );

    // Mapping of {y => gy}
    let mut grads = FnvHashMap::default();

    // Mapping of {node => must visit or not (boolean)}
    let contrib = contributed_to_grads(objective, variables);

    // Prepare a heap with tensor's rank numbers for reversed
    // topological sort.
    let mut heap = BinaryHeap::new();
    heap.push(objective.clone());
    grads.insert(objective.clone(), initial_grad);
    let mut grad_done = HashSet::<Tensor>::new();

    // BackProp implementation
    while let Some(target) = heap.pop() {
        let borrowed_target = target.borrow();

        // Vec<Tensor> to Vec<&Tensor>
        let xs: Vec<&Tensor> = borrowed_target.inputs.iter().map(|a| a).collect();

        // time to call grad
        let gxs = {
            // Safe unwrapping is guaranteed by topological ordering
            let gy = grads.get(&target).unwrap();
            borrowed_target.op.grad(gy, xs.as_slice(), &target)
        };

        debug_assert_eq!(
            xs.len(),
            gxs.len(),
            "Wrong `grad` implementation of {}.",
            target
        );

        // cuts the backward path if gx is None.
        for (x, maybe_gx) in xs.into_iter().zip(gxs) {
            if !contrib.contains_key(x) {
                continue;
            }
            if let Some(gx) = maybe_gx {
                // memo gx
                if let Some(g) = grads.remove(x) {
                    grads.insert(x.clone(), g + &gx);
                } else {
                    grads.insert(x.clone(), gx);
                }
                // update heap
                if !x.is_source() && !grad_done.contains(x) {
                    grad_done.insert(x.clone());
                    heap.push(x.clone());
                }
            }
        }
    }

    variables
        .iter()
        .map(|v| {
            grads.remove(v).expect(
                "Input variable(s) didn't contributed to gradient computation",
            )
        })
        .collect::<Vec<Tensor>>()
}


// This is used for tests for now
pub fn collect_nodes_from(end_point: &Tensor) -> HashSet<Tensor>
{
    let mut collected = HashSet::new();
    end_point.visit_once(&mut |arg| { collected.insert(arg.clone()); });
    collected
}
