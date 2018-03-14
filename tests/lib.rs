#[allow(unused_imports)]
#[macro_use]
extern crate ndarray;
extern crate autograd as ag;

#[cfg(test)]
pub mod test_binary_op_eval;
pub mod test_binary_op_grad;
pub mod test_tensor_ops_grad;
pub mod test_tensor_ops_eval;
pub mod test_array_gen;
pub mod test_core;
