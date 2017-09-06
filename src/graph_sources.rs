extern crate ndarray;

use std::rc::Rc;
use std::cell::RefCell;
use tensor::{Tensor, RawTensor};
use ndarray_ext::NdArray;
use ops;

/// Constructor of a tensor placeholder.
///
/// `shape[0]` can be -1, which means dynamic batch size.
#[inline]
pub fn placeholder(shape: &[isize]) -> Tensor {
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp{ name: "Placeholder".to_string() }),
        inputs: vec![],
        param: None,
        rank: 0,
    })))
}

/// Creates a constant tensor.
#[inline]
pub fn constant(array: ndarray::Array<f32, ndarray::IxDyn>) -> Tensor {
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp{ name: "Constant".to_string() }),
        inputs: vec![],
        param: Some(array),
        rank: 0,
    })))
}

/// Creates a constant tensor.
#[inline]
pub fn scalar(a: f32) -> Tensor {
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp{ name: "Scalar".to_string() }),
        inputs: vec![],
        param: Some(NdArray::from_elem(ndarray::IxDyn(&[1]), a)),
        rank: 0,
    })))
}

/// Creates a shared variable.
#[inline]
pub fn variable(array: ndarray::Array<f32, ndarray::IxDyn>) -> Tensor {
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp{ name: "Variable".to_string() }),
        inputs: vec![],
        param: Some(array),
        rank: 0,
    })))
}
