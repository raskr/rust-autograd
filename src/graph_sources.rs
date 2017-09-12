extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use std::cell::RefCell;
use std::rc::Rc;
use tensor::{RawTensor, Tensor};


/// Constructor of a tensor placeholder.
///
/// `shape[*]` can be -1, which means dynamic dim size.
#[inline]
pub fn placeholder(shape: &[isize]) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp { name: "Placeholder".to_string() }),
        inputs: vec![],
        param: None,
        rank: 0,
    })))
}

/// Creates a shared variable.
#[inline]
pub fn variable<T: ndarray::Dimension>(array: ndarray::Array<f32, T>) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp { name: "Variable".to_string() }),
        inputs: vec![],
        param: Some(array.into_dyn()),
        rank: 0,
    })))
}

/// Returns a constant tensor
pub fn zeros(shape: &[usize]) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp { name: "Zeros".to_string() }),
        inputs: vec![],
        param: Some(::init::zeros(shape)),
        rank: 0,
    })))
}

/// Returns a constant tensor
pub fn ones(shape: &[usize]) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp { name: "Ones".to_string() }),
        inputs: vec![],
        param: Some(::init::ones(shape)),
        rank: 0,
    })))
}

/// Creates a constant tensor.
#[inline]
pub fn constant<T: ndarray::Dimension>(array: ndarray::Array<f32, T>) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp { name: "Constant".to_string() }),
        inputs: vec![],
        param: Some(array.into_dyn()),
        rank: 0,
    })))
}

/// Creates a constant tensor.
#[inline]
pub fn scalar(a: f32) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp { name: "Scalar".to_string() }),
        inputs: vec![],
        param: Some(NdArray::from_elem(ndarray::IxDyn(&[1]), a)),
        rank: 0,
    })))
}

/// Creates a constant tensor.
#[inline]
pub fn range(start: usize, end: usize, step: usize) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp { name: "Scalar".to_string() }),
        inputs: vec![],
        param: Some(
            ndarray::Array1::range(start as f32, end as f32, step as f32).into_dyn(),
        ),
        rank: 0,
    })))
}

/// Outputs values sampled from the normal distribution.
pub fn random_normal(shape: &[usize], mean: f64, stddev: f64) -> Tensor
{
    let op = ops::random_ops::RandomNormal {
        shape: shape.to_vec(),
        mean: mean,
        stddev: stddev,
    };
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(op),
        inputs: vec![],
        param: None,
        rank: 0,
    })))
}

/// Outputs values sampled from the uniform distribution.
pub fn random_uniform(shape: &[usize], min: f64, max: f64) -> Tensor
{
    let op = ops::random_ops::RandomUniform {
        shape: shape.to_vec(),
        min: min,
        max: max,
    };
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(op),
        inputs: vec![],
        param: None,
        rank: 0,
    })))
}

/// Outputs values sampled from the standard normal distribution.
pub fn standard_normal(shape: &[usize]) -> Tensor
{
    let op = ops::random_ops::StandardNormal { shape: shape.to_vec() };
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(op),
        inputs: vec![],
        param: None,
        rank: 0,
    })))
}

/// Outputs values sampled from the standard uniform distribution.
pub fn standard_uniform(shape: &[usize]) -> Tensor
{
    let op = ops::random_ops::StandardUniform { shape: shape.to_vec() };
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(op),
        inputs: vec![],
        param: None,
        rank: 0,
    })))
}

/// Outputs values sampled from the bernoulli distribution.
pub fn bernoulli(shape: &[usize], p: f64) -> Tensor
{
    let op = ops::random_ops::Bernoulli {
        shape: shape.to_vec(),
        p: p,
    };
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(op),
        inputs: vec![],
        param: None,
        rank: 0,
    })))
}

/// Outputs values sampled from the exponential distribution.
pub fn random_exp(shape: &[usize], lambda: f64) -> Tensor
{
    let op = ops::random_ops::Exponential {
        shape: shape.to_vec(),
        lambda: lambda,
    };
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(op),
        inputs: vec![],
        param: None,
        rank: 0,
    })))
}

/// Outputs values sampled from the gamma distribution.
pub fn gamma(shape: &[usize], shape_param: f64, scale: f64) -> Tensor
{
    let op = ops::random_ops::Gamma {
        shape: shape.to_vec(),
        shape_param: shape_param,
        scale: scale,
    };
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(op),
        inputs: vec![],
        param: None,
        rank: 0,
    })))
}

/// Outputs values sampled from the log normal distribution.
pub fn log_normal(shape: &[usize], mean: f64, stddev: f64) -> Tensor
{
    let op = ops::random_ops::LogNormal {
        shape: shape.to_vec(),
        mean: mean,
        stddev: stddev,
    };
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(op),
        inputs: vec![],
        param: None,
        rank: 0,
    })))
}
