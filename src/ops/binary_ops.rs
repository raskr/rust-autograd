/// Implement +, -, *, / operators for Tensor
extern crate ndarray;

use std::ops::{Add, Sub, Mul, Div};
use std::rc::Rc;
use std::cell::RefCell;
use tensor::{Tensor, RawTensor};
use ndarray_ext::NdArray;
use ops;


#[inline(always)]
fn scalar_tensor_add(a: &NdArray, b: &NdArray) -> NdArray {
    // comparing the rank of tensors doesn't solve the problem here.
    let len_a = a.len();
    let len_b = b.len();
    if len_a > len_b { a + b } else { b + a }
}

#[inline(always)]
fn scalar_tensor_mul(a: &NdArray, b: &NdArray) -> NdArray {
    // comparing the rank of tensors doesn't solve the problem here.
    let len_a = a.len();
    let len_b = b.len();
    if len_a > len_b { a * b } else { b * a }
}


pub struct ElementwiseAdd;
pub struct ElementwiseSub;
pub struct ElementwiseMul;
pub struct ElementwiseDiv;


impl ops::Op for ElementwiseAdd {
    fn name(&self) -> &str {
        "ElementwiseAdd"
    }

    fn compute(&mut self, mut xs: &[&NdArray], _: bool) -> NdArray {
        let a = xs[0];
        let b = xs[1];
        scalar_tensor_add(a, b)
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(gy.clone()), Some(gy.clone())]
    }
}


impl ops::Op for ElementwiseSub {
    fn name(&self) -> &str {
        "ElementwiseSub"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        let a = xs[0];
        let b = xs[1];
        if a.shape() == &[1] {  // a is scalar
            // unwrap is safe
            let a = NdArray::from_elem(b.shape(), *a.get(0).unwrap());
            a - b
        } else {
            a - b
        }
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(gy.clone()), Some(-1 * gy)]
    }
}

impl ops::Op for ElementwiseMul {
    fn name(&self) -> &str {
        "ElementwiseMul"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        let a = xs[0];
        let b = xs[1];
        scalar_tensor_mul(a, b)
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        let x0 = inputs[0];
        let x1 = inputs[1];
        vec![Some(gy * x1), Some(gy * x0)]
    }
}

impl ops::Op for ElementwiseDiv {
    fn name(&self) -> &str {
        "ElementwiseDiv"
    }

    fn compute(&mut self, mut xs: &[&NdArray], _: bool) -> NdArray {
        let x0 = xs[0];
        let x1 = xs[1];
        if x0.shape() == &[1] {
            // a is scalar
            // unwrap is safe
            NdArray::from_elem(x1.shape(), *x0.get(0).unwrap()) / x1
        } else {
            x0 / x1
        }
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        let x0 = inputs[0];
        let x1 = inputs[1];
        let a: Tensor = gy / x1;
        let b: Tensor = -1 * x0 * ops::pow(x1, -2.) * gy;
        vec![Some(a), Some(b)]
    }
}


fn f32_to_tensor(arg: f32) -> Tensor {
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp{ name: "from_f32".to_string() }),
        inputs: vec![],
        param: Some(NdArray::from_elem(ndarray::IxDyn(&[1]), arg)),
        rank: 0,
    })))
}

fn f64_to_tensor(arg: f64) -> Tensor {
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp{ name: "from_f64".to_string() }),
        inputs: vec![],
        param: Some(NdArray::from_elem(ndarray::IxDyn(&[1]), arg as f32)),
        rank: 0,
    })))
}

fn i32_to_tensor(arg: i32) -> Tensor {
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp{ name: "from_i32".to_string() }),
        inputs: vec![],
        param: Some(NdArray::from_elem(ndarray::IxDyn(&[1]), arg as f32)),
        rank: 0,
    })))
}

fn i64_to_tensor(arg: i64) -> Tensor {
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp{ name: "from_i64".to_string() }),
        inputs: vec![],
        param: Some(NdArray::from_elem(ndarray::IxDyn(&[1]), arg as f32)),
        rank: 0,
    })))
}

fn u32_to_tensor(arg: u32) -> Tensor {
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp{ name: "from_u32".to_string() }),
        inputs: vec![],
        param: Some(NdArray::from_elem(ndarray::IxDyn(&[1]), arg as f32)),
        rank: 0,
    })))
}

fn u64_to_tensor(arg: u64) -> Tensor {
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp{ name: "from_u64".to_string() }),
        inputs: vec![],
        param: Some(NdArray::from_elem(ndarray::IxDyn(&[1]), arg as f32)),
        rank: 0,
    })))
}

fn usize_to_tensor(arg: usize) -> Tensor {
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp{ name: "from_usize".to_string() }),
        inputs: vec![],
        param: Some(NdArray::from_elem(ndarray::IxDyn(&[1]), arg as f32)),
        rank: 0,
    })))
}

fn isize_to_tensor(arg: isize) -> Tensor {
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(ops::dummy_op::DummyOp{ name: "from_isize".to_string() }),
        inputs: vec![],
        param: Some(NdArray::from_elem(ndarray::IxDyn(&[1]), arg as f32)),
        rank: 0,
    })))
}


// -- std::ops::{Add, Sub, Mul, Div} implementations --


macro_rules! impl_elementwise_between_tensor_and_scalar {
    (
        $trt:ident,
        $func:ident,
        $op:ident,
        $scalar_type:ty,
        $conversion_fn:ident
    ) => {

        // scalar op Tensor
        impl $trt<Tensor> for $scalar_type {
            type Output = Tensor;
            fn $func(self, rhs: Tensor) -> Self::Output
            {
                ops::apply_op($op, &[&$conversion_fn(self), &rhs])
            }
        }

        // scalar op &Tensor
        impl<'a> $trt<&'a Tensor> for $scalar_type {
            type Output = Tensor;
            fn $func(self, rhs: &'a Tensor) -> Self::Output
            {
                ops::apply_op($op, &[&$conversion_fn(self), rhs])
            }
        }

        // Tensor op scalar
        impl $trt<$scalar_type> for Tensor {
            type Output = Tensor;
            fn $func(self, rhs: $scalar_type) -> Self::Output
            {
                ops::apply_op($op, &[&self, &$conversion_fn(rhs)])
            }
        }

        // &Tensor op scalar
        impl<'a> $trt<$scalar_type> for &'a Tensor {
            type Output = Tensor;
            fn $func(self, rhs: $scalar_type) -> Self::Output
            {
                ops::apply_op($op, &[self, &$conversion_fn(rhs)])
            }
        }
    }
}

macro_rules! impl_elementwise_between_two_tensors {
    (
        $trt:ident,
        $func:ident,
        $op:ident
    ) => {
        // Tensor op Tensor
        impl $trt for Tensor {
            type Output = Tensor;
            fn $func(self, rhs: Tensor) -> Self::Output
            {
                ops::apply_op($op, &[&self, &rhs])
            }
        }

        // Tensor op &Tensor
        impl<'a> $trt<&'a Tensor> for Tensor {
            type Output = Tensor;
            fn $func(self, rhs: &Tensor) -> Self::Output
            {
                ops::apply_op($op, &[&self, rhs])
            }
        }

        // &Tensor op Tensor
        impl<'a> $trt<Tensor> for &'a Tensor {
            type Output = Tensor;
            fn $func(self, rhs: Tensor) -> Self::Output
            {
                ops::apply_op($op, &[self, &rhs])
            }
        }

        // &Tensor op &Tensor
        // lifetime of the two tensors are unrelated
        impl<'a, 'b> $trt<&'a Tensor> for &'b Tensor {
            type Output = Tensor;
            fn $func(self, rhs: &Tensor) -> Self::Output
            {
                ops::apply_op($op, &[self, rhs])
            }
        }
    };
}



impl_elementwise_between_two_tensors!(Add, add, ElementwiseAdd);
impl_elementwise_between_two_tensors!(Sub, sub, ElementwiseSub);
impl_elementwise_between_two_tensors!(Mul, mul, ElementwiseMul);
impl_elementwise_between_two_tensors!(Div, div, ElementwiseDiv);

impl_elementwise_between_tensor_and_scalar!(Add, add, ElementwiseAdd, i32, i32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, ElementwiseSub, i32, i32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, ElementwiseMul, i32, i32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, ElementwiseDiv, i32, i32_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, ElementwiseAdd, i64, i64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, ElementwiseSub, i64, i64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, ElementwiseMul, i64, i64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, ElementwiseDiv, i64, i64_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, ElementwiseAdd, f32, f32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, ElementwiseSub, f32, f32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, ElementwiseMul, f32, f32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, ElementwiseDiv, f32, f32_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, ElementwiseAdd, f64, f64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, ElementwiseSub, f64, f64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, ElementwiseMul, f64, f64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, ElementwiseDiv, f64, f64_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, ElementwiseAdd, u32, u32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, ElementwiseSub, u32, u32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, ElementwiseMul, u32, u32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, ElementwiseDiv, u32, u32_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, ElementwiseAdd, u64, u64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, ElementwiseSub, u64, u64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, ElementwiseMul, u64, u64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, ElementwiseDiv, u64, u64_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, ElementwiseAdd, usize, usize_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, ElementwiseSub, usize, usize_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, ElementwiseMul, usize, usize_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, ElementwiseDiv, usize, usize_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, ElementwiseAdd, isize, isize_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, ElementwiseSub, isize, isize_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, ElementwiseMul, isize, isize_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, ElementwiseDiv, isize, isize_to_tensor);