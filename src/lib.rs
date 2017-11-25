#[macro_use(s)]
extern crate ndarray;

#[macro_use]
pub mod test_helper;

pub mod tensor;

#[doc(hidden)]
pub mod eval;

#[doc(hidden)]
pub mod gradient;

pub mod context;

pub mod ops;

pub mod gradient_descent;

pub mod ndarray_ext;

pub use ndarray_ext::array_gen;

pub use tensor::Tensor;

pub use ops::*;

pub use context::Context;

#[doc(hidden)]
pub use ndarray_ext::NdArray;

pub use eval::eval;

pub use eval::run;
