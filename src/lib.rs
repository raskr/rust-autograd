#[macro_use(s)]
extern crate ndarray;
extern crate rand;

#[macro_use]
#[doc(hidden)]
pub mod test_helper;

pub mod tensor;

#[doc(hidden)]
pub mod runtime;

#[doc(hidden)]
pub mod gradient;

pub mod ops;

pub mod ndarray_ext;

pub mod errors;

pub mod op;

pub use ndarray_ext::array_gen;

pub use tensor::Tensor;

pub use ops::*;

pub use ops::gradient_descent_ops;

pub use errors::*;

#[doc(hidden)]
pub use ndarray_ext::NdArray;

pub use runtime::eval;

pub use runtime::run;
