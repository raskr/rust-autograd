use crate::op;
use crate::tensor::Tensor;
use crate::Float;

macro_rules! impl_op {
    ($name:ident) => {
        pub struct $name;
        impl<T: Float> crate::op::Op<T> for $name {
            fn name(&self) -> &str {
                stringify!($name)
            }

            fn grad(
                &self,
                _: &Tensor<T>,
                _: &[&Tensor<T>],
                _: &Tensor<T>,
            ) -> Vec<Option<Tensor<T>>> {
                unreachable!()
            }

            fn compute<'v>(
                &self,
                _: crate::runtime::OpComputeContext<'v, T>,
            ) -> op::ComputeResults<'v, T> {
                unreachable!()
            }
        }
    };
}

impl_op!(Variable);
impl_op!(Const);
impl_op!(Placeholder);
