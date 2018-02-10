
use ndarray_ext::NdArray;
use tensor::Tensor;

macro_rules! impl_op {
    ($name:ident) => {
        pub struct $name;
        impl ::ops::Op for $name {
            fn name(&self) -> &str
            {
                stringify!($name)
            }

            fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
            {
                unreachable!()
            }

            fn compute(&self, _: ::runtime::OpComputeContext)
                -> Result<NdArray, ::errors::OpComputeErrorStatus>
            {
                unreachable!()
            }
        }
    }
}

impl_op!(Variable);
impl_op!(Const);
impl_op!(Placeholder);
