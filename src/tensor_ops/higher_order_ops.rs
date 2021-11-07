use crate::Float;
use crate::{op, NdArray, NdArrayView};
use std::marker::PhantomData;

pub(crate) struct MapOp<T: Float> {
    pub(crate) phantom: PhantomData<T>,
    pub(crate) f: fn(NdArrayView<T>) -> NdArray<T>,
}

impl<F: Float> op::Op<F> for MapOp<F> {
    fn compute(&self, ctx: &mut op::ComputeContext<F>) -> Result<(), op::OpError> {
        let f = self.f;
        let x = ctx.input(0);
        ctx.append_output(f(x));
        Ok(())
    }
    fn grad(&self, _: &mut op::GradientContext<F>) {}
}
