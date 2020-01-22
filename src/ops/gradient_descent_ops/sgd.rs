use crate::Float;

pub(crate) struct SGDOp<T: Float> {
    pub lr: T,
}

impl<T: Float> SGDOp<T> {
    pub(crate) fn new(lr: T) -> Self {
        SGDOp { lr }
    }
}

impl<T: Float> crate::op::Op<T> for SGDOp<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        ctx.input_mut(0).scaled_add(-self.lr, &ctx.input(1));
        ctx.append_output(Err(crate::op::ComputeException::NoOutput));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None])
    }
}
