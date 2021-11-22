use crate::Float;

// mutable op
pub(crate) struct SGDOp<F> {
    pub(crate) alpha: F,
}

// mutable op
pub(crate) struct MomentumSGDOp<T: Float> {
    pub(crate) lr: T,
    pub(crate) momentum: T,
}

impl<F: Float> crate::op::Op<F> for SGDOp<F> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<F>) -> Result<(), crate::op::OpError> {
        let mut var = ctx.input_mut(0);
        let update = ctx.input(1);
        var.zip_mut_with(&update, move |l, &r| *l -= self.alpha * r);
        ctx.append_empty_output();
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<F>) {
        ctx.append_input_grad(None);
        ctx.append_input_grad(None);
    }
}

impl<T: Float> crate::op::Op<T> for MomentumSGDOp<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let mut param = ctx.input_mut(0);
        let grad = ctx.input(1);
        let mut v = ctx.input_mut(2);

        v.zip_mut_with(&grad, move |v, &g| *v = *v * self.momentum - self.lr * g);
        param.zip_mut_with(&v, move |p, &v| *p += v);
        ctx.append_empty_output();
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
        ctx.append_input_grad(None);
    }
}
