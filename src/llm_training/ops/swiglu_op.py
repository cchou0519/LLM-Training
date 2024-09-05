import torch
import torch.nn.functional as F

swiglu_fwd_codestring = """
template <typename T> T swiglu_fwd(T x, T y) {
    return float(x) * float(y) / (1.0f + ::exp(-float(x)));
}
"""
swiglu_bwd_codestring = """
template <typename T> T swiglu_bwd(T x, T y, T g, T& dx, T& dy) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = float(x) * x_sigmoid * float(g);
}
"""
swiglu_fwd = torch.cuda.jiterator._create_jit_fn(swiglu_fwd_codestring)
swiglu_bwd = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_bwd_codestring, num_outputs=2)


class SwiGLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return swiglu_fwd(x, y)

    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        return swiglu_bwd(x, y, dout)


def swiglu(
    x: torch.Tensor,
    *,
    w3: torch.Tensor,
    w1w2: torch.Tensor | None = None,
    w1: torch.Tensor | None = None,
    w2: torch.Tensor | None = None
) -> torch.Tensor:
    assert (
        w1w2 is not None and w1 is None and w2 is None
        or w1w2 is None and w1 is not None and w2 is not None
    )

    if w1w2 is not None:
        x1x2 = F.linear(x, w1w2)
        x1, x2 = torch.chunk(x1x2, chunks=2, dim=-1)
    else:
        x1 = F.linear(x, w1)
        x2 = F.linear(x, w2)
    
    return F.linear(SwiGLUFunction.apply(x1, x2), w3)
