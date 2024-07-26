class Net(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  fc1 : __torch__.torch.nn.modules.linear.Linear
  fc2 : __torch__.torch.nn.modules.linear.___torch_mangle_0.Linear
  def forward(self: __torch__.Net,
    x: Tensor) -> Tensor:
    fc2 = self.fc2
    fc1 = self.fc1
    input = torch.relu((fc1).forward(x, ))
    _0 = torch.softmax((fc2).forward(input, ), 1)
    return _0
