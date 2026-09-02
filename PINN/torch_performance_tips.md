


## Q: what does contiguous mean?

“Contiguous” means the tensor’s elements are laid out in memory in the simple expected order, with no gaps or strided jumping.

For a 2D tensor shaped (N, 3), a contiguous layout means memory is arranged roughly like:

[[a, b, c],
 [d, e, f],
 [g, h, i]]

stored as:

a b c d e f g h i

Some PyTorch operations, especially slicing, don’t copy data. They create a view into the same memory with strides. That view can be
non-contiguous.

Example:

x = torch.randn(5, 3)
print(x.is_contiguous())      # True

y = x[:, -1]
print(y.is_contiguous())      # Often False

Why? Because y is the last column of a 2D array, so in memory PyTorch reaches elements by stepping every 3 values, not by reading one
flat block.

So:

- contiguous tensor = one clean block of memory
- non-contiguous tensor = a view that uses strides to skip around memory

Why it matters:

- most ops work fine on non-contiguous tensors
- some ops need contiguous memory internally
- then PyTorch makes a copy, which costs time and memory


