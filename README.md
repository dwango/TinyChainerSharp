# ChainerCSharp
Pure C# Reimplementation of chainer, works with unity

Usage
===============

backward
-----------

```
var x = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose());
var constant = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose());
var target = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}).Transpose());
var loss = functions.MeanSquaredError.ForwardStatic(
    x + constant,
    target
);
Assert.IsNull(x.Grad);
loss.Backward();
Assert.IsNotNull(x.Grad);
```

