# TinyChainerSharp
Pure C# Reimplementation of [chainer](https://github.com/chainer/chainer), works with Unity.

Currently, this only support full-connected layers.

Dependencies
==============

- Math.NET Numerics
  - https://github.com/mathnet/mathnet-numerics
- SimpleJSON
  - https://github.com/Bunny83/SimpleJSON

Usage
===============

backward
-----------

```csharp
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

Chain & Optimizer
-------------------

```csharp
internal class LogicalOperationChain : Chain
{
    public LogicalOperationChain(
    ) : base(new Dictionary<string, Link>()
    {
        {"fc1", new links.Linear(2, 6)},
        {"fc2", new links.Linear(6, 1)}
    })
    {
    }

    public Variable Forward(Variable x)
    {
        var h = x;
        h = functions.Sigmoid.ForwardStatic(Children["fc1"].Forward(h));
        h = Children["fc2"].Forward(h);
        return h;
    }
}
```

```csharp
var data = new Matrix<float>[,] // XOR logic
{
    {
        builder.DenseOfArray(new float[,] {{1, 1}}),
        builder.DenseOfArray(new float[,] {{0}}),
    },
    {
        builder.DenseOfArray(new float[,] {{0, 1}}),
        builder.DenseOfArray(new float[,] {{1}}),
    },
    {
        builder.DenseOfArray(new float[,] {{1, 0}}),
        builder.DenseOfArray(new float[,] {{1}}),
    },
    {
        builder.DenseOfArray(new float[,] {{0, 0}}),
        builder.DenseOfArray(new float[,] {{0}}),
    },
};

var logic = new LogicalOperationChain();
var optimizer = new SGD(lr: 0.5f);
optimizer.Setup(logic);

for (int epoch = 0; epoch < 300; epoch++)
{
    for (int i = 0; i < data.GetLength(0); i++)
    {
        var input = new Variable(data[i, 0]);
        var output = new Variable(data[i, 1]);
        var loss = functions.MeanSquaredError.ForwardStatic(
            logic.Forward(input),
            output
        );
        optimizer.ZeroGrads();
        loss.Backward();
        optimizer.Update();
    }
}
```

References
====================
- Tokui, S., Oono, K., Hido, S. and Clayton, J., Chainer: a Next-Generation Open Source Framework for Deep Learning, Proceedings of Workshop on Machine Learning Systems(LearningSys) in The Twenty-ninth Annual Conference on Neural Information Processing Systems (NIPS), (2015)
- Chainer
  - https://github.com/chainer/chainer

