using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.functions
{
    public class ReLU: FunctionBase<ReLU>
    {
        protected override Variable _forward(List<Variable> inputs)
        {
            var input = inputs[0].Value.Clone();
            var result = input.Map(x => Math.Max(x, 0));
            return new Variable(result);
        }

        protected override List<Matrix<float>> _backward(List<Matrix<float>> inputs, Matrix<float> gy)
        {
            var y = Output.Value;
            var y_positive = y.Clone().Map(x => x > 0 ? 1f : 0f);;
            return new List<Matrix<float>>() {gy.PointwiseMultiply(y_positive)};
        }
    }
}